import re

import fairseq
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cfg(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


class SSLModel(nn.Module):
    def __init__(self, device, checkpoint_path, freeze=True):
        super().__init__()
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        self.model = models[0].to(device)
        self.output_dim = 1024
        self.freeze = bool(freeze)

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def extract_layers(self, waveform):
        if waveform.ndim == 3:
            waveform = waveform[:, :, 0]
        if waveform.ndim != 2:
            raise RuntimeError(f"Expected waveform shape [B, S], got {tuple(waveform.shape)}.")

        reference = next(self.model.parameters())
        waveform = waveform.to(device=reference.device, dtype=reference.dtype)

        with torch.set_grad_enabled(not self.freeze):
            output = self.model(waveform, mask=False, features_only=True)

        layer_results = output.get("layer_results")
        if not layer_results:
            raise RuntimeError("XLS-R did not return layer_results.")

        batch_size = waveform.size(0)
        layers = []
        for item in layer_results:
            feature = item[0] if isinstance(item, (tuple, list)) else item
            if feature.ndim != 3:
                continue
            if feature.size(1) == batch_size:
                feature = feature.transpose(0, 1)
            elif feature.size(0) != batch_size:
                raise RuntimeError(f"Unexpected XLS-R layer shape: {tuple(feature.shape)}.")
            layers.append(feature)

        if len(layers) < 2:
            raise RuntimeError(f"At least two XLS-R layers are required, got {len(layers)}.")
        return layers


class LayerFusion(nn.Module):
    def __init__(self, num_layers, input_dim):
        super().__init__()
        self.num_layers = int(num_layers)
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, layers):
        if len(layers) != self.num_layers:
            raise RuntimeError(f"Expected {self.num_layers} XLS-R layers, got {len(layers)}.")

        weights = torch.softmax(self.layer_logits.float(), dim=0)
        fused = layers[0] * weights[0].to(layers[0].dtype)
        for index in range(1, self.num_layers):
            fused = fused + layers[index] * weights[index].to(layers[index].dtype)
        return self.norm(fused), weights


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden_dim
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((hidden_dim,), 0.1))

    def forward(self, inputs):
        residual = inputs
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = self.pointwise(F.gelu(self.depthwise(hidden))).transpose(1, 2)
        return residual + self.dropout(hidden) * self.scale


class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout):
        super().__init__()
        attention_dim = max(32, hidden_dim // 2)
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        scores = self.attention(inputs).squeeze(-1).float()
        weights = torch.softmax(scores, dim=1).to(inputs.dtype)
        mean = torch.sum(inputs * weights.unsqueeze(-1), dim=1)
        second = torch.sum(inputs.float().square() * weights.float().unsqueeze(-1), dim=1)
        std = torch.sqrt((second - mean.float().square()).clamp_min(1e-5)).to(inputs.dtype)
        return self.output(torch.cat([mean, std], dim=-1)), weights


class SLSBranch(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, embedding_dim, dropout):
        super().__init__()
        self.layer_fusion = LayerFusion(num_layers, input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.temporal = nn.Sequential(
            TemporalConvBlock(hidden_dim, dilation=1, dropout=dropout),
            TemporalConvBlock(hidden_dim, dilation=2, dropout=dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.pool = AttentiveStatsPooling(hidden_dim, embedding_dim, dropout)

    def forward(self, layers):
        fused, layer_weights = self.layer_fusion(layers)
        hidden = self.temporal(self.projection(fused))
        embedding, temporal_attention = self.pool(hidden)
        return embedding, layer_weights, temporal_attention


class TransitionTemporalPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        attention_dim = max(32, hidden_dim // 2)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, transitions):
        hidden = self.projection(transitions)
        scores = self.attention(hidden).squeeze(-1).float()
        weights = torch.softmax(scores, dim=-1).to(hidden.dtype)
        tokens = torch.sum(hidden * weights.unsqueeze(-1), dim=2)
        return tokens, weights


class TransitionBranch(nn.Module):
    def __init__(
        self,
        num_transitions,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_heads,
        num_encoder_layers,
        feedforward_dim,
        dropout
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.num_transitions = int(num_transitions)
        self.transition_norm = nn.LayerNorm(input_dim)
        self.temporal_pool = TransitionTemporalPooling(input_dim, hidden_dim, dropout)
        self.layer_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.layer_position = nn.Parameter(torch.zeros(1, self.num_transitions + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.layer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        nn.init.trunc_normal_(self.layer_cls, std=0.02)
        nn.init.trunc_normal_(self.layer_position, std=0.02)

    def forward(self, layers):
        if len(layers) - 1 != self.num_transitions:
            raise RuntimeError(f"Expected {self.num_transitions + 1} layers, got {len(layers)}.")

        transitions = [layers[index + 1] - layers[index] for index in range(len(layers) - 1)]
        transitions = self.transition_norm(torch.stack(transitions, dim=1))
        tokens, temporal_attention = self.temporal_pool(transitions)

        cls = self.layer_cls.expand(tokens.size(0), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        sequence = sequence + self.layer_position[:, :sequence.size(1)]
        sequence = self.layer_encoder(sequence)
        embedding = self.output(sequence[:, 0])
        return embedding, temporal_attention


class DualBranchDetector(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim=1024,
        hidden_dim=192,
        embedding_dim=192,
        num_heads=6,
        num_encoder_layers=2,
        feedforward_dim=384,
        dropout=0.1,
        gate_init=-1.5
    ):
        super().__init__()
        self.sls_branch = SLSBranch(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        self.transition_branch = TransitionBranch(
            num_transitions=num_layers - 1,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )
        self.raw_norm = nn.LayerNorm(embedding_dim)
        self.transition_norm = nn.LayerNorm(embedding_dim)
        self.transition_adapter = nn.Linear(embedding_dim, embedding_dim)
        self.fusion_gate = nn.Linear(embedding_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2)
        )

        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, float(gate_init))

    def forward(self, layers, return_details=False):
        raw_embedding, layer_weights, raw_attention = self.sls_branch(layers)
        transition_embedding, transition_attention = self.transition_branch(layers)

        raw_feature = self.raw_norm(raw_embedding)
        transition_feature = self.transition_norm(transition_embedding)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([raw_feature, transition_feature], dim=-1)))
        fused_embedding = raw_embedding + gate * self.transition_adapter(transition_feature)
        logits = self.classifier(fused_embedding)

        if return_details:
            details = {
                "fusion_gate": gate,
                "sls_layer_weights": layer_weights,
                "sls_temporal_attention": raw_attention,
                "transition_temporal_attention": transition_attention,
                "sls_embedding": raw_embedding,
                "transition_embedding": transition_embedding
            }
            return logits, fused_embedding, details
        return logits, fused_embedding


class Model(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        checkpoint_path = get_cfg(config, "xlsr_ckpt_path", "/root/rivermind-fs/xlsr2_300m.pt")
        freeze_ssl = as_bool(get_cfg(config, "freeze_ssl", True))
        hidden_dim = int(get_cfg(config, "hidden_dim", 192))
        embedding_dim = int(get_cfg(config, "embedding_dim", 192))
        num_heads = int(get_cfg(config, "num_heads", 6))
        num_encoder_layers = int(get_cfg(config, "num_encoder_layers", 2))
        feedforward_dim = int(get_cfg(config, "feedforward_dim", hidden_dim * 2))
        dropout = float(get_cfg(config, "dropout", 0.1))
        gate_init = float(get_cfg(config, "fusion_gate_init", -1.5))

        self.ssl_model = SSLModel(device, checkpoint_path, freeze=freeze_ssl)
        try:
            self.num_layers = len(self.ssl_model.model.encoder.layers)
        except (AttributeError, TypeError):
            self.num_layers = 24

        self.detector = DualBranchDetector(
            num_layers=self.num_layers,
            input_dim=self.ssl_model.output_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            gate_init=gate_init
        )

    def forward(self, waveform, return_embedding=False, return_details=False):
        layers = self.ssl_model.extract_layers(waveform)
        logits, embedding, *extra = self.detector(layers, return_details=return_details)

        if return_details:
            return logits, embedding, extra[0]
        if return_embedding:
            return logits, embedding
        return logits


def build_optimizer(model, ssl_lr=5e-7, head_lr=1e-4, layer_decay=0.95, weight_decay=1e-4):
    groups = {}
    used = set()
    last_layer = max(0, model.num_layers - 1)

    def add_parameter(parameter, lr, decay):
        if not parameter.requires_grad or id(parameter) in used:
            return
        groups.setdefault((float(lr), float(decay)), []).append(parameter)
        used.add(id(parameter))

    for name, parameter in model.ssl_model.model.named_parameters():
        match = re.search(r"encoder\.layers\.(\d+)\.", name)
        if match:
            depth = last_layer - int(match.group(1))
            lr = ssl_lr * layer_decay ** max(0, depth)
        else:
            lr = ssl_lr * layer_decay ** (last_layer + 1)
        decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else weight_decay
        add_parameter(parameter, lr, decay)

    for name, parameter in model.detector.named_parameters():
        decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else weight_decay
        add_parameter(parameter, head_lr, decay)

    parameter_groups = [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in groups.items()
    ]
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were found.")
    return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.98), eps=1e-8)