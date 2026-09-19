import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils


class XLSRBackbone(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        if len(models) != 1:
            raise RuntimeError(f"Expected one XLS-R model, got {len(models)}")

        self.model = models[0]

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layers"):
            raise RuntimeError("Cannot find XLS-R transformer layers")

        self.num_layers = len(self.model.encoder.layers)
        self.hidden_dim = self._infer_hidden_dim()

        print(f"Loaded XLS-R: {checkpoint_path}")
        print(f"Transformer layers: {self.num_layers}")
        print(f"Hidden dim: {self.hidden_dim}")

    def _infer_hidden_dim(self):
        if hasattr(self.model.cfg, "encoder_embed_dim"):
            return int(self.model.cfg.encoder_embed_dim)

        first_layer = self.model.encoder.layers[0]

        if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "embed_dim"):
            return int(first_layer.self_attn.embed_dim)

        return 1024

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _prepare_waveform(waveform):
        if waveform.ndim == 3:
            if waveform.size(-1) == 1:
                waveform = waveform[..., 0]
            elif waveform.size(1) == 1:
                waveform = waveform[:, 0]
            else:
                raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [B, T], got {tuple(waveform.shape)}")

        return waveform

    @torch.no_grad()
    def extract_layers(self, waveform):
        waveform = self._prepare_waveform(waveform)

        model_parameter = next(self.model.parameters())
        waveform = waveform.to(device=model_parameter.device, dtype=model_parameter.dtype)

        result = self.model(waveform, mask=False, features_only=True)

        if "layer_results" not in result:
            raise RuntimeError(
                f"XLS-R output does not contain layer_results. Keys: {list(result.keys())}"
            )

        layers = []

        for item in result["layer_results"]:
            x = item[0] if isinstance(item, (tuple, list)) else item

            if x.ndim != 3:
                raise RuntimeError(f"Unexpected XLS-R layer shape: {tuple(x.shape)}")

            if x.size(1) == waveform.size(0):
                x = x.transpose(0, 1)
            elif x.size(0) != waveform.size(0):
                raise RuntimeError(
                    f"Cannot infer batch dimension from layer shape {tuple(x.shape)}"
                )

            layers.append(x)

        if len(layers) != self.num_layers:
            raise RuntimeError(f"Expected {self.num_layers} layers, got {len(layers)}")

        return layers


class TransitionTemporalPool(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=192, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: [B, N, T, D]
        feature = self.projection(x)

        score = self.attention(feature).squeeze(-1)
        attention = torch.softmax(score.float(), dim=-1).to(feature.dtype)

        token = torch.sum(feature * attention.unsqueeze(-1), dim=2)

        return token, attention


class TransitionHead(nn.Module):
    def __init__(
        self,
        num_transitions,
        input_dim=1024,
        hidden_dim=192,
        embedding_dim=192,
        num_heads=6,
        num_encoder_layers=2,
        feedforward_dim=384,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )

        self.num_transitions = num_transitions

        self.transition_norm = nn.LayerNorm(input_dim)

        self.temporal_pool = TransitionTemporalPool(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.layer_cls = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.layer_position = nn.Parameter(
            torch.empty(1, num_transitions + 1, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.layer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )

        nn.init.normal_(self.layer_cls, mean=0.0, std=0.02)
        nn.init.normal_(self.layer_position, mean=0.0, std=0.02)

    def forward(
        self,
        transitions,
        return_details=False,
        embedding_residual=None,
    ):
        transitions = self.transition_norm(transitions)

        transition_tokens, temporal_attention = self.temporal_pool(transitions)

        batch_size = transition_tokens.size(0)

        cls = self.layer_cls.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, transition_tokens], dim=1)
        sequence = sequence + self.layer_position[:, :sequence.size(1)]

        encoded = self.layer_encoder(sequence)

        embedding = self.output(encoded[:, 0])
        if embedding_residual is not None:
            if embedding_residual.shape != embedding.shape:
                raise RuntimeError(
                    "Embedding residual shape mismatch: "
                    f"expected {tuple(embedding.shape)}, "
                    f"got {tuple(embedding_residual.shape)}"
                )
            embedding = embedding + embedding_residual
        logits = self.classifier(embedding)

        if not return_details:
            return logits, embedding

        details = {
            "temporal_attention": temporal_attention,
            "transition_tokens": transition_tokens,
            "encoded_tokens": encoded[:, 1:],
            "embedding": embedding,
        }

        return logits, embedding, details


class DualTransitionConsistencyDetector(nn.Module):
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
    ):
        super().__init__()

        self.num_transitions = num_layers - 1

        self.head_a = TransitionHead(
            num_transitions=self.num_transitions,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        self.head_b = TransitionHead(
            num_transitions=self.num_transitions,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def build_transitions(self, layers):
        transitions = [layers[i + 1] - layers[i] for i in range(len(layers) - 1)]
        transitions = torch.stack(transitions, dim=1)

        if transitions.size(1) != self.num_transitions:
            raise RuntimeError(
                f"Expected {self.num_transitions} transitions, got {transitions.size(1)}"
            )

        return transitions

    @staticmethod
    def relation_matrix(embedding):
        embedding = F.normalize(embedding.float(), p=2, dim=-1)
        return embedding @ embedding.transpose(0, 1)

    @staticmethod
    def relation_consistency_loss(embedding_a, embedding_b):
        relation_a = DualTransitionConsistencyDetector.relation_matrix(embedding_a)
        relation_b = DualTransitionConsistencyDetector.relation_matrix(embedding_b)

        return F.mse_loss(relation_a, relation_b)

    def forward_train(self, layers, return_details=False):
        transitions = self.build_transitions(layers)

        logits_a, embedding_a, details_a = self.head_a(
            transitions,
            return_details=True,
        )

        logits_b, embedding_b, details_b = self.head_b(
            transitions,
            return_details=True,
        )

        relation_loss = self.relation_consistency_loss(
            embedding_a,
            embedding_b,
        )

        if not return_details:
            return logits_a, logits_b, embedding_a, embedding_b, relation_loss

        details = {
            "relation_loss": relation_loss,
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
            "relation_a": self.relation_matrix(embedding_a),
            "relation_b": self.relation_matrix(embedding_b),
            "temporal_attention_a": details_a["temporal_attention"],
            "temporal_attention_b": details_b["temporal_attention"],
            "transition_tokens_a": details_a["transition_tokens"],
            "transition_tokens_b": details_b["transition_tokens"],
            "encoded_tokens_a": details_a["encoded_tokens"],
            "encoded_tokens_b": details_b["encoded_tokens"],
        }

        return logits_a, logits_b, embedding_a, embedding_b, relation_loss, details

    def forward(self, layers, inference_head="a", return_details=False):
        transitions = self.build_transitions(layers)

        if inference_head == "a":
            logits, embedding, details = self.head_a(
                transitions,
                return_details=True,
            )

            if return_details:
                return logits, embedding, details

            return logits

        if inference_head == "b":
            logits, embedding, details = self.head_b(
                transitions,
                return_details=True,
            )

            if return_details:
                return logits, embedding, details

            return logits

        if inference_head == "ensemble":
            logits_a, embedding_a = self.head_a(transitions)
            logits_b, embedding_b = self.head_b(transitions)

            logits = (logits_a + logits_b) / 2.0
            embedding = (embedding_a + embedding_b) / 2.0

            if return_details:
                return logits, embedding, {
                    "logits_a": logits_a,
                    "logits_b": logits_b,
                    "embedding_a": embedding_a,
                    "embedding_b": embedding_b,
                }

            return logits

        raise ValueError(f"Unknown inference_head: {inference_head}")


class LowHighConsistencyBranch(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim=1024,
        embedding_dim=192,
        split_after_layer=21,
    ):
        super().__init__()

        if not 1 <= split_after_layer < num_layers:
            raise ValueError(
                f"split_after_layer must be in [1, {num_layers - 1}], "
                f"got {split_after_layer}"
            )

        self.num_layers = int(num_layers)
        self.split_after_layer = int(split_after_layer)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(4 * input_dim),
            nn.Linear(4 * input_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    @staticmethod
    def temporal_mean_std(view):
        mean = view.mean(dim=1)
        std = view.var(dim=1, unbiased=False).add(1e-6).sqrt()
        return mean, std

    def forward(self, layers):
        if len(layers) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} XLS-R layers, got {len(layers)}"
            )

        normalized = [self.layer_norm(layer) for layer in layers]
        low_view = torch.stack(
            normalized[:self.split_after_layer],
            dim=0,
        ).mean(dim=0)
        high_view = torch.stack(
            normalized[self.split_after_layer:],
            dim=0,
        ).mean(dim=0)

        difference = high_view - low_view
        agreement = high_view * low_view
        difference_mean, difference_std = self.temporal_mean_std(difference)
        agreement_mean, agreement_std = self.temporal_mean_std(agreement)

        statistics = torch.cat(
            [
                difference_mean,
                difference_std,
                agreement_mean,
                agreement_std,
            ],
            dim=-1,
        )
        return self.output(statistics), {
            "low_view": low_view,
            "high_view": high_view,
            "difference_mean": difference_mean,
            "difference_std": difference_std,
            "agreement_mean": agreement_mean,
            "agreement_std": agreement_std,
        }


class DualTransitionLowHighConsistencyDetector(
    DualTransitionConsistencyDetector
):
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
        split_after_layer=21,
        residual_scale=0.1,
    ):
        super().__init__(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

        self.residual_scale = float(residual_scale)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.consistency_branch = LowHighConsistencyBranch(
            num_layers=num_layers,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            split_after_layer=split_after_layer,
        )

    def current_residual_strength(self):
        return self.residual_scale * torch.tanh(self.residual_gate)

    def build_residual(self, layers):
        consistency_embedding, consistency_details = self.consistency_branch(
            layers
        )
        residual = self.current_residual_strength() * consistency_embedding
        return residual, consistency_embedding, consistency_details

    def forward_train(self, layers, return_details=False):
        transitions = self.build_transitions(layers)
        residual, consistency_embedding, consistency_details = self.build_residual(
            layers
        )

        logits_a, embedding_a, details_a = self.head_a(
            transitions,
            return_details=True,
            embedding_residual=residual,
        )
        logits_b, embedding_b, details_b = self.head_b(
            transitions,
            return_details=True,
            embedding_residual=residual,
        )
        relation_loss = self.relation_consistency_loss(
            embedding_a,
            embedding_b,
        )

        if not return_details:
            return logits_a, logits_b, embedding_a, embedding_b, relation_loss

        return (
            logits_a,
            logits_b,
            embedding_a,
            embedding_b,
            relation_loss,
            {
                "relation_loss": relation_loss,
                "embedding_a": embedding_a,
                "embedding_b": embedding_b,
                "temporal_attention_a": details_a["temporal_attention"],
                "temporal_attention_b": details_b["temporal_attention"],
                "transition_tokens_a": details_a["transition_tokens"],
                "transition_tokens_b": details_b["transition_tokens"],
                "consistency_embedding": consistency_embedding,
                "consistency_details": consistency_details,
                "residual_strength": self.current_residual_strength(),
            },
        )

    def forward(self, layers, inference_head="a", return_details=False):
        transitions = self.build_transitions(layers)
        residual, consistency_embedding, consistency_details = self.build_residual(
            layers
        )

        if inference_head in {"a", "b"}:
            head = self.head_a if inference_head == "a" else self.head_b
            logits, embedding, details = head(
                transitions,
                return_details=True,
                embedding_residual=residual,
            )
            if return_details:
                details.update({
                    "consistency_embedding": consistency_embedding,
                    "consistency_details": consistency_details,
                    "residual_strength": self.current_residual_strength(),
                })
                return logits, embedding, details
            return logits

        if inference_head == "ensemble":
            logits_a, embedding_a = self.head_a(
                transitions,
                embedding_residual=residual,
            )
            logits_b, embedding_b = self.head_b(
                transitions,
                embedding_residual=residual,
            )
            logits = (logits_a + logits_b) / 2.0
            embedding = (embedding_a + embedding_b) / 2.0

            if return_details:
                return logits, embedding, {
                    "logits_a": logits_a,
                    "logits_b": logits_b,
                    "embedding_a": embedding_a,
                    "embedding_b": embedding_b,
                    "consistency_embedding": consistency_embedding,
                    "consistency_details": consistency_details,
                    "residual_strength": self.current_residual_strength(),
                }
            return logits

        raise ValueError(f"Unknown inference_head: {inference_head}")


class Model(nn.Module):
    def __init__(self, device, config):
        super().__init__()

        self.ssl_model = XLSRBackbone(config["ssl_checkpoint"])

        self.detector_type = str(config.get("detector_type", "dual_transition"))
        common_arguments = {
            "num_layers": self.ssl_model.num_layers,
            "input_dim": self.ssl_model.hidden_dim,
            "hidden_dim": int(config.get("hidden_dim", 192)),
            "embedding_dim": int(config.get("embedding_dim", 192)),
            "num_heads": int(config.get("num_heads", 6)),
            "num_encoder_layers": int(config.get("num_encoder_layers", 2)),
            "feedforward_dim": int(config.get("feedforward_dim", 384)),
            "dropout": float(config.get("dropout", 0.1)),
        }

        if self.detector_type == "dual_transition":
            self.detector = DualTransitionConsistencyDetector(**common_arguments)
        elif self.detector_type == "dual_transition_low_high_consistency":
            self.detector = DualTransitionLowHighConsistencyDetector(
                **common_arguments,
                split_after_layer=int(config.get("split_after_layer", 21)),
                residual_scale=float(config.get("residual_scale", 0.1)),
            )
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type}")

        self.inference_head = str(config.get("inference_head", "a"))

    def train(self, mode=True):
        super().train(mode)
        self.ssl_model.eval()
        return self

    def forward_train(self, waveform, return_details=False):
        self.ssl_model.eval()

        with torch.no_grad():
            layers = self.ssl_model.extract_layers(waveform)

        return self.detector.forward_train(
            layers,
            return_details=return_details,
        )

    def forward(self, waveform, return_details=False):
        self.ssl_model.eval()

        with torch.no_grad():
            layers = self.ssl_model.extract_layers(waveform)

        return self.detector(
            layers,
            inference_head=self.inference_head,
            return_details=return_details,
        )


def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    parameters = [
        parameter
        for parameter in model.detector.parameters()
        if parameter.requires_grad
    ]

    return torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
