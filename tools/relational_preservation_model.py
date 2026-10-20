"""Source-only XLS-R adaptation with bona fide temporal relation preservation.

Independent of the dirty project model/trainer. Uses the existing raw-layer
dual head unchanged; the teacher is used only for a training regularizer.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fairseq import checkpoint_utils
from src.model import DualTransitionConsistencyDetector


def temporal_relation(hidden):
    """[B,T,D] -> centered cosine relations [B,T,T], computed in FP32."""
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        centered = hidden.float() - hidden.float().mean(dim=1, keepdim=True)
        unit = F.normalize(centered, dim=-1, eps=1e-6)
        return unit @ unit.transpose(1, 2)


def relation_errors(task, reference):
    delta = temporal_relation(task) - temporal_relation(reference.detach())
    t = delta.size(-1)
    if t < 2:
        raise ValueError("At least two frames are required")
    mask = ~torch.eye(t, device=delta.device, dtype=torch.bool)
    return delta.square()[:, mask].mean(dim=1)


def raw_layers(self, layers):
    tokens = torch.stack(layers[1:], dim=1)
    if tokens.size(1) != self.num_transitions:
        raise ValueError(tuple(tokens.shape))
    return tokens


class SSLPath(nn.Module):
    def __init__(self, path, trainable=False):
        super().__init__()
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([path])
        self.model = models[0]
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_dim = int(self.model.cfg.encoder_embed_dim)
        for name, p in self.model.named_parameters():
            # Convolutional extractor/unused pretraining heads remain frozen.
            enabled = name.startswith(("encoder.", "post_extract_proj.", "layer_norm."))
            p.requires_grad_(bool(trainable and enabled))
        self.model.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def layers(self, waveform):
        result = self.model(waveform, mask=False, features_only=True)
        states = [item[0].transpose(0, 1) for item in result["layer_results"]]
        if len(states) != self.num_layers:
            raise RuntimeError("Missing Transformer layers")
        return states

    @torch.no_grad()
    def boundary(self, waveform, layer):
        captured = []
        handle = self.model.encoder.layers[layer - 1].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[0])
        )
        # Fairseq draws numpy layerdrop numbers even in eval mode. Preserve the
        # caller's RNG so teacher diagnostics cannot perturb unrelated draws.
        numpy_state = np.random.get_state()
        try:
            result = self.model(waveform, mask=False, features_only=True, layer=layer - 1)
        finally:
            handle.remove()
            np.random.set_state(numpy_state)
        if len(captured) != 1:
            raise RuntimeError("Teacher did not reach the requested boundary")
        # Hook captures the exact pre-final-normalization block output. Remove
        # any internal Fairseq sequence padding, just as layer_results does.
        return captured[0][:result["x"].size(1)].transpose(0, 1)


class RelationalDetector(nn.Module):
    def __init__(self, config, arm, training_reference=True):
        super().__init__()
        self.arm = arm
        self.task = SSLPath(config["ssl_checkpoint"], trainable=arm in {"ft", "rp"})
        self.teacher = (SSLPath(config["ssl_checkpoint"]) if
                        training_reference and arm in {"ft", "rp"} else None)
        self.anchor_layer = int(config["anchor_layer"])
        self.detector = DualTransitionConsistencyDetector(
            num_layers=self.task.num_layers, input_dim=self.task.hidden_dim,
            hidden_dim=int(config["hidden_dim"]), embedding_dim=int(config["embedding_dim"]),
            num_heads=int(config["num_heads"]), num_encoder_layers=int(config["num_encoder_layers"]),
            feedforward_dim=int(config["feedforward_dim"]), dropout=float(config["dropout"]),
        )
        self.detector.build_transitions = types.MethodType(raw_layers, self.detector)

    def forward_train(self, waveform, labels):
        states = self.task.layers(waveform)
        logits_a, logits_b, emb_a, emb_b, head_relation = self.detector.forward_train(states)
        real = labels == 1
        error_sum = logits_a.new_zeros((), dtype=torch.float32)
        if self.teacher is not None and bool(real.any()):
            reference = self.teacher.boundary(waveform[real], self.anchor_layer)
            error_sum = relation_errors(states[self.anchor_layer - 1][real], reference).sum()
        return logits_a, logits_b, head_relation, error_sum

    def predict(self, waveform):
        states = self.task.layers(waveform)
        return self.detector(states, inference_head="ensemble", return_details=True)[2]

    def payload(self):
        value = {"detector": self.detector.state_dict()}
        if self.arm in {"ft", "rp"}:
            value["task_ssl"] = self.task.model.state_dict()
        return value

    def load_payload(self, saved):
        self.detector.load_state_dict(saved["detector"], strict=True)
        if "task_ssl" in saved:
            self.task.model.load_state_dict(saved["task_ssl"], strict=True)


def loss_terms(model, outputs, labels, config, real_fraction):
    a, b, head_relation, temporal_sum = outputs
    weights = labels.new_tensor([config["fake_weight"], config["real_weight"]], dtype=torch.float32)
    ce_a = F.cross_entropy(a.float(), labels, weight=weights)
    ce_b = F.cross_entropy(b.float(), labels, weight=weights)
    classification = (ce_a + ce_b) * 0.5
    temporal = temporal_sum / (labels.numel() * real_fraction)
    coefficient = float(config["lambda_temporal"]) if model.arm == "rp" else 0.0
    total = classification + float(config["lambda_head_relation"]) * head_relation + coefficient * temporal
    return {"train_loss": total, "classification_loss": classification,
            "ce_a": ce_a, "ce_b": ce_b, "head_relation_loss": head_relation,
            "temporal_scaled": temporal, "temporal_sum": temporal_sum}
