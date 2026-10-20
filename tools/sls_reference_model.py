"""Full-width SLS backend with self or frozen-pretrained layer routing.

No early feature projection, dual classifier, or representation matching loss.
The reference supplies gate descriptors only, never classifier values.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class XLSRStates(nn.Module):
    def __init__(self, checkpoint, trainable):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        self.model = models[0]
        for name, parameter in self.model.named_parameters():
            enabled = name.startswith(("encoder.", "post_extract_proj.", "layer_norm."))
            parameter.requires_grad_(trainable and enabled)
        self.model.eval()

    def train(self, mode=True):
        # Evaluation mode disables SSL dropout/layerdrop, not task gradients.
        super().train(False)
        return self

    def forward(self, waveform):
        result = self.model(waveform, mask=False, features_only=True)
        states = [entry[0].transpose(0, 1) for entry in result["layer_results"]]
        if len(states) != 24 or any(h.shape[-1] != 1024 for h in states):
            raise RuntimeError("Expected all 24 XLS-R layers of width 1024")
        return states


class FullWidthSLS(nn.Module):
    """Official SLS classifier dimensions/activations; gates use detached inputs."""
    def __init__(self):
        super().__init__()
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU()
        self.fc0 = nn.Linear(1024, 1)
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)

    @staticmethod
    def descriptors(states):
        return torch.stack([h.mean(dim=1) for h in states], dim=1)

    def forward(self, states, reference_descriptors=None, return_details=False):
        if len(states) != 24:
            raise ValueError("SLS requires all 24 Transformer layers")
        expected = (states[0].size(0), 201, 1024)
        if any(tuple(h.shape) != expected for h in states):
            raise ValueError(f"Expected each layer [B,201,1024], got {states[0].shape}")
        descriptors = (self.descriptors(states) if reference_descriptors is None
                       else reference_descriptors)
        if tuple(descriptors.shape) != (expected[0], 24, 1024):
            raise ValueError("Reference descriptors must have shape [B,24,1024]")
        if not bool(torch.isfinite(descriptors).all()):
            raise ValueError("Nonfinite reference descriptors")
        # Both controlled arms detach the gate-input path; gate weights train.
        gates = torch.sigmoid(self.fc0(descriptors.detach().to(states[0])))
        feature = sum(h * gates[:, i:i+1] for i, h in enumerate(states))
        pooled = F.max_pool2d(self.selu(self.first_bn(feature.unsqueeze(1))), 3, 3)
        flattened = pooled.flatten(1)
        if flattened.size(1) != 22847:
            raise RuntimeError("Unexpected SLS pooling shape; do not change audio pad")
        logits = self.selu(self.fc3(self.selu(self.fc1(flattened))))
        log_probs = F.log_softmax(logits.float(), dim=1)
        if return_details:
            return {"log_probs": log_probs, "logits": logits,
                    "gates": gates.squeeze(-1), "aggregate": feature,
                    "pooled_shape": tuple(pooled.shape)}
        return log_probs


class ReferenceRoutedSLS(nn.Module):
    def __init__(self, checkpoint=None, reference_source="pretrained_reference",
                 task=None, reference=None):
        super().__init__()
        if reference_source not in {"self_reference", "pretrained_reference"}:
            raise ValueError(reference_source)
        self.reference_source = reference_source
        self.task = task if task is not None else XLSRStates(checkpoint, trainable=True)
        self.reference = None
        if reference_source == "pretrained_reference":
            self.reference = (reference if reference is not None else
                              XLSRStates(checkpoint, trainable=False))
            self.reference.requires_grad_(False)
            self.reference.eval()
        self.backend = FullWidthSLS()

    def train(self, mode=True):
        super().train(mode)
        if self.reference is not None:
            self.reference.eval()
        return self

    @torch.no_grad()
    def extract_reference_descriptors(self, waveform):
        if self.reference is None:
            raise RuntimeError("Self-reference arm has no pretrained teacher")
        # Fairseq samples NumPy layerdrop draws even in eval mode.
        state = np.random.get_state()
        try:
            return self.backend.descriptors(self.reference(waveform))
        finally:
            np.random.set_state(state)

    def forward(self, waveform, reference_descriptors=None, return_details=False):
        states = self.task(waveform)
        if self.reference_source == "pretrained_reference":
            descriptors = (self.extract_reference_descriptors(waveform)
                           if reference_descriptors is None else reference_descriptors)
        else:
            if reference_descriptors is not None:
                raise ValueError("Cannot inject pretrained descriptors into self-reference")
            descriptors = self.backend.descriptors(states)
        return self.backend(states, descriptors, return_details)

    def trainable_payload(self, pretrained_sha256):
        """Best-model payload: no teacher, frozen base, or optimizer copies.

        Reconstruct with the original verified pretrained checkpoint first.
        Training-resume serialization is a separate trainer responsibility.
        """
        if len(pretrained_sha256) != 64:
            raise ValueError("A verified original checkpoint SHA256 is required")
        names = {name for name, p in self.task.named_parameters() if p.requires_grad}
        return {"format": "sls_reference_trainable_v1",
                "reference_source": self.reference_source,
                "pretrained_sha256": pretrained_sha256,
                "backend": self.backend.state_dict(),
                "task_trainable": {name: tensor for name, tensor in
                                   self.task.state_dict().items() if name in names}}

    def load_trainable_payload(self, saved, pretrained_sha256):
        if (saved["format"] != "sls_reference_trainable_v1" or
            saved["pretrained_sha256"] != pretrained_sha256 or
            saved["reference_source"] != self.reference_source):
            raise ValueError("Checkpoint/base/reference-source mismatch")
        expected = {name for name, p in self.task.named_parameters() if p.requires_grad}
        if set(saved["task_trainable"]) != expected:
            raise ValueError("Updated-parameter set mismatch")
        self.backend.load_state_dict(saved["backend"], strict=True)
        self.task.load_state_dict(saved["task_trainable"], strict=False)
