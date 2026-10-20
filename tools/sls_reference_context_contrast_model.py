"""Unchanged native classifier and QK guidance; training-only context readout."""
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


class ReferenceContextContrastSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.register_buffer("context_center", torch.zeros(24, 1024), persistent=False)
        self.center_ready = False
        self.capture_aux = False
        self.aux_features = self.context_descriptors = None
        owner = weakref.ref(self)

        def capture(_module, _args, output):
            model = owner()
            if model.capture_aux:
                model.aux_features = model.backend.selu(output)

        self._handles.append(self.backend.fc1.register_forward_hook(capture))

    @torch.no_grad()
    def reference_descriptors(self):
        if len(self._reference_inputs) != 24:
            raise RuntimeError("Actual 24-layer reference input cache missing")
        return torch.stack([h.float().mean(dim=0) for h, _ in self._reference_inputs], dim=1)

    def extract_reference(self, wave):
        super().extract_reference(wave)
        if self.capture_aux:
            if not self.center_ready:
                raise RuntimeError("Source-real reference mean not fitted")
            self.context_descriptors = F.normalize(
                self.reference_descriptors()-self.context_center[None], dim=-1, eps=1e-6).detach()

    def auxiliary_forward(self, wave):
        if not self.training:
            raise RuntimeError("Auxiliary matching is training only")
        self.aux_features = self.context_descriptors = None
        self.capture_aux = True
        try:
            output = super().forward(wave)
        finally:
            self.capture_aux = False
        if self.aux_features is None or self.context_descriptors is None:
            raise RuntimeError("Auxiliary feature wiring missing")
        # Outer BF16 autocast must not quantize the weak context-similarity differences.
        with torch.autocast(device_type=wave.device.type, enabled=False):
            c = self.context_descriptors.float()
            u = ((torch.einsum("ild,jld->ij", c, c)/24 + 1)/2).clamp(0, 1).detach()
        return output, self.aux_features, u

    def clear_aux(self):
        self.aux_features = self.context_descriptors = None

    def trainable_payload(self, sha):
        saved = super().trainable_payload(sha)
        saved.update(format="sls_reference_context_contrast_v1", context_center=self.context_center,
                     center_ready=self.center_ready)
        return saved

    def load_trainable_payload(self, saved, sha):
        if saved["format"] != "sls_reference_context_contrast_v1":
            raise ValueError("Wrong context-contrast payload")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_attention_v1"), sha)
        self.context_center.copy_(saved["context_center"].to(self.context_center))
        self.center_ready = bool(saved["center_ready"])
