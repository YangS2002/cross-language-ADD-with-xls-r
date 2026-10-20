"""Full-width SLS with genuine-source conditional residual correction."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools.sls_reference_model import XLSRStates, FullWidthSLS


class ConditionalResidualSLS(nn.Module):
    def __init__(self, checkpoint, arm):
        super().__init__()
        if arm not in {"sls", "self_residual", "pretrained_residual"}:
            raise ValueError(arm)
        self.arm = arm
        self.task = XLSRStates(checkpoint, True)
        self.backend = FullWidthSLS()
        self.predictor = self.update = self.reference = None
        self.correction_enabled = True
        self.reference_calls = []
        if arm != "sls":
            self.predictor = nn.Linear(1024, 1024)
            self.update = nn.Linear(1024, 1024)
            nn.init.zeros_(self.update.weight)
            nn.init.zeros_(self.update.bias)
        if arm == "pretrained_residual":
            self.reference = XLSRStates(checkpoint, False)

    @torch.no_grad()
    def reference_frames(self, wave):
        captured, calls, hooks = [], [], []
        for index, layer in enumerate(self.reference.model.encoder.layers):
            def capture(_module, _inputs, output, index=index):
                calls.append(index + 1)
                if 8 <= index < 16:
                    captured.append(output[0])
            hooks.append(layer.register_forward_hook(capture))
        state = np.random.get_state()
        try:
            result = self.reference.model(wave, mask=False, features_only=True, layer=15)
        finally:
            np.random.set_state(state)
            for hook in hooks:
                hook.remove()
        if calls != list(range(1, 17)) or len(captured) != 8:
            raise RuntimeError(f"Reference early exit violated: {calls}")
        self.reference_calls = calls
        frames = result["x"].size(1)
        return sum(h[:frames].transpose(0, 1) for h in captured) / 8

    def forward(self, wave, return_details=False):
        # Compute the frozen reference before retaining task autograd activations.
        pretrained_frames = self.reference_frames(wave) if self.reference is not None else None
        states = self.task(wave)
        if any(tuple(h.shape[1:]) != (201, 1024) for h in states):
            raise RuntimeError("Keep original pad64600 / 24 layers")
        # Original SLS gate-input path is NOT detached.
        gates = torch.sigmoid(self.backend.fc0(self.backend.descriptors(states)))
        z = sum(h * gates[:, i:i+1] for i, h in enumerate(states))
        prediction = reference = None
        delta = torch.zeros_like(z)
        if self.predictor is not None:
            reference = (pretrained_frames if self.reference is not None
                         else sum(h.detach() for h in states[8:16]) / 8)
            reference = F.layer_norm(reference.detach(), (1024,))
            prediction = self.predictor(reference)
            residual = z - prediction.detach()
            raw = self.update(residual)
            with torch.autocast(device_type=wave.device.type, enabled=False):
                limit = z.detach().float().norm(dim=-1, keepdim=True)
                scale = (limit / (raw.float().norm(dim=-1, keepdim=True) + 1e-8)).clamp(max=1)
                delta = (0.1 * raw.float() * scale).to(z.dtype)
        corrected = z + delta if self.correction_enabled else z
        b = self.backend
        pooled = F.max_pool2d(b.selu(b.first_bn(corrected.unsqueeze(1))), 3, 3)
        flat = pooled.flatten(1)
        if flat.size(1) != 22847:
            raise RuntimeError("SLS pooling dimension changed")
        output = F.log_softmax(b.selu(b.fc3(b.selu(b.fc1(flat)))).float(), dim=1)
        if return_details:
            return dict(log_probs=output, aggregate=z, prediction=prediction,
                        reference=reference, delta=delta)
        return output

    def trainable_payload(self, sha):
        names = {n for n, p in self.task.named_parameters() if p.requires_grad}
        return dict(format="sls_conditional_residual_v1", arm=self.arm,
                    pretrained_sha256=sha, backend=self.backend.state_dict(),
                    predictor=None if self.predictor is None else self.predictor.state_dict(),
                    update=None if self.update is None else self.update.state_dict(),
                    task_trainable={n:t for n,t in self.task.state_dict().items() if n in names})

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_conditional_residual_v1" or saved["arm"] != self.arm
                or saved["pretrained_sha256"] != sha):
            raise ValueError("Payload/base/arm mismatch")
        expected = {n for n,p in self.task.named_parameters() if p.requires_grad}
        if set(saved["task_trainable"]) != expected:
            raise ValueError("Trainable parameter mismatch")
        self.backend.load_state_dict(saved["backend"])
        self.task.load_state_dict(saved["task_trainable"], strict=False)
        if self.predictor is not None:
            self.predictor.load_state_dict(saved["predictor"])
            self.update.load_state_dict(saved["update"])
