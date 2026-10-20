"""Native fine-tuned XLS-R/SLS plus frozen-relation variation evidence."""
import math
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.sls_reference_model import XLSRStates, FullWidthSLS


def weighted_variation(h, p):
    """sum_s p_ts*(h_t-h_s)^2, without a [B,T,T,C] allocation.

    The common temporal offset is removed only inside this translation-invariant
    statistic, not from native SLS. FP64 is retained for the analytical tests.
    """
    with torch.autocast(device_type=h.device.type, enabled=False):
        dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        x = h.to(dtype)
        x = x - x.mean(dim=1, keepdim=True)
        p = p.to(dtype)
        mu = torch.bmm(p, x)
        raw = x.square() - 2 * x * mu + torch.bmm(p, x.square())
        return raw.clamp_min(0), raw


class ReferenceVariationSLS(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.task = XLSRStates(checkpoint, True)
        self.reference = XLSRStates(checkpoint, False)
        self.backend = FullWidthSLS()
        # Constructing a zero matrix must not consume the existing RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.backend.variation_fc = nn.Linear(22847, 1024, bias=False)
        nn.init.zeros_(self.backend.variation_fc.weight)
        self.backend.variation_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1)
        self.evidence_enabled = True
        self._relations = []
        self._handles = []
        self.forward_count = 0
        self.diagnostic_sink = None
        self.last_diagnostic = None
        owner = weakref.ref(self)
        if len(self.reference.model.encoder.layers) != 24:
            raise RuntimeError("Expected all 24 original XLS-R layers")
        for index, layer in enumerate(self.reference.model.encoder.layers):
            def capture(_module, _args, output, index=index):
                model = owner()
                if index != len(model._relations):
                    raise RuntimeError("Teacher attention executed out of order")
                p = output[1]
                if p is None or p.ndim != 3 or p.shape[1:] != (201, 201):
                    raise RuntimeError("Expected native head-averaged [B,201,201] attention")
                # Native attention is averaged across heads. Renormalize in FP32
                # to remove BF16 row-sum roundoff, not to learn another relation.
                p = p.detach().float()
                p = p / p.sum(dim=-1, keepdim=True)
                model._relations.append(p)
            self._handles.append(layer.self_attn.register_forward_hook(capture))

    def load_warm_backend(self, warm):
        absent, unexpected = self.backend.load_state_dict(warm["backend"], strict=False)
        expected = {"variation_fc.weight", "variation_bn.weight", "variation_bn.bias",
                    "variation_bn.running_mean", "variation_bn.running_var",
                    "variation_bn.num_batches_tracked"}
        # Older PyTorch may omit num_batches_tracked from the missing-key list.
        if set(absent) not in (expected, expected - {"variation_bn.num_batches_tracked"}) or unexpected:
            raise RuntimeError(f"Warmup backend mismatch: {absent}, {unexpected}")
        if bool(self.backend.variation_fc.weight.detach().abs().sum()):
            raise RuntimeError("Supplementary classifier must start exactly at zero")

    @torch.no_grad()
    def extract_relations(self, wave):
        self._relations = []
        state = np.random.get_state()
        try:
            self.reference(wave)
        finally:
            np.random.set_state(state)
        if len(self._relations) != 24:
            raise RuntimeError("Missing original attention layers")

    def forward(self, wave, return_details=False):
        self.forward_count += 1
        diagnose = return_details or (self.training and self.diagnostic_sink is not None
                                     and (self.forward_count == 20 or self.forward_count % 200 == 0))
        layer_rows = []
        try:
            if self.evidence_enabled:
                self.extract_relations(wave)
            states = self.task(wave)  # Unmodified task Q/K/V, FFN, residual stream.
            if len(states) != 24 or any(tuple(h.shape[1:]) != (201, 1024) for h in states):
                raise RuntimeError("Keep all layers, native width1024 and pad64600")
            b = self.backend
            gates = torch.sigmoid(b.fc0(b.descriptors(states)))  # No detach.
            aggregate = sum(h * gates[:, i:i+1] for i, h in enumerate(states))
            native_pool = F.max_pool2d(b.selu(b.first_bn(aggregate.unsqueeze(1))), 3, 3).flatten(1)
            native_hidden = b.fc1(native_pool)
            delta = torch.zeros_like(native_hidden)
            evidence = None
            if self.evidence_enabled:
                for index, (h, p) in enumerate(zip(states, self._relations)):
                    d, raw = weighted_variation(h, p)
                    term = d * gates[:, index:index+1].float()
                    evidence = term if evidence is None else evidence + term
                    if diagnose:
                        with torch.no_grad():
                            row = dict(layer=index+1, row_sum_max_error=float((p.sum(-1)-1).abs().max()),
                                       entropy=float((-(p.clamp_min(1e-12).log()*p).sum(-1)
                                                     / math.log(p.size(-1))).mean()),
                                       diagonal_mass=float(p.diagonal(dim1=-2, dim2=-1).mean()),
                                       relative_distance_to_uniform=float((p-1/p.size(-1)).norm()
                                                                          / p.norm().clamp_min(1e-12)),
                                       variation_mean=float(d.mean()), variation_std=float(d.std()),
                                       negative_fraction=float((raw < 0).float().mean()),
                                       minimum_unclamped=float(raw.min()))
                            if not bool(torch.isfinite(p).all() and torch.isfinite(d).all()):
                                raise FloatingPointError("Nonfinite frozen relation or variation")
                            if row["minimum_unclamped"] < -1e-4 * max(1., float(h.detach().float().square().max())):
                                raise FloatingPointError("Variation cancellation exceeded roundoff bound")
                            layer_rows.append(row)
                pooled_d = F.max_pool2d(b.selu(b.variation_bn(evidence.unsqueeze(1))), 3, 3).flatten(1)
                delta = b.variation_fc(pooled_d)
            hidden = b.selu(native_hidden + delta)
            output = F.log_softmax(b.selu(b.fc3(hidden)).float(), dim=1)
            if diagnose:
                with torch.no_grad():
                    native_output = F.log_softmax(b.selu(b.fc3(b.selu(native_hidden))).float(), dim=1)
                    self.last_diagnostic = dict(forward=self.forward_count, layers=layer_rows,
                        delta_to_native_hidden_norm=float(delta.float().norm()
                                                          / native_hidden.float().norm().clamp_min(1e-12)),
                        mean_absolute_score_contribution=float((output[:, 1]-native_output[:, 1]).abs().mean()))
                    if self.training and self.diagnostic_sink is not None:
                        self.diagnostic_sink(self.last_diagnostic)
            if return_details:
                return dict(log_probs=output, states=states, evidence=evidence,
                            relations=list(self._relations), diagnostic=self.last_diagnostic)
            return output
        finally:
            self._relations = []  # Autograd owns needed tensors; no next-batch cache.

    def trainable_payload(self, sha):
        names = {n for n, p in self.task.named_parameters() if p.requires_grad}
        return dict(format="sls_reference_variation_v1", pretrained_sha256=sha,
                    backend=self.backend.state_dict(),
                    task_trainable={n: t for n, t in self.task.state_dict().items() if n in names})

    def load_trainable_payload(self, saved, sha):
        if saved["format"] != "sls_reference_variation_v1" or saved["pretrained_sha256"] != sha:
            raise ValueError("Payload/base mismatch")
        expected = {n for n, p in self.task.named_parameters() if p.requires_grad}
        if set(saved["task_trainable"]) != expected:
            raise ValueError("Trainable parameter set mismatch")
        self.backend.load_state_dict(saved["backend"], strict=True)
        self.task.load_state_dict(saved["task_trainable"], strict=False)
