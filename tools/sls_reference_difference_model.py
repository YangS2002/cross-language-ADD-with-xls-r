"""Original Q/K routing plus signed task-Value aggregation complement."""
from types import MethodType
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


def mix_context(context, local, gain):
    # FP32 mixing lets a small gain remain observable under BF16 attention.
    return (context.float() + gain * (local.float() - context.float())).to(context.dtype)


class ReferenceDifferenceSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint, gain_cap=.25):
        super().__init__(checkpoint)
        if not 0 < gain_cap < 1:
            raise ValueError("Keep a nonzero original-context contribution")
        self.gain_cap = float(gain_cap)
        self.difference_gain = nn.Parameter(torch.zeros(24))
        self.difference_enabled = True
        self.record_difference = False
        self.difference_diagnostics = []
        owner = weakref.ref(self)
        for index, layer in enumerate(self.task.model.encoder.layers):
            guided = layer.self_attn.forward

            def complemented(module, query, key, value, *args,
                             index=index, guided=guided, **kwargs):
                model = owner()
                context, weights = guided(query, key, value, *args, **kwargs)
                if not model.difference_enabled:
                    return context, weights
                if (module.bias_k is not None or module.bias_v is not None
                        or module.add_zero_attn or query.shape != value.shape):
                    raise RuntimeError("Identity complement requires native same-length self attention")
                # One coefficient per layer, shared by heads: projection commutes
                # with this mixture. Both branches use the SAME task V/out/bias.
                local = F.linear(F.linear(value, module.v_proj.weight, module.v_proj.bias),
                                 module.out_proj.weight, module.out_proj.bias)
                gain = model.difference_gain[index]
                output = mix_context(context, local, gain)
                if model.record_difference:
                    with torch.no_grad():
                        delta = local.float() - context.float()
                        denominator = context.float().square().sum().clamp_min(1e-20)
                        model.difference_diagnostics.append(torch.stack((
                            (delta.square().sum()/denominator).sqrt(),
                            (gain.detach().square()*delta.square().sum()/denominator).sqrt())))
                if weights is not None:
                    # Returned weights describe EFFECTIVE aggregation, not A0.
                    eye = torch.eye(weights.shape[-1], device=weights.device, dtype=weights.dtype)
                    weights = mix_context(weights, eye.expand_as(weights), gain)
                return output, weights

            layer.self_attn.forward = MethodType(complemented, layer.self_attn)

    @torch.no_grad()
    def project_gains(self):
        # Project AFTER optimizer update. A clamp in forward would block a
        # zero-initialized gate's boundary gradient in some parameterizations.
        self.difference_gain.clamp_(0., self.gain_cap)

    def forward(self, wave, return_details=False):
        self.difference_diagnostics = []
        return super().forward(wave, return_details)

    def trainable_payload(self, sha):
        saved = super().trainable_payload(sha)
        saved.update(format="sls_reference_difference_v1",
                     routing="original_QK_with_identity_complement",
                     difference_gain=self.difference_gain.detach().clone(), gain_cap=self.gain_cap)
        return saved

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_reference_difference_v1"
                or saved["routing"] != "original_QK_with_identity_complement"
                or saved["gain_cap"] != self.gain_cap):
            raise ValueError("Wrong relation-complement payload")
        gain = saved["difference_gain"]
        if (tuple(gain.shape) != (24,) or not bool(torch.isfinite(gain).all())
                or bool((gain < 0).any()) or bool((gain > self.gain_cap).any())):
            raise ValueError("Invalid bounded complement coefficients")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_attention_v1",
                                            routing="actual_frozen_reference_QK_all24"), sha)
        with torch.no_grad():
            self.difference_gain.copy_(gain)
