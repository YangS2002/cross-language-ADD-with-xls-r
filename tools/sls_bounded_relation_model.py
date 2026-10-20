"""Frozen-context, bounded Q/K adaptation; task values and native SLS learn."""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_query_model import PretrainedQuerySLS


def bounded_relation_logits(original, adapted, tau, valid=None):
    """Row-centered logit correction, before padding/attention masking.

    Both inputs are UNMASKED logits on the same frozen reference inputs.
    A valid mask is broadcastable to the logits; masked positions never enter
    the row mean. FP64 is retained for the small analytical gradient checks.
    """
    if tau <= 0 or original.shape != adapted.shape:
        raise ValueError("Positive bound and equal-shaped logits required")
    with torch.autocast(device_type=adapted.device.type, enabled=False):
        dtype = torch.float64 if adapted.dtype == torch.float64 else torch.float32
        base, task = original.to(dtype), adapted.to(dtype)
        delta = task - base
        if valid is None:
            centered = delta - delta.mean(-1, keepdim=True)
        else:
            valid = valid.to(device=delta.device, dtype=torch.bool).expand_as(delta)
            count = valid.sum(-1, keepdim=True)
            if bool((count == 0).any()):
                raise ValueError("A query has no valid keys")
            average = torch.where(valid, delta, 0).sum(-1, keepdim=True) / count
            centered = torch.where(valid, delta - average, 0)
        correction = tau * torch.tanh(centered / tau)
        logits = base + correction
        if valid is not None:
            logits = logits.masked_fill(~valid, float("-inf"))
        return logits, correction


class BoundedRelationSLS(PretrainedQuerySLS):
    def __init__(self, checkpoint, tau=0.5):
        super().__init__(checkpoint)
        if tau <= 0:
            raise ValueError("Positive relation bound required")
        self.tau = float(tau)
        self.reference_copy_only = False  # Initialization/wiring test only.
        self.forward_count = 0
        self.diagnostic_sink = None
        self.last_diagnostic = None
        self._diagnose = False
        self._relation_rows = []
        owner = weakref.ref(self)
        for index, layer in enumerate(self.task.model.encoder.layers):
            layer.self_attn.q_proj.requires_grad_(True)
            layer.self_attn.k_proj.requires_grad_(True)
            native = self._native_forwards[index]

            def guided(module, query, key, value, key_padding_mask=None,
                       incremental_state=None, need_weights=True, static_kv=False,
                       attn_mask=None, before_softmax=False, need_head_weights=False,
                       index=index, native=native):
                model = owner()
                if not model.guidance_enabled:
                    return native(query, key, value, key_padding_mask, incremental_state,
                                  need_weights, static_kv, attn_mask, before_softmax, need_head_weights)
                if (incremental_state is not None or static_kv or before_softmax
                        or need_head_weights or module.onnx_trace or attn_mask is not None
                        or module.bias_k is not None or module.bias_v is not None
                        or module.add_zero_attn or query.device.type != "cuda"):
                    raise RuntimeError("Unsupported attention path; no silent fallback")
                if index != len(model.query_calls) or len(model._reference_inputs) != 24:
                    raise RuntimeError("Frozen context missing or out of layer order")
                context, reference_mask = model._reference_inputs[index]
                if (context.shape != query.shape or key is None or value is None
                        or key.shape != query.shape or value.shape != query.shape):
                    raise RuntimeError("Reference/task frame shapes differ")
                if (reference_mask is None) != (key_padding_mask is None):
                    raise RuntimeError("Reference/task mask presence differs")
                if reference_mask is not None and not torch.equal(reference_mask, key_padding_mask):
                    raise RuntimeError("Reference/task masks differ")
                if model.query_check_mode is not None:
                    raise RuntimeError("Use explicit reference-copy initialization check")
                teacher = model.reference.model.encoder.layers[index].self_attn
                model.query_calls.append(index + 1)
                # The existing Attention path is the numerical reference. There
                # are no task Q/K gradients in this SELF-TEST-ONLY mode.
                if model.reference_copy_only:
                    return F.multi_head_attention_forward(
                        context, context, value, module.embed_dim, module.num_heads,
                        torch.empty(0), torch.cat((teacher.q_proj.bias, teacher.k_proj.bias,
                                                  module.v_proj.bias)),
                        module.bias_k, module.bias_v, module.add_zero_attn,
                        module.dropout_module.p, module.out_proj.weight, module.out_proj.bias,
                        module.training or module.dropout_module.apply_during_inference,
                        key_padding_mask, need_weights, attn_mask, use_separate_proj_weight=True,
                        q_proj_weight=teacher.q_proj.weight, k_proj_weight=teacher.k_proj.weight,
                        v_proj_weight=module.v_proj.weight)

                length, batch, width = context.shape
                heads, head_dim = module.num_heads, width // module.num_heads

                def reshape(x):
                    return x.contiguous().view(length, batch * heads, head_dim).transpose(0, 1)

                # Same linear/scaling/BMM order as installed PyTorch MHA. Cast
                # logits to FP32 only after BMM so R=0 reproduces the BF16 copy
                # path, rather than changing its numerical protocol silently.
                with torch.no_grad():
                    q0 = reshape(teacher.q_proj(context) * head_dim ** -0.5)
                    k0 = reshape(teacher.k_proj(context))
                    base = torch.bmm(q0, k0.transpose(1, 2)).float()
                q = reshape(module.q_proj(context) * head_dim ** -0.5)
                k = reshape(module.k_proj(context))
                adapted = torch.bmm(q, k.transpose(1, 2)).float()
                valid = None
                if key_padding_mask is not None:
                    if key_padding_mask.dtype != torch.bool:
                        raise RuntimeError("Expected boolean padding mask")
                    valid = (~key_padding_mask[:, None, None, :]).expand(
                        batch, heads, length, length).reshape(batch * heads, length, length)
                logits, correction = bounded_relation_logits(base, adapted, model.tau, valid)
                probabilities = logits.softmax(-1)
                v = reshape(module.v_proj(value))  # Task input, NOT frozen context.
                weights = probabilities.to(v.dtype)
                dropout = module.dropout_module.p if (
                    module.training or module.dropout_module.apply_during_inference) else 0.0
                if dropout:
                    weights = F.dropout(weights, p=dropout, training=True)
                attended = torch.bmm(weights, v).transpose(0, 1).contiguous().view(
                    length * batch, width)
                output = module.out_proj(attended).view(length, batch, width)
                if model._diagnose:
                    with torch.no_grad():
                        original = base if valid is None else base.masked_fill(~valid, float("-inf"))
                        prior = original.softmax(-1)
                        log_prior = original.log_softmax(-1)
                        log_actual = logits.log_softmax(-1)
                        log_difference = log_actual - log_prior
                        if valid is not None:
                            log_difference = torch.where(valid, log_difference, 0)
                        kl = (probabilities * log_difference).sum(-1)
                        r = correction if valid is None else correction[valid]
                        row = dict(layer=index + 1, correction_mean_abs=float(r.abs().mean()),
                                   correction_max_abs=float(r.abs().max()),
                                   saturation_fraction=float((r.abs() >= .95 * model.tau).float().mean()),
                                   mean_attention_kl=float(kl.mean()),
                                   mean_attention_l1=float((probabilities-prior).abs().sum(-1).mean()),
                                   row_sum_max_error=float((probabilities.sum(-1)-1).abs().max()))
                        if (not bool(torch.isfinite(output).all() and torch.isfinite(probabilities).all())
                                or row["correction_max_abs"] > model.tau + 1e-6):
                            raise FloatingPointError("Nonfinite attention or violated relation bound")
                        model._relation_rows.append(row)
                averaged = weights.view(batch, heads, length, length).mean(1) if need_weights else None
                return output, averaged

            layer.self_attn.forward = MethodType(guided, layer.self_attn)

    def forward(self, wave, return_details=False):
        self.forward_count += 1
        self._diagnose = return_details or (self.training and self.diagnostic_sink is not None
                                           and (self.forward_count == 20 or self.forward_count % 200 == 0))
        self._relation_rows = []
        result = super().forward(wave, return_details)
        if self._diagnose and self.guidance_enabled and not self.reference_copy_only:
            if len(self._relation_rows) != 24:
                raise RuntimeError("Missing relation diagnostics")
            self.last_diagnostic = dict(forward=self.forward_count, layers=self._relation_rows,
                                       tau=self.tau, frozen_context=True)
            if self.training and self.diagnostic_sink is not None:
                self.diagnostic_sink(self.last_diagnostic)
            if return_details:
                result["diagnostic"] = self.last_diagnostic
        self._relation_rows = []
        return result

    def trainable_payload(self, sha):
        saved = super().trainable_payload(sha)
        saved.update(format="sls_bounded_relation_v1", tau=self.tau,
                     routing="same_frozen_context_QK_bounded_logit_correction_all24")
        return saved

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_bounded_relation_v1" or saved["tau"] != self.tau
                or saved["routing"] != "same_frozen_context_QK_bounded_logit_correction_all24"):
            raise ValueError("Wrong bounded-relation checkpoint")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_query_v1"), sha)
