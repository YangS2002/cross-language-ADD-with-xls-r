"""Original functional Value base plus full task-parameter Value correction.

U = A0 [V0(Z0) + (Vtheta - V0)(Ztheta)].  The existing full-width
trainable V parameters are retained; no extra adapter, loss or random path.
"""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


def reference_value_projection(reference_input, task_input, original, task):
    # Preserve the original projection GEMM shape (concatenating projections
    # changes rounding even at zero delta). Subtract FP32 parameters BEFORE
    # autocast; mix projected values in FP32, then restore the native dtype.
    fixed = F.linear(reference_input, original.weight, original.bias)
    increment = F.linear(task_input, task.weight-original.weight,
                         task.bias-original.bias)
    return (fixed.float()+increment.float()).to(fixed.dtype) if fixed.dtype != torch.float64 else fixed+increment


class ReferenceValueIncrementSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        owner = weakref.ref(self)
        for index, layer in enumerate(self.task.model.encoder.layers):
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
                        or need_head_weights or module.onnx_trace or query.device.type != "cuda"):
                    raise RuntimeError("Unsupported attention path; no silent fallback")
                if index != len(model.query_calls) or len(model._reference_inputs) != 24:
                    raise RuntimeError("Original Q/K missing or out of order")
                reference_input, reference_mask = model._reference_inputs[index]
                if (reference_input.shape != query.shape or key is None or value is None
                        or key.shape != query.shape or value.shape != query.shape):
                    raise RuntimeError("Reference/task shape mismatch")
                if (reference_mask is None) != (key_padding_mask is None):
                    raise RuntimeError("Padding-mask presence mismatch")
                if reference_mask is not None and not torch.equal(reference_mask, key_padding_mask):
                    raise RuntimeError("Padding masks differ")
                original = model.reference.model.encoder.layers[index].self_attn
                external_query = reference_input
                if model.query_check_mode == "zero":
                    external_query = torch.zeros_like(reference_input)
                elif model.query_check_mode is not None:
                    raise ValueError("Unknown source-only check mode")
                projected = reference_value_projection(
                    reference_input, value, original.v_proj, module.v_proj)
                length, batch, width = projected.shape
                static_value = projected.contiguous().view(
                    length, batch*module.num_heads, width//module.num_heads).transpose(0, 1)
                model.query_calls.append(index+1)
                # static_v is the native PyTorch preprojected-Value API. Empty
                # dummy projection avoids recomputing an unused full task V.
                # Q/K, softmax/masks, attention bmm and task out stay unchanged.
                return F.multi_head_attention_forward(
                    external_query, reference_input, value[..., :0],
                    module.embed_dim, module.num_heads, torch.empty(0),
                    torch.cat((original.q_proj.bias, original.k_proj.bias,
                               torch.zeros_like(module.v_proj.bias))),
                    module.bias_k, module.bias_v, module.add_zero_attn,
                    module.dropout_module.p, module.out_proj.weight, module.out_proj.bias,
                    module.training or module.dropout_module.apply_during_inference,
                    key_padding_mask, need_weights, attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=original.q_proj.weight,
                    k_proj_weight=original.k_proj.weight,
                    v_proj_weight=module.v_proj.weight[:, :0], static_v=static_value)

            layer.self_attn.forward = MethodType(guided, layer.self_attn)

    def trainable_payload(self, sha):
        saved = super().trainable_payload(sha)
        saved.update(format="sls_reference_value_increment_v1",
                     routing="original_QK_original_context_Value_base_task_delta_Value")
        return saved

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_reference_value_increment_v1"
                or saved["routing"] != "original_QK_original_context_Value_base_task_delta_Value"):
            raise ValueError("Wrong Value-increment payload")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_attention_v1",
                                           routing="actual_frozen_reference_QK_all24"), sha)
