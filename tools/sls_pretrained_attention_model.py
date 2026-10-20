"""Frozen original Q/K routing, jointly adapted task V/out/FFN and full-width SLS."""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_query_model import PretrainedQuerySLS


class PretrainedAttentionSLS(PretrainedQuerySLS):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        owner = weakref.ref(self)
        for index, layer in enumerate(self.task.model.encoder.layers):
            layer.self_attn.k_proj.requires_grad_(False)
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
                    raise RuntimeError("Original Q/K missing or out of layer order")
                reference_input, reference_mask = model._reference_inputs[index]
                if (reference_input.shape != query.shape or key is None or value is None
                        or key.shape != query.shape or value.shape != query.shape):
                    raise RuntimeError("Original/task attention shape mismatch")
                if (reference_mask is None) != (key_padding_mask is None):
                    raise RuntimeError("Padding-mask presence mismatch")
                if reference_mask is not None and not torch.equal(reference_mask, key_padding_mask):
                    raise RuntimeError("Padding masks differ")
                reference_attention = model.reference.model.encoder.layers[index].self_attn
                external_query = reference_input
                if model.query_check_mode == "zero":
                    external_query = torch.zeros_like(reference_input)
                elif model.query_check_mode is not None:
                    raise ValueError("Unknown source-only wiring-check mode")
                model.query_calls.append(index+1)
                # Actual teacher inputs and weights fix A0(x), not just Wq/Wk.
                # All values, output projection, residual stream and FFN are task computations.
                return F.multi_head_attention_forward(
                    external_query, reference_input, value,
                    module.embed_dim, module.num_heads, torch.empty(0),
                    torch.cat((reference_attention.q_proj.bias,
                               reference_attention.k_proj.bias, module.v_proj.bias)),
                    module.bias_k, module.bias_v, module.add_zero_attn,
                    module.dropout_module.p, module.out_proj.weight, module.out_proj.bias,
                    module.training or module.dropout_module.apply_during_inference,
                    key_padding_mask, need_weights, attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=reference_attention.q_proj.weight,
                    k_proj_weight=reference_attention.k_proj.weight,
                    v_proj_weight=module.v_proj.weight)

            layer.self_attn.forward = MethodType(guided, layer.self_attn)

    def trainable_payload(self, sha):
        saved = super().trainable_payload(sha)
        saved["format"] = "sls_pretrained_attention_v1"
        saved["routing"] = "actual_frozen_reference_QK_all24"
        return saved

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_pretrained_attention_v1"
                or saved["routing"] != "actual_frozen_reference_QK_all24"):
            raise ValueError("Wrong attention-copy payload")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_query_v1"), sha)
