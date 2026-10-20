"""Original Q/K routing with training-only stochastic original Value access."""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


class ReferenceValueSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint, probability=0.1, replacement_seed=902020):
        super().__init__(checkpoint)
        if not 0 <= probability <= 1:
            raise ValueError("Replacement probability outside [0,1]")
        self.probability = float(probability)
        self.replacement_seed = int(replacement_seed)
        self.value_rng = torch.Generator(device="cpu").manual_seed(self.replacement_seed)
        self.replacement_override = None  # Short source test only; None in formal run.
        self.last_value_mask = [False] * 24
        self.replacement_counts = [0] * 24
        self.training_forwards = 0
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
                    raise RuntimeError("Unsupported attention path")
                if index != len(model.query_calls) or len(model._reference_inputs) != 24:
                    raise RuntimeError("Reference layer order mismatch")
                reference_input, reference_mask = model._reference_inputs[index]
                if (reference_input.shape != query.shape or key is None or value is None
                        or key.shape != query.shape or value.shape != query.shape):
                    raise RuntimeError("Reference/task shape mismatch")
                if (reference_mask is None) != (key_padding_mask is None):
                    raise RuntimeError("Padding-mask presence mismatch")
                if reference_mask is not None and not torch.equal(reference_mask, key_padding_mask):
                    raise RuntimeError("Padding masks differ")
                reference_attention = model.reference.model.encoder.layers[index].self_attn
                projection = reference_attention if model.last_value_mask[index] else module
                value_input = reference_input if model.last_value_mask[index] else value
                model.query_calls.append(index + 1)
                # Both reference INPUT and original Wv/bv are selected together.
                # Q/K always reference; out/residual/FFN always task.
                return F.multi_head_attention_forward(
                    reference_input, reference_input, value_input,
                    module.embed_dim, module.num_heads, torch.empty(0),
                    torch.cat((reference_attention.q_proj.bias,
                               reference_attention.k_proj.bias, projection.v_proj.bias)),
                    module.bias_k, module.bias_v, module.add_zero_attn,
                    module.dropout_module.p, module.out_proj.weight, module.out_proj.bias,
                    module.training or module.dropout_module.apply_during_inference,
                    key_padding_mask, need_weights, attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=reference_attention.q_proj.weight,
                    k_proj_weight=reference_attention.k_proj.weight,
                    v_proj_weight=projection.v_proj.weight)

            layer.self_attn.forward = MethodType(guided, layer.self_attn)

    def forward(self, wave, return_details=False):
        # XLSRStates deliberately stays eval to disable SSL dropout, so use the
        # outer model.training flag rather than the attention module's flag.
        if self.replacement_override is not None:
            self.last_value_mask = list(self.replacement_override)
            if len(self.last_value_mask) != 24:
                raise ValueError("Exactly 24 override decisions required")
        elif self.training and self.guidance_enabled:
            self.last_value_mask = (torch.rand(24, generator=self.value_rng) < self.probability).tolist()
        else:
            self.last_value_mask = [False] * 24
        if self.training and self.guidance_enabled:
            self.training_forwards += 1
            for i, selected in enumerate(self.last_value_mask):
                self.replacement_counts[i] += int(selected)
        return super().forward(wave, return_details=return_details)

    def trainable_payload(self, sha):
        saved = super().trainable_payload(sha)
        saved.update(format="sls_reference_value_v1", routing="original_QK_training_value_replacement",
                     replacement_probability=self.probability, replacement_seed=self.replacement_seed,
                     replacement_rng_state=self.value_rng.get_state())
        return saved

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_reference_value_v1"
                or saved["routing"] != "original_QK_training_value_replacement"
                or saved["replacement_probability"] != self.probability
                or saved["replacement_seed"] != self.replacement_seed):
            raise ValueError("Value replacement checkpoint/config mismatch")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_attention_v1",
                                           routing="actual_frozen_reference_QK_all24"), sha)
        self.value_rng.set_state(saved["replacement_rng_state"].cpu())
