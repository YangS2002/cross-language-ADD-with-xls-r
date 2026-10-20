"""Original attention routing with a task-state compatibility loss.

The detection forward is identical to PretrainedAttentionSLS. During training,
the frozen original Q/K projections additionally read the task attention input.
Their attention distribution is compared with the frozen teacher distribution.
Only task-state gradients flow through this auxiliary loss.
"""
from types import MethodType

import torch
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


class QKRelationSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.collect_relation_loss = False
        self.relation_loss = None
        self._relation_terms = []
        for index, layer in enumerate(self.task.model.encoder.layers):
            original_forward = layer.self_attn.forward

            def with_relation(module, query, key, value, key_padding_mask=None,
                              incremental_state=None, need_weights=True,
                              static_kv=False, attn_mask=None,
                              before_softmax=False, need_head_weights=False,
                              index=index, original_forward=original_forward):
                if self.collect_relation_loss:
                    if not self.guidance_enabled or len(self._reference_inputs) != 24:
                        raise RuntimeError("Relation loss requires all original attention inputs")
                    reference_input, reference_mask = self._reference_inputs[index]
                    if query.shape != reference_input.shape:
                        raise RuntimeError("Task/reference attention input mismatch")
                    if (reference_mask is None) != (key_padding_mask is None):
                        raise RuntimeError("Task/reference attention mask mismatch")
                    if reference_mask is not None and not torch.equal(reference_mask, key_padding_mask):
                        raise RuntimeError("Task/reference attention masks differ")
                    original = self.reference.model.encoder.layers[index].self_attn
                    heads = module.num_heads
                    width = module.embed_dim // heads

                    def logits(states):
                        q = F.linear(states, original.q_proj.weight, original.q_proj.bias)
                        k = F.linear(states, original.k_proj.weight, original.k_proj.bias)
                        steps, batch, _ = states.shape
                        q = q.reshape(steps, batch, heads, width).permute(1, 2, 0, 3).float()
                        k = k.reshape(steps, batch, heads, width).permute(1, 2, 0, 3).float()
                        scores = torch.matmul(q, k.transpose(-2, -1)) * (width ** -0.5)
                        if key_padding_mask is not None:
                            scores = scores.masked_fill(key_padding_mask[:, None, None, :],
                                                        torch.finfo(scores.dtype).min)
                        return scores

                    with torch.no_grad():
                        target_log = F.log_softmax(logits(reference_input), dim=-1)
                        target = target_log.exp()
                    current_log = F.log_softmax(logits(query), dim=-1)
                    term = (target * (target_log - current_log)).sum(dim=-1).mean()
                    self._relation_terms.append(term)
                return original_forward(query, key, value, key_padding_mask,
                                        incremental_state, need_weights, static_kv,
                                        attn_mask, before_softmax, need_head_weights)

            layer.self_attn.forward = MethodType(with_relation, layer.self_attn)

    def forward(self, wave, return_details=False):
        self._relation_terms = []
        try:
            output = super().forward(wave, return_details=return_details)
            if self.collect_relation_loss:
                if len(self._relation_terms) != 24:
                    raise RuntimeError("Expected 24 per-layer relation losses")
                self.relation_loss = torch.stack(self._relation_terms).mean()
            else:
                self.relation_loss = None
            return output
        finally:
            self._relation_terms = []

