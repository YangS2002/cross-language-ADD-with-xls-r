"""The same original-Q/K encoder with a last-layer-only MLP backend."""
import torch
from torch import nn
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS
from tools.sls_qk_relation_model import QKRelationSLS


class LastLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(2048)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, last_state):
        pooled = torch.cat((last_state.mean(dim=1),
                            last_state.float().std(dim=1, unbiased=False).to(last_state)), dim=1)
        logits = self.fc2(F.gelu(self.fc1(self.norm(pooled))))
        return F.log_softmax(logits.float(), dim=1)


class _LastLayerForward:
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.backend = LastLayerMLP()
        self.collect_relation_loss = False
        self.relation_loss = None
        self._relation_terms = []

    def forward(self, wave, return_details=False):
        self.query_calls = []
        self._relation_terms = []
        try:
            if self.guidance_enabled:
                self.extract_reference(wave)
            states = self.task(wave)
        finally:
            self._reference_inputs = []
        if self.guidance_enabled and self.query_calls != list(range(1, 25)):
            raise RuntimeError("Original Q/K guidance incomplete")
        if len(states) != 24 or tuple(states[-1].shape[1:]) != (201, 1024):
            raise RuntimeError("Last-layer output shape changed")
        output = self.backend(states[-1])
        if self.collect_relation_loss:
            if len(self._relation_terms) != 24:
                raise RuntimeError("Relation loss incomplete")
            self.relation_loss = torch.stack(self._relation_terms).mean()
        else:
            self.relation_loss = None
        self._relation_terms = []
        if return_details:
            return dict(log_probs=output, states=states)
        return output

    def trainable_payload(self, sha):
        payload = super().trainable_payload(sha)
        payload["backend_kind"] = "last_layer_mean_std_mlp_2048_512_2"
        return payload

    def load_trainable_payload(self, saved, sha):
        if saved.get("backend_kind") != "last_layer_mean_std_mlp_2048_512_2":
            raise ValueError("Wrong MLP backend checkpoint")
        super().load_trainable_payload({k: v for k, v in saved.items()
                                        if k != "backend_kind"}, sha)


class QKLastMLP(_LastLayerForward, PretrainedAttentionSLS):
    pass


class QKRelationLastMLP(_LastLayerForward, QKRelationSLS):
    pass
