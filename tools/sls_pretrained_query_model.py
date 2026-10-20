"""Original XLS-R queries, trainable task keys/values, unchanged full-width SLS."""
from types import MethodType
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.sls_reference_model import XLSRStates, FullWidthSLS


class PretrainedQuerySLS(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.task = XLSRStates(checkpoint, True)
        self.reference = XLSRStates(checkpoint, False)
        self.backend = FullWidthSLS()
        self.guidance_enabled = True  # Native path is only for initialization checks.
        self.query_check_mode = None  # Source-only wiring test; never formal training.
        self.reference_calls = []
        self.query_calls = []
        self._reference_inputs = []
        self._handles = []
        self._native_forwards = []
        owner = weakref.ref(self)
        layers = self.task.model.encoder.layers
        if len(layers) != 24 or len(self.reference.model.encoder.layers) != 24:
            raise ValueError("Exactly 24 native XLS-R blocks required")
        for index, layer in enumerate(layers):
            attention = layer.self_attn
            attention.q_proj.requires_grad_(False)
            original = attention.forward
            self._native_forwards.append(original)

            def guided(module, query, key, value, key_padding_mask=None,
                       incremental_state=None, need_weights=True, static_kv=False,
                       attn_mask=None, before_softmax=False, need_head_weights=False,
                       index=index, native=original):
                model = owner()
                if not model.guidance_enabled:
                    return native(query, key, value, key_padding_mask,
                                  incremental_state, need_weights, static_kv,
                                  attn_mask, before_softmax, need_head_weights)
                if (incremental_state is not None or static_kv or before_softmax
                        or need_head_weights or module.onnx_trace
                        or query.device.type != "cuda"):
                    raise RuntimeError("Unsupported attention path; no silent fallback")
                if index != len(model.query_calls) or len(model._reference_inputs) != 24:
                    raise RuntimeError("External queries missing or out of layer order")
                reference_query, reference_mask = model._reference_inputs[index]
                if (reference_query.shape != query.shape or key is None or value is None
                        or key.shape != query.shape or value.shape != query.shape):
                    raise RuntimeError("Reference/task attention shapes differ")
                if (reference_mask is None) != (key_padding_mask is None):
                    raise RuntimeError("Reference/task padding-mask presence differs")
                if reference_mask is not None and not torch.equal(reference_mask, key_padding_mask):
                    raise RuntimeError("Reference/task padding masks differ")
                if model.query_check_mode == "zero":
                    reference_query = torch.zeros_like(reference_query)
                elif model.query_check_mode is not None:
                    raise ValueError("Unknown wiring-check mode")
                model.query_calls.append(index + 1)
                # Match installed fairseq fast path, replacing only query INPUT.
                # Frozen task q_proj equals original Wq/bq; k/v/out remain task.
                return F.multi_head_attention_forward(
                    reference_query, key, value, module.embed_dim, module.num_heads,
                    torch.empty(0),
                    torch.cat((module.q_proj.bias, module.k_proj.bias, module.v_proj.bias)),
                    module.bias_k, module.bias_v, module.add_zero_attn,
                    module.dropout_module.p, module.out_proj.weight, module.out_proj.bias,
                    module.training or module.dropout_module.apply_during_inference,
                    key_padding_mask, need_weights, attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=module.q_proj.weight,
                    k_proj_weight=module.k_proj.weight,
                    v_proj_weight=module.v_proj.weight)

            attention.forward = MethodType(guided, attention)

            def capture(_module, args, kwargs, index=index):
                model = owner()
                if index != len(model._reference_inputs):
                    raise RuntimeError("Reference blocks not executed sequentially")
                query = kwargs["query"] if "query" in kwargs else args[0]
                mask = kwargs.get("key_padding_mask")
                model._reference_inputs.append((query.detach(), mask))
                model.reference_calls.append(index + 1)

            self._handles.append(self.reference.model.encoder.layers[index].self_attn
                                 .register_forward_pre_hook(capture, with_kwargs=True))

    @torch.no_grad()
    def extract_reference(self, wave):
        self._reference_inputs = []
        self.reference_calls = []
        state = np.random.get_state()
        try:
            self.reference(wave)
        finally:
            np.random.set_state(state)
        if self.reference_calls != list(range(1, 25)):
            raise RuntimeError("Reference must execute all 24 blocks")

    def forward(self, wave, return_details=False):
        self.query_calls = []
        try:
            if self.guidance_enabled:
                self.extract_reference(wave)
            states = self.task(wave)
        finally:
            # Autograd holds any needed tensors, but no batch cache leaks forward.
            self._reference_inputs = []
        if self.guidance_enabled and self.query_calls != list(range(1, 25)):
            raise RuntimeError("Some task blocks bypassed external Query")
        if any(tuple(h.shape[1:]) != (201, 1024) for h in states):
            raise RuntimeError("Preserve pad64600 and native 1024 dimensions")
        b = self.backend
        # Pure SLS gradient path: do NOT detach task gate descriptors.
        gates = torch.sigmoid(b.fc0(b.descriptors(states)))
        aggregate = sum(h * gates[:, i:i+1] for i, h in enumerate(states))
        pooled = F.max_pool2d(b.selu(b.first_bn(aggregate.unsqueeze(1))), 3, 3)
        flat = pooled.flatten(1)
        if flat.size(1) != 22847:
            raise RuntimeError("Unexpected full-width SLS pooling shape")
        output = F.log_softmax(b.selu(b.fc3(b.selu(b.fc1(flat)))).float(), dim=1)
        if return_details:
            return dict(log_probs=output, states=states, aggregate=aggregate)
        return output

    def trainable_payload(self, sha):
        names = {n for n, p in self.task.named_parameters() if p.requires_grad}
        return dict(format="sls_pretrained_query_v1", pretrained_sha256=sha,
                    backend=self.backend.state_dict(),
                    task_trainable={n: t for n, t in self.task.state_dict().items() if n in names})

    def load_trainable_payload(self, saved, sha):
        if saved["format"] != "sls_pretrained_query_v1" or saved["pretrained_sha256"] != sha:
            raise ValueError("Payload/base mismatch")
        expected = {n for n, p in self.task.named_parameters() if p.requires_grad}
        if set(saved["task_trainable"]) != expected:
            raise ValueError("Trainable parameter set mismatch")
        self.backend.load_state_dict(saved["backend"], strict=True)
        self.task.load_state_dict(saved["task_trainable"], strict=False)
