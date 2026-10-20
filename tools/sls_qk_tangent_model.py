"""First-order adaptation of the complete original-QK backbone.

Absolute task parameters retain the baseline AdamW semantics. The frozen
reference supplies primal activations; only directional increments propagate
through the task branch. The SLS backend is unchanged and nonlinear.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


def linear_tangent(delta, primal, task, reference):
    bias = None if task.bias is None else task.bias - reference.bias
    return (F.linear(delta, reference.weight)
            + F.linear(primal, task.weight - reference.weight, bias))


def norm_tangent(delta, primal, task, reference, output_dtype):
    # The primal statistics and all derivative coefficients are constants.
    with torch.no_grad():
        z = primal.float()
        centered = z - z.mean(-1, keepdim=True)
        inv = torch.rsqrt(centered.square().mean(-1, keepdim=True) + reference.eps)
        normalized = centered * inv
    d = delta.float()
    dcentered = d - d.mean(-1, keepdim=True)
    response = inv * (dcentered - normalized * (normalized * dcentered).mean(-1, keepdim=True))
    return (reference.weight * response
            + (task.weight - reference.weight) * normalized
            + (task.bias - reference.bias)).to(output_dtype)


def activation_tangent(delta, primal, kind="gelu"):
    with torch.no_grad():
        z = primal.float()
        if kind == "gelu":
            derivative = .5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-.5*z*z) / math.sqrt(2*math.pi)
        elif kind in ("gelu_accurate", "gelu_fast"):
            a = math.sqrt(2/math.pi)
            t = torch.tanh(a*(z + .044715*z.pow(3)))
            derivative = .5*(1+t) + .5*z*(1-t*t)*a*(1 + 3*.044715*z*z)
        elif kind == "relu":
            derivative = (z > 0).float()
        else:
            raise ValueError(f"Unsupported activation: {kind}")
    return (delta.float() * derivative).to(primal.dtype)


class QKTangentSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.tangent_enabled = True
        self._capture_tangent = False
        self._primal = {}
        names = ["layer_norm", "encoder.layer_norm", "encoder.pos_conv.0", "encoder.pos_conv.2"]
        if self.reference.model.post_extract_proj is not None:
            names.append("post_extract_proj")
        for index in range(24):
            names.extend(f"encoder.layers.{index}.{suffix}" for suffix in
                         ("fc1", "fc2", "self_attn_layer_norm", "final_layer_norm"))
        modules = dict(self.reference.model.named_modules())
        for name in names:
            def capture(module, args, output, name=name):
                if self._capture_tangent:
                    self._primal[name] = (args[0].detach(), output.detach())
            self._handles.append(modules[name].register_forward_hook(capture))

    def _norm(self, name, delta):
        z, y = self._primal[name]
        return norm_tangent(delta, z, self.task.model.get_submodule(name),
                            self.reference.model.get_submodule(name), y.dtype)

    def _linear(self, name, delta):
        z, _ = self._primal[name]
        return linear_tangent(delta, z, self.task.model.get_submodule(name),
                              self.reference.model.get_submodule(name))

    def _position(self, delta):
        task = self.task.model.encoder.pos_conv[0]
        ref = self.reference.model.encoder.pos_conv[0]
        z, _ = self._primal["encoder.pos_conv.0"]
        # Original fairseq weight_norm(dim=2) coordinates are g and v.
        if not hasattr(task, "weight_g") or not hasattr(task, "weight_v"):
            raise RuntimeError("Expected original weight_norm coordinates")
        v, g = ref.weight_v, ref.weight_g
        with torch.no_grad():
            radius = v.square().sum((0, 1), keepdim=True).sqrt()
            unit = v / radius
        dv, dg = task.weight_v - v, task.weight_g - g
        dw = dg * unit + (g/radius) * (dv - unit*(unit*dv).sum((0, 1), keepdim=True))
        args = dict(stride=task.stride, padding=task.padding,
                    dilation=task.dilation, groups=task.groups)
        result = (F.conv1d(delta.transpose(1, 2), ref.weight, None, **args)
                  + F.conv1d(z, dw, task.bias-ref.bias, **args))
        result = self.reference.model.encoder.pos_conv[1](result)
        return activation_tangent(result, self._primal["encoder.pos_conv.2"][0]).transpose(1, 2)

    def _attention(self, index, delta):
        ref = self.reference.model.encoder.layers[index].self_attn
        task = self.task.model.encoder.layers[index].self_attn
        z, padding = self._reference_inputs[index]
        steps, batch, width = z.shape
        heads = ref.num_heads
        head_width = width // heads

        def split(t):
            return t.reshape(steps, batch*heads, head_width).transpose(0, 1)

        def merge(t):
            return t.transpose(0, 1).contiguous().reshape(steps, batch, width)

        if ref.bias_k is not None or ref.bias_v is not None or ref.add_zero_attn:
            raise RuntimeError("Unsupported reference attention tokens")
        with torch.no_grad():
            q, k, v = split(ref.q_proj(z)), split(ref.k_proj(z)), split(ref.v_proj(z))
            # Match the baseline's need_weights=True bmm/softmax path.
            scores = torch.bmm(q * (head_width ** -.5), k.transpose(1, 2))
            if padding is not None:
                scores = scores.view(batch, heads, steps, steps).masked_fill(
                    padding[:, None, None, :], float("-inf")).view(batch*heads, steps, steps)
            attention = F.softmax(scores, dim=-1)
            context = merge(torch.bmm(attention, v))
        dv = split(linear_tangent(delta, z, task.v_proj, ref.v_proj))
        dc = merge(torch.bmm(attention, dv))
        self.query_calls.append(index+1)
        return linear_tangent(dc, context, task.out_proj, ref.out_proj)

    def _ffn(self, index, delta):
        prefix = f"encoder.layers.{index}."
        layer = self.reference.model.encoder.layers[index]
        result = self._linear(prefix+"fc1", delta)
        result = activation_tangent(result, self._primal[prefix+"fc1"][1], layer.activation_fn.__name__)
        return self._linear(prefix+"fc2", result)

    def forward(self, wave, return_details=False):
        if not self.tangent_enabled:
            return super().forward(wave, return_details=return_details)
        if not self.guidance_enabled or self.query_check_mode is not None:
            raise RuntimeError("Tangent path requires actual, unchanged original QK")
        self._primal = {}
        self._reference_inputs = []
        self.reference_calls = []
        self.query_calls = []
        numpy_state = np.random.get_state()
        self._capture_tangent = True
        try:
            with torch.no_grad():
                original_states = self.reference(wave)
        finally:
            self._capture_tangent = False
            np.random.set_state(numpy_state)
        try:
            if len(self._reference_inputs) != 24:
                raise RuntimeError("Incomplete reference inputs")
            delta = self._norm("layer_norm", torch.zeros_like(self._primal["layer_norm"][0]))
            if self.task.model.post_extract_proj is not None:
                delta = self._linear("post_extract_proj", delta)
            delta = delta + self._position(delta)
            encoder = self.task.model.encoder
            if not encoder.layer_norm_first:
                delta = self._norm("encoder.layer_norm", delta)
            length = delta.shape[1]
            pad = self._reference_inputs[0][0].shape[0] - length
            if pad < 0:
                raise RuntimeError("Reference padding mismatch")
            if pad:
                delta = F.pad(delta, (0, 0, 0, pad))
            delta = delta.transpose(0, 1)
            states = []
            for index, layer in enumerate(encoder.layers):
                # Match fairseq's eval-mode NumPy draws in the task branch.
                np.random.random()
                prefix = f"encoder.layers.{index}."
                if layer.layer_norm_first:
                    delta = delta + self._attention(index, self._norm(prefix+"self_attn_layer_norm", delta))
                    delta = delta + self._ffn(index, self._norm(prefix+"final_layer_norm", delta))
                else:
                    delta = self._norm(prefix+"self_attn_layer_norm", delta + self._attention(index, delta))
                    delta = self._norm(prefix+"final_layer_norm", delta + self._ffn(index, delta))
                primal = original_states[index]
                addition = delta[:length].transpose(0, 1)
                states.append((primal.float() + addition.float()).to(primal.dtype))
            b = self.backend
            gates = torch.sigmoid(b.fc0(b.descriptors(states)))
            aggregate = sum(h*gates[:, i:i+1] for i, h in enumerate(states))
            pooled = F.max_pool2d(b.selu(b.first_bn(aggregate.unsqueeze(1))), 3, 3)
            flat = pooled.flatten(1)
            if flat.shape[1] != 22847:
                raise RuntimeError("Unexpected SLS feature dimensions")
            output = F.log_softmax(b.selu(b.fc3(b.selu(b.fc1(flat)))).float(), dim=1)
            if return_details:
                return dict(log_probs=output, states=states, aggregate=aggregate)
            return output
        finally:
            self._primal = {}
            self._reference_inputs = []

    def trainable_payload(self, sha):
        result = super().trainable_payload(sha)
        result["format"] = "sls_qk_tangent_v1"
        return result

    def load_trainable_payload(self, saved, sha):
        if saved["format"] != "sls_qk_tangent_v1":
            raise ValueError("Tangent inference requires a tangent checkpoint")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_attention_v1"), sha)
