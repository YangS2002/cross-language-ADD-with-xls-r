"""Source-only adversarial neighborhood around actual frozen XLS-R relations."""
from types import MethodType
import math
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_attention_model import PretrainedAttentionSLS


def bounded_tilt(q, k, e, rho, padding_mask=None):
    """Projected Q/K [B,H,T,D]; E [H,D,D]. FP32 controls the logit bound."""
    with torch.autocast(q.device.type, enabled=False):
        qf, kf = q.detach().float(), k.detach().float()
        qmax = qf.norm(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        knorm = kf.norm(dim=-1)
        if padding_mask is not None:
            if bool(padding_mask.all(dim=-1).any()):
                raise ValueError("All keys are masked")
            knorm = knorm.masked_fill(padding_mask[:, None, :], 0)
        kmax = knorm.amax(dim=-1, keepdim=True).unsqueeze(-1)
        qhat = qf / qmax.clamp_min(1e-12)
        khat = kf / kmax.clamp_min(1e-12)
        return rho * torch.matmul(torch.matmul(qhat, e.unsqueeze(0)), khat.transpose(-1, -2))


class RelationNeighborhoodSLS(PretrainedAttentionSLS):
    def __init__(self, checkpoint, rho=.2):
        super().__init__(checkpoint)
        self.rho = float(rho)
        self.neighborhood = None  # Not a parameter, buffer, optimizer state or saved weight.
        self._batch_reference = None
        self._wave_pointer = None
        self._projected_reference = {}
        self._reference_extractions = 0
        self.collect_diagnostics = False
        self.relation_diagnostics = []
        self.last_step_info = {}
        owner = weakref.ref(self)
        layers = self.task.model.encoder.layers
        shapes = {(l.self_attn.num_heads, l.self_attn.head_dim) for l in layers}
        if len(shapes) != 1:
            raise ValueError("This candidate requires equal native head shapes")
        self.heads, self.head_dim = shapes.pop()
        self.neighborhood_shape = (len(layers), self.heads, self.head_dim, self.head_dim)
        for index, layer in enumerate(layers):
            original_guided = layer.self_attn.forward

            def guided(module, query, key, value, key_padding_mask=None,
                       incremental_state=None, need_weights=True, static_kv=False,
                       attn_mask=None, before_softmax=False, need_head_weights=False,
                       index=index, original_guided=original_guided):
                model = owner()
                if model.neighborhood is None:
                    return original_guided(query, key, value, key_padding_mask,
                                           incremental_state, need_weights, static_kv,
                                           attn_mask, before_softmax, need_head_weights)
                if (not model.guidance_enabled or incremental_state is not None
                        or static_kv or before_softmax or need_head_weights
                        or module.onnx_trace or module.bias_k is not None
                        or module.bias_v is not None or module.add_zero_attn
                        or module.training or module.dropout_module.apply_during_inference):
                    raise RuntimeError("Unsupported neighborhood attention path")
                if index != len(model.query_calls) or len(model._reference_inputs) != 24:
                    raise RuntimeError("Actual reference Q/K missing or out of order")
                source, source_mask = model._reference_inputs[index]
                if source.shape != query.shape or key.shape != query.shape or value.shape != query.shape:
                    raise RuntimeError("Reference/task shapes differ")
                if ((source_mask is None) != (key_padding_mask is None)
                        or (source_mask is not None and not torch.equal(source_mask, key_padding_mask))):
                    raise RuntimeError("Reference/task masks differ")
                model.query_calls.append(index+1)
                t, b, width = query.shape
                heads, dim = module.num_heads, module.head_dim
                if index not in model._projected_reference:
                    ref = model.reference.model.encoder.layers[index].self_attn
                    with torch.no_grad():
                        q0 = F.linear(source, ref.q_proj.weight, ref.q_proj.bias)
                        k0 = F.linear(source, ref.k_proj.weight, ref.k_proj.bias)
                        q0 = q0.view(t, b, heads, dim).permute(1, 2, 0, 3).contiguous()
                        k0 = k0.view(t, b, heads, dim).permute(1, 2, 0, 3).contiguous()
                    model._projected_reference[index] = (q0, k0)
                q0, k0 = model._projected_reference[index]
                v = F.linear(value, module.v_proj.weight, module.v_proj.bias)
                v = v.view(t, b*heads, dim).transpose(0, 1)
                q = q0.reshape(b*heads, t, dim) * math.sqrt(1./dim)
                k = k0.reshape(b*heads, t, dim)
                # Match native need_weights=True bmm path. Only the additive tilt is new.
                base_logits = torch.bmm(q, k.transpose(1, 2))
                delta = bounded_tilt(q0, k0, model.neighborhood[index], model.rho, key_padding_mask)
                logits = base_logits.float() + delta.reshape(b*heads, t, t)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        logits = logits.masked_fill(attn_mask.unsqueeze(0), -torch.inf)
                    else:
                        logits = logits + attn_mask.float().unsqueeze(0)
                if key_padding_mask is not None:
                    logits = logits.view(b, heads, t, t).masked_fill(
                        key_padding_mask[:, None, None, :], -torch.inf).view(b*heads, t, t)
                probabilities = torch.softmax(logits, dim=-1)
                weights = probabilities.to(base_logits.dtype)
                context = torch.bmm(weights, v)
                context = context.transpose(0, 1).contiguous().view(t*b, width)
                output = F.linear(context, module.out_proj.weight, module.out_proj.bias).view(t, b, width)
                if model.collect_diagnostics:
                    model._diagnose(index, base_logits, delta, probabilities, key_padding_mask, attn_mask)
                returned = weights.view(b, heads, t, t).mean(1) if need_weights else None
                return output, returned

            layer.self_attn.forward = MethodType(guided, layer.self_attn)

    def begin_batch(self, wave):
        if self._batch_reference is not None or self.neighborhood is not None:
            raise RuntimeError("Previous batch cache not cleared")
        super().extract_reference(wave)
        self._reference_extractions += 1
        self._batch_reference = tuple(self._reference_inputs)
        self._wave_pointer = wave.data_ptr()
        self._projected_reference = {}

    def extract_reference(self, wave):
        if self._batch_reference is None:
            self._reference_extractions += 1
            return super().extract_reference(wave)
        if wave.data_ptr() != self._wave_pointer:
            raise RuntimeError("Cannot reuse reference for a different waveform batch")
        self._reference_inputs = list(self._batch_reference)
        self.reference_calls = list(range(1, 25))

    def zeros(self, requires_grad=True):
        device = next(self.task.parameters()).device
        self.neighborhood = torch.zeros(self.neighborhood_shape, device=device,
                                        dtype=torch.float32, requires_grad=requires_grad)
        return self.neighborhood

    def end_batch(self):
        self.neighborhood = None
        self._batch_reference = None
        self._wave_pointer = None
        self._projected_reference = {}
        self._reference_inputs = []
        self.collect_diagnostics = False

    @torch.no_grad()
    def _diagnose(self, index, base, delta, perturbed, padding, mask):
        b, h, t, _ = delta.shape
        sample = torch.tensor(sorted({0, b-1}), device=delta.device)
        rows = torch.tensor(sorted({0, t//2, t-1}), device=delta.device)
        old = base.float().view(b, h, t, t)[sample][:, :, rows, :]
        new = perturbed.view(b, h, t, t)[sample][:, :, rows, :]
        if mask is not None:
            selected_mask = mask[rows]
            old = old.masked_fill(selected_mask, -torch.inf) if mask.dtype == torch.bool else old+selected_mask
        if padding is not None:
            old = old.masked_fill(padding[sample, None, None, :], -torch.inf)
        old_log = F.log_softmax(old, -1)
        old_p = old_log.exp()
        safe = old_p > 0
        difference = torch.where(safe, old_log-new.clamp_min(1e-30).log(), 0)
        kl = (old_p*difference).sum(-1)
        tv = .5*(old_p-new).abs().sum(-1)
        d = delta[sample][:, :, rows, :]
        record = dict(layer=index+1, max_abs_tilt=float(d.abs().max()),
                      mean_abs_tilt=float(d.abs().mean()), mean_row_kl=float(kl.mean()),
                      max_row_kl=float(kl.max()), mean_row_tv=float(tv.mean()),
                      max_row_tv=float(tv.max()))
        if not all(math.isfinite(v) for k, v in record.items() if k != "layer"):
            raise FloatingPointError("Nonfinite relation diagnostic")
        if record["max_abs_tilt"] > self.rho+1e-5 or record["max_row_kl"] > self.rho**2/2+1e-5:
            raise RuntimeError("FP32 relation neighborhood exceeded its bound")
        self.relation_diagnostics.append(record)

    def trainable_payload(self, sha):
        result = super().trainable_payload(sha)
        result.update(format="sls_relation_neighborhood_v1", neighborhood_rho=self.rho,
                      training_neighborhood="bounded_normalized_QK_bilinear_one_step",
                      inference_neighborhood="zero_original_attention")
        return result

    def load_trainable_payload(self, saved, sha):
        if (saved["format"] != "sls_relation_neighborhood_v1"
                or saved["neighborhood_rho"] != self.rho
                or saved["inference_neighborhood"] != "zero_original_attention"):
            raise ValueError("Wrong relation-neighborhood payload")
        super().load_trainable_payload(dict(saved, format="sls_pretrained_attention_v1"), sha)
