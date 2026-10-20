"""Original per-unit attention mass, adaptive within-unit task Key routing."""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from tools.sls_pretrained_query_model import PretrainedQuerySLS


class _UnitMassAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, logits, groups, reference_logits):
        index=groups[:,None,None,:].expand_as(logits)
        b=logits.float()
        maxima=torch.full_like(b,-torch.inf)
        maxima.scatter_reduce_(-1,index,b,reduce="amax",include_self=True)
        exp=torch.exp(b-maxima.gather(-1,index))
        totals=torch.zeros_like(b).scatter_add(-1,index,exp)
        masses=torch.zeros_like(b).scatter_add(-1,index,original.float())
        attention=masses.gather(-1,index)*(exp/totals.gather(-1,index))
        if reference_logits is not None:
            # Exact mathematical identity at equal logits. Avoid group summation
            # roundoff being amplified through 24 blocks. The backward below is
            # the actual conditional-softmax Jacobian, including at this point.
            equal=(logits==reference_logits).all(-1,keepdim=True)
            attention=torch.where(equal,original.float(),attention)
        ctx.save_for_backward(attention,masses,groups)
        ctx.logit_dtype=logits.dtype
        return attention

    @staticmethod
    def backward(ctx, gradient):
        attention,masses,groups=ctx.saved_tensors
        index=groups[:,None,None,:].expand_as(attention)
        g=gradient.float()
        sums=torch.zeros_like(attention).scatter_add(-1,index,attention*g)
        safe_mass=torch.where(masses>0,masses,torch.ones_like(masses))
        mean=(sums/safe_mass).gather(-1,index)
        return None,(attention*(g-mean)).to(ctx.logit_dtype),None,None


def unit_mass_attention(original, logits, groups, reference_logits=None):
    """Exact conditional-softmax Jacobian; frozen original probabilities/groups.

    Reference logits enable the exact initialization identity. This is neither
    a detached task-attention path nor a straight-through surrogate gradient.
    """
    if original.shape!=logits.shape or logits.shape[-1]!=groups.shape[-1]:
        raise ValueError("Attention/partition shape mismatch")
    return _UnitMassAttention.apply(original.detach(),logits,groups,reference_logits)


class UnitMassSLS(PretrainedQuerySLS):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.partition_override = None  # Diagnostic endpoints only, never formal training.
        self.routing_mode = "unit_mass"  # Diagnostic reference/query paths are explicit.
        self.audit_attention = False
        self.attention_audit = []
        self._unit_groups = None
        owner = weakref.ref(self)
        quantizer = self.reference.model.quantizer
        if (quantizer is None or not quantizer.time_first
                or self.reference.model.input_quantizer is not None):
            raise RuntimeError("Expected original target quantizer on normalized BTC features")

        def capture_units(_module, _args, features):
            model = owner()
            q = model.reference.model.quantizer
            drop = features.shape[1] % model.reference.model.crop_seq_to_multiple
            if drop:
                features = features[:,:-drop]
            scores = q.weight_proj(features).reshape(*features.shape[:2],q.groups,q.num_vars)
            codes = scores.argmax(-1)
            joint = torch.zeros_like(codes[...,0])
            for g in range(q.groups):
                joint = joint*q.num_vars+codes[...,g]
            model._unit_groups = torch.stack([
                torch.unique(row, sorted=True, return_inverse=True)[1] for row in joint])

        self._handles.append(self.reference.model.layer_norm.register_forward_hook(capture_units))
        for index, layer in enumerate(self.task.model.encoder.layers):
            native = self._native_forwards[index]

            def guided(module, query, key, value, key_padding_mask=None,
                       incremental_state=None, need_weights=True, static_kv=False,
                       attn_mask=None, before_softmax=False, need_head_weights=False,
                       index=index, native=native):
                model = owner()
                if not model.guidance_enabled:
                    return native(query,key,value,key_padding_mask,incremental_state,need_weights,
                                  static_kv,attn_mask,before_softmax,need_head_weights)
                if (incremental_state is not None or static_kv or before_softmax
                        or need_head_weights or module.onnx_trace or query.device.type!="cuda"
                        or attn_mask is not None or key_padding_mask is not None
                        or module.bias_k is not None or module.bias_v is not None
                        or module.add_zero_attn or module.training):
                    raise RuntimeError("Unsupported path; keep original eval SSL and raw pad64600")
                if index!=len(model.query_calls) or len(model._reference_inputs)!=24:
                    raise RuntimeError("Missing or out-of-order original contexts")
                z0, mask = model._reference_inputs[index]
                if mask is not None or z0.shape!=query.shape or key.shape!=query.shape or value.shape!=query.shape:
                    raise RuntimeError("Original/task contexts differ")
                t,b,d = query.shape
                h = module.num_heads
                head = d//h
                def split(x):
                    return x.reshape(t,b,h,head).permute(1,2,0,3)
                original_module = model.reference.model.encoder.layers[index].self_attn
                with torch.no_grad():
                    q0 = split(original_module.q_proj(z0))*(head**-.5)
                    k0 = split(original_module.k_proj(z0))
                    original_logits=torch.matmul(q0,k0.transpose(-1,-2))
                    original = torch.softmax(original_logits.float(),dim=-1)
                if model.query_check_mode=="zero":
                    q_task = torch.zeros_like(q0)
                elif model.query_check_mode is None:
                    q_task = q0
                else:
                    raise ValueError("Unknown diagnostic query mode")
                logits = torch.matmul(q_task,split(module.k_proj(key)).transpose(-1,-2))
                groups = model._unit_groups
                if groups is None or groups.shape!=(b,t):
                    raise RuntimeError("Missing aligned original units")
                if model.partition_override=="singleton":
                    groups = torch.arange(t,device=query.device)[None,:].expand(b,-1)
                elif model.partition_override=="one_group":
                    groups = torch.zeros_like(groups)
                elif model.partition_override is not None:
                    raise ValueError("Unknown partition endpoint")
                if model.routing_mode=="unit_mass":
                    attention = unit_mass_attention(original,logits,groups,original_logits)
                elif model.routing_mode=="reference":
                    attention = original
                elif model.routing_mode=="query":
                    attention = torch.softmax(logits.float(),dim=-1)
                else:
                    raise ValueError("Unknown routing mode")
                if model.audit_attention:
                    with torch.no_grad():
                        idx=groups[:,None,None,:].expand_as(attention)
                        p0=torch.zeros_like(original).scatter_add(-1,idx,original)
                        p=torch.zeros_like(attention).scatter_add(-1,idx,attention.detach())
                        model.attention_audit.append(dict(layer=index+1,
                            mass_max_error=float((p-p0).abs().max()),
                            row_sum_max_error=float((attention.detach().sum(-1)-1).abs().max()),
                            attention_max_change=float((attention.detach()-original).abs().max())))
                model.query_calls.append(index+1)
                task_values=split(module.v_proj(value))
                probabilities=attention.to(task_values.dtype)
                message=torch.matmul(probabilities,task_values)
                message=message.permute(2,0,1,3).reshape(t,b,d)
                output=module.out_proj(message)
                return output, probabilities.mean(1) if need_weights else None

            layer.self_attn.forward=MethodType(guided,layer.self_attn)

    def forward(self, wave, return_details=False):
        self.attention_audit=[]
        try:
            return super().forward(wave,return_details)
        finally:
            self._unit_groups=None

    def trainable_payload(self, sha):
        payload=super().trainable_payload(sha)
        payload.update(format="sls_unit_mass_v1",unit_policy="original_quantizer_joint_argmax")
        return payload

    def load_trainable_payload(self,saved,sha):
        if saved.get("format")!="sls_unit_mass_v1" or saved.get("unit_policy")!="original_quantizer_joint_argmax":
            raise ValueError("Wrong method/partition payload")
        super().load_trainable_payload({**saved,"format":"sls_pretrained_query_v1"},sha)
