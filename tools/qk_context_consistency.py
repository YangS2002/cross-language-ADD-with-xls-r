"""Prepared auxiliary probe; not connected to any trainer or launch script.

The detector still routes with actual reference Q/K. This module probes *task
contexts* with the frozen reference projections. Only capture() enables hooks.
Real CUDA/fairseq equivalence, gradients and costs must pass before training.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def context_relation_kl(reference_context, task_context, reference_attention,
                        key_padding_mask=None):
    """Per-utterance KL(A0 || Rcheck), [T,B,D] contexts, frozen Q/K probe.

    Compute all valid query/key positions and heads in FP32, without averaging
    attention heads before KL. Caller selects real source utterances for training.
    Gradients reach task_context, never the teacher context or its projections.
    """
    if reference_context.shape != task_context.shape or task_context.ndim != 3:
        raise ValueError("Reference/task context shape mismatch")
    t, batch, width = task_context.shape
    heads = reference_attention.num_heads
    if batch == 0 or t == 0 or width % heads:
        raise ValueError("Nonempty contexts and integral head size required")
    if reference_context.device != task_context.device:
        raise ValueError("Reference/task contexts must share a device")
    if key_padding_mask is None:
        valid = torch.ones(batch, t, device=task_context.device, dtype=torch.bool)
    else:
        if key_padding_mask.shape != (batch, t) or key_padding_mask.dtype != torch.bool:
            raise ValueError("Expected boolean [B,T] padding mask")
        valid = ~key_padding_mask.to(task_context.device)
    if not bool(valid.any(-1).all()):
        raise ValueError("All-masked utterance")

    def project(context):
        x = context.transpose(0, 1).float()
        q = reference_attention.q_proj
        k = reference_attention.k_proj
        qx = F.linear(x, q.weight.detach().float(),
                      None if q.bias is None else q.bias.detach().float())
        kx = F.linear(x, k.weight.detach().float(),
                      None if k.bias is None else k.bias.detach().float())
        size = width // heads
        qx = qx.reshape(batch, t, heads, size).transpose(1, 2)
        kx = kx.reshape(batch, t, heads, size).transpose(1, 2)
        logits = torch.matmul(qx, kx.transpose(-2, -1)) / math.sqrt(size)
        return F.log_softmax(logits.masked_fill(~valid[:, None, None, :], -torch.inf), -1)

    with torch.autocast(device_type=task_context.device.type, enabled=False):
        with torch.no_grad():
            log_teacher = project(reference_context.detach())
            probability = log_teacher.exp()
            log_teacher = log_teacher.masked_fill(~valid[:, None, None, :], 0)
        log_task = project(task_context)
        # Avoid 0*(-inf - -inf) on excluded keys.
        log_task = log_task.masked_fill(~valid[:, None, None, :], 0)
        row_kl = (probability * (log_teacher - log_task)).sum(-1)
        row_kl = row_kl.masked_fill(~valid[:, None, :], 0)
        result = row_kl.sum((1, 2)) / (heads * valid.sum(-1))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite context relation KL")
    return result


class ContextRelationRecorder:
    """Opt-in hooks for existing PretrainedAttentionSLS; no model mutation.

    Usage (future trainer integration, not an existing launch path):
        with recorder.capture(labels):
            output = model(wave)
            relation_loss = recorder.loss()
        total = original_classification_nll + fixed_lambda * relation_loss

    Default: only label==1 participates in auxiliary gradients. record_all=True
    is an inference-only diagnostic and requires torch.no_grad(). Labels remain
    available for reporting; no target diagnostic is used to fit/select models.
    """
    def __init__(self, model):
        self.model = model
        self.active = False
        self.labels = None
        self.selector = None
        self.values = []
        self.handles = []
        layers = model.task.model.encoder.layers
        if len(layers) != 24 or len(model.reference.model.encoder.layers) != 24:
            raise ValueError("Prepared protocol requires 24 native layers")
        for index, layer in enumerate(layers):
            def capture_input(_module, args, kwargs, index=index):
                if not self.active:
                    return
                if index != len(self.values) or len(model._reference_inputs) != 24:
                    raise RuntimeError("Incomplete reference/layer ordering")
                current = kwargs["query"] if "query" in kwargs else args[0]
                mask = kwargs.get("key_padding_mask")
                original, original_mask = model._reference_inputs[index]
                if current.shape != original.shape or current.shape[1] != len(self.labels):
                    raise RuntimeError("Context/label shapes differ")
                if (mask is None) != (original_mask is None):
                    raise RuntimeError("Reference/task mask presence differs")
                if mask is not None and not torch.equal(mask, original_mask):
                    raise RuntimeError("Reference/task masks differ")
                if not bool(self.selector.any()):
                    self.values.append(current.new_empty(0, dtype=torch.float32))
                    return
                teacher = model.reference.model.encoder.layers[index].self_attn
                selected_mask = None if mask is None else mask[self.selector]
                self.values.append(context_relation_kl(
                    original[:, self.selector], current[:, self.selector],
                    teacher, selected_mask))
            self.handles.append(layer.self_attn.register_forward_pre_hook(capture_input, with_kwargs=True))

    @contextmanager
    def capture(self, labels, *, record_all=False):
        if self.active:
            raise RuntimeError("Nested capture forbidden")
        if labels.ndim != 1 or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Expected binary [B] labels, real=1")
        if record_all and torch.is_grad_enabled():
            raise RuntimeError("All-class diagnostic requires torch.no_grad()")
        device = next(self.model.task.parameters()).device
        self.labels = labels.detach().to(device)
        self.selector = torch.ones_like(self.labels, dtype=torch.bool) if record_all else self.labels == 1
        self.values = []
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            self.labels = None
            self.selector = None
            self.values = []

    def per_layer_values(self):
        if not self.active or len(self.values) != 24:
            raise RuntimeError("Read completed 24-layer capture before context exit")
        return torch.stack(self.values)  # [24,R]; remains empty for all-fake batch.

    def loss(self):
        values = self.per_layer_values()
        return values.mean() if values.numel() else values.new_zeros(())

    def close(self):
        if self.active:
            raise RuntimeError("Cannot remove hooks during capture")
        for handle in self.handles:
            handle.remove()
        self.handles = []
