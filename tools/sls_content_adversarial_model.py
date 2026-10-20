"""Unmodified SLS inference; class-conditional content suppression during training."""
import torch
from torch import nn
from torch.nn import functional as F
from tools.sls_reference_model import XLSRStates, FullWidthSLS


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def balanced_content_kl(logits, target, labels, valid):
    """Equal class contributions without altering main sampling or class loss."""
    if target.requires_grad:
        raise ValueError('Content reference must be detached')
    if logits.shape != target.shape or labels.shape != valid.shape:
        raise ValueError('Content batch shapes disagree')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite content input')
    if (target < 0).any() or not torch.allclose(target[valid].sum(-1),
            torch.ones_like(target[valid].sum(-1)), atol=1e-4):
        raise ValueError('Valid reference must be a probability distribution')
    per_item = F.kl_div(F.log_softmax(logits.float(), -1), target.float(),
                         reduction='none').sum(-1)
    terms = [per_item[valid & (labels == y)].mean() for y in (0, 1)
             if bool((valid & (labels == y)).any())]
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def sls_embedding(backend, states):
    if len(states) != 24 or any(tuple(h.shape[1:]) != (201, 1024) for h in states):
        raise ValueError('Keep SLS 24 layers, 1024 width and pad64600')
    # NO detached gate descriptor: identical to fixed ConditionalResidualSLS arm=sls.
    gates = torch.sigmoid(backend.fc0(backend.descriptors(states)))
    z = sum(h * gates[:, i:i+1] for i, h in enumerate(states))
    pooled = F.max_pool2d(backend.selu(backend.first_bn(z.unsqueeze(1))), 3, 3)
    flat = pooled.flatten(1)
    if flat.shape[1] != 22847:
        raise ValueError('Main SLS classifier shape changed')
    return backend.selu(backend.fc1(flat))


class ContentAdversarialSLS(nn.Module):
    def __init__(self, checkpoint=None, task=None, strength=0.1, classes=26):
        super().__init__()
        self.task = XLSRStates(checkpoint, True) if task is None else task
        self.backend = FullWidthSLS()
        # Auxiliary initialization consumes no main-path/loader RNG state.
        with torch.random.fork_rng():
            torch.manual_seed(1731)
            self.content = nn.Sequential(nn.Linear(1026, 256), nn.ReLU(),
                                         nn.Linear(256, classes))
        self.strength = strength

    def forward(self, wave, labels=None, return_details=False):
        embedding = sls_embedding(self.backend, self.task(wave))
        log_probs = F.log_softmax(self.backend.selu(self.backend.fc3(embedding)).float(), 1)
        if not return_details:
            return log_probs
        logits = None
        if labels is not None:
            reversed_embedding = ReverseGradient.apply(embedding.float(), self.strength)
            conditions = F.one_hot(labels, 2).to(reversed_embedding)
            logits = self.content(torch.cat([reversed_embedding, conditions], 1))
        return dict(log_probs=log_probs, embedding=embedding, content_logits=logits)


class FrozenContentReadout(nn.Module):
    """Original XLS-R plus fixed warmup gate and trainable CTC readout ONLY."""
    def __init__(self, checkpoint, gate_state, classes=29):
        super().__init__()
        self.reference = XLSRStates(checkpoint, False)
        self.gate = nn.Linear(1024, 1)
        self.gate.load_state_dict(gate_state)
        self.gate.requires_grad_(False)
        self.readout = nn.Linear(1024, classes)

    @torch.no_grad()
    def frames(self, wave):
        states = self.reference(wave)
        gates = torch.sigmoid(self.gate(FullWidthSLS.descriptors(states)))
        return sum(h * gates[:, i:i+1] for i, h in enumerate(states)).float()

    def forward(self, frames):
        return self.readout(frames.float())

    @torch.no_grad()
    def content_distribution(self, frames, letter_indices):
        posterior = self(frames).softmax(-1)[..., letter_indices].mean(1)
        mass = posterior.sum(-1)
        valid = torch.isfinite(posterior).all(-1) & (mass > 1e-6)
        q = posterior / mass.clamp_min(1e-6).unsqueeze(-1)
        return q, valid, mass


def content_optimizer(model, head_lr=1e-6, ssl_lr=1e-6,
                      adversary_lr=1e-4, weight_decay=1e-4):
    return torch.optim.AdamW([
        dict(params=list(model.backend.parameters()), lr=head_lr),
        dict(params=[p for p in model.task.parameters() if p.requires_grad], lr=ssl_lr),
        dict(params=list(model.content.parameters()), lr=adversary_lr)],
        weight_decay=weight_decay, foreach=False)


def content_training_step(model, optimizer, wave, labels, reference, valid, diagnostics=False):
    """Single-source step; auxiliary targets must be from a validated frozen cache.

    This function does not authorize a formal run with an unvalidated pilot cache.
    GRL scales only upstream gradients; do NOT multiply content loss by lambda again.
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=wave.device.type, dtype=torch.bfloat16,
                        enabled=wave.is_cuda):
        d = model(wave, labels=labels, return_details=True)
    detection = F.nll_loss(d['log_probs'], labels,
                           weight=d['log_probs'].new_tensor([0.1, 0.9]))
    content = balanced_content_kl(d['content_logits'], reference, labels, valid)
    total = detection + content
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite source loss')
    extra = {}
    if diagnostics:
        detection_gradient = torch.autograd.grad(detection, d['embedding'], retain_graph=True)[0].float()
        content_gradient = torch.autograd.grad(content, d['embedding'], retain_graph=True)[0].float()
        extra = dict(detection_embedding_gradient_norm=float(detection_gradient.norm()),
            reversed_content_embedding_gradient_norm=float(content_gradient.norm()),
            content_to_detection_embedding_gradient_ratio=float(
                content_gradient.norm()/detection_gradient.norm().clamp_min(1e-12)))
    total.backward()
    main = list(model.backend.parameters()) + [p for p in model.task.parameters() if p.requires_grad]
    main_norm = torch.nn.utils.clip_grad_norm_(main, 5., error_if_nonfinite=True)
    auxiliary_norm = torch.nn.utils.clip_grad_norm_(model.content.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return dict(classification_loss=float(detection.detach()),content_loss=float(content.detach()),
                main_gradient_norm=float(main_norm),auxiliary_gradient_norm=float(auxiliary_norm),
                log_probs=d['log_probs'].detach(),valid_content=int(valid.sum()),**extra)
