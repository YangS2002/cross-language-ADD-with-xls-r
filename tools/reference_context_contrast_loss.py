"""Bounded frozen-context weights on the existing classification embedding."""
import math

import torch
from torch.nn import functional as F


def context_contrast(features, labels, similarity, tau=.1, beta=math.log(2)):
    if tau <= 0 or beta < 0 or beta > math.log(2) + 1e-8:
        raise ValueError("Invalid temperature or registered weight bound")
    n = features.shape[0]
    if features.ndim != 2 or labels.shape != (n,) or similarity.shape != (n, n):
        raise ValueError("Contrast shapes do not match")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Binary labels required")
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    f = features.to(dtype)
    u = similarity.detach().to(dtype).clamp(0, 1)
    off = ~torch.eye(n, device=f.device, dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & off
    neg = (labels[:, None] != labels[None, :]) & off
    valid = pos.any(1) & neg.any(1)
    info = dict(valid_fake=int((valid & (labels == 0)).sum()),
                valid_real=int((valid & (labels == 1)).sum()),
                samples_fake=int((labels == 0).sum()), samples_real=int((labels == 1).sum()),
                empty_valid=int(not bool(valid.any())),
                context_std=float(u[off].std(unbiased=False)) if bool(off.any()) else 0.,
                negative_weight_mean=1., negative_weight_min=1., negative_weight_max=1.,
                positive_weight_ratio=1.)
    if not bool(valid.any()):
        return f.sum() * 0, info
    z = F.normalize(f, dim=1, eps=1e-6)
    a = z @ z.T / tau
    ubar = (u * off).sum(1, keepdim=True) / max(n-1, 1)
    log_b = torch.where(neg, beta * (u-ubar), torch.zeros_like(u))
    q_raw = torch.exp(-beta*u) * pos
    q = q_raw / q_raw.sum(1, keepdim=True).clamp_min(1e-20)
    log_den = torch.logsumexp((a + log_b).masked_fill(~off, -torch.inf), dim=1)
    ell = -(q * (a - log_den[:, None])).sum(1)
    classes = [ell[valid & (labels == label)].mean() for label in (0, 1)
               if bool((valid & (labels == label)).any())]
    loss = torch.stack(classes).mean()
    weights = log_b[neg].exp()
    ratios = [float(q_raw[i, pos[i]].max()/q_raw[i, pos[i]].min())
              for i in range(n) if bool(valid[i])]
    info.update(negative_weight_mean=float(weights.mean()),
                negative_weight_min=float(weights.min()), negative_weight_max=float(weights.max()),
                positive_weight_ratio=max(ratios))
    return loss, info


def self_test():
    g = torch.Generator().manual_seed(902020)
    f = torch.randn(6, 5, generator=g, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    u = torch.rand(6, 6, generator=g, dtype=torch.float64)
    u = ((u + u.T)/2).requires_grad_()
    torch.autograd.gradcheck(lambda x: context_contrast(x, y, u)[0], (f,),
                             eps=1e-6, atol=1e-4, rtol=1e-3)
    loss, info = context_contrast(f, y, u)
    loss.backward()
    assert f.grad is not None and torch.isfinite(f.grad).all() and u.grad is None
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    permuted, _ = context_contrast(f.detach()[permutation], y[permutation],
                                 u.detach()[permutation][:, permutation])
    torch.testing.assert_close(loss.detach(), permuted)
    constant, _ = context_contrast(f.detach(), y, torch.full((6, 6), .5))
    z = F.normalize(f.detach(), dim=1)
    a = z @ z.T / .1
    off = ~torch.eye(6, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & off
    expected = -(pos * (a-torch.logsumexp(a.masked_fill(~off, -torch.inf), 1)[:, None])).sum(1)/pos.sum(1)
    torch.testing.assert_close(constant, expected.mean())
    for labels in (torch.zeros(6, dtype=torch.long), torch.tensor([0, 1])):
        x = f.detach()[:len(labels)].clone().requires_grad_()
        empty, empty_info = context_contrast(x, labels, torch.zeros(len(labels), len(labels)))
        empty.backward()
        assert float(empty) == 0 and empty_info["empty_valid"] == 1 and torch.isfinite(x.grad).all()
    one_real, _ = context_contrast(f.detach(), torch.tensor([0, 0, 0, 0, 0, 1]), u.detach())
    huge, _ = context_contrast(f.detach()*1e8, y, u.detach())
    assert torch.isfinite(one_real) and torch.isfinite(huge)
    assert .5-1e-6 <= info["negative_weight_min"] <= info["negative_weight_max"] <= 2+1e-6
    assert info["positive_weight_ratio"] <= 2+1e-6
    return dict(gradcheck=True, permutation_invariant=True, constant_equals_supcon=True,
                empty_pairs_finite=True, one_real_finite=True, large_features_finite=True,
                reference_gradient=False, bounded_weights=True)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
