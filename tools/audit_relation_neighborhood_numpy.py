"""CPU-only synthetic math audit; no XLS-R, audio, GPU, or training.

Shapes: Q/K/V [batch, heads, tokens, head_dim], E [heads, dim, dim].
This verifies the proposed neighborhood, not a real-model implementation.
"""

import json
import math

import numpy as np


def bounded_inputs(q, k, valid):
    if not valid.any(axis=-1).all():
        raise ValueError("Every sample needs at least one valid key")
    qmax = np.linalg.norm(q, axis=-1).max(axis=-1, keepdims=True)[..., None]
    knorm = np.linalg.norm(k, axis=-1)
    kmax = np.where(valid[:, None, :], knorm, 0).max(axis=-1, keepdims=True)[..., None]
    qhat = np.divide(q, qmax, out=np.zeros_like(q), where=qmax > 0)
    khat = np.divide(k, kmax, out=np.zeros_like(k), where=kmax > 0)
    return qhat, khat, kmax


def tilt(qhat, khat, e, rho):
    return rho * np.einsum("bhid,hde,bhje->bhij", qhat, e, khat)


def attention(logits, valid):
    masked = np.where(valid[:, None, None, :], logits, -np.inf)
    maximum = masked.max(axis=-1, keepdims=True)
    z = masked - maximum
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return np.exp(logp), logp


def objective(e, qhat, khat, base, v, w, y, valid, rho):
    a, _ = attention(base + tilt(qhat, khat, e, rho), valid)
    context = a @ v
    pooled = context.mean(axis=2).reshape(y.size, -1)
    logits = pooled @ w
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    weights = np.where(y == 1, .9, .1)
    loss = -np.dot(weights, logp[np.arange(y.size), y]) / weights.sum()
    dz = np.exp(logp)
    dz[np.arange(y.size), y] -= 1
    dz *= (weights / weights.sum())[:, None]
    dc = (dz @ w.T).reshape(y.size, v.shape[1], 1, v.shape[-1]) / v.shape[2]
    da = np.einsum("bhid,bhjd->bhij", np.broadcast_to(dc, context.shape), v)
    dl = a * (da - (a * da).sum(axis=-1, keepdims=True))
    gradient = rho * np.einsum("bhid,bhij,bhje->hde", qhat, dl, khat)
    return float(loss), gradient


def run():
    rng = np.random.default_rng(902020)
    b, h, t, d = 16, 3, 9, 4
    rho = .2
    q = rng.normal(size=(b, h, t, d))
    k = rng.normal(size=q.shape)
    v = rng.normal(size=q.shape)
    valid = np.ones((b, t), dtype=bool)
    valid[::2, -2:] = False
    qhat, khat, kmax = bounded_inputs(q, k, valid)
    base = q @ k.swapaxes(-1, -2) / math.sqrt(d)
    zero = np.zeros((h, d, d))
    a0, log0 = attention(base, valid)
    azero, _ = attention(base + tilt(qhat, khat, zero, rho), valid)
    assert np.array_equal(a0, azero)
    w = rng.normal(size=(h*d, 2))
    y = np.zeros(b, dtype=int)
    y[:2] = 1
    loss0, g = objective(zero, qhat, khat, base, v, w, y, valid, rho)
    numerical = np.zeros_like(g)
    eps = 1e-6
    for index in np.ndindex(g.shape):
        direction = np.zeros_like(zero)
        direction[index] = eps
        plus = objective(direction, qhat, khat, base, v, w, y, valid, rho)[0]
        minus = objective(-direction, qhat, khat, base, v, w, y, valid, rho)[0]
        numerical[index] = (plus-minus)/(2*eps)
    np.testing.assert_allclose(g, numerical, atol=2e-9, rtol=2e-5)
    norms = np.linalg.norm(g, axis=(-2, -1), keepdims=True)
    e = np.divide(g, norms, out=np.zeros_like(g), where=norms > 0)
    delta = tilt(qhat, khat, e, rho)
    assert np.abs(delta).max() <= rho + 1e-12
    anew, lognew = attention(base + delta, valid)
    np.testing.assert_allclose(anew.sum(axis=-1), 1, atol=1e-14)
    expanded = np.broadcast_to(valid[:, None, None, :], anew.shape)
    assert not anew[~expanded].any()
    logratio = lognew[expanded]-log0[expanded]
    assert np.abs(logratio).max() <= 2*rho + 1e-12
    difference = np.zeros_like(a0)
    difference[expanded] = log0[expanded]-lognew[expanded]
    kl = (a0*difference).sum(axis=-1)
    assert kl.min() >= -1e-12
    assert kl.max() <= rho*rho/2 + 1e-12
    tv = .5*np.abs(anew-a0).sum(axis=-1)
    assert tv.max() <= rho/2 + 1e-12
    # The same tilt can be implemented with a dimension-preserving Query change.
    coefficient = np.divide(rho*math.sqrt(d), kmax,
                            out=np.zeros_like(kmax), where=kmax > 0)
    qnew = q + coefficient*np.einsum("bhid,hde->bhie", qhat, e)
    equivalent = qnew @ k.swapaxes(-1, -2)/math.sqrt(d)
    np.testing.assert_allclose(equivalent, base+delta, atol=1e-13, rtol=1e-12)
    loss_adv, _ = objective(e, qhat, khat, base, v, w, y, valid, rho)
    loss_small, _ = objective(.001*e, qhat, khat, base, v, w, y, valid, rho)
    assert loss_small > loss0
    # No contrastive positive/negative requirement: one-class CE is defined.
    one_class_loss, one_class_grad = objective(zero, qhat, khat, base, v, w,
                                             np.zeros_like(y), valid, rho)
    assert np.isfinite(one_class_loss) and np.isfinite(one_class_grad).all()
    # Zero projections must not create NaNs in the normalization.
    qz, kz, _ = bounded_inputs(np.zeros_like(q), np.zeros_like(k), valid)
    assert np.isfinite(tilt(qz, kz, e, rho)).all()
    rejected = False
    try:
        bounded_inputs(q, k, np.zeros_like(valid))
    except ValueError:
        rejected = True
    assert rejected
    shifted, _ = attention(base+6., valid)
    np.testing.assert_allclose(shifted, a0, atol=2e-15, rtol=2e-14)
    qs, ks, _ = bounded_inputs(q*10000., k*.001, valid)
    assert np.abs(tilt(qs, ks, e, rho)).max() <= rho+1e-12
    result = dict(
        scope="synthetic_CPU_FP64_math_only_not_XLSR_or_cross_lingual_validation",
        rho=rho, zero_equivalence=True, gradient_max_error=float(np.abs(g-numerical).max()),
        query_factorization_max_error=float(np.abs(equivalent-base-delta).max()),
        maximum_logit_change=float(np.abs(delta).max()),
        maximum_absolute_log_probability_ratio=float(np.abs(logratio).max()),
        maximum_row_kl=float(kl.max()), row_kl_bound=rho*rho/2,
        maximum_total_variation=float(tv.max()), total_variation_bound=rho/2,
        synthetic_clean_nll=loss0, synthetic_one_step_nll=loss_adv,
        one_step_is_approximate_not_exact_inner_max=True,
        one_class_loss_defined=True, zero_norm_safe=True, all_masked_rejected=True,
        row_shift_invariant=True, rescaled_QK_neighborhood_bounded=True,
        real_model_BF16_gradient_BN_memory_speed_checks="not_performed")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
