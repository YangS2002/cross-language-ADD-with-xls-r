"""Independent synthetic FP64 audit, no speech, weights, Torch or training.

Checks the auxiliary objective and a diagnostic null control mathematically.
It does not verify the CUDA/fairseq integration in qk_context_consistency.py.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np


def logsoftmax(logits, valid):
    logits = np.where(valid[None, :], logits, -np.inf)
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def relation_kl_and_gradient(original, current, wq, wk, bq, bk, valid):
    """One utterance/head independent analytic derivative, including Q/K bias."""
    if not valid.any():
        raise ValueError("All-masked utterance")
    scale = np.sqrt(wq.shape[1])
    q0, k0 = original @ wq + bq, original @ wk + bk
    q, k = current @ wq + bq, current @ wk + bk
    lp0 = logsoftmax(q0 @ k0.T / scale, valid)
    lp = logsoftmax(q @ k.T / scale, valid)
    p0, p = np.exp(lp0), np.exp(lp)
    difference = np.zeros_like(lp)
    difference[:, valid] = lp0[:, valid] - lp[:, valid]
    value = (p0 * difference)[valid].sum() / valid.sum()
    g = (p - p0) / valid.sum()
    g[~valid, :] = 0
    gradient = (g @ k @ wq.T + g.T @ q @ wk.T) / scale
    return float(value), gradient


def signed_log_bucket(offset):
    if offset == 0:
        return (0, 0)
    return ((1 if offset > 0 else -1), abs(int(offset)).bit_length() - 1)


def within_bucket_indices(length, valid):
    """Per-row deterministic reversal inside signed log-distance buckets.

    Each [2^k,2^(k+1)-1] bucket on each side is separate; self is singleton.
    Masked keys are singleton/unchanged. No RNG or content-dependent selection.
    """
    if not valid.any():
        raise ValueError("All-masked utterance")
    mapping = np.tile(np.arange(length), (length, 1))
    for i in range(length):
        buckets = {}
        for j in np.flatnonzero(valid):
            buckets.setdefault(signed_log_bucket(j - i), []).append(j)
        for js in buckets.values():
            mapping[i, js] = js[::-1]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing audit")
    rng = np.random.default_rng(20261011)
    t, width, head = 19, 7, 4
    original = rng.normal(size=(t, width))
    current = original + .2 * rng.normal(size=(t, width))
    wq, wk = rng.normal(size=(width, head)), rng.normal(size=(width, head))
    bq, bk = rng.normal(size=head), rng.normal(size=head)
    valid = np.ones(t, dtype=bool)
    valid[[2, 16, 18]] = False  # Include non-prefix padding.
    value, gradient = relation_kl_and_gradient(original, current, wq, wk, bq, bk, valid)
    direction = rng.normal(size=current.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = relation_kl_and_gradient(original, current + eps * direction, wq, wk, bq, bk, valid)[0]
    minus = relation_kl_and_gradient(original, current - eps * direction, wq, wk, bq, bk, valid)[0]
    fd = (plus - minus) / (2 * eps)
    derivative = float((gradient * direction).sum())
    error = abs(fd - derivative)
    assert error < 1e-7, (fd, derivative)
    initial, initial_gradient = relation_kl_and_gradient(original, original, wq, wk, bq, bk, valid)
    assert abs(initial) < 1e-14 and np.linalg.norm(initial_gradient) < 1e-14
    assert np.linalg.norm(gradient[~valid]) == 0
    try:
        relation_kl_and_gradient(original, current, wq, wk, bq, bk, np.zeros(t, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("All-masked input accepted")

    # Generic hidden Gram permits rotations that a fixed learned Q/K probe need not.
    rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
    rotated = original @ rotation
    centered = original - original.mean(0)
    centered_rotated = rotated - rotated.mean(0)
    u = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    ur = centered_rotated / np.linalg.norm(centered_rotated, axis=1, keepdims=True)
    gram_error = float(np.abs(u @ u.T - ur @ ur.T).max())
    rotated_kl = relation_kl_and_gradient(original, rotated, wq, wk, bq, bk, valid)[0]
    assert gram_error < 1e-12 and rotated_kl > 1e-4

    p = np.exp(logsoftmax((original @ wq + bq) @ (original @ wk + bk).T / np.sqrt(head), valid))
    mapping = within_bucket_indices(t, valid)
    permuted = np.take_along_axis(p, mapping, axis=-1)
    restored = np.take_along_axis(permuted, mapping, axis=-1)
    assert np.array_equal(restored, p)  # Exact sham/round-trip.
    assert np.array_equal(np.sort(p, axis=-1), np.sort(permuted, axis=-1))
    assert np.array_equal(np.diag(p), np.diag(permuted))
    assert np.array_equal(permuted[:, ~valid], p[:, ~valid])
    for i in range(t):
        for bucket in set(signed_log_bucket(j - i) for j in np.flatnonzero(valid)):
            js = [j for j in np.flatnonzero(valid) if signed_log_bucket(j - i) == bucket]
            assert abs(p[i, js].sum() - permuted[i, js].sum()) < 1e-14
    entropy = lambda a: -(np.where(a > 0, a, 1) * np.log(np.where(a > 0, a, 1))).sum(-1)
    assert np.max(np.abs(entropy(p) - entropy(permuted))) < 1e-13
    messages = rng.normal(size=(t, head))
    constant = np.ones((t, head))
    assert np.max(np.abs(p @ constant - permuted @ constant)) < 1e-14
    tv = .5 * np.abs(p - permuted).sum(-1)
    changed = float(np.linalg.norm((p - permuted) @ messages))
    assert float(tv[valid].mean()) > 1e-4 and changed > 1e-4

    # Parse prepared Torch implementation without importing absent Torch.
    core = Path(__file__).with_name("qk_context_consistency.py")
    ast.parse(core.read_text(encoding="utf-8"))
    result = dict(
        scope="synthetic FP64 mathematics only; not Torch/CUDA/fairseq or detector performance",
        kl=value, directional_gradient_error=error, zero_initial_kl=initial,
        zero_initial_gradient_norm=float(np.linalg.norm(initial_gradient)),
        hidden_gram_rotation_error=gram_error, frozen_qk_rotation_kl=rotated_kl,
        null_mean_tv_valid_queries=float(tv[valid].mean()), null_message_change_l2=changed,
        checks=["fixed-QK KL gradient including both input paths and projection bias",
                "masked query/key exclusion and all-masked rejection", "zero initialization",
                "hidden Gram is not fixed learned QK functionality",
                "null row probabilities/entropy/diagonal/mask/signed bucket mass preserved",
                "null round-trip exact", "constant messages unchanged", "nonconstant messages changed",
                "Torch module AST parses; runtime remains unverified"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
