"""CPU synthetic mathematics only; no speech data, training, or server access."""
import json
from pathlib import Path

import numpy as np


def objective(us, uq, ys, yq, cs, cq, rho=1.0):
    ns = np.linalg.norm(us, axis=1, keepdims=True)
    nq = np.linalg.norm(uq, axis=1, keepdims=True)
    rs, rq = us / ns, uq / nq
    x = np.sqrt(cs)[:, None] * rs
    w = x.T @ np.linalg.solve(x @ x.T + rho * np.eye(len(x)), np.sqrt(cs) * ys)
    margins = yq * (rq @ w)
    loss = float(cq @ np.logaddexp(0.0, -margins))
    ds = -cq * yq / (1.0 + np.exp(margins))
    u = rq.T @ ds
    a = np.linalg.solve(rs.T @ (cs[:, None] * rs) + rho * np.eye(rs.shape[1]), u)
    grs = cs[:, None] * ((ys - rs @ w)[:, None] * a - (rs @ a)[:, None] * w)
    grq = ds[:, None] * w
    gus = (grs - rs * np.sum(rs * grs, axis=1, keepdims=True)) / ns
    guq = (grq - rq * np.sum(rq * grq, axis=1, keepdims=True)) / nq
    return loss, gus, guq, w


def main():
    rng = np.random.default_rng(12)
    us, uq = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    ys, yq = np.array([-1, 1, -1, 1, -1, 1]), np.array([1, -1, 1, -1, 1, -1])
    cs = np.where(ys == 1, .9, .1)
    cq = np.where(yq == 1, .9, .1)
    cs, cq = cs / cs.sum(), cq / cq.sum()
    loss, gs, gq, _ = objective(us, uq, ys, yq, cs, cq)
    h, errors = 1e-6, []
    for array, gradient in ((us, gs), (uq, gq)):
        for index in np.ndindex(array.shape):
            saved = array[index]
            array[index] = saved + h
            plus = objective(us, uq, ys, yq, cs, cq)[0]
            array[index] = saved - h
            minus = objective(us, uq, ys, yq, cs, cq)[0]
            array[index] = saved
            errors.append(abs((plus - minus) / (2 * h) - gradient[index]))
    labels = np.array([-1., -1., 1., 1.])
    weights = np.full(4, .25)
    e1 = np.array([1., 0., 0.])
    e2 = np.array([0., 1., 0.])
    common = labels[:, None] * e1
    cases = {}
    for name, support, query in (
        ("shared_authenticity_cue", common, common),
        ("shared_source_shortcut_indistinguishable", common, common),
        ("orthogonal_attack_specific_cues", common, labels[:, None] * e2),
        ("opposite_query_cue", common, -common),
        ("balanced_collapsed_features", np.tile(e1, (4, 1)), np.tile(e1, (4, 1))),
        ("distinct_fake_modes_shared_cue", common,
         common + np.array([[0., 0., 0.], [0., 0., 0.], [0., .5, 0.], [0., -.5, 0.]])),
    ):
        value, g_support, g_query, fitted = objective(support, query, labels, labels, weights, weights)
        cases[name] = dict(loss=value, support_gradient_norm=float(np.linalg.norm(g_support)),
                           query_gradient_norm=float(np.linalg.norm(g_query)),
                           fitted_weight_norm=float(np.linalg.norm(fitted)))
    result = dict(status="synthetic_mathematics_only_not_speech_validation", rho=1.0,
                  gradient_check=dict(max_abs_error=max(errors), support_gradient_norm=float(np.linalg.norm(gs)),
                                      query_gradient_norm=float(np.linalg.norm(gq)), loss=loss),
                  cases=cases,
                  limitations=["Same source shortcut gives exactly same objective as a transferable cue.",
                               "Nonzero loss does not guarantee nonzero representation gradients.",
                               "No evidence about XLS-R, cross-language EER, or seed stability."])
    if max(errors) > 1e-7:
        raise AssertionError("Analytic support/query derivatives failed finite differences")
    destination = Path(__file__).resolve().parents[1] / "experiments/QK_CROSS_ATTACK_READOUT_MATH_2026-10-12.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
