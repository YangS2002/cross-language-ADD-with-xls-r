"""One candidate; reuse matching Attention training/evaluation without edits."""
import argparse
import csv
import gc
import json
from functools import partial
from pathlib import Path
import shutil
import time

import torch
import yaml

from tools import train_sls_pretrained_attention as base
from tools.sls_reference_difference_model import ReferenceDifferenceSLS, mix_context

ARM = "reference_difference"
_signature = base.signature
_step = base.step
_train_epoch = base.train_epoch


def signature(c, path):
    saved = _signature(c, path)
    saved["shared_trainer"] = saved["trainer"]
    saved["trainer"] = base.digest(__file__)
    saved["model"] = base.digest("tools/sls_reference_difference_model.py")
    saved["dependencies"]["tools/sls_pretrained_attention_model.py"] = base.digest(
        "tools/sls_pretrained_attention_model.py")
    return saved


def make_optimizer(model, c):
    return torch.optim.AdamW([
        dict(params=list(model.backend.parameters()), lr=c["head_lr"]),
        dict(params=[p for p in model.task.parameters() if p.requires_grad], lr=c["ssl_lr"]),
        dict(params=[model.difference_gain], lr=c["gain_lr"], weight_decay=0.)],
        weight_decay=c["weight_decay"], foreach=False)


def step(model, opt, wave, labels):
    model.source_steps = getattr(model, "source_steps", 0)+1
    model.record_difference = model.source_steps == 1 or model.source_steps % 200 == 0
    result = _step(model, opt, wave, labels)
    model.project_gains()
    if model.record_difference:
        values = torch.stack(model.difference_diagnostics).cpu().tolist()
        print("DIFFERENCE_ACCESS", json.dumps(dict(
            source_step=model.source_steps, gain=model.difference_gain.detach().cpu().tolist(),
            delta_over_context=[v[0] for v in values],
            contribution_over_context=[v[1] for v in values],
            ratios_are_projected_branch_norms_not_content_measures=True)), flush=True)
    model.record_difference = False
    return result


def preflight(c, dataset, sha):
    start = time.perf_counter()
    half = c["batch_size"]//2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"][:half])
    rows = [dataset[i] for i in ids]
    wave = torch.stack([r["audio"] for r in rows]).cuda()
    labels = torch.tensor([int(r["label"]) for r in rows], device="cuda")
    base.set_seed(c["seed"])
    model = base.PretrainedAttentionSLS(c["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    comparisons = []
    for dtype, prefix in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            model.difference_enabled = False
            original = model(wave[:2], return_details=True)
            expected = [h.float().cpu() for h in original["states"]]
            expected_score = original["log_probs"].cpu()
            del original
            model.difference_enabled = True
            proposed = model(wave[:2], return_details=True)
            errors = []
            for old, new in zip(expected, proposed["states"]):
                new = new.float().cpu()
                torch.testing.assert_close(old, new, atol=0, rtol=0)
                errors.append(float((old-new).abs().max()))
            torch.testing.assert_close(expected_score, proposed["log_probs"].cpu(), atol=0, rtol=0)
            comparisons.append(dict(dtype=prefix, layer_max_errors=errors, score_max_error=0.))
            del proposed, expected, expected_score
    # Tiny analytic identity check, unrelated to any target data.
    v = torch.tensor([[2., 1.], [0., 3.]])
    a = torch.full((2, 2), .5)
    gamma = torch.tensor(.2, requires_grad=True)
    mixed = mix_context(a@v, v, gamma)
    torch.testing.assert_close(mixed, ((1-gamma)*a+gamma*torch.eye(2))@v)
    mixed[0, 0].backward()
    torch.testing.assert_close(gamma.grad, torch.tensor(1.))
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    opt = make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    loss, output, norm = step(model, opt, wave, labels)
    if any(p.grad is not None for p in model.reference.parameters()):
        raise RuntimeError("Frozen reference gradient leak")
    gradients = model.difference_gain.grad.detach()
    if not bool(torch.isfinite(gradients).all()) or not bool((gradients.abs() > 0).all()):
        raise RuntimeError("Some zero-initialized gates have no source gradient")
    if not bool((model.difference_gain > 0).any()):
        raise RuntimeError("Projected complement never activates in short source check")
    for index in (0, 23):
        layer = model.task.model.encoder.layers[index]
        for frozen in (layer.self_attn.q_proj, layer.self_attn.k_proj):
            if any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
                raise RuntimeError("Task Q/K unexpectedly trained")
        for module in (layer.self_attn.v_proj, layer.self_attn.out_proj, layer.fc1):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("Task V/out/FFN gradient absent")
    payload = model.trainable_payload(sha)
    expected_gains = model.difference_gain.detach().clone()
    with torch.no_grad():
        model.difference_gain.zero_()
    model.load_trainable_payload(payload, sha)
    torch.testing.assert_close(model.difference_gain, expected_gains, atol=0, rtol=0)
    record = dict(seconds=time.perf_counter()-start, source_updates=1,
                  initial_equivalence=comparisons, formula_and_boundary_gradient=True,
                  source_nll=float(loss), grad_norm=norm,
                  gain_gradient=gradients.cpu().tolist(), gains_after_update=expected_gains.cpu().tolist(),
                  teacher_gradient=False, task_QK_gradient=False, payload_round_trip=True,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  self_test_updates_discarded_before_formal_training=True)
    if record["reserved_gib"] > c["max_reserved_gib"] or shutil.disk_usage(c["output_dir"]).free < 4*2**30:
        raise RuntimeError("Source preflight exceeds memory/disk budget")
    base.atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SHORT_PREFLIGHT_OK", json.dumps(record), flush=True)
    del payload, model, opt, output, loss, wave, labels
    gc.collect()
    torch.cuda.empty_cache()
    return record


def train_epoch(model, opt, dataset, c, epoch, deadline):
    metrics = _train_epoch(model, opt, dataset, c, epoch, deadline)
    gains = model.difference_gain.detach().cpu()
    metrics.update(gain_mean=float(gains.mean()), gain_max=float(gains.max()),
                   gain_active_layers=int((gains > 0).sum()),
                   gain_at_cap_layers=int((gains >= model.gain_cap-1e-7).sum()))
    base.atomic_json(Path(c["output_dir"])/f"difference_access_epoch{epoch}.json",
                     dict(gains=gains.tolist(), cap=model.gain_cap,
                          forward="(1-gamma)A0 V + gamma V", effective_routing_modified=True))
    return metrics


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    with open(c["matched_attention_results"], newline="") as stream:
        previous = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                    if r["arm"] == "pretrained_attention" and int(r["milestone"]) == epoch}
    if set(values) != set(base.PAPER_SLS) or set(previous) != set(values):
        raise ValueError("Missing full five-domain comparison")
    mean = sum(values[k] for k in base.CROSS)/4
    gaps = {k: values[k]-previous[k] for k in values}
    severe = epoch == 4 and ((sum(gaps[k] for k in base.CROSS)/4 > 3
                              and sum(gaps[k] > 5 for k in base.CROSS) >= 2)
                             or (gaps["itw"] > 3 and sum(gaps[k] for k in base.CROSS) >= 0)
                             or any(values[k] > 35 and gaps[k] > 8 for k in base.CROSS))
    return dict(epoch=epoch, stop=severe,
                reason="registered_gross_failure" if severe else "complete" if epoch == 8 else "continue_to8",
                per_domain=values, historical_attention=previous, gaps_vs_attention=gaps,
                cross_mean=mean, worst_cross=max(values[k] for k in base.CROSS),
                ch_de_joint=(values["ch"]+values["de"])/2,
                difference_from_historical_sls_epoch4=mean-12.9235,
                paper_sls_gaps={k: values[k]-base.PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-base.PAPER_MAMBA[k] for k in values},
                eligible_for_fixed_multiseed=False, further_seeds_require_user_authorization=True,
                single_seed_is_not_stability_evidence=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    root = Path(c["output_dir"]).resolve()
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no overwrite/restart")
    base.ARM = ARM
    base.PretrainedAttentionSLS = partial(ReferenceDifferenceSLS, gain_cap=c["gain_cap"])
    base.signature, base.preflight = signature, preflight
    base.make_optimizer, base.step = make_optimizer, step
    base.train_epoch, base.decide = train_epoch, decide
    try:
        base.run(c, args.config)
    except Exception as error:
        if root.is_relative_to(Path("/root/rivermind-data")):
            prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
            base.atomic_json(root/"status.json", dict(
                status="stopped_budget" if isinstance(error, TimeoutError) else "failed",
                error=repr(error), previous_status=prior, ended_at=base.utc_now(), automatic_restart=False))
        raise


if __name__ == "__main__":
    main()
