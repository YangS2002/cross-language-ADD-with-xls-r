"""One registered seed20 candidate, clean-source checkpoint selection."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from data import simple_collate_fn
from tools import train_sls_pretrained_attention as base
from tools.audit_relation_neighborhood_numpy import run as audit_math
from tools.sls_relation_neighborhood_model import RelationNeighborhoodSLS

ARM = "relation_neighborhood"
CONFIG = {}
STAT = {}
_signature, _train_epoch = base.signature, base.train_epoch


def signature(c, path):
    result = _signature(c, path)
    result["shared_trainer"] = result["trainer"]
    result["trainer"] = base.digest(__file__)
    result["model"] = base.digest("tools/sls_relation_neighborhood_model.py")
    for p in ("tools/sls_pretrained_attention_model.py", "tools/audit_relation_neighborhood_numpy.py"):
        result["dependencies"][p] = base.digest(p)
    return result


def build_model(checkpoint):
    return RelationNeighborhoodSLS(checkpoint, CONFIG["neighborhood_rho"])


def same_rng(a, b):
    return (a["python"] == b["python"] and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:]
            and torch.equal(a["torch"], b["torch"])
            and len(a["cuda"]) == len(b["cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))


def step(model, opt, wave, labels):
    opt.zero_grad(set_to_none=True)
    bn = model.backend.first_bn
    if not bn.training or not bn.track_running_stats:
        raise RuntimeError("Expected native train BN with running statistics")
    old_track = bn.track_running_stats
    before_count = int(bn.num_batches_tracked)
    before_extractions = model._reference_extractions
    collect = not STAT or STAT["steps"] % 200 == 0
    model.relation_diagnostics = []
    after = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.begin_batch(wave)
        before = base.rng_state()
        e = model.zeros(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(wave)
        weights = output.new_tensor([.1, .9])
        clean = F.nll_loss(output, labels, weight=weights)
        if not bool(torch.isfinite(clean)):
            raise FloatingPointError("Nonfinite clean source NLL")
        (.5*clean).backward()
        after = base.rng_state()
        if e.grad is None or not bool(torch.isfinite(e.grad).all()):
            raise FloatingPointError("Missing/nonfinite relation gradient")
        g = e.grad.detach()
        gnorm = g.norm(dim=(-2, -1), keepdim=True)
        direction = torch.where(gnorm > 0, g/gnorm.clamp_min(1e-30), torch.zeros_like(g))
        model.neighborhood = direction.detach()
        model.collect_diagnostics = collect
        bn.track_running_stats = False
        base.restore_rng(before)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            perturbed_output = model(wave)
        perturbed = F.nll_loss(perturbed_output, labels, weight=weights)
        if not bool(torch.isfinite(perturbed)):
            raise FloatingPointError("Nonfinite perturbed source NLL")
        (.5*perturbed).backward()
        margin = output[:, 1]-output[:, 0]
        margin_adv = perturbed_output[:, 1]-perturbed_output[:, 0]
        change = (margin_adv-margin).detach()
        info = dict(clean_nll=float(clean.detach()), perturbed_nll=float(perturbed.detach()),
                    total_objective=float(.5*(clean.detach()+perturbed.detach())),
                    nll_increment=float(perturbed.detach()-clean.detach()),
                    ascent_increased_loss=bool(perturbed.detach() > clean.detach()),
                    zero_gradient_heads=int((gnorm == 0).sum()),
                    heads=int(gnorm.numel()), gradient_norm_mean=float(gnorm.mean()),
                    score_change_mean_absolute=float(change.abs().mean()),
                    normalized_score_change=float(change.abs().mean()/margin.detach().std(unbiased=False).clamp_min(1e-6)),
                    clean_margin_std=float(margin.detach().std(unbiased=False)),
                    bn_updates=int(bn.num_batches_tracked)-before_count,
                    teacher_forward_count=model._reference_extractions-before_extractions)
        if info["bn_updates"] != 1 or info["teacher_forward_count"] != 1:
            raise RuntimeError("Two-pass BN/reference state mismatch")
        model.last_step_info = info
    finally:
        bn.track_running_stats = old_track
        if after is not None:
            base.restore_rng(after)
        model.end_batch()
    if not same_rng(after, base.rng_state()):
        raise RuntimeError("Extra pass advanced the future random stream")
    params = [p for group in opt.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
    opt.step()
    if STAT:
        n = labels.numel()
        STAT["steps"] += 1
        STAT["samples"] += n
        for name in ("perturbed_nll", "total_objective", "nll_increment",
                     "score_change_mean_absolute", "normalized_score_change"):
            STAT[name] += info[name]*n
        STAT["not_increased"] += int(not info["ascent_increased_loss"])
        STAT["zero_gradient_heads"] += info["zero_gradient_heads"]
        STAT["heads"] += info["heads"]
        if collect:
            record = dict(batch=STAT["steps"], **info, layers=model.relation_diagnostics)
            STAT["diagnostics"].append(record)
            print("RELATION_DIAGNOSTIC", json.dumps(record), flush=True)
    return clean.detach(), output.detach(), float(norm)


def preflight(c, dataset, sha):
    start = time.perf_counter()
    math_result = audit_math()
    base.set_seed(c["seed"])
    model = build_model(c["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    half = c["batch_size"]//2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"][:half])
    batch = simple_collate_fn([dataset[i] for i in ids])
    wave = batch["audio"].cuda()
    labels = batch["label"].view(-1).long().cuda()
    comparisons = []
    for dtype, key in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        subwave = wave[:2]
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                model.begin_batch(subwave)
                normal = model(subwave, return_details=True)
                expected = [s.float().cpu() for s in normal["states"]]
                expected_output = normal["log_probs"].cpu()
                del normal
                model.zeros(False)
                changed = model(subwave, return_details=True)
                errors = []
                for old, new in zip(expected, changed["states"]):
                    new = new.float().cpu()
                    torch.testing.assert_close(old, new, atol=c[key+"_atol"], rtol=c[key+"_rtol"])
                    errors.append(float((old-new).abs().max()))
                torch.testing.assert_close(expected_output, changed["log_probs"].cpu(),
                                           atol=c[key+"_atol"], rtol=c[key+"_rtol"])
                comparisons.append(dict(dtype=key, layer_max_errors=errors,
                                        score_max_error=float((expected_output-changed["log_probs"].cpu()).abs().max())))
                del changed, expected, expected_output
        finally:
            model.end_batch()
    model.train()
    opt = base.make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    loss, output, norm = step(model, opt, wave, labels)
    step_seconds = time.perf_counter()-begin
    if any(p.grad is not None for p in model.reference.parameters()):
        raise RuntimeError("Reference gradient leak")
    for index in (0, 23):
        layer = model.task.model.encoder.layers[index]
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj):
            if any(p.requires_grad or p.grad is not None for p in projection.parameters()):
                raise RuntimeError("Task Q/K was trained")
        for module in (layer.self_attn.v_proj, layer.self_attn.out_proj, layer.fc1):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("Task evidence gradient missing")
    for module in (model.backend.fc0, model.backend.fc1, model.backend.fc3):
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
            raise RuntimeError("SLS gradient missing")
    if model.last_step_info["zero_gradient_heads"] == model.last_step_info["heads"]:
        raise RuntimeError("No relation direction was connected")
    if len(model.relation_diagnostics) != 24 or max(r["max_abs_tilt"] for r in model.relation_diagnostics) <= 1e-5:
        raise RuntimeError("Perturbed relation had no observable effect")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(wave[:2])
    if model.neighborhood is not None or model._batch_reference is not None:
        raise RuntimeError("Training-only neighborhood leaked into eval")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 284246149:
        raise RuntimeError("Permanent parameter capacity changed")
    payload = model.trainable_payload(sha)
    model.load_trainable_payload(payload, sha)
    record = dict(seconds=time.perf_counter()-start, source_updates=1,
                  self_test_update_discarded=True, math=math_result, comparisons=comparisons,
                  source_nll=float(loss), grad_norm=norm, source_step_seconds=step_seconds,
                  diagnostics=model.last_step_info, layers=model.relation_diagnostics,
                  reference_gradient=False, task_QK_gradient=False,
                  inference_original_path=True, trainable_parameters=trainable,
                  neighborhood_shape=list(model.neighborhood_shape),
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  kernel="native_bmm_base_logits_plus_FP32_bounded_tilt")
    if record["reserved_gib"] > c["max_reserved_gib"] or shutil.disk_usage(c["output_dir"]).free < 4*2**30:
        raise RuntimeError("Short preflight memory/disk budget exceeded")
    base.atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SHORT_PREFLIGHT_OK", json.dumps(record), flush=True)
    del payload, model, opt, loss, output, wave, labels
    gc.collect()
    torch.cuda.empty_cache()
    return record


def train_epoch(model, opt, dataset, c, epoch, deadline):
    global STAT
    STAT = {k: 0 for k in ("steps", "samples", "perturbed_nll", "total_objective", "nll_increment",
                           "score_change_mean_absolute", "normalized_score_change", "not_increased",
                           "zero_gradient_heads", "heads")}
    STAT["diagnostics"] = []
    metrics = _train_epoch(model, opt, dataset, c, epoch, deadline)
    metrics.update({k: STAT[k]/STAT["samples"] for k in ("perturbed_nll", "total_objective", "nll_increment",
                    "score_change_mean_absolute", "normalized_score_change")})
    metrics.update(ascent_not_increased_fraction=STAT["not_increased"]/STAT["steps"],
                   zero_gradient_head_fraction=STAT["zero_gradient_heads"]/STAT["heads"])
    base.atomic_json(Path(c["output_dir"])/f"relation_diagnostic_epoch{epoch}.json",
                     dict(epoch=epoch, metrics=metrics, sampled_batches=STAT["diagnostics"]))
    STAT = {}
    return metrics


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    with open(c["matched_attention_results"], newline="") as stream:
        historical = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                      if r["arm"] == "pretrained_attention" and int(r["milestone"]) == epoch}
    if set(values) != set(base.PAPER_SLS) or set(historical) != set(values):
        raise ValueError("Incomplete five-domain comparison")
    gaps = {k: values[k]-historical[k] for k in values}
    mean = sum(values[k] for k in base.CROSS)/4
    worst = max(values[k] for k in base.CROSS)
    severe = epoch == 4 and ((sum(gaps[k] for k in base.CROSS)/4 > 3
                              and sum(gaps[k] > 5 for k in base.CROSS) >= 2)
                             or (gaps["itw"] > 3 and sum(gaps[k] for k in base.CROSS) >= 0)
                             or any(values[k] > 35 and gaps[k] > 8 for k in base.CROSS))
    return dict(epoch=epoch, stop=severe, reason="registered_gross_failure" if severe
                else "complete" if epoch == 8 else "continue_to8", per_domain=values,
                historical_attention=historical, gaps_vs_attention=gaps, cross_mean=mean,
                worst_cross=worst, ch_de_joint=(values["ch"]+values["de"])/2,
                paper_sls_gaps={k: values[k]-base.PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-base.PAPER_MAMBA[k] for k in values},
                eligible_for_fixed_multiseed=epoch == 8 and mean <= 10.8029 and worst <= 15.2643
                and values["itw"] <= 8.46, further_seeds_require_user_authorization=True,
                single_seed_is_not_stability_evidence=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    CONFIG = yaml.safe_load(Path(args.config).read_text())
    c = CONFIG
    root = Path(c["output_dir"]).resolve()
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no overwrite/restart")
    base.ARM, base.PretrainedAttentionSLS = ARM, build_model
    base.signature, base.preflight = signature, preflight
    base.step, base.train_epoch, base.decide = step, train_epoch, decide
    try:
        base.run(c, args.config)
    except Exception as error:
        if root.is_relative_to(Path("/root/rivermind-data")):
            prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
            base.atomic_json(root/"status.json", dict(status="stopped_budget" if isinstance(error, TimeoutError)
                             else "failed", error=repr(error), previous_status=prior,
                             ended_at=base.utc_now(), automatic_restart=False))
        raise


if __name__ == "__main__":
    main()
