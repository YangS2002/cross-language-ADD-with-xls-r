"""One registered candidate; reuse source-best selection and full stage evaluation."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from data import simple_collate_fn
from tools import train_sls_pretrained_attention as base
from tools.reference_context_contrast_loss import context_contrast, self_test
from tools.sls_reference_context_contrast_model import ReferenceContextContrastSLS

ARM = "reference_context_contrast"
_signature, _train_epoch = base.signature, base.train_epoch
CONFIG = {}
CENTER = None
STAT = {}


def signature(c, path):
    result = _signature(c, path)
    result["shared_trainer"] = result["trainer"]
    result["trainer"] = base.digest(__file__)
    result["model"] = base.digest("tools/sls_reference_context_contrast_model.py")
    for p in ("tools/sls_pretrained_attention_model.py", "tools/reference_context_contrast_loss.py"):
        result["dependencies"][p] = base.digest(p)
    return result


def build_model(checkpoint):
    model = ReferenceContextContrastSLS(checkpoint)
    if CENTER is not None:
        model.context_center.copy_(CENTER)
        model.center_ready = True
    return model


def step(model, opt, wave, labels):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, features, u = model.auxiliary_forward(wave)
    ce = F.nll_loss(output, labels, weight=output.new_tensor([.1, .9]))
    contrast, info = context_contrast(features, labels, u,
                                      tau=CONFIG["contrast_temperature"], beta=CONFIG["contrast_beta"])
    total = ce + CONFIG["contrast_weight"]*contrast
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite classification/contrastive objective")
    total.backward()
    params = [p for group in opt.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
    opt.step()
    if STAT:
        n = labels.numel()
        for k in ("valid_fake", "valid_real", "samples_fake", "samples_real", "empty_valid"):
            STAT[k] += info[k]
        STAT["steps"] += 1
        STAT["samples"] += n
        STAT["contrast_sum"] += float(contrast.detach())*n
        STAT["total_sum"] += float(total.detach())*n
        for k in ("context_std", "negative_weight_mean", "positive_weight_ratio"):
            STAT[k] += info[k]
        STAT["negative_weight_min"] = min(STAT["negative_weight_min"], info["negative_weight_min"])
        STAT["negative_weight_max"] = max(STAT["negative_weight_max"], info["negative_weight_max"])
    model.last_aux_info = dict(**info, classification_loss=float(ce.detach()),
                               contrast_loss=float(contrast.detach()), total_loss=float(total.detach()))
    model.clear_aux()
    # Return classification ONLY: existing epoch accounting and best-selection remain unchanged.
    return ce.detach(), output.detach(), float(norm)


def preflight(c, dataset, sha):
    global CENTER
    start = time.perf_counter()
    tests = self_test()
    base.set_seed(c["seed"])
    model = build_model(c["ssl_checkpoint"]).cuda().eval()
    ids = [i for i, label in enumerate(dataset.synthesizerlabels) if label == "bonafide"]
    if len(ids) != 2580:
        raise RuntimeError("Source-real centering set mismatch")
    begin = time.perf_counter()
    total = torch.zeros(24, 1024, device="cuda", dtype=torch.float32)
    count = 0
    loader = DataLoader(Subset(dataset, ids), batch_size=c["batch_size"], shuffle=False,
                        num_workers=c["num_workers"], collate_fn=simple_collate_fn, pin_memory=True)
    with torch.no_grad():
        for b in loader:
            if not bool((b["label"] == 1).all()):
                raise RuntimeError("Non-real sample in mean estimation")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.extract_reference(b["audio"].cuda(non_blocking=True))
            d = model.reference_descriptors()
            total += d.sum(0)
            count += d.shape[0]
            model._reference_inputs = []
            if count % 512 == 0:
                print("SOURCE_CENTER", count, flush=True)
    if count != 2580 or not bool(torch.isfinite(total).all()):
        raise RuntimeError("Incomplete/nonfinite source-real mean")
    CENTER = (total/count).cpu()
    model.context_center.copy_(CENTER.cuda())
    model.center_ready = True
    center_seconds = time.perf_counter()-begin
    base.atomic_save(Path(c["output_dir"])/"source_reference_mean.pt",
                     dict(mean=CENTER, samples=count, source_labels_sha256=base.digest(c["train_label_path"]),
                          pretrained_sha256=sha, definition="mean_actual_attention_input_24layers_bf16_fp32_accum"))
    base.atomic_json(Path(c["output_dir"])/"source_reference_mean.json",
                     dict(samples=count, seconds=center_seconds, shape=list(CENTER.shape),
                          bytes=CENTER.numel()*CENTER.element_size(), source_only=True))
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    test_ids = ids[:c["batch_size"]//2] + [i for i, label in enumerate(dataset.synthesizerlabels)
                                         if label != "bonafide"][:c["batch_size"]//2]
    b = simple_collate_fn([dataset[i] for i in test_ids])
    wave, y = b["audio"].cuda(), b["label"].view(-1).long().cuda()
    model.train()
    # BN train path and inputs identical, no parameter update: auxiliary reads must not change scores.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        normal = model(wave[:2])
        aux, f, u = model.auxiliary_forward(wave[:2])
        torch.testing.assert_close(normal, aux, atol=0, rtol=0)
        output_error = float((normal-aux).abs().max())
    model.clear_aux()
    opt = base.make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    loss, output, norm = step(model, opt, wave, y)
    if any(p.grad is not None for p in model.reference.parameters()):
        raise RuntimeError("Frozen reference gradient leak")
    for index in (0, 23):
        layer = model.task.model.encoder.layers[index]
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj):
            if any(p.requires_grad or p.grad is not None for p in projection.parameters()):
                raise RuntimeError("Task Q/K accidentally trained")
        for module in (layer.self_attn.v_proj, layer.self_attn.out_proj, layer.fc1):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("Task gradient absent")
    for module in (model.backend.fc0, model.backend.fc1, model.backend.fc3):
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
            raise RuntimeError("SLS gradient absent")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(wave[:2])
    if model.aux_features is not None or model.context_descriptors is not None:
        raise RuntimeError("Evaluation computes training-only matching")
    payload = model.trainable_payload(sha)
    model.load_trainable_payload(payload, sha)
    record = dict(seconds=time.perf_counter()-start, source_mean_seconds=center_seconds,
                  source_updates=1, self_tests=tests, auxiliary_max_score_error=output_error,
                  source_nll=float(loss), grad_norm=norm, auxiliary=model.last_aux_info,
                  inference_auxiliary_disabled=True, reference_gradient=False,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    if record["reserved_gib"] > c["max_reserved_gib"] or shutil.disk_usage(c["output_dir"]).free < 4*2**30:
        raise RuntimeError("Preflight memory/disk budget exceeded")
    base.atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SHORT_PREFLIGHT_OK", json.dumps(record), flush=True)
    del payload, model, opt, loss, output, wave, y, normal, aux, f, u, total, d
    gc.collect()
    torch.cuda.empty_cache()
    return record


def train_epoch(model, opt, dataset, c, epoch, deadline):
    global STAT
    STAT = {k: 0 for k in ("valid_fake", "valid_real", "samples_fake", "samples_real", "empty_valid",
                           "steps", "samples", "contrast_sum", "total_sum", "context_std",
                           "negative_weight_mean", "positive_weight_ratio")}
    STAT.update(negative_weight_min=1., negative_weight_max=1.)
    metrics = _train_epoch(model, opt, dataset, c, epoch, deadline)
    metrics.update(contrast_loss=STAT["contrast_sum"]/STAT["samples"],
                   total_objective=STAT["total_sum"]/STAT["samples"],
                   valid_real_anchors=STAT["valid_real"], valid_fake_anchors=STAT["valid_fake"],
                   real_anchor_coverage=STAT["valid_real"]/max(STAT["samples_real"], 1),
                   fake_anchor_coverage=STAT["valid_fake"]/max(STAT["samples_fake"], 1),
                   batches_no_valid_anchor=STAT["empty_valid"],
                   context_std=STAT["context_std"]/STAT["steps"],
                   negative_weight_mean=STAT["negative_weight_mean"]/STAT["steps"],
                   negative_weight_min=STAT["negative_weight_min"],
                   negative_weight_max=STAT["negative_weight_max"],
                   positive_weight_ratio=STAT["positive_weight_ratio"]/STAT["steps"])
    base.atomic_json(Path(c["output_dir"])/f"context_diagnostic_epoch{epoch}.json", metrics)
    STAT = {}
    return metrics


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    with open(c["matched_attention_results"], newline="") as stream:
        attention = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                     if r["arm"] == "pretrained_attention" and int(r["milestone"]) == epoch}
    if set(values) != set(base.PAPER_SLS) or set(attention) != set(values):
        raise ValueError("Incomplete five-domain comparison")
    gaps = {k: values[k]-attention[k] for k in values}
    mean = sum(values[k] for k in base.CROSS)/4
    worst = max(values[k] for k in base.CROSS)
    severe = epoch == 4 and ((sum(gaps[k] for k in base.CROSS)/4 > 3
                              and sum(gaps[k] > 5 for k in base.CROSS) >= 2)
                             or (gaps["itw"] > 3 and sum(gaps[k] for k in base.CROSS) >= 0)
                             or any(values[k] > 35 and gaps[k] > 8 for k in base.CROSS))
    return dict(epoch=epoch, stop=severe, reason="registered_gross_failure" if severe
                else "complete" if epoch == 8 else "continue_to8", per_domain=values,
                historical_attention=attention, gaps_vs_attention=gaps, cross_mean=mean,
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
