"""Single seed20 Query-guided candidate; reused warmup, total epochs4..8."""
import argparse
import csv
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

from tools.sls_pretrained_query_model import PretrainedQuerySLS
from tools.train_relational_preservation import (
    atomic_json, atomic_save, append_csv, digest, dataset_for, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains
from tools.train_sls_reference import evaluate
from train import set_seed, make_train_loader

ARM = "pretrained_query"
CROSS = ("ch", "de", "uk", "es")
PAPER_SLS = dict(ch=12.26, de=19.71, uk=21.13, es=9.69, itw=7.46)
PAPER_MAMBA = dict(ch=17.32, de=23.57, uk=11.89, es=15.83, itw=6.71)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_optimizer(model, c):
    return torch.optim.AdamW([
        dict(params=list(model.backend.parameters()), lr=c["head_lr"]),
        dict(params=[p for p in model.task.parameters() if p.requires_grad], lr=c["ssl_lr"])],
        weight_decay=c["weight_decay"], foreach=False)


def cpu_payload(model, sha):
    def clone(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        return value
    return clone(model.trainable_payload(sha))


def signature(c, config_path):
    from fairseq.modules import MultiheadAttention
    from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
    modules = (MultiheadAttention.forward, TransformerEncoder.extract_features)
    installed = [hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in modules]
    expected = ["84410048dc06fc75ae4f0b5e3b8d23660588f6276fecce5a5a67b65ba2167030",
                "78d2dfde3b2f790ab380d4b9e4f1a582442495689c2f83f80776a7b7e2b6e6f3"]
    if installed != expected:
        raise RuntimeError("Installed attention implementation changed; inspect before running")
    result = dict(config=digest(config_path), trainer=digest(__file__),
                  model=digest("tools/sls_pretrained_query_model.py"),
                  pretrained=digest(c["ssl_checkpoint"]), warmup=digest(c["shared_warmup"]),
                  labels=digest(c["train_label_path"]), torch_version=torch.__version__,
                  fairseq_function_sha256=installed,
                  dependencies={p: digest(p) for p in (
                      "data.py", "train.py", "tools/sls_reference_model.py",
                      "tools/train_relational_preservation.py", "tools/train_sls_reference.py",
                      "tools/evaluate_relational_preservation.py",
                      "tools/evaluate_four_domains_shared_xlsr.py")})
    if (result["pretrained"] != c["expected_pretrained_sha256"]
            or result["warmup"] != c["expected_warmup_sha256"]):
        raise ValueError("Original/warmup SHA mismatch")
    return result


def step(model, opt, wave, labels):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(wave)
    weights = output.new_tensor([0.1, 0.9])
    loss = torch.nn.functional.nll_loss(output, labels, weight=weights)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite source NLL")
    loss.backward()
    params = [p for group in opt.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
    opt.step()
    return loss.detach(), output.detach(), float(norm)


def preflight(c, dataset, sha):
    start = time.perf_counter()
    half = c["batch_size"] // 2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"]
           [:c["batch_size"]-half])
    batches = [dataset[i] for i in ids]
    wave = torch.stack([b["audio"] for b in batches]).cuda()
    labels = torch.tensor([int(b["label"]) for b in batches], device="cuda")
    set_seed(c["seed"])
    model = PretrainedQuerySLS(c["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    comparisons = []
    # Compare the exact same task/SLS weights, toggling native vs external Query.
    for dtype in (torch.float32, torch.bfloat16):
        tolerance = "fp32" if dtype == torch.float32 else "bf16"
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            model.guidance_enabled = False
            native = model(wave[:2], return_details=True)
            expected = {"layers": [s.float().cpu() for s in native["states"]],
                        "output": native["log_probs"].cpu()}
            del native
            model.guidance_enabled = True
            guided = model(wave[:2], return_details=True)
            max_errors = []
            for old, new in zip(expected["layers"], guided["states"]):
                new = new.float().cpu()
                torch.testing.assert_close(old, new, atol=c[tolerance+"_atol"], rtol=c[tolerance+"_rtol"])
                max_errors.append(float((old-new).abs().max()))
            torch.testing.assert_close(expected["output"], guided["log_probs"].cpu(),
                                       atol=c[tolerance+"_atol"], rtol=c[tolerance+"_rtol"])
            row = dict(dtype=tolerance, layer_max_absolute_errors=max_errors,
                       output_max_absolute_error=float((expected["output"]-guided["log_probs"].cpu()).abs().max()),
                       reference_calls=model.reference_calls, query_calls=model.query_calls)
            comparisons.append(row)
            print("INITIAL_EQUIVALENCE", json.dumps(row), flush=True)
            del guided, expected
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        normal = model(wave[:2])
        model.query_check_mode = "zero"
        changed = model(wave[:2])
        wiring_effect = float((normal-changed).abs().max())
        model.query_check_mode = None
        if not np.isfinite(wiring_effect) or wiring_effect <= 1e-6:
            raise RuntimeError("External Query wiring check had no finite observable effect")
        del normal, changed
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    opt = make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for i in range(3):
        loss, output, norm = step(model, opt, wave, labels)
        if any(p.grad is not None for p in model.reference.parameters()):
            raise RuntimeError("Gradient leaked into original reference")
        for layer in model.task.model.encoder.layers:
            if any(p.requires_grad or p.grad is not None for p in layer.self_attn.q_proj.parameters()):
                raise RuntimeError("Task Query projection was trained")
        for index in (0, 23):
            layer = model.task.model.encoder.layers[index]
            for module in (layer.self_attn.k_proj, layer.self_attn.v_proj, layer.fc1):
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for p in module.parameters()):
                    raise RuntimeError(f"Task gradients absent at layer {index+1}")
        for module in (model.backend.fc0, model.backend.fc1, model.backend.fc3):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("SLS backend gradient path absent")
        losses.append(float(loss))
        print("SELF_TEST_STEP", i, float(loss), norm, flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    record = dict(batch=c["batch_size"], comparisons=comparisons,
                  query_wiring_max_score_change=wiring_effect,
                  wiring_check_is_not_multilingual_mechanism_evidence=True,
                  trainable_parameters=trainable, model_bytes=model_bytes,
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  disk_budget_bytes=max(4*2**30, int(model_bytes*2.1)+512*2**20),
                  losses=losses, seconds=time.perf_counter()-start,
                  reference_gradient=False, task_query_gradient=False,
                  source_ids=[b["file_name"] for b in batches])
    if (record["reserved_gib"] > c["max_reserved_gib"]
            or shutil.disk_usage(c["output_dir"]).free < record["disk_budget_bytes"]):
        raise RuntimeError("Preflight memory/disk limit exceeded; no silent batch change")
    # In-memory serialization schema round trip; no additional on-disk weights.
    payload = model.trainable_payload(sha)
    model.load_trainable_payload(payload, sha)
    del payload, model, opt, loss, output, wave, labels
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SELF_TEST_OK", json.dumps(record), flush=True)
    return record


def train_epoch(model, opt, dataset, c, epoch, deadline):
    model.train()
    torch.cuda.reset_peak_memory_stats()
    loader = make_train_loader(dataset, c["batch_size"], c["seed"]+100000*epoch,
                               c["num_workers"], torch.device("cuda"))
    start = time.perf_counter()
    count = real = correct = 0
    legacy_total = weighted_numerator = weight_denominator = 0.
    by_class = {k: dict(count=0, correct=0, nll=0.) for k in (0, 1)}
    for i, b in enumerate(loader, 1):
        if time.monotonic() > deadline:
            raise TimeoutError("Hard budget reached; incomplete epoch not eligible for selection")
        y = b["label"].view(-1).long().cuda(non_blocking=True)
        loss, output, norm = step(model, opt, b["audio"].cuda(non_blocking=True), y)
        n = y.numel()
        count += n
        real += int((y == 1).sum())
        correct += int((output.argmax(1) == y).sum())
        legacy_total += float(loss)*n
        nll = -output.gather(1, y[:, None]).squeeze(1)
        weights = torch.where(y == 1, .9, .1)
        weighted_numerator += float((nll*weights).sum())
        weight_denominator += float(weights.sum())
        for label in (0, 1):
            mask = y == label
            by_class[label]["count"] += int(mask.sum())
            by_class[label]["correct"] += int((output.argmax(1)[mask] == label).sum())
            by_class[label]["nll"] += float(nll[mask].sum())
        if i == 20:
            elapsed = time.perf_counter()-start
            timing = dict(epoch=epoch, steps=i, seconds=elapsed,
                          estimated_epoch_seconds=elapsed/i*len(loader),
                          estimate_only=True, reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            atomic_json(Path(c["output_dir"])/f"timing_epoch{epoch}.json", timing)
            print("TIMING", json.dumps(timing), flush=True)
        if i % 200 == 0:
            progress = dict(epoch=epoch, batch=i, samples=count, train_loss=legacy_total/count,
                            seconds=time.perf_counter()-start, updated_at=utc_now())
            atomic_json(Path(c["output_dir"])/"progress.json", progress)
            print("PROGRESS", json.dumps(progress), flush=True)
    if (count, real) != (25380, 2580):
        raise RuntimeError("Incomplete source epoch")
    return dict(train_loss=legacy_total/count, exact_weighted_nll=weighted_numerator/weight_denominator,
                accuracy=correct/count, recall_fake=by_class[0]["correct"]/by_class[0]["count"],
                recall_real=by_class[1]["correct"]/by_class[1]["count"],
                nll_fake=by_class[0]["nll"]/by_class[0]["count"],
                nll_real=by_class[1]["nll"]/by_class[1]["count"],
                samples=count, real=real, fake=count-real, steps=i, grad_norm_last=norm,
                seconds=time.perf_counter()-start, reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    with open(c["matched_baseline_results"], newline="") as stream:
        baseline = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                    if r["arm"] == "sls" and int(r["milestone"]) == epoch}
    if set(baseline) != set(values) or set(values) != set(PAPER_SLS):
        raise ValueError("Incomplete five-domain comparison")
    gaps = {k: values[k]-baseline[k] for k in values}
    mean = float(np.mean([values[k] for k in CROSS]))
    base_mean = float(np.mean([baseline[k] for k in CROSS]))
    severe = ((mean-base_mean > 3 and sum(gaps[k] > 5 for k in CROSS) >= 2)
              or (gaps["itw"] > 3 and mean >= base_mean)
              or any(values[k] > 35 and gaps[k] > 8 for k in CROSS))
    gate = (epoch == c["epochs"] and mean < base_mean
            and mean < np.mean([PAPER_SLS[k] for k in CROSS])
            and values["itw"] <= c["itw_eer_limit"]
            and max(values[k] for k in CROSS) <= 25
            and all(values[k] <= PAPER_SLS[k]+3 for k in CROSS))
    return dict(epoch=epoch, stop=bool(severe), reason="severe_degradation" if severe else
                "fixed_epoch8_complete" if epoch == c["epochs"] else "continue_to_fixed_epoch8",
                per_domain=values, matched_sls=baseline, gaps=gaps,
                cross_mean=mean, matched_cross_mean=base_mean,
                difference_from_historical_sls_epoch4=mean-12.9235,
                worst_cross=max(values[k] for k in CROSS),
                ch_de_joint=(values["ch"]+values["de"])/2,
                paper_sls_gaps={k: values[k]-PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-PAPER_MAMBA[k] for k in values},
                eligible_for_fixed_multiseed=bool(gate),
                single_seed_is_not_stability_evidence=True)


def run(c, config_path, preflight_only=False):
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("All outputs must be on persistent data disk")
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing run: no automatic overwrite/replay")
    root.mkdir(parents=True, exist_ok=True)
    (root/ARM).mkdir(exist_ok=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    sig = signature(c, config_path)
    dataset, _ = dataset_for(c)
    domains = make_domains(c)
    atomic_json(root/"inventory.json", [d.validate_paths() for d in domains.values()])
    experiment = dict(config=c, signature=sig, pid=os.getpid(), started_at=utc_now(),
                      source_only_training=True, single_candidate=True,
                      target_use="full evaluation and predeclared termination only",
                      recovery="one best; NO Adam state; no exact resume",
                      UK_is_previously_used_diagnostic_domain=True)
    if not preflight_only:
        atomic_json(root/"experiment.json", experiment)
    atomic_json(root/"status.json", dict(status="preflight", seed=c["seed"], updated_at=utc_now()))
    test = preflight(c, dataset, sig["pretrained"])
    if preflight_only:
        atomic_json(root/"status.json", dict(status="preflight_only_complete", updated_at=utc_now()))
        return
    # Throw away all self-test updates and construct a clean formal model.
    deadline = time.monotonic()+c["max_run_seconds"]
    set_seed(c["seed"])
    model = PretrainedQuerySLS(c["ssl_checkpoint"]).cuda()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    opt = make_optimizer(model, c)
    restore_rng(warm["rng"])
    del warm
    best = float("inf")
    best_epoch = None
    epoch_estimate = 0.
    eval_estimate = float(c["initial_eval_estimate_seconds"])
    for epoch in range(c["warmup_epochs"]+1, c["epochs"]+1):
        estimated_cost = epoch_estimate + (eval_estimate if epoch in c["milestones"] else 0)
        if time.monotonic()+estimated_cost > deadline:
            raise TimeoutError("Insufficient budget for next complete epoch/stage")
        atomic_json(root/"status.json", dict(status="training", seed=c["seed"], epoch=epoch,
                                              updated_at=utc_now()))
        metrics = train_epoch(model, opt, dataset, c, epoch, deadline)
        epoch_estimate = metrics["seconds"]
        selected = metrics["train_loss"] < best
        if selected:
            best = metrics["train_loss"]
            best_epoch = epoch
            atomic_save(root/ARM/"best.pth", dict(**model.trainable_payload(sig["pretrained"]),
                        epoch=epoch, loss=best, seed=c["seed"], signature=sig))
        row = dict(arm=ARM, seed=c["seed"], epoch=epoch, **metrics, selected=selected,
                   free_bytes=shutil.disk_usage(root).free)
        append_csv(root/"epoch_results.csv", row)
        print("EPOCH", json.dumps(row), flush=True)
        if epoch in c["milestones"]:
            if time.monotonic()+eval_estimate > deadline:
                raise TimeoutError("Insufficient budget for complete five-domain evaluation")
            atomic_json(root/"status.json", dict(status="evaluating", seed=c["seed"], epoch=epoch,
                                                  best_epoch=best_epoch, updated_at=utc_now()))
            last = cpu_payload(model, sig["pretrained"])
            state = rng_state()
            begin = time.perf_counter()
            rows = evaluate(model, c, domains, ARM, epoch, sig["pretrained"])
            eval_estimate = time.perf_counter()-begin
            decision = decide(rows, c, epoch)
            atomic_json(root/f"decision_epoch{epoch}.json", decision)
            print("DECISION", json.dumps(decision), flush=True)
            if decision["stop"]:
                atomic_json(root/"status.json", dict(status="stopped_poor_results", seed=c["seed"],
                            epoch=epoch, best_epoch=best_epoch, best_source_loss=best,
                            reason=decision["reason"], ended_at=utc_now(), automatic_restart=False))
                print("STOPPED_POOR_RESULTS", flush=True)
                return
            model.load_trainable_payload(last, sig["pretrained"])
            restore_rng(state)
            del last
    atomic_json(root/f"completed_{ARM}.json", dict(seed=c["seed"], epoch=c["epochs"],
                best_epoch=best_epoch, best_loss=best, ended_at=utc_now()))
    atomic_json(root/"status.json", dict(status="complete", seed=c["seed"], epoch=c["epochs"],
                best_epoch=best_epoch, best_source_loss=best, ended_at=utc_now(),
                eligible_for_fixed_multiseed=decision["eligible_for_fixed_multiseed"]))
    print("ALL_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    if (Path(c["output_dir"])/"experiment.json").exists():
        raise RuntimeError("Existing experiment: do not restart or overwrite status")
    try:
        run(c, args.config, args.preflight_only)
    except Exception as error:
        root = Path(c["output_dir"]).resolve()
        if root.is_relative_to(Path("/root/rivermind-data")):
            prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
            atomic_json(root/"status.json", dict(status="stopped_budget" if isinstance(error, TimeoutError)
                        else "failed", error=repr(error), previous_status=prior,
                        ended_at=utc_now(), automatic_restart=False))
        print("FAILED", repr(error), flush=True)
        raise


if __name__ == "__main__":
    main()
