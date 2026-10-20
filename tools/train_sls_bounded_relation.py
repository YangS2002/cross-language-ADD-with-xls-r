"""One bounded-relation candidate, seed20, source-only epochs4..8."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from tools.sls_bounded_relation_model import BoundedRelationSLS, bounded_relation_logits
from tools.train_sls_pretrained_attention import (
    signature as original_signature, make_optimizer, step, cpu_payload,
    train_epoch, evaluate, utc_now, CROSS, PAPER_SLS, PAPER_MAMBA)
from tools.train_relational_preservation import (
    atomic_json, atomic_save, append_csv, digest, dataset_for, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains
from train import set_seed

ARM = "bounded_relation"


def signature(c, config_path):
    sig = original_signature(c, config_path)
    sig.update(trainer=digest(__file__), model=digest("tools/sls_bounded_relation_model.py"))
    sig["dependencies"].update({p: digest(p) for p in (
        "tools/train_sls_pretrained_attention.py", "tools/sls_pretrained_attention_model.py",
        "tools/launch_sls_bounded_relation.py")})
    sig["historical_results"] = {key: digest(c[key]) for key in (
        "matched_baseline_results", "matched_query_results", "matched_attention_results")}
    warm_root = Path(c["shared_warmup"]).parent
    sig["warmup_provenance"] = {name: digest(warm_root/name)
                                for name in ("experiment.json", "epoch_results.csv")}
    return sig


def analytical_tests(tau):
    generator = torch.Generator().manual_seed(919)
    base = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
    adapted = (base + torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)).requires_grad_()
    valid = torch.ones_like(base, dtype=torch.bool)
    valid[..., -1] = False
    actual, correction = bounded_relation_logits(base, adapted, tau, valid)
    delta = adapted - base
    mean = torch.where(valid, delta, 0).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
    expected_r = torch.where(valid, tau * torch.tanh((delta - mean) / tau), 0)
    torch.testing.assert_close(correction, expected_r, atol=1e-12, rtol=1e-12)
    probabilities = actual.softmax(-1)
    prior = base.masked_fill(~valid, float("-inf")).softmax(-1)
    assert bool((probabilities[..., -1] == 0).all())
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 5, dtype=torch.float64))
    # Odds distortion, excluding masked keys: log ratios change by R_i-R_j.
    change = ((probabilities[..., 0].log()-probabilities[..., 1].log())
              -(prior[..., 0].log()-prior[..., 1].log()))
    if bool((change.abs() > 2*tau+1e-12).any()):
        raise RuntimeError("Analytical odds bound failed")
    shifted, _ = bounded_relation_logits(base, adapted+17, tau, valid)
    torch.testing.assert_close(probabilities, shifted.softmax(-1), atol=1e-12, rtol=1e-12)
    zero, r0 = bounded_relation_logits(base, base, tau, valid)
    torch.testing.assert_close(zero.softmax(-1), prior, atol=0, rtol=0)
    assert float(r0.abs().max()) == 0
    def differentiable(value):
        return bounded_relation_logits(base, value, tau, valid)[0].softmax(-1)
    if not torch.autograd.gradcheck(differentiable, (adapted,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise RuntimeError("Bounded relation gradient check failed")
    initial = base.clone().requires_grad_()
    initial_logits, _ = bounded_relation_logits(base, initial, tau, valid)
    gradient = torch.autograd.grad(initial_logits.softmax(-1)[..., 0].sum(), initial)[0]
    if float(gradient.abs().sum()) <= 0:
        raise RuntimeError("Zero correction blocked initial gradients")
    return dict(formula_max_error=float((correction-expected_r).abs().max()),
                initial_correction_max=0., initial_gradient_sum=float(gradient.abs().sum()),
                maximum_log_odds_change=float(change.abs().max()),
                masked_keys_zero=True, fp64_gradcheck=True, row_shift_invariant=True)


def load_warm_backend(model, c):
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    # Historical warmup stores backend+RNG only. Verify its registered producer
    # and all three COMPLETE source rows, rather than inventing payload metadata.
    warm_root = Path(c["shared_warmup"]).parent
    producer = json.loads((warm_root/"experiment.json").read_text())["config"]
    if producer["seed"] != c["seed"] or producer["warmup_epochs"] != c["warmup_epochs"]:
        raise RuntimeError("Warmup producer seed/epoch mismatch")
    with (warm_root/"epoch_results.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["arm"] == "warmup"]
    if ([int(r["epoch"]) for r in rows] != [1, 2, 3]
            or any((int(r["seed"]), int(r["samples"]), int(r["real"]), int(r["fake"]))
                   != (c["seed"], 25380, 2580, 22800) for r in rows)):
        raise RuntimeError("Warmup provenance is incomplete")
    if set(warm) != {"backend", "rng"}:
        raise RuntimeError("Unexpected warmup payload")
    model.backend.load_state_dict(warm["backend"], strict=True)
    return warm


def preflight(c, dataset, sha):
    start = time.perf_counter()
    math_tests = analytical_tests(c["relation_tau"])
    half = c["batch_size"]//2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"]
           [:c["batch_size"]-half])
    batches = [dataset[i] for i in ids]
    wave = torch.stack([b["audio"] for b in batches]).cuda()
    labels = torch.tensor([int(b["label"]) for b in batches], device="cuda")
    set_seed(c["seed"])
    model = BoundedRelationSLS(c["ssl_checkpoint"], c["relation_tau"]).cuda().eval()
    warm = load_warm_backend(model, c)
    metadata = dict(verified_seed=c["seed"], verified_complete_source_epochs=[1, 2, 3],
                    verified_sha256=c["expected_warmup_sha256"], producer="SLS_conditional_residual_seed20",
                    payload_keys=sorted(warm))
    print("WARMUP_METADATA", json.dumps(metadata), flush=True)
    del warm
    comparisons = []
    for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=name != "fp32"):
            model.reference_copy_only = True
            copied = model(wave[:2], return_details=True)
            expected_states = [h.float().cpu() for h in copied["states"]]
            expected_score = copied["log_probs"].cpu()
            del copied
            model.reference_copy_only = False
            bounded = model(wave[:2], return_details=True)
            errors = []
            for old, new in zip(expected_states, bounded["states"]):
                new = new.float().cpu()
                torch.testing.assert_close(old, new, atol=c[name+"_atol"], rtol=c[name+"_rtol"])
                errors.append(float((old-new).abs().max()))
            torch.testing.assert_close(expected_score, bounded["log_probs"].cpu(),
                                       atol=c[name+"_atol"], rtol=c[name+"_rtol"])
            initial_r = max(row["correction_max_abs"] for row in bounded["diagnostic"]["layers"])
            if initial_r != 0:
                raise RuntimeError("Original/task QK were not identically initialized")
            row = dict(dtype=name, layer_max_errors=errors,
                       score_max_error=float((expected_score-bounded["log_probs"].cpu()).abs().max()),
                       initial_correction_max=initial_r, reference_calls=model.reference_calls,
                       task_calls=model.query_calls)
            comparisons.append(row)
            print("INITIAL_COPY_EQUIVALENCE", json.dumps(row), flush=True)
            del bounded, expected_states, expected_score
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    opt = make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for index in range(3):
        loss, output, norm = step(model, opt, wave, labels)
        if any(p.grad is not None or p.requires_grad for p in model.reference.parameters()):
            raise RuntimeError("Reference gradients leaked")
        if any(p.grad is not None or p.requires_grad
               for p in model.task.model.feature_extractor.parameters()):
            raise RuntimeError("Original acoustic convolution was unfrozen")
        for layer_index in (0, 23):
            layer = model.task.model.encoder.layers[layer_index]
            for module in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                           layer.self_attn.out_proj, layer.fc1):
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for p in module.parameters()):
                    raise RuntimeError(f"Task QKV/FFN gradients absent at layer{layer_index+1}")
        for module in (model.backend.fc0, model.backend.fc1, model.backend.fc3):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("SLS gradients absent")
        losses.append(float(loss))
        print("SELF_TEST_STEP", index, float(loss), norm, flush=True)
    qk_gradients = []
    for layer in model.task.model.encoder.layers:
        qk_gradients.append({key: float(getattr(layer.self_attn, key).weight.grad.abs().sum())
                             for key in ("q_proj", "k_proj")})
    if any(min(row.values()) <= 0 for row in qk_gradients):
        raise RuntimeError("Some bounded-relation layers lack QK gradients")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        details = model(wave[:2], return_details=True)
        contribution = details["diagnostic"]
        if max(row["correction_max_abs"] for row in contribution["layers"]) <= 0:
            raise RuntimeError("No learned correction after source self-test")
        normal = details["log_probs"].detach().cpu()
        model.reference_copy_only = True
        copied = model(wave[:2]).detach().cpu()
        model.reference_copy_only = False
        wiring_score_change = float((normal-copied).abs().max())
    model_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    record = dict(analytical_tests=math_tests, warmup_metadata=metadata, comparisons=comparisons,
                  batch=c["batch_size"], trainable_parameters=sum(p.numel() for p in model.parameters()
                                                                if p.requires_grad),
                  added_trainable_parameters=0, model_bytes=model_bytes,
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  disk_budget_bytes=max(5*2**30, int(model_bytes*3.2)+512*2**20),
                  source_test_losses=losses, all_layer_qk_gradient_sums=qk_gradients,
                  contribution_after_test_updates=contribution,
                  wiring_log_probability_change=wiring_score_change,
                  seconds=time.perf_counter()-start, source_ids=[b["file_name"] for b in batches],
                  original_context_and_reference_frozen=True, task_qkv_out_ffn_sls_trainable=True,
                  test_updates_will_be_discarded=True)
    if (record["reserved_gib"] > c["max_reserved_gib"]
            or shutil.disk_usage(c["output_dir"]).free < record["disk_budget_bytes"]):
        raise RuntimeError("Preflight memory/disk limit exceeded; no silent batch change")
    model.load_trainable_payload(model.trainable_payload(sha), sha)
    del details, model, opt, loss, output, wave, labels
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SELF_TEST_OK", json.dumps({k: record[k] for k in (
        "trainable_parameters", "added_trainable_parameters", "reserved_gib", "seconds")}), flush=True)
    return record


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    references = {}
    for key, path, arm in (("sls", c["matched_baseline_results"], "sls"),
                          ("query", c["matched_query_results"], "pretrained_query"),
                          ("attention", c["matched_attention_results"], "pretrained_attention")):
        with open(path, newline="") as stream:
            references[key] = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                               if r["arm"] == arm and int(r["milestone"]) == epoch}
    if set(values) != set(PAPER_SLS) or any(set(v) != set(values) for v in references.values()):
        raise RuntimeError("Incomplete stage/reference comparison")
    mean = float(np.mean([values[k] for k in CROSS]))
    baseline = references["sls"]
    gaps = {k: values[k]-baseline[k] for k in values}
    base_mean = float(np.mean([baseline[k] for k in CROSS]))
    # Cost screen, not target checkpoint/head/seed selection. Ordinary local
    # regressions do not trigger this rule. Identical to the last preregistered
    # severe-failure rule; no new post-result thresholds.
    severe = epoch == 4 and ((mean-base_mean > 3 and sum(gaps[k] > 5 for k in CROSS) >= 2)
             or (gaps["itw"] > 3 and mean >= base_mean)
             or any(values[k] > 35 and gaps[k] > 8 for k in CROSS))
    promising = (epoch == 8 and mean < 11.1029 and max(values[k] for k in CROSS) <= 15.2643
                 and values["itw"] <= c["itw_eer_limit"])
    return dict(epoch=epoch, stop=bool(severe),
                reason="severe_preregistered_stage4_failure" if severe else
                ("fixed_epoch8_complete" if epoch == 8 else "continue_to_epoch8"),
                per_domain=values, matched_references=references,
                per_domain_gaps={key: {k: values[k]-ref[k] for k in values}
                                 for key, ref in references.items()},
                cross_mean=mean, worst_cross=max(values[k] for k in CROSS),
                ch_de_joint=(values["ch"]+values["de"])/2,
                difference_from_historical_sls_epoch4=mean-12.92350705,
                paper_sls_gaps={k: values[k]-PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-PAPER_MAMBA[k] for k in values},
                worth_discussing_fixed_multiseed=bool(promising), automatic_multiseed=False,
                further_seeds_require_user_authorization=True,
                single_seed_is_not_stability_evidence=True)


def run(c, config_path, preflight_only=False):
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("All outputs must remain on the persistent disk")
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment; no overwrite or automatic replay")
    if c["seed"] != 20 or c["relation_tau"] != 0.5 or c["epochs"] != 8 or c["milestones"] != [4, 8]:
        raise ValueError("Only the registered seed20, bound0.5, stages4/8 candidate is authorized")
    root.mkdir(parents=True, exist_ok=True)
    (root/ARM).mkdir(exist_ok=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    sig = signature(c, config_path)
    dataset, _ = dataset_for(c)
    domains = make_domains(c)
    atomic_json(root/"inventory.json", [d.validate_paths() for d in domains.values()])
    if not preflight_only:
        atomic_json(root/"experiment.json", dict(config=c, signature=sig, pid=os.getpid(),
                    started_at=utc_now(), source_only_training=True, single_candidate=True,
                    target_use="evaluation and registered severe termination only",
                    recovery="best and epoch4 snapshot; NO Adam; no exact resume",
                    UK_is_previously_used_diagnostic_domain=True))
    checked = root/"self_test_signature.json"
    if (not preflight_only and checked.exists() and (root/"self_test.json").exists()
            and json.loads(checked.read_text()) == sig):
        print("REUSE_VERIFIED_PREFLIGHT", flush=True)
    else:
        atomic_json(root/"status.json", dict(status="preflight", seed=c["seed"], updated_at=utc_now()))
        preflight(c, dataset, sig["pretrained"])
        atomic_json(checked, sig)
    if preflight_only:
        atomic_json(root/"status.json", dict(status="preflight_only_complete", updated_at=utc_now()))
        return
    # Discard every self-test update, restore the independent same-seed warmup RNG.
    deadline = time.monotonic()+c["max_run_seconds"]
    set_seed(c["seed"])
    model = BoundedRelationSLS(c["ssl_checkpoint"], c["relation_tau"]).cuda()
    warm = load_warm_backend(model, c)
    opt = make_optimizer(model, c)
    restore_rng(warm["rng"])
    del warm
    best, best_epoch = float("inf"), None
    epoch_estimate = 0.
    eval_estimate = float(c["initial_eval_estimate_seconds"])
    for epoch in range(c["warmup_epochs"]+1, c["epochs"]+1):
        estimated_cost = epoch_estimate + (eval_estimate if epoch in c["milestones"] else 0)
        if time.monotonic()+estimated_cost > deadline:
            raise TimeoutError("Insufficient budget for next complete epoch/stage")
        atomic_json(root/"status.json", dict(status="training", seed=c["seed"], epoch=epoch,
                                              updated_at=utc_now()))
        model.forward_count = 0
        model.diagnostic_sink = lambda record, epoch=epoch: atomic_json(
            root/f"source_diagnostic_epoch{epoch}.json", dict(epoch=epoch, **record))
        metrics = train_epoch(model, opt, dataset, c, epoch, deadline)
        epoch_estimate = metrics["seconds"]
        selected = metrics["train_loss"] < best
        if selected:
            best, best_epoch = metrics["train_loss"], epoch
            atomic_save(root/ARM/"best.pth", dict(**model.trainable_payload(sig["pretrained"]),
                        epoch=epoch, loss=best, seed=c["seed"], signature=sig))
        row = dict(arm=ARM, seed=c["seed"], epoch=epoch, **metrics, selected=selected,
                   free_bytes=shutil.disk_usage(root).free)
        append_csv(root/"epoch_results.csv", row)
        print("EPOCH", json.dumps(row), flush=True)
        if epoch in c["milestones"]:
            if epoch == 4:
                snapshot = root/ARM/"source_best_at_epoch4.pth"
                if snapshot.exists():
                    raise RuntimeError("Existing stage snapshot")
                shutil.copyfile(root/ARM/"best.pth", snapshot)
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
                worth_discussing_fixed_multiseed=decision["worth_discussing_fixed_multiseed"]))
    print("ALL_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    if (Path(c["output_dir"])/"experiment.json").exists():
        raise RuntimeError("Already launched; no restart/overwrite")
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
