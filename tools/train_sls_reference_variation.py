"""One source-only seed20 candidate; reused warmup, five-domain stages4/8."""
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

import numpy as np
import torch
import yaml

from tools.sls_reference_variation_model import ReferenceVariationSLS, weighted_variation
from tools.train_relational_preservation import (
    atomic_json, atomic_save, append_csv, digest, dataset_for, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains
from tools.train_sls_pretrained_attention import (
    make_optimizer, step, cpu_payload, train_epoch, evaluate, utc_now,
    CROSS, PAPER_SLS, PAPER_MAMBA)
from train import set_seed

ARM = "reference_variation"


def signature(c, config_path):
    from fairseq.modules import MultiheadAttention
    from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
    installed = [hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                 for fn in (MultiheadAttention.forward, TransformerEncoder.extract_features)]
    if installed != ["84410048dc06fc75ae4f0b5e3b8d23660588f6276fecce5a5a67b65ba2167030",
                     "78d2dfde3b2f790ab380d4b9e4f1a582442495689c2f83f80776a7b7e2b6e6f3"]:
        raise RuntimeError("Installed fairseq changed; inspect before execution")
    result = dict(config=digest(config_path), trainer=digest(__file__),
                  model=digest("tools/sls_reference_variation_model.py"),
                  pretrained=digest(c["ssl_checkpoint"]), warmup=digest(c["shared_warmup"]),
                  labels=digest(c["train_label_path"]), torch_version=torch.__version__,
                  fairseq_function_sha256=installed,
                  dependencies={p: digest(p) for p in (
                      "data.py", "train.py", "tools/sls_reference_model.py",
                      "tools/train_relational_preservation.py", "tools/train_sls_pretrained_attention.py",
                      "tools/evaluate_relational_preservation.py",
                      "tools/evaluate_four_domains_shared_xlsr.py")},
                  historical_results={k: digest(c[k]) for k in (
                      "matched_baseline_results", "matched_query_results", "matched_attention_results")})
    if (result["pretrained"] != c["expected_pretrained_sha256"]
            or result["warmup"] != c["expected_warmup_sha256"]):
        raise ValueError("Original/warmup SHA mismatch")
    return result


def analytical_tests():
    generator = torch.Generator().manual_seed(919)
    h = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    p = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64).softmax(-1)
    d, _ = weighted_variation(h, p)
    brute = (p[..., None]*(h[:, :, None, :]-h[:, None, :, :]).square()).sum(2)
    torch.testing.assert_close(d, brute, atol=1e-10, rtol=1e-10)
    actual_grad = torch.autograd.grad(d.sum(), h, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(brute.sum(), h)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)
    shifted, _ = weighted_variation(h.detach()+100, p)
    torch.testing.assert_close(d.detach(), shifted, atol=1e-10, rtol=1e-10)
    identity, _ = weighted_variation(h.detach(), torch.eye(5, dtype=h.dtype).expand(2, -1, -1))
    torch.testing.assert_close(identity, torch.zeros_like(identity), atol=1e-10, rtol=0)
    return dict(brute_force_max_error=float((d-brute).abs().max()),
                gradient_max_error=float((actual_grad-expected_grad).abs().max()),
                shift_max_error=float((d-shifted).abs().max()), identity_max=float(identity.max()))


def preflight(c, dataset, sha):
    start = time.perf_counter()
    math_tests = analytical_tests()
    half = c["batch_size"]//2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"]
           [:c["batch_size"]-half])
    batches = [dataset[i] for i in ids]
    wave = torch.stack([b["audio"] for b in batches]).cuda()
    labels = torch.tensor([int(b["label"]) for b in batches], device="cuda")
    set_seed(c["seed"])
    model = ReferenceVariationSLS(c["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.load_warm_backend(warm)
    metadata = {k: v for k, v in warm.items() if isinstance(v, (str, int, float, bool))}
    print("WARMUP_METADATA", json.dumps(metadata), flush=True)
    if "epoch" in warm and warm["epoch"] != c["warmup_epochs"]:
        raise RuntimeError("Wrong warmup epoch")
    if "seed" in warm and warm["seed"] != c["seed"]:
        raise RuntimeError("Warmup seed mismatch")
    del warm
    comparisons = []
    for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=name != "fp32"):
            model.evidence_enabled = False
            expected = model(wave[:2], return_details=True)
            old_states = [h.float().cpu() for h in expected["states"]]
            old_output = expected["log_probs"].cpu()
            del expected
            model.evidence_enabled = True
            actual = model(wave[:2], return_details=True)
            errors = []
            for old, new in zip(old_states, actual["states"]):
                new = new.float().cpu()
                torch.testing.assert_close(old, new, atol=c[name+"_atol"], rtol=c[name+"_rtol"])
                errors.append(float((old-new).abs().max()))
            torch.testing.assert_close(old_output, actual["log_probs"].cpu(),
                                       atol=c[name+"_atol"], rtol=c[name+"_rtol"])
            if float(actual["evidence"].abs().mean()) < 1e-5:
                raise RuntimeError("Variation evidence degenerated to zero")
            comparisons.append(dict(dtype=name, layer_errors=errors,
                score_max_error=float((old_output-actual["log_probs"].cpu()).abs().max()),
                relation_diagnostics=actual["diagnostic"]["layers"]))
            print("INITIAL_EQUIVALENCE", name, comparisons[-1]["score_max_error"], flush=True)
            del actual, old_states, old_output
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    opt = make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for index in range(3):
        loss, output, norm = step(model, opt, wave, labels)
        if any(p.grad is not None or p.requires_grad for p in model.reference.parameters()):
            raise RuntimeError("Frozen original route received gradients")
        for layer_index in (0, 23):
            layer = model.task.model.encoder.layers[layer_index]
            for module in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj, layer.self_attn.out_proj, layer.fc1):
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                    raise RuntimeError(f"Native task gradients absent at layer{layer_index+1}")
        for module in (model.backend.fc0, model.backend.fc1, model.backend.fc3,
                       model.backend.variation_fc):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("Classifier/variation gradient absent")
        losses.append(float(loss))
        print("SELF_TEST_STEP", index, float(loss), norm, flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        details = model(wave[:2], return_details=True)
        contribution = details["diagnostic"]
        if contribution["delta_to_native_hidden_norm"] <= 0:
            raise RuntimeError("Zero contribution after source test updates")
    model_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    record = dict(analytical_tests=math_tests, warmup_metadata=metadata, comparisons=comparisons,
                  batch=c["batch_size"], trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  supplemental_parameters=sum(p.numel() for p in model.backend.variation_fc.parameters()),
                  model_bytes=model_bytes, reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  disk_budget_bytes=max(5*2**30, int(model_bytes*3.2)+512*2**20),
                  source_test_losses=losses, contribution_after_test_updates=contribution,
                  seconds=time.perf_counter()-start, source_ids=[b["file_name"] for b in batches],
                  task_qkv_and_ffn_trainable=True, frozen_reference_gradient=False,
                  test_updates_will_be_discarded=True)
    if (record["reserved_gib"] > c["max_reserved_gib"]
            or shutil.disk_usage(c["output_dir"]).free < record["disk_budget_bytes"]):
        raise RuntimeError("Preflight memory/disk limit exceeded")
    payload = model.trainable_payload(sha)
    model.load_trainable_payload(payload, sha)
    del payload, details, model, opt, loss, output, wave, labels
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SELF_TEST_OK", json.dumps({k: record[k] for k in (
        "trainable_parameters", "supplemental_parameters", "reserved_gib", "seconds")}), flush=True)
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
    baseline = references["sls"]
    mean = float(np.mean([values[k] for k in CROSS]))
    base_mean = float(np.mean([baseline[k] for k in CROSS]))
    gaps = {k: values[k]-baseline[k] for k in values}
    severe = epoch == 4 and ((mean-base_mean > 3 and sum(gaps[k] > 5 for k in CROSS) >= 2)
             or (gaps["itw"] > 3 and mean >= base_mean)
             or any(values[k] > 35 and gaps[k] > 8 for k in CROSS))
    promising = (epoch == 8 and mean < 11.1029 and max(values[k] for k in CROSS) <= 15.2643
                 and values["itw"] <= c["itw_eer_limit"])
    return dict(epoch=epoch, stop=bool(severe),
                reason="preregistered_severe_stage4_failure" if severe else
                "fixed_epoch8_complete" if epoch == 8 else "continue_to_epoch8",
                per_domain=values, historical=references,
                gaps={name: {k: values[k]-reference[k] for k in values}
                      for name, reference in references.items()},
                cross_mean=mean, worst_cross=max(values[k] for k in CROSS),
                ch_de_joint=(values["ch"]+values["de"])/2,
                paper_sls_gaps={k: values[k]-PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-PAPER_MAMBA[k] for k in values},
                difference_from_historical_sls_epoch4=mean-12.9235,
                promising_for_fixed_multiseed=bool(promising), eligible_for_automatic_multiseed=False,
                single_seed_is_not_stability_evidence=True)


def run(c, config_path, preflight_only=False):
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent data disk required")
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no replay")
    root.mkdir(parents=True, exist_ok=True)
    (root/ARM).mkdir(exist_ok=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    sig = signature(c, config_path)
    dataset, _ = dataset_for(c)
    domains = make_domains(c)
    atomic_json(root/"inventory.json", [d.validate_paths() for d in domains.values()])
    atomic_json(root/"status.json", dict(status="preflight", updated_at=utc_now()))
    preflight(c, dataset, sig["pretrained"])
    if preflight_only:
        atomic_json(root/"status.json", dict(status="preflight_only_complete", updated_at=utc_now()))
        return
    deadline = time.monotonic()+c["max_run_seconds"]
    atomic_json(root/"experiment.json", dict(config=c, signature=sig, pid=os.getpid(), started_at=utc_now(),
                source_only_training=True, single_candidate=True,
                recovery="best plus stage4 source-best snapshot; NO Adam; cannot exactly resume",
                UK_is_previously_used_diagnostic_domain=True,
                target_use="evaluation and preregistered severe termination only"))
    set_seed(c["seed"])
    model = ReferenceVariationSLS(c["ssl_checkpoint"]).cuda()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.load_warm_backend(warm)
    opt = make_optimizer(model, c)
    restore_rng(warm["rng"])
    del warm
    best = float("inf")
    best_epoch = None
    epoch_estimate = 0.
    eval_estimate = float(c["initial_eval_estimate_seconds"])
    for epoch in range(c["warmup_epochs"]+1, c["epochs"]+1):
        estimated = epoch_estimate+(eval_estimate if epoch in c["milestones"] else 0)
        if time.monotonic()+estimated > deadline:
            raise TimeoutError("Budget insufficient for complete next source/stage")
        atomic_json(root/"status.json", dict(status="training", seed=c["seed"], epoch=epoch, updated_at=utc_now()))
        model.forward_count = 0
        def record_diagnostic(row, epoch=epoch):
            atomic_json(root/f"source_diagnostic_epoch{epoch}.json", dict(epoch=epoch, **row))
            print("EVIDENCE", epoch, row["forward"], row["delta_to_native_hidden_norm"],
                  row["mean_absolute_score_contribution"], flush=True)
        model.diagnostic_sink = record_diagnostic
        metrics = train_epoch(model, opt, dataset, c, epoch, deadline)
        model.diagnostic_sink = None
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
        if epoch == 4:
            shutil.copy2(root/ARM/"best.pth", root/ARM/"stage4_source_best.pth")
        if epoch in c["milestones"]:
            if time.monotonic()+eval_estimate > deadline:
                raise TimeoutError("Budget insufficient for full five-domain evaluation")
            atomic_json(root/"status.json", dict(status="evaluating", epoch=epoch, best_epoch=best_epoch,
                                                  updated_at=utc_now()))
            last = cpu_payload(model, sig["pretrained"])
            state = rng_state()
            begin = time.perf_counter()
            rows = evaluate(model, c, domains, ARM, epoch, sig["pretrained"])
            eval_estimate = time.perf_counter()-begin
            decision = decide(rows, c, epoch)
            atomic_json(root/f"decision_epoch{epoch}.json", decision)
            print("DECISION", json.dumps(decision), flush=True)
            if decision["stop"]:
                atomic_json(root/"status.json", dict(status="stopped_poor_results", epoch=epoch,
                    best_epoch=best_epoch, best_source_loss=best, ended_at=utc_now(), automatic_restart=False))
                print("STOPPED_POOR_RESULTS", flush=True)
                return
            model.load_trainable_payload(last, sig["pretrained"])
            restore_rng(state)
            del last
    atomic_json(root/f"completed_{ARM}.json", dict(seed=c["seed"], epoch=c["epochs"],
                best_epoch=best_epoch, best_loss=best, ended_at=utc_now()))
    atomic_json(root/"status.json", dict(status="complete", seed=c["seed"], epoch=c["epochs"],
                best_epoch=best_epoch, best_source_loss=best, ended_at=utc_now(), automatic_multiseed=False))
    print("ALL_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    root = Path(c["output_dir"]).resolve()
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no automatic restart")
    try:
        run(c, args.config, args.preflight_only)
    except Exception as error:
        if root.is_relative_to(Path("/root/rivermind-data")):
            prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
            atomic_json(root/"status.json", dict(status="stopped_budget" if isinstance(error, TimeoutError)
                        else "failed", error=repr(error), previous_status=prior, ended_at=utc_now(),
                        automatic_restart=False))
        print("FAILED", repr(error), flush=True)
        raise


if __name__ == "__main__":
    main()
