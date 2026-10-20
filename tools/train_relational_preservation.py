"""Staged, resumable seed20 experiment; minimum source loss, target stopping only."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data import DetectDataset
from train import make_train_loader, set_seed
from tools.relational_preservation_model import RelationalDetector, loss_terms, relation_errors


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def tensor_bytes(item):
        if isinstance(item, torch.Tensor):
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(tensor_bytes(x) for x in item.values())
        if isinstance(item, (list, tuple)):
            return sum(tensor_bytes(x) for x in item)
        return 0
    # Account for the additional temporary checkpoint, not a fixed threshold
    # that incorrectly treats tiny warmup and full Adam resumes identically.
    needed = int(tensor_bytes(value) * 1.02) + 256 * 1024**2
    if shutil.disk_usage(path.parent).free < needed:
        raise RuntimeError(f"Insufficient space for atomic checkpoint ({needed} bytes): {path.parent}")
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def append_csv(path, value):
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(value))
        if not exists:
            writer.writeheader()
        writer.writerow(value)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def optimizer_for(model, config):
    ssl = [p for p in model.task.parameters() if p.requires_grad]
    groups = [{"params": list(model.detector.parameters()), "lr": float(config["head_lr"])}]
    if ssl:
        groups.append({"params": ssl, "lr": float(config["ssl_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]), foreach=False)


def dataset_for(config):
    dataset = DetectDataset(argparse.Namespace(algo=0, exp_name=config["exp_name"]),
                            config["train_data_path"], config["train_label_path"], stage="train")
    n_real = sum(label == "bonafide" for label in dataset.synthesizerlabels)
    if (len(dataset), n_real) != (25380, 2580):
        raise ValueError((len(dataset), n_real))
    return dataset, n_real / len(dataset)


def train_epoch(model, optimizer, dataset, fraction, config, epoch):
    model.train()
    loader = make_train_loader(dataset, int(config["batch_size"]),
                               int(config["seed"]) + 100000 * epoch,
                               int(config["num_workers"]), torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()
    totals = {key: 0.0 for key in ["train_loss", "classification_loss", "ce_a", "ce_b",
                                  "head_relation_loss", "temporal_scaled"]}
    real_count = fake_count = empty_real = steps = count = 0
    temporal_sum = ce_real_sum = ce_fake_sum = 0.0
    start = time.perf_counter()
    for batch in loader:
        waveform = batch["audio"].cuda(non_blocking=True)
        labels = batch["label"].view(-1).long().cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_train(waveform, labels)
        losses = loss_terms(model, output, labels, config, fraction)
        if not bool(torch.isfinite(losses["train_loss"])):
            raise FloatingPointError("Nonfinite loss")
        losses["train_loss"].backward()
        ssl_parameters = [p for p in model.task.parameters() if p.grad is not None]
        head_parameters = [p for p in model.detector.parameters() if p.grad is not None]
        ssl_norm = torch.nn.utils.clip_grad_norm_(ssl_parameters, float("inf"))
        head_norm = torch.nn.utils.clip_grad_norm_(head_parameters, float("inf"))
        torch.nn.utils.clip_grad_norm_(ssl_parameters + head_parameters,
                                      float(config["grad_clip"]), error_if_nonfinite=True)
        optimizer.step()
        n = labels.numel()
        real = labels == 1
        nr = int(real.sum())
        per_sample = 0.5 * (F.cross_entropy(output[0].float(), labels, reduction="none") +
                            F.cross_entropy(output[1].float(), labels, reduction="none"))
        for key in totals:
            totals[key] += float(losses[key].detach()) * n
        temporal_sum += float(losses["temporal_sum"].detach())
        ce_real_sum += float(per_sample[real].sum().detach())
        ce_fake_sum += float(per_sample[~real].sum().detach())
        real_count += nr
        fake_count += n - nr
        empty_real += int(nr == 0)
        count += n
        steps += 1
        if steps % 200 == 0:
            print("PROGRESS", model.arm, epoch, steps, count,
                  "loss", totals["train_loss"] / count, flush=True)
    if (count, real_count, fake_count) != (25380, 2580, 22800):
        raise RuntimeError("Source epoch did not cover all samples")
    return {**{key: value / count for key, value in totals.items()},
            "real_samples": real_count, "fake_samples": fake_count,
            "temporal_per_real": temporal_sum / real_count,
            "ce_real": ce_real_sum / real_count, "ce_fake": ce_fake_sum / fake_count,
            "empty_real_batches": empty_real, "optimizer_steps": steps,
            "ssl_grad_norm_last": float(ssl_norm), "head_grad_norm_last": float(head_norm),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "seconds": time.perf_counter() - start}


def run_block(config, dataset, fraction, arm, end_epoch):
    output = Path(config["output_dir"])
    resume = Path(config["resume_dir"]) / f"{arm}.pth"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        start_epoch = saved["epoch"] + 1
    elif arm == "warmup":
        saved = None
        start_epoch = 1
    else:
        saved = torch.load(output / "warmup.pth", map_location="cpu", weights_only=False)
        start_epoch = int(config["warmup_epochs"]) + 1
    if start_epoch > end_epoch:
        return
    print("START_BLOCK", arm, start_epoch, end_epoch, flush=True)
    set_seed(int(config["seed"]))
    model = RelationalDetector(config, arm).cuda()
    optimizer = optimizer_for(model, config)
    best = float("inf")
    if saved is not None:
        model.load_payload(saved)
        if saved["arm"] == arm:
            optimizer.load_state_dict(saved["optimizer"])
            best = saved["best_loss"]
        restore_rng(saved["rng"])
    del saved
    for epoch in range(start_epoch, end_epoch + 1):
        metrics = train_epoch(model, optimizer, dataset, fraction, config, epoch)
        selected = metrics["train_loss"] < best
        best = min(best, metrics["train_loss"])
        payload = {**model.payload(), "arm": arm, "seed": config["seed"], "epoch": epoch,
                   "config": config, "selection_metric": "source_train_loss",
                   "selection_value": metrics["train_loss"]}
        if selected and arm != "warmup":
            atomic_save(output / arm / "best_train_loss.pth", payload)
        resume_payload = {**payload, "best_loss": best, "optimizer": optimizer.state_dict(), "rng": rng_state()}
        atomic_save(resume, resume_payload)
        if arm == "warmup" and epoch == int(config["warmup_epochs"]):
            atomic_save(output / "warmup.pth", resume_payload)
        row = {"arm": arm, "seed": config["seed"], "epoch": epoch, **metrics,
               "selected": selected, "best_loss": best,
               "disk_free_gib": shutil.disk_usage(output).free / 1024**3}
        append_csv(output / "epoch_results.csv", row)
        print("EPOCH", json.dumps(row), flush=True)
        del payload, resume_payload
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def unit_test():
    torch.manual_seed(20)
    x = torch.randn(2, 7, 9)
    q = torch.linalg.qr(torch.randn(9, 9)).Q
    assert float(relation_errors(x @ q + 2, x).max()) < 1e-10
    altered = (x + torch.randn_like(x) * 0.2).requires_grad_()
    loss = relation_errors(altered, x).mean()
    loss.backward()
    assert float(altered.grad.norm()) > 0
    print("UNIT_TEST_OK", flush=True)


def self_test(config, dataset, fraction):
    unit_test()
    device = torch.device("cuda")
    real_indices = [i for i, label in enumerate(dataset.synthesizerlabels) if label == "bonafide"]
    batch = int(config["batch_size"])
    wave = torch.stack([dataset[i]["audio"] for i in real_indices[:batch]]).cuda()
    labels = torch.ones(batch, device=device, dtype=torch.long)
    results = []
    initial = None
    for arm in ["ft", "rp"]:
        gc.collect()
        torch.cuda.empty_cache()
        set_seed(config["seed"])
        model = RelationalDetector(config, arm).cuda()
        model.detector.eval()
        opt = optimizer_for(model, config)
        torch.cuda.reset_peak_memory_stats()
        calls = []
        hooks = [layer.register_forward_hook(lambda _m, _i, _o, k=k: calls.append(k))
                 for k, layer in enumerate(model.teacher.model.encoder.layers, 1)]
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_train(wave, labels)
            terms = loss_terms(model, out, labels, config, fraction)
            if step == 0:
                if float(terms["temporal_sum"].detach()) > 1e-5:
                    raise RuntimeError("Identical SSL paths disagree at anchor")
                current = out[0].detach().float().cpu()
                if initial is None:
                    initial = current
                else:
                    torch.testing.assert_close(initial, current, atol=1e-5, rtol=1e-5)
            terms["train_loss"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5,
                                          error_if_nonfinite=True)
            assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                       for p in model.task.model.encoder.layers[0].parameters())
            assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                       for p in model.task.model.encoder.layers[-1].parameters())
            assert all(p.grad is None for p in model.teacher.parameters())
            assert all(p.grad is None for p in model.task.model.feature_extractor.parameters())
            opt.step()
            torch.cuda.synchronize()
            print("SELF_TEST_STEP", arm, step, float(terms["train_loss"].detach()),
                  float(terms["temporal_sum"].detach()), time.perf_counter() - t, flush=True)
        assert calls == list(range(1, 13)) * 3, calls
        for hook in hooks:
            hook.remove()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            isolated = model.forward_train(wave, labels)
        isolated[3].backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                   for p in model.task.model.encoder.layers[0].parameters())
        assert all(p.grad is None for p in model.task.model.encoder.layers[-1].parameters())
        assert all(p.grad is None for p in model.teacher.parameters())
        print("RELATION_GRADIENT_SCOPE_OK", arm, flush=True)
        result = {"arm": arm, "batch": batch, "pure_real_stress": True,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        results.append(result)
        print("SELF_TEST_MEMORY", json.dumps(result), flush=True)
        # Release tensors retaining graphs before constructing the next arm.
        del out, isolated, terms, model, opt, current
    atomic_json(Path(config["output_dir"]) / "self_test.json", results)
    if max(r["peak_reserved_gib"] for r in results) > 20:
        raise RuntimeError("Pressure test exceeded the registered 20 GiB budget")
    print("SELF_TEST_OK", flush=True)


def stopping_decision(config, milestone, rows):
    values = {(r["arm"], r["domain"]): r["eer"] for r in rows if r["head"] == "ensemble"}
    domains = [d for d in config["domains"] if d != "itw"]
    gaps = [values[("rp", d)] - values[("ft", d)] for d in domains]
    failed = sum(gap > 5 for gap in gaps) >= 2 and np.mean(gaps) > 3
    if milestone >= 8 and np.mean(gaps) >= 0 and values[("rp", "itw")] - values[("ft", "itw")] > 2:
        failed = True
    return {"milestone": milestone, "stop_rp": bool(failed), "cross_language_gaps": dict(zip(domains, gaps)),
            "cross_language_mean_gap": float(np.mean(gaps)),
            "itw_gap": values[("rp", "itw")] - values[("ft", "itw")]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["precision"] != "bf16" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This registered run requires BF16")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    dataset, fraction = dataset_for(config)
    if args.self_test:
        self_test(config, dataset, fraction)
        return
    from tools.evaluate_relational_preservation import make_domains, evaluate
    datasets = make_domains(config)
    inventory = []
    for name, target in datasets.items():
        if not args.preflight and name == "itw" and not target.archive.is_file():
            inventory.append({"domain": "itw", "status": "download_pending"})
            print("ITW_DOWNLOAD_PENDING_SOURCE_TRAINING_ALLOWED", flush=True)
        else:
            inventory.append(target.validate_paths())
    atomic_json(output / "inventory.json", inventory)
    if args.preflight:
        print("PREFLIGHT_OK", json.dumps(inventory), flush=True)
        return
    signature = {"config_sha256": digest(config_path), "labels_sha256": digest(config["train_label_path"]),
                 "trainer_sha256": digest(__file__),
                 "model_sha256": digest(REPO / "tools/relational_preservation_model.py"),
                 "existing_model_sha256": digest(REPO / "src/model.py"),
                 "data_sha256": digest(REPO / "data.py"),
                 "ssl_sha256": digest(config["ssl_checkpoint"])}
    provenance = output / "experiment.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous["signature"] != signature:
            raise RuntimeError("Cannot resume with changed code/config/data")
    else:
        atomic_json(provenance, {"config": config, "signature": signature,
                                "torch_version": torch.__version__, "target_use": "exploratory stopping only"})
    run_block(config, dataset, fraction, "warmup", int(config["warmup_epochs"]))
    for milestone in config["milestones"]:
        decision_path = output / f"decision_epoch{milestone}.json"
        if decision_path.exists():
            if json.loads(decision_path.read_text())["stop_rp"]:
                print("PREVIOUSLY_STOPPED", flush=True)
                return
            continue
        rows = []
        for arm in config["arms"]:
            run_block(config, dataset, fraction, arm, milestone)
        wait_started = time.monotonic()
        while not datasets["itw"].archive.is_file():
            if time.monotonic() - wait_started > 3600:
                raise RuntimeError("ITW not ready after one hour; source checkpoint retained")
            print("WAITING_FOR_ITW", milestone, int(time.monotonic() - wait_started), flush=True)
            time.sleep(30)
        # Update inventory after verified download is promoted to final name.
        atomic_json(output / "inventory.json", [d.validate_paths() for d in datasets.values()])
        for arm in config["arms"]:
            rows.extend(evaluate(config, arm, milestone, datasets))
        decision = stopping_decision(config, milestone, rows)
        atomic_json(decision_path, decision)
        print("MILESTONE", json.dumps(decision), flush=True)
        if decision["stop_rp"]:
            atomic_json(output / "status.json", {"status": "stopped_for_poor_results", **decision})
            print("STOPPED_FOR_POOR_RESULTS", flush=True)
            return
    atomic_json(output / "status.json", {"status": "complete", "seed": config["seed"]})
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED", type(exc).__name__, str(exc), flush=True)
        raise
