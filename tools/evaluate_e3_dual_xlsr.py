"""Four-domain E3 evaluation; checkpoint choice is source train loss only."""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.evaluate_four_domains_shared_xlsr import (
    AudioDomain, DOMAINS, atomic_json, collate, digest, eer, write_csv,
)
from tools.train_dual_xlsr_reference import DualReferenceDetector


def resolve_checkpoint(path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("selection_metric") != "source_train_loss":
        raise ValueError(f"Not source-train-loss checkpoint: {checkpoint_path}")
    config = payload["config"]
    arm = payload["arm"]
    model = DualReferenceDetector(config, arm)
    model.task_ssl.model.load_state_dict(payload["task_ssl"], strict=True)
    for name in (
        "task_projection", "reference_projection", "cross_attention",
        "fusion_norm", "detector",
    ):
        getattr(model, name).load_state_dict(payload[name], strict=True)
    model.to(device).eval()
    details = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "arm": arm, "seed": int(payload["seed"]),
        "epoch": int(payload["epoch"]),
        "source_train_loss": float(payload["selection_value"]),
    }
    del payload
    return model, details


def evaluate_domain(model, domain, batch_size, workers, device):
    dataset = AudioDomain(domain)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, pin_memory=True, persistent_workers=workers > 0,
    )
    labels_all = []
    scores = {head: [] for head in ("ensemble", "a", "b")}
    started = time.monotonic()
    with torch.inference_mode():
        for waveform, labels, _names in tqdm(loader, desc=domain):
            waveform = waveform.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                fused, _embedding, details = model(waveform, return_details=True)
            for head, logits in (
                ("ensemble", fused), ("a", details["logits_a"]),
                ("b", details["logits_b"]),
            ):
                score = torch.softmax(logits.float(), dim=1)[:, 1]
                scores[head].extend(score.cpu().numpy().tolist())
            labels_all.extend(labels.tolist())
    labels_all = np.asarray(labels_all)
    if len(labels_all) != len(dataset):
        raise RuntimeError(f"Incomplete {domain} scoring")
    result = []
    for head in ("ensemble", "a", "b"):
        value, threshold = eer(labels_all, scores[head])
        result.append({
            "domain": domain, "head": head, "eer": value,
            "threshold": threshold, "samples": len(dataset),
            "seconds_domain": time.monotonic() - started,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS),
                        default=list(DOMAINS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inventory = [AudioDomain(domain).validate_paths() for domain in args.domains]
    atomic_json(output / "inventory.json", inventory)
    for row in inventory:
        print("DOMAIN", row, flush=True)
    if args.preflight:
        for domain in args.domains:
            dataset = AudioDomain(domain)
            dataset[0]
            dataset[-1]
            if dataset.handle is not None:
                dataset.handle.close()
        print("PREFLIGHT_OK", flush=True)
        return
    manifest = json.loads(Path(args.checkpoint_manifest).read_text(encoding="utf-8"))
    expected = {
        f"{arm}_seed{seed}"
        for arm in ("self_reference", "frozen_reference")
        for seed in (20, 1001, 7)
    }
    if set(manifest) != expected:
        raise ValueError(f"Expected all six predeclared models, got {sorted(manifest)}")
    checkpoints = {key: resolve_checkpoint(value) for key, value in manifest.items()}
    results_path = output / "results.csv"
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
    else:
        results = []
    provenance = {
        "input_protocol": "torchaudio.load plus unchanged data.pad; no resampling/augmentation",
        "score": "softmax(mean(head_a_logits,head_b_logits))[:,1]",
        "head_selection": "none; ensemble is fixed primary score",
        "checkpoint_selection": "minimum source train loss",
        "checkpoint_manifest": str(args.checkpoint_manifest),
        "batch_size": args.batch_size, "workers": args.workers,
        "checkpoints": {},
    }
    device = torch.device("cuda")
    for key in sorted(checkpoints):
        complete = all(
            any(r["model"] == key and r["domain"] == domain and
                r["head"] == head for r in results)
            for domain in args.domains for head in ("ensemble", "a", "b")
        )
        if complete:
            print("SKIP_COMPLETE", key, flush=True)
            continue
        gc.collect()
        torch.cuda.empty_cache()
        model, info = load_model(checkpoints[key], device)
        if key != f'{info["arm"]}_seed{info["seed"]}':
            raise ValueError(f"Manifest key/checkpoint mismatch: {key}, {info}")
        provenance["checkpoints"][key] = info
        atomic_json(output / "provenance.json", provenance)
        for domain in args.domains:
            if all(any(r["model"] == key and r["domain"] == domain and
                       r["head"] == head for r in results)
                   for head in ("ensemble", "a", "b")):
                continue
            rows = evaluate_domain(model, domain, args.batch_size,
                                   args.workers, device)
            results = [r for r in results if not (r["model"] == key and r["domain"] == domain)]
            for row in rows:
                results.append({"model": key, "arm": info["arm"],
                                "seed": info["seed"], **row})
            write_csv(results_path, results)
            atomic_json(output / "status.json", {
                "last_model": key, "last_domain": domain,
                "completed_model_domains": len({(r["model"], r["domain"])
                                                for r in results if r["head"] == "ensemble"}),
                "expected_model_domains": len(checkpoints) * len(args.domains),
            })
            print("COMPLETE", key, domain,
                  [r for r in rows if r["head"] == "ensemble"], flush=True)
        del model
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
