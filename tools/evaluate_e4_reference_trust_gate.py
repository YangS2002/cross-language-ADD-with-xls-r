"""E4 four-domain evaluation of two predeclared seed-20 arms."""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.evaluate_e3_dual_xlsr import evaluate_domain
from tools.evaluate_four_domains_shared_xlsr import (
    AudioDomain, DOMAINS, atomic_json, digest, write_csv,
)
from tools.train_e4_reference_trust_gate import E4Detector


def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("selection_metric") != "source_train_loss":
        raise ValueError(f"Not a source-training-loss checkpoint: {checkpoint_path}")
    config, arm = payload["config"], payload["arm"]
    model = E4Detector(config, arm)
    model.task_ssl.model.load_state_dict(payload["task_ssl"], strict=True)
    model.gate.load_state_dict(payload["gate"], strict=True)
    model.detector.load_state_dict(payload["detector"], strict=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inventory = [AudioDomain(domain).validate_paths() for domain in DOMAINS]
    atomic_json(output / "inventory.json", inventory)
    if args.preflight:
        for domain in DOMAINS:
            dataset = AudioDomain(domain)
            dataset[0]
            dataset[-1]
            if dataset.handle is not None:
                dataset.handle.close()
        print("PREFLIGHT_OK", flush=True)
        return
    manifest = json.loads(Path(args.checkpoint_manifest).read_text(encoding="utf-8"))
    expected = {"task_only_seed20", "reference_gate_seed20"}
    if set(manifest) != expected:
        raise ValueError(f"Expected two predeclared E4 checkpoints, got {sorted(manifest)}")
    checkpoints = {}
    for key, value in manifest.items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = REPO / candidate
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        checkpoints[key] = candidate
    results_path = output / "results.csv"
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
    else:
        results = []
    provenance = {
        "input_protocol": "torchaudio.load plus unchanged data.pad; no resampling/augmentation",
        "score": "softmax(mean(head_a_logits,head_b_logits))[:,1]",
        "checkpoint_selection": "minimum source train loss",
        "head_selection": "none; fixed ensemble is primary",
        "checkpoint_manifest": str(args.checkpoint_manifest),
        "batch_size": args.batch_size, "workers": args.workers,
        "checkpoints": {},
    }
    device = torch.device("cuda")
    for key in ("task_only_seed20", "reference_gate_seed20"):
        complete = all(
            any(r["model"] == key and r["domain"] == domain and r["head"] == head
                for r in results)
            for domain in DOMAINS for head in ("ensemble", "a", "b")
        )
        if complete:
            print("SKIP_COMPLETE", key, flush=True)
            continue
        gc.collect()
        torch.cuda.empty_cache()
        model, info = load_model(checkpoints[key], device)
        if key != f'{info["arm"]}_seed{info["seed"]}':
            raise ValueError(f"Manifest/checkpoint mismatch: {key}, {info}")
        provenance["checkpoints"][key] = info
        atomic_json(output / "provenance.json", provenance)
        for domain in DOMAINS:
            if all(any(r["model"] == key and r["domain"] == domain and r["head"] == head
                       for r in results) for head in ("ensemble", "a", "b")):
                continue
            scored = evaluate_domain(model, domain, args.batch_size, args.workers, device)
            results = [r for r in results if not (r["model"] == key and r["domain"] == domain)]
            results.extend({"model": key, "arm": info["arm"], "seed": info["seed"], **r}
                           for r in scored)
            write_csv(results_path, results)
            atomic_json(output / "status.json", {
                "last_model": key, "last_domain": domain,
                "completed_model_domains": len({(r["model"], r["domain"])
                                                for r in results if r["head"] == "ensemble"}),
                "expected_model_domains": len(checkpoints) * len(DOMAINS),
            })
            print("COMPLETE", key, domain,
                  [r for r in scored if r["head"] == "ensemble"], flush=True)
        del model
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
