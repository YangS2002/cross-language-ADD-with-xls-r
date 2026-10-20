"""Evaluate existing M3/M4 checkpoints with one frozen XLS-R pass per batch.

The audio operation is exactly the project's ``data.pad``. Zip files are read
in memory, so no extracted dataset or resampling is introduced.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import eval_metrics_DF as em
from data import pad
from src.model import (
    DualTransitionConsistencyDetector,
    DualTransitionLowHighConsistencyDetector,
    DualTransitionLowHighVectorAdapterDetector,
    XLSRBackbone,
)


FS = Path("/root/rivermind-fs")
DATA = Path("/root/rivermind-data/data")
DOMAINS = {
    "de": (DATA / "exp/mlaad_eval_de.txt", DATA / "mlaad", None, ""),
    "uk": (FS / "exp/mlaad_eval_uk.txt", None, FS / "mlaad_eval_uk_clip.zip", "data/ys/"),
    "es": (FS / "exp/mlaad_eval_es.txt", None, FS / "mlaad_eval_es_clip.zip", "data/ys/"),
    "itw": (FS / "exp/itw.txt", None, FS / "release_in_the_wild.zip", "release_in_the_wild/"),
}
CHECKPOINTS = {
    "M3_seed20": "M3_dual_transition_low_high_consistency_seed20_10ep/checkpoints/seed_20_best_train_loss.pth",
    "M3_seed1001": "M3_dual_transition_low_high_consistency_seed1001_10ep/checkpoints/seed_1001_best_train_loss.pth",
    "M4_seed20": "M4_dual_transition_low_high_vector_adapter_seed20_10ep/checkpoints/seed_20_best_train_loss.pth",
    "M4_seed1001": "M4_dual_transition_low_high_vector_adapter_seed1001_10ep/checkpoints/seed_1001_best_train_loss.pth",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


class AudioDomain(Dataset):
    def __init__(self, name):
        self.name = name
        self.label_path, self.root, self.archive, self.prefix = DOMAINS[name]
        self.rows = []
        self.handle = None
        self.handle_pid = None
        for line in self.label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue  # Same rule as data.getlabels.
            if fields[-1] not in {"spoof", "bonafide"}:
                raise ValueError((name, fields[-1], line))
            self.rows.append((fields[1], 0 if fields[-1] == "spoof" else 1))
        labels = [label for _, label in self.rows]
        if not self.rows or 0 not in labels or 1 not in labels:
            raise ValueError(f"Missing genuine or fake samples: {name}")

    def validate_paths(self):
        if self.archive:
            with zipfile.ZipFile(self.archive) as z:
                names = set(z.namelist())
            missing = [name for name, _ in self.rows if self.prefix + name not in names]
        else:
            missing = [name for name, _ in self.rows if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{self.name}: {len(missing)} files missing, first: {missing[:3]}")
        return {"domain": self.name, "samples": len(self.rows),
                "real": sum(y == 1 for _, y in self.rows),
                "fake": sum(y == 0 for _, y in self.rows),
                "label_sha256": digest(self.label_path),
                "archive": str(self.archive) if self.archive else None}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, label = self.rows[index]
        if self.archive:
            if self.handle_pid != os.getpid():
                if self.handle is not None:
                    self.handle.close()
                self.handle = zipfile.ZipFile(self.archive)
                self.handle_pid = os.getpid()
            with self.handle.open(self.prefix + name) as f:
                waveform, _sample_rate = torchaudio.load(io.BytesIO(f.read()))
        else:
            waveform, _sample_rate = torchaudio.load(str(self.root / name))
        # Deliberately retain original first-64600-samples/repeat protocol.
        return pad(waveform).squeeze(0), label, name


def collate(batch):
    return torch.stack([item[0] for item in batch]), np.asarray([item[1] for item in batch]), [item[2] for item in batch]


def make_detector(backbone, checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location="cpu")
    config = saved["config"]
    kind = config["detector_type"]
    cls = {"dual_transition": DualTransitionConsistencyDetector,
           "dual_transition_low_high_consistency": DualTransitionLowHighConsistencyDetector,
           "dual_transition_low_high_vector_adapter": DualTransitionLowHighVectorAdapterDetector}[kind]
    kwargs = dict(
        num_layers=backbone.num_layers,
        input_dim=backbone.hidden_dim,
        hidden_dim=int(config.get("hidden_dim", 192)),
        embedding_dim=int(config.get("embedding_dim", 192)),
        num_heads=int(config.get("num_heads", 6)),
        num_encoder_layers=int(config.get("num_encoder_layers", 2)),
        feedforward_dim=int(config.get("feedforward_dim", 384)),
        dropout=float(config.get("dropout", 0.1)),
    )
    if kind != "dual_transition":
        kwargs.update(split_after_layer=int(config.get("split_after_layer", 21)),
                      residual_scale=float(config.get("residual_scale", 0.1)))
    detector = cls(**kwargs)
    if config.get("feature_type") == "raw_layers":
        from tools.train_raw_vs_transition import layer_state_tokens
        detector.build_transitions = types.MethodType(layer_state_tokens, detector)
    detector.load_state_dict(saved["detector"], strict=True)
    detector.to(device).eval()
    return detector, {"checkpoint": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
                      "seed": saved["seed"], "epoch": saved["epoch"],
                      "selection_metric": saved["selection_metric"], "detector_type": kind}


def eer(labels, scores):
    real = np.asarray(scores, dtype=np.float64)[labels == 1]
    fake = np.asarray(scores, dtype=np.float64)[labels == 0]
    value, threshold = em.compute_eer(real, fake)
    return 100.0 * float(value), float(threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--checkpoint-manifest", help="JSON mapping model key to checkpoint path relative to repo")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {domain: AudioDomain(domain) for domain in args.domains}
    inventory = [datasets[domain].validate_paths() for domain in args.domains]
    inventory_path = out / "inventory.json"
    if inventory_path.exists():
        old = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory = list({row["domain"]: row for row in old + inventory}.values())
    for row in inventory:
        print("DOMAIN", row, flush=True)
    atomic_json(out / "inventory.json", inventory)
    for domain in args.domains:
        dataset = datasets[domain]
        dataset[0]
        dataset[-1]
        if dataset.handle is not None:
            dataset.handle.close()
            dataset.handle = None
            dataset.handle_pid = None
    if args.preflight:
        print("PREFLIGHT_OK", flush=True)
        return

    device = torch.device("cuda")
    backbone_path = FS / "xlsr2_300m.pt"
    backbone = XLSRBackbone(str(backbone_path)).to(device).eval()
    checkpoint_map = CHECKPOINTS
    if args.checkpoint_manifest:
        checkpoint_map = json.loads(Path(args.checkpoint_manifest).read_text(encoding="utf-8"))
    detectors, provenance = {}, {}
    for key, relative in checkpoint_map.items():
        checkpoint_ref = Path(relative)
        if checkpoint_ref.is_absolute():
            checkpoint_path = checkpoint_ref
        else:
            # Manifests may name paths relative to the repository root, while
            # the built-in historical map above is relative to ``models/``.
            repo_relative = REPO / checkpoint_ref
            models_relative = REPO / "models" / checkpoint_ref
            checkpoint_path = repo_relative if repo_relative.is_file() else models_relative
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {key!r} not found. Tried {checkpoint_path}; "
                f"manifest value was {relative!r}."
            )
        detector, info = make_detector(backbone, checkpoint_path, device)
        detectors[key], provenance[key] = detector, info
    atomic_json(out / "provenance.json", {
        "models": provenance, "ssl_checkpoint": str(backbone_path),
        "input_protocol": "torchaudio.load, data.pad first 64600 samples or repeat; no resampling; no augmentation",
        "main_score": "softmax(mean(head_a_logits, head_b_logits))[:,1]",
        "heads": "A and B diagnostics only; no selection",
        "batch_size": args.batch_size, "workers": args.workers,
    })
    result_path = out / "results.csv"
    if result_path.exists():
        with open(result_path, newline="", encoding="utf-8") as f:
            results = [row for row in csv.DictReader(f) if row["domain"] not in args.domains]
    else:
        results = []
    for domain in args.domains:
        started = time.time()
        loader = DataLoader(datasets[domain], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=args.workers > 0)
        labels_all = []
        scores = {(key, head): [] for key in detectors for head in ("ensemble", "a", "b")}
        with torch.no_grad():
            for waveform, labels, _names in tqdm(loader, desc=domain):
                waveform = waveform.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    layers = backbone.extract_layers(waveform)
                    for key, detector in detectors.items():
                        fused, _embedding, details = detector(layers, inference_head="ensemble", return_details=True)
                        for head, logits in (("ensemble", fused), ("a", details["logits_a"]),
                                             ("b", details["logits_b"])):
                            values = torch.softmax(logits.float(), dim=1)[:, 1]
                            scores[key, head].extend(values.cpu().numpy().tolist())
                labels_all.extend(labels.tolist())
        labels_all = np.asarray(labels_all)
        if len(labels_all) != len(datasets[domain]):
            raise RuntimeError(f"Incomplete scoring: {domain}")
        for key in detectors:
            for head in ("ensemble", "a", "b"):
                value, threshold = eer(labels_all, scores[key, head])
                results.append({"domain": domain, "model": key, "head": head,
                                "eer": value, "threshold": threshold,
                                "samples": len(labels_all), "seconds_domain": time.time() - started})
        write_csv(out / "results.csv", results)
        atomic_json(out / "status.json", {"completed_domains": [r["domain"] for r in results if r["head"] == "ensemble"],
                                          "latest_domain": domain, "seconds": time.time() - started})
        print("COMPLETE", domain, [r for r in results if r["domain"] == domain and r["head"] == "ensemble"], flush=True)
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
