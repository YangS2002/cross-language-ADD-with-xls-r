"""Paired source-only training for E2 without editing existing model/train modules."""

import argparse
import gc
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import DetectDataset
from src.model import DualTransitionConsistencyDetector, Model, build_optimizer
from train import make_train_loader, save_checkpoint, set_seed, train_one_epoch, write_csv


def layer_state_tokens(self, layers):
    """Use the 23 contextual states aligned to the 23 original transitions."""
    tokens = torch.stack(layers[1:], dim=1)
    if tokens.size(1) != self.num_transitions:
        raise RuntimeError((tokens.shape, self.num_transitions))
    return tokens


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def make_detector(backbone, config, feature_type):
    detector = DualTransitionConsistencyDetector(
        num_layers=backbone.num_layers,
        input_dim=backbone.hidden_dim,
        hidden_dim=int(config["hidden_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        num_heads=int(config["num_heads"]),
        num_encoder_layers=int(config["num_encoder_layers"]),
        feedforward_dim=int(config["feedforward_dim"]),
        dropout=float(config["dropout"]),
    )
    if feature_type == "raw_layers":
        detector.build_transitions = types.MethodType(layer_state_tokens, detector)
    return detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    methods = list(config["methods"])
    seeds = [int(s) for s in config["seeds"]]
    if methods != ["transition", "raw_layers"] or len(set(seeds)) != len(seeds):
        raise ValueError("E2 methods/seeds must match the registered config")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps({
        "config": config, "config_sha256": hash_file(config_path),
        "source_labels_sha256": hash_file(config["train_label_path"]),
        "trainer_sha256": hash_file(__file__),
        "torch_version": torch.__version__,
        "checkpoint_selection": "minimum source train loss",
        "target_evaluation_during_training": False,
        "augmentation": "none; original data.pad",
    }, indent=2) + "\n", encoding="utf-8")
    device = torch.device("cuda")
    amp = bool(config["amp"])
    dataset_args = argparse.Namespace(algo=0, exp_name=config["exp_name"])
    train_dataset = DetectDataset(dataset_args, config["train_data_path"],
                                  config["train_label_path"], stage="train")
    # XLS-R is frozen and shared. Reset the detector seed before *each* run.
    model = Model(device, config).to(device)
    if args.self_test:
        loader = make_train_loader(train_dataset, 2, seeds[0], 0, device)
        waveform = next(iter(loader))["audio"].to(device)
        for feature_type in methods:
            set_seed(seeds[0])
            model.detector = make_detector(model.ssl_model, config, feature_type).to(device)
            logits_a, logits_b, _a, _b, relation = model.forward_train(waveform)
            loss = logits_a.float().square().mean() + logits_b.float().square().mean() + relation
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {feature_type} self-test loss")
            print("SELF_TEST", feature_type, tuple(logits_a.shape), float(loss), flush=True)
        print("SELF_TEST_OK", flush=True)
        return
    rows = []
    manifest = {}
    for seed in seeds:
        for feature_type in methods:
            run_dir = output / f"{feature_type}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "best_train_loss.pth"
            print("START", feature_type, seed, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            set_seed(seed)
            loader = make_train_loader(train_dataset, int(config["batch_size"]), seed,
                                       int(config["num_workers"]), device)
            model.detector = make_detector(model.ssl_model, config, feature_type).to(device)
            optimizer = build_optimizer(model, float(config["lr"]), float(config["weight_decay"]))
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(
                [float(config["fake_weight"]), float(config["real_weight"])],
                dtype=torch.float32, device=device))
            scaler = torch.cuda.amp.GradScaler(enabled=amp)
            best = float("inf")
            run_config = dict(config, feature_type=feature_type, seeds=[seed],
                              output_dir=str(run_dir))
            for epoch in range(1, int(config["epochs"]) + 1):
                start = time.time()
                metrics = train_one_epoch(model, loader, optimizer, criterion, scaler,
                                          device, amp, float(config["lambda_relation"]),
                                          float(config["grad_clip"]))
                loss = float(metrics["train_loss"])
                selected = loss < best
                if selected:
                    best = loss
                    save_checkpoint(str(checkpoint_path), {
                        "seed": seed, "epoch": epoch,
                        "selection_metric": "train_loss",
                        "selection_value": loss,
                        "config": run_config,
                        "detector": model.detector.state_dict(),
                    })
                row = {"method": feature_type, "seed": seed, "epoch": epoch,
                       **metrics, "is_best_train_loss": selected,
                       "best_train_loss_so_far": best,
                       "seconds": time.time() - start}
                rows.append(row)
                write_csv(str(output / "epoch_results.csv"), rows)
                print("EPOCH", feature_type, seed, epoch, "loss", loss,
                      "best", selected, "seconds", row["seconds"], flush=True)
            print("FINISH", feature_type, seed, "best_loss", best,
                  "checkpoint", checkpoint_path, flush=True)
            manifest[f"{feature_type}_seed{seed}"] = str(checkpoint_path)
            (output / "checkpoint_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            del loader, optimizer, criterion, scaler
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
