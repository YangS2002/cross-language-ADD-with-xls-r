"""E3: task-query / original-reference layer attention, with paired self-reference control.

Standalone on purpose: the project's existing Model freezes XLS-R and is dirty on
the remote machine. This script does not change that model or the audio loader.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from fairseq import checkpoint_utils

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import DetectDataset
from src.model import DualTransitionConsistencyDetector, XLSRBackbone
from train import make_train_loader, save_checkpoint, set_seed, write_csv


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_token_interface(self, layers):
    if isinstance(layers, torch.Tensor):
        tokens = layers
    else:
        tokens = torch.stack(layers[1:], dim=1)
    if tokens.ndim != 4 or tokens.size(1) != self.num_transitions:
        raise RuntimeError(f"Unexpected fused token shape: {tuple(tokens.shape)}")
    return tokens


class TrainableXLSR(nn.Module):
    """Fairseq XLS-R with gradients enabled and SSL dropout/masking disabled."""

    def __init__(self, checkpoint_path):
        super().__init__()
        models, _cfg, _task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        if len(models) != 1:
            raise RuntimeError(f"Expected one XLS-R model, got {len(models)}")
        self.model = models[0]
        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layers"):
            raise RuntimeError("Cannot find XLS-R transformer layers")
        self.num_layers = len(self.model.encoder.layers)
        self.hidden_dim = int(getattr(self.model.cfg, "encoder_embed_dim", 1024))
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.model.eval()

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def extract_layers(self, waveform):
        waveform = XLSRBackbone._prepare_waveform(waveform)
        model_parameter = next(self.model.parameters())
        waveform = waveform.to(device=model_parameter.device, dtype=model_parameter.dtype)
        result = self.model(waveform, mask=False, features_only=True)
        if "layer_results" not in result:
            raise RuntimeError(f"XLS-R returned {list(result)} without layer_results")
        layers = []
        for item in result["layer_results"]:
            layer = item[0] if isinstance(item, (tuple, list)) else item
            if layer.ndim != 3:
                raise RuntimeError(f"Unexpected layer shape: {tuple(layer.shape)}")
            if layer.size(1) == waveform.size(0):
                layer = layer.transpose(0, 1)
            elif layer.size(0) != waveform.size(0):
                raise RuntimeError(f"Cannot find batch axis: {tuple(layer.shape)}")
            layers.append(layer)
        if len(layers) != self.num_layers:
            raise RuntimeError(f"Expected {self.num_layers} layers, got {len(layers)}")
        return layers


class DualReferenceDetector(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        if arm not in {"self_reference", "frozen_reference"}:
            raise ValueError(arm)
        self.arm = arm
        self.task_ssl = TrainableXLSR(config["ssl_checkpoint"])
        # Build in both arms so RNG consumption and parameter topology are paired.
        self.original_ssl = XLSRBackbone(config["ssl_checkpoint"])
        if (self.task_ssl.num_layers, self.task_ssl.hidden_dim) != (
            self.original_ssl.num_layers, self.original_ssl.hidden_dim
        ):
            raise RuntimeError("XLS-R branch dimensions differ")
        dim = int(config["hidden_dim"])
        input_dim = self.task_ssl.hidden_dim
        self.task_projection = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim))
        self.reference_projection = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim))
        self.cross_attention = nn.MultiheadAttention(
            dim, int(config["num_heads"]), dropout=0.0, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.detector = DualTransitionConsistencyDetector(
            num_layers=self.task_ssl.num_layers,
            input_dim=dim,
            hidden_dim=dim,
            embedding_dim=int(config["embedding_dim"]),
            num_heads=int(config["num_heads"]),
            num_encoder_layers=int(config["num_encoder_layers"]),
            feedforward_dim=int(config["feedforward_dim"]),
            dropout=float(config["dropout"]),
        )
        self.detector.build_transitions = types.MethodType(raw_token_interface, self.detector)

    def train(self, mode=True):
        super().train(mode)
        self.task_ssl.eval()  # SSL dropout/masking off; parameters retain gradients.
        self.original_ssl.eval()
        return self

    def fused_tokens(self, waveform):
        task_layers = self.task_ssl.extract_layers(waveform)
        task_states = torch.stack(task_layers[1:], dim=1)
        if self.arm == "frozen_reference":
            with torch.no_grad():
                reference_layers = self.original_ssl.extract_layers(waveform)
                reference_states = torch.stack(reference_layers[1:], dim=1)
        else:
            reference_states = task_states.detach()
        task_tokens = self.task_projection(task_states)
        reference_tokens = self.reference_projection(reference_states)
        b, n, t, d = task_tokens.shape
        query = task_tokens.permute(0, 2, 1, 3).reshape(b * t, n, d)
        reference = reference_tokens.permute(0, 2, 1, 3).reshape(b * t, n, d)
        context, _weights = self.cross_attention(
            query, reference, reference, need_weights=False
        )
        fused = self.fusion_norm(query + context)
        return fused.reshape(b, t, n, d).permute(0, 2, 1, 3).contiguous()

    def forward_train(self, waveform):
        return self.detector.forward_train(self.fused_tokens(waveform))

    def forward(self, waveform, return_details=False):
        return self.detector(
            self.fused_tokens(waveform), inference_head="ensemble",
            return_details=return_details,
        )


def make_optimizer(model, config):
    ssl_parameters = [p for p in model.task_ssl.parameters() if p.requires_grad]
    head_parameters = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("task_ssl.")
    ]
    ssl_ids = {id(p) for p in ssl_parameters}
    head_ids = {id(p) for p in head_parameters}
    if ssl_ids & head_ids or not ssl_ids or not head_ids:
        raise RuntimeError("Invalid SSL/head optimizer partition")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if ssl_ids | head_ids != expected:
        raise RuntimeError("Optimizer does not cover all trainable parameters")
    return torch.optim.AdamW([
        {"params": ssl_parameters, "lr": float(config["ssl_lr"])},
        {"params": head_parameters, "lr": float(config["head_lr"])},
    ], weight_decay=float(config["weight_decay"]))


def train_epoch(model, loader, optimizer, criterion, scaler, config, device,
                max_optimizer_steps=None):
    model.train()
    effective = int(config["effective_batch_size"])
    amp = bool(config["amp"])
    total_samples = len(loader.dataset)
    if max_optimizer_steps is not None:
        total_samples = min(total_samples, int(max_optimizer_steps) * effective)
    processed = 0
    group_seen = 0
    group_target = min(effective, total_samples)
    totals = {k: 0.0 for k in (
        "train_loss", "classification_loss", "ce_a", "ce_b", "relation_loss",
        "correct_a", "correct_b",
    )}
    steps = 0
    skipped_steps = 0
    consecutive_skips = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        waveform = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].view(-1).long().to(device, non_blocking=True)
        count = int(labels.size(0))
        if count + group_seen > group_target:
            raise RuntimeError("Effective batch boundary crossed")
        with torch.cuda.amp.autocast(enabled=amp):
            logits_a, logits_b, _embedding_a, _embedding_b, relation = model.forward_train(waveform)
        ce_a = criterion(logits_a.float(), labels)
        ce_b = criterion(logits_b.float(), labels)
        classification = 0.5 * (ce_a + ce_b)
        loss = classification + float(config["lambda_relation"]) * relation
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite E3 training loss")
        scaler.scale(loss * (count / group_target)).backward()
        for key, value in (
            ("train_loss", loss), ("classification_loss", classification),
            ("ce_a", ce_a), ("ce_b", ce_b), ("relation_loss", relation),
        ):
            totals[key] += float(value.detach()) * count
        totals["correct_a"] += int((logits_a.detach().argmax(1) == labels).sum())
        totals["correct_b"] += int((logits_b.detach().argmax(1) == labels).sum())
        processed += count
        group_seen += count
        if group_seen == group_target:
            scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            norm = nn.utils.clip_grad_norm_(
                parameters, float("inf"), error_if_nonfinite=False,
            )
            if math.isfinite(float(norm)):
                nn.utils.clip_grad_norm_(
                    parameters, float(config["grad_clip"]), error_if_nonfinite=True,
                )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            was_skipped = scaler.get_scale() < scale_before
            if was_skipped:
                skipped_steps += 1
                consecutive_skips += 1
            else:
                consecutive_skips = 0
            if not math.isfinite(float(norm)) and not was_skipped:
                raise FloatingPointError("Nonfinite gradients were not skipped by AMP")
            if consecutive_skips > 10:
                raise FloatingPointError("More than 10 consecutive AMP-overflow steps")
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            group_seen = 0
            group_target = min(effective, total_samples - processed)
        if processed >= total_samples:
            break
    if processed != total_samples or group_seen != 0:
        raise RuntimeError((processed, total_samples, group_seen))
    return {
        "train_loss": totals["train_loss"] / total_samples,
        "classification_loss": totals["classification_loss"] / total_samples,
        "ce_a": totals["ce_a"] / total_samples,
        "ce_b": totals["ce_b"] / total_samples,
        "relation_loss": totals["relation_loss"] / total_samples,
        "train_acc_a": totals["correct_a"] / total_samples,
        "train_acc_b": totals["correct_b"] / total_samples,
        "optimizer_steps": steps,
        "amp_skipped_steps": skipped_steps,
        "last_grad_norm": float(norm),
    }


def checkpoint_payload(model, config, arm, seed, epoch, train_loss):
    return {
        "arm": arm, "seed": seed, "epoch": epoch,
        "selection_metric": "source_train_loss", "selection_value": train_loss,
        "config": config,
        "task_ssl": model.task_ssl.model.state_dict(),
        "task_projection": model.task_projection.state_dict(),
        "reference_projection": model.reference_projection.state_dict(),
        "cross_attention": model.cross_attention.state_dict(),
        "fusion_norm": model.fusion_norm.state_dict(),
        "detector": model.detector.state_dict(),
    }


def self_test(config, dataset, device, batch_size):
    # One step in each arm, identical seed and first source minibatch.
    for arm in config["arms"]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        set_seed(int(config["seeds"][0]))
        model = DualReferenceDetector(config, arm).to(device)
        optimizer = make_optimizer(model, config)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [float(config["fake_weight"]), float(config["real_weight"])], device=device
        ))
        loader = make_train_loader(dataset, batch_size,
                                   int(config["seeds"][0]), 0, device)
        batch = next(iter(loader))
        waveform = batch["audio"].to(device)
        labels = batch["label"].view(-1).long().to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(config["amp"])):
            a, b, _ea, _eb, relation = model.forward_train(waveform)
        loss = 0.5 * (criterion(a.float(), labels) + criterion(b.float(), labels))
        loss = loss + float(config["lambda_relation"]) * relation
        loss.backward()
        ssl_grad = sum(float(p.grad.float().norm()) for p in model.task_ssl.parameters() if p.grad is not None)
        frozen_grad = sum(p.grad is not None for p in model.original_ssl.parameters())
        if not math.isfinite(float(loss)) or ssl_grad <= 0 or frozen_grad != 0:
            raise RuntimeError((arm, loss, ssl_grad, frozen_grad))
        optimizer.step()
        print("SELF_TEST", arm, "loss", float(loss), "ssl_grad", ssl_grad,
              "frozen_grad_tensors", frozen_grad,
              "peak_gb", torch.cuda.max_memory_allocated() / 2**30, flush=True)
        del model, optimizer, loader
    print("SELF_TEST_OK", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-batch-size", type=int)
    parser.add_argument("--benchmark-steps", type=int, help="Source-only speed test; no files written")
    parser.add_argument("--seeds", nargs="+", type=int, help="Predeclared prefix for staged execution")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["arms"] != ["self_reference", "frozen_reference"]:
        raise ValueError("Unexpected E3 arms")
    seeds = [int(seed) for seed in config["seeds"]]
    if len(set(seeds)) != len(seeds) or (args.seeds and args.seeds != seeds[:len(args.seeds)]):
        raise ValueError("Only a prefix of predeclared seeds may be staged")
    run_seeds = args.seeds or seeds
    if int(config["effective_batch_size"]) % int(config["microbatch_size"]) != 0:
        raise ValueError("effective batch must divide into microbatches")
    device = torch.device("cuda")
    dataset_args = argparse.Namespace(algo=0, exp_name=config["exp_name"])
    dataset = DetectDataset(dataset_args, config["train_data_path"],
                            config["train_label_path"], stage="train")
    if args.self_test:
        self_test(config, dataset, device,
                  args.self_test_batch_size or int(config["microbatch_size"]))
        return
    if args.benchmark_steps:
        for arm in config["arms"]:
            gc.collect()
            torch.cuda.empty_cache()
            set_seed(seeds[0])
            model = DualReferenceDetector(config, arm).to(device)
            optimizer = make_optimizer(model, config)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(
                [float(config["fake_weight"]), float(config["real_weight"])], device=device
            ))
            scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"]), init_scale=1024)
            loader = make_train_loader(dataset, int(config["microbatch_size"]),
                                       seeds[0], int(config["num_workers"]), device)
            started = time.monotonic()
            metrics = train_epoch(model, loader, optimizer, criterion, scaler,
                                  config, device, args.benchmark_steps)
            seconds = time.monotonic() - started
            print("BENCHMARK", arm, "seconds", seconds,
                  "seconds_per_step", seconds / metrics["optimizer_steps"],
                  json.dumps(metrics), flush=True)
            del model, optimizer, criterion, scaler, loader
        return
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": config, "config_sha256": sha256(config_path),
        "trainer_sha256": sha256(__file__),
        "source_labels_sha256": sha256(config["train_label_path"]),
        "torch_version": torch.__version__,
        "input_protocol": "torchaudio.load plus unchanged data.pad; no resampling/augmentation",
        "checkpoint_selection": "minimum source train loss",
        "target_evaluation_during_training": False,
    }
    (output / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    rows = []
    csv_path = output / "epoch_results.csv"
    if csv_path.exists():
        raise FileExistsError(f"Refuse to overwrite existing E3 run: {csv_path}")
    manifest = {}
    for seed in run_seeds:
        for arm in config["arms"]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            set_seed(seed)
            model = DualReferenceDetector(config, arm).to(device)
            optimizer = make_optimizer(model, config)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(
                [float(config["fake_weight"]), float(config["real_weight"])], device=device
            ))
            scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"]), init_scale=1024)
            loader = make_train_loader(dataset, int(config["microbatch_size"]),
                                       seed, int(config["num_workers"]), device)
            run_dir = output / f"{arm}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "best_train_loss.pth"
            best = float("inf")
            print("START", arm, seed, flush=True)
            for epoch in range(1, int(config["epochs"]) + 1):
                started = time.monotonic()
                metrics = train_epoch(model, loader, optimizer, criterion, scaler,
                                      config, device)
                loss = metrics["train_loss"]
                selected = loss < best
                if selected:
                    best = loss
                    save_checkpoint(str(checkpoint), checkpoint_payload(
                        model, config, arm, seed, epoch, loss
                    ))
                row = {
                    "arm": arm, "seed": seed, "epoch": epoch, **metrics,
                    "selected": selected, "best_train_loss_so_far": best,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "seconds": time.monotonic() - started,
                }
                rows.append(row)
                write_csv(str(csv_path), rows)
                print("EPOCH", json.dumps(row), flush=True)
            manifest[f"{arm}_seed{seed}"] = str(checkpoint)
            (output / "checkpoint_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print("FINISH", arm, seed, "checkpoint", checkpoint, flush=True)
            del model, optimizer, criterion, scaler, loader
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
