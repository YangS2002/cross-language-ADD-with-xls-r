"""E4 single-seed paired test of reference-relative same-layer adaptation.

New file by design: do not alter the dirty project model or trainer.
"""

import argparse
import gc
import json
import math
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data import DetectDataset
from src.model import DualTransitionConsistencyDetector, XLSRBackbone
from train import make_eval_loader, make_train_loader, save_checkpoint, set_seed, write_csv
from tools.train_dual_xlsr_reference import (
    TrainableXLSR, make_optimizer, raw_token_interface, sha256, train_epoch,
)


class RelativeTrustGate(nn.Module):
    """Bounded interpolation driven only by dimensionless adaptation statistics."""

    def __init__(self, num_layer_tokens):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.layer_bias = nn.Parameter(torch.zeros(num_layer_tokens))

    def forward(self, task, reference, return_diagnostics=False):
        if task.shape != reference.shape or task.ndim != 4:
            raise ValueError((tuple(task.shape), tuple(reference.shape)))
        delta = task - reference
        # Calculate norms and cosine in FP32 even when the SSL forward uses AMP.
        task32, ref32, delta32 = task.float(), reference.float(), delta.float()
        relative_norm = delta32.norm(dim=-1) / ref32.norm(dim=-1).clamp_min(1e-6)
        cosine_distance = 1.0 - F.cosine_similarity(task32, ref32, dim=-1, eps=1e-6)
        statistics = torch.stack(
            (relative_norm.log1p(), cosine_distance.clamp(0.0, 2.0)), dim=-1
        )
        gate = torch.sigmoid(
            self.net(statistics).squeeze(-1)
            + self.layer_bias.view(1, -1, 1)
        )
        fused = reference + gate.to(delta.dtype).unsqueeze(-1) * delta
        if not return_diagnostics:
            return fused
        return fused, {
            "gate_mean": float(gate.detach().mean()),
            "gate_min": float(gate.detach().amin()),
            "gate_max": float(gate.detach().amax()),
            "relative_drift_mean": float(relative_norm.detach().mean()),
        }


class E4Detector(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        if arm not in {"task_only", "reference_gate"}:
            raise ValueError(arm)
        self.arm = arm
        self.task_ssl = TrainableXLSR(config["ssl_checkpoint"])
        # Construct in both arms to keep downstream initialization RNG matched.
        self.original_ssl = XLSRBackbone(config["ssl_checkpoint"])
        if (self.task_ssl.num_layers, self.task_ssl.hidden_dim) != (
            self.original_ssl.num_layers, self.original_ssl.hidden_dim
        ):
            raise RuntimeError("Task and original XLS-R dimensions differ")
        self.gate = RelativeTrustGate(self.task_ssl.num_layers - 1)
        self.detector = DualTransitionConsistencyDetector(
            num_layers=self.task_ssl.num_layers,
            input_dim=self.task_ssl.hidden_dim,
            hidden_dim=int(config["hidden_dim"]),
            embedding_dim=int(config["embedding_dim"]),
            num_heads=int(config["num_heads"]),
            num_encoder_layers=int(config["num_encoder_layers"]),
            feedforward_dim=int(config["feedforward_dim"]),
            dropout=float(config["dropout"]),
        )
        self.detector.build_transitions = types.MethodType(raw_token_interface, self.detector)

    def train(self, mode=True):
        super().train(mode)
        self.task_ssl.eval()  # Gradients remain enabled; disable SSL dropout/masking.
        self.original_ssl.eval()
        return self

    def _states(self, waveform):
        return torch.stack(self.task_ssl.extract_layers(waveform)[1:], dim=1)

    def tokens(self, waveform):
        task = self._states(waveform)
        if self.arm == "task_only":
            return task
        with torch.no_grad():
            reference = torch.stack(self.original_ssl.extract_layers(waveform)[1:], dim=1)
        return self.gate(task, reference)

    def forward_train(self, waveform):
        return self.detector.forward_train(self.tokens(waveform))

    def forward(self, waveform, return_details=False):
        return self.detector(
            self.tokens(waveform), inference_head="ensemble", return_details=return_details
        )

    @torch.no_grad()
    def source_diagnostics(self, waveform):
        task = self._states(waveform)
        reference = torch.stack(self.original_ssl.extract_layers(waveform)[1:], dim=1)
        _fused, values = self.gate(task, reference, return_diagnostics=True)
        return values


def checkpoint_payload(model, config, arm, seed, epoch, train_loss):
    return {
        "arm": arm, "seed": seed, "epoch": epoch,
        "selection_metric": "source_train_loss", "selection_value": train_loss,
        "config": config,
        "task_ssl": model.task_ssl.model.state_dict(),
        "gate": model.gate.state_dict(),
        "detector": model.detector.state_dict(),
    }


def one_batch_check(config, dataset, device):
    initial_outputs = {}
    for arm in config["arms"]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        set_seed(int(config["seed"]))
        model = E4Detector(config, arm).to(device)
        loader = make_eval_loader(dataset, int(config["batch_size"]), 0, device)
        batch = next(iter(loader))
        waveform = batch["audio"].to(device)
        labels = batch["label"].view(-1).long().to(device)
        model.train()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(config["amp"])):
            initial_outputs[arm] = model(waveform).detach().float().cpu()
        optimizer = make_optimizer(model, config)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [float(config["fake_weight"]), float(config["real_weight"])], device=device
        ))
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(config["amp"])):
            a, b, _ea, _eb, relation = model.forward_train(waveform)
        loss = 0.5 * (criterion(a.float(), labels) + criterion(b.float(), labels))
        loss = loss + float(config["lambda_relation"]) * relation
        loss.backward()
        task_grad = sum(float(p.grad.float().norm()) for p in model.task_ssl.parameters() if p.grad is not None)
        gate_grad = sum(float(p.grad.float().norm()) for p in model.gate.parameters() if p.grad is not None)
        original_grads = sum(p.grad is not None for p in model.original_ssl.parameters())
        if not math.isfinite(float(loss)) or task_grad <= 0 or original_grads:
            raise RuntimeError((arm, float(loss), task_grad, original_grads))
        # At initialization task == reference, so dZ/dgate == 0 by design.
        if gate_grad != 0:
            raise RuntimeError((arm, gate_grad))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if arm == "reference_gate":
            with torch.cuda.amp.autocast(enabled=bool(config["amp"])):
                a, b, _ea, _eb, relation = model.forward_train(waveform)
            second_loss = 0.5 * (criterion(a.float(), labels) + criterion(b.float(), labels))
            second_loss = second_loss + float(config["lambda_relation"]) * relation
            second_loss.backward()
            gate_grad = sum(float(p.grad.float().norm()) for p in model.gate.parameters() if p.grad is not None)
            if gate_grad <= 0 or not math.isfinite(gate_grad):
                raise RuntimeError(f"Gate lacks gradient after task SSL moves: {gate_grad}")
        print("SELF_TEST", arm, "loss", float(loss), "task_grad", task_grad,
              "gate_grad", gate_grad, "peak_gb",
              torch.cuda.max_memory_allocated() / 2**30, flush=True)
        del model, optimizer, loader
    difference = (initial_outputs["task_only"] - initial_outputs["reference_gate"]).abs().max().item()
    if difference > 1e-3:
        raise RuntimeError(f"Initial arms differ before training: {difference}")
    print("SELF_TEST_OK initial_max_logit_difference", difference, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark-steps", type=int)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["arms"] != ["task_only", "reference_gate"] or int(config["seed"]) != 20:
        raise ValueError("E4 is predeclared as two arms at seed 20")
    device = torch.device("cuda")
    dataset_args = argparse.Namespace(algo=0, exp_name=config["exp_name"])
    dataset = DetectDataset(dataset_args, config["train_data_path"],
                            config["train_label_path"], stage="train")
    if args.self_test:
        one_batch_check(config, dataset, device)
        return
    if args.benchmark_steps is not None and args.benchmark_steps <= 0:
        raise ValueError("--benchmark-steps must be positive")
    output = Path(config["output_dir"])
    if args.benchmark_steps is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refuse to overwrite existing E4 output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        provenance = {
            "config": config, "config_sha256": sha256(config_path),
            "trainer_sha256": sha256(__file__),
            "source_labels_sha256": sha256(config["train_label_path"]),
            "input_protocol": "torchaudio.load plus unchanged data.pad; no resampling/augmentation",
            "checkpoint_selection": "minimum source train loss",
            "target_evaluation_during_training": False,
            "torch_version": torch.__version__,
        }
        (output / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    rows, manifest = [], {}
    for arm in config["arms"]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        set_seed(int(config["seed"]))
        model = E4Detector(config, arm).to(device)
        optimizer = make_optimizer(model, config)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [float(config["fake_weight"]), float(config["real_weight"])], device=device
        ))
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"]), init_scale=1024)
        loader = make_train_loader(dataset, int(config["batch_size"]),
                                   int(config["seed"]), int(config["num_workers"]), device)
        diagnostic_loader = make_eval_loader(dataset, min(8, int(config["batch_size"])), 0, device)
        diagnostic_batch = next(iter(diagnostic_loader))["audio"].to(device)
        if args.benchmark_steps is not None:
            started = time.monotonic()
            metrics = train_epoch(model, loader, optimizer, criterion, scaler, config,
                                  device, args.benchmark_steps)
            print("BENCHMARK", arm, "seconds_per_step",
                  (time.monotonic() - started) / metrics["optimizer_steps"],
                  json.dumps(metrics), flush=True)
            continue
        run_dir = output / f"{arm}_seed{config['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best_train_loss.pth"
        best = float("inf")
        print("START", arm, config["seed"], flush=True)
        for epoch in range(1, int(config["epochs"]) + 1):
            started = time.monotonic()
            metrics = train_epoch(model, loader, optimizer, criterion, scaler, config, device)
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(config["amp"])):
                metrics.update(model.source_diagnostics(diagnostic_batch))
            loss = metrics["train_loss"]
            selected = loss < best
            if selected:
                best = loss
                save_checkpoint(str(checkpoint), checkpoint_payload(
                    model, config, arm, int(config["seed"]), epoch, loss
                ))
            row = {
                "arm": arm, "seed": int(config["seed"]), "epoch": epoch, **metrics,
                "selected": selected, "best_train_loss_so_far": best,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                "seconds": time.monotonic() - started,
            }
            rows.append(row)
            write_csv(str(output / "epoch_results.csv"), rows)
            print("EPOCH", json.dumps(row), flush=True)
        manifest[f"{arm}_seed{config['seed']}"] = str(checkpoint)
        (output / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print("FINISH", arm, config["seed"], "checkpoint", checkpoint, flush=True)
        del model, optimizer, criterion, scaler, loader, diagnostic_loader, diagnostic_batch
    if args.benchmark_steps is None:
        print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
