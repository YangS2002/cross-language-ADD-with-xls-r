import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_metrics_DF as em
from core_scripts.startup_config import set_random_seed
from data import DetectDataset, simple_collate_fn
from src.model import Model, build_optimizer
from tools.utils import parse_addarguments, readconfig


REAL_LABEL = 1
FAKE_LABEL = 0


def cfg_int(config, key, default):
    return int(config.get(key, default))


def cfg_float(config, key, default):
    return float(config.get(key, default))


def cfg_bool(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def train_epoch(loader, model, optimizer, criterion, scaler, device, grad_clip, accumulation_steps, amp):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loop = tqdm(loader, desc="Train")

    for step, batch in enumerate(loop):
        waveform = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].view(-1).long().to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(waveform)
            loss = criterion(logits, labels)
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)

        if should_step:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        loop.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples)
    }


@torch.no_grad()
def evaluate(loader, model, criterion, device, amp, name="Eval"):
    model.eval()
    bona_scores = []
    spoof_scores = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc=name):
        waveform = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].view(-1).long().to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(waveform)
            loss = criterion(logits, labels)

        scores = torch.softmax(logits.float(), dim=1)[:, REAL_LABEL]
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        bona_scores.extend(scores_np[labels_np == REAL_LABEL])
        spoof_scores.extend(scores_np[labels_np == FAKE_LABEL])

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if not bona_scores or not spoof_scores:
        raise RuntimeError("Evaluation requires at least one bona fide and one spoof sample.")

    eer, threshold = em.compute_eer(np.asarray(bona_scores), np.asarray(spoof_scores))
    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples),
        "eer": eer * 100.0,
        "threshold": float(threshold)
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, metrics, config):
    torch.save({
        "epoch": epoch,
        "metrics": metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dict(config)
    }, path)


def main():
    args = parse_addarguments()
    config = readconfig(args.config)
    seed = cfg_int(config, "seed", 1234)
    set_random_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg_bool(config, "amp", True) and device.type == "cuda"
    print(f"Device: {device}, AMP: {amp}")

    train_dataset = DetectDataset(
        args,
        datapath=config["train_data_path"],
        labelspath=config["train_label_path"],
        stage="train"
    )
    dev_dataset = DetectDataset(
        args,
        datapath=config["dev_data_path"],
        labelspath=config["dev_label_path"],
        stage="test"
    )
    test_dataset = DetectDataset(
        args,
        datapath=config["test_data_path"],
        labelspath=config["test_label_path"],
        stage="test"
    )

    num_workers = cfg_int(config, "num_workers", 4)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_int(config, "batch_size", 16),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg_int(config, "eval_batch_size", 16),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg_int(config, "eval_batch_size", 16),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    model = Model(device, config).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([
        cfg_float(config, "fake_weight", 0.1),
        cfg_float(config, "real_weight", 0.9)
    ], device=device))
    optimizer = build_optimizer(
        model,
        ssl_lr=cfg_float(config, "ssl_lr", 5e-7),
        head_lr=cfg_float(config, "head_lr", 1e-4),
        layer_decay=cfg_float(config, "layer_decay", 0.95),
        weight_decay=cfg_float(config, "weight_decay", 1e-4)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    epochs = cfg_int(config, "epochs", 20)
    patience = cfg_int(config, "patience", 3)
    grad_clip = cfg_float(config, "grad_clip", 5.0)
    accumulation_steps = max(1, cfg_int(config, "accumulation_steps", 1))
    save_dir = os.path.join("models", config["exp_name"] + "_xlsr_transition")
    os.makedirs(save_dir, exist_ok=True)

    best_dev_eer = float("inf")
    best_test_eer = float("inf")
    test_eer_at_best_dev = float("nan")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_result = train_epoch(
            train_loader,
            model,
            optimizer,
            criterion,
            scaler,
            device,
            grad_clip,
            accumulation_steps,
            amp
        )
        dev_result = evaluate(dev_loader, model, criterion, device, amp, name="Dev")
        test_result = evaluate(test_loader, model, criterion, device, amp, name="Test")

        dev_improved = dev_result["eer"] < best_dev_eer
        test_improved = test_result["eer"] < best_test_eer
        if dev_improved:
            best_dev_eer = dev_result["eer"]
            test_eer_at_best_dev = test_result["eer"]
            bad_epochs = 0
        else:
            bad_epochs += 1
        if test_improved:
            best_test_eer = test_result["eer"]

        metrics = {
            "dev_eer": dev_result["eer"],
            "test_eer": test_result["eer"],
            "best_dev_eer": best_dev_eer,
            "best_test_eer_oracle": best_test_eer,
            "test_eer_at_best_dev": test_eer_at_best_dev
        }

        save_checkpoint(
            os.path.join(save_dir, "last.pth"),
            model,
            optimizer,
            scaler,
            epoch,
            metrics,
            config
        )
        if dev_improved:
            save_checkpoint(
                os.path.join(save_dir, "best_dev.pth"),
                model,
                optimizer,
                scaler,
                epoch,
                metrics,
                config
            )
        if test_improved:
            save_checkpoint(
                os.path.join(save_dir, "best_test_oracle.pth"),
                model,
                optimizer,
                scaler,
                epoch,
                metrics,
                config
            )

        print(
            f"epoch:{epoch} "
            f"train_loss:{train_result['loss']:.6f} "
            f"train_acc:{train_result['acc'] * 100:.2f}% "
            f"dev_loss:{dev_result['loss']:.6f} "
            f"dev_acc:{dev_result['acc'] * 100:.2f}% "
            f"dev_eer:{dev_result['eer']:.4f}% "
            f"test_loss:{test_result['loss']:.6f} "
            f"test_acc:{test_result['acc'] * 100:.2f}% "
            f"test_eer:{test_result['eer']:.4f}% "
            f"best_dev_eer:{best_dev_eer:.4f}% "
            f"best_test_eer_oracle:{best_test_eer:.4f}% "
            f"time:{time.ctime()}"
        )

        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} epochs without EER improvement.")
            break

    print(f"Training finished. Best dev EER: {best_dev_eer:.4f}%")
    print(f"Test EER at best dev: {test_eer_at_best_dev:.4f}%")
    print(f"Best test EER (oracle, diagnostic only): {best_test_eer:.4f}%")
    print(f"Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()