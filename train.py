import csv
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_metrics_DF as em
from data import DetectDataset, simple_collate_fn
from src.model import Model, build_optimizer
from tools.utils import parse_addarguments


REAL_LABEL = 1
FAKE_LABEL = 0


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_loader(dataset, batch_size, seed, num_workers, device):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def make_eval_loader(dataset, batch_size, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def compute_eer(real_scores, fake_scores):
    if len(real_scores) == 0 or len(fake_scores) == 0:
        return float("nan"), float("nan")

    eer, threshold = em.compute_eer(
        np.asarray(real_scores, dtype=np.float64),
        np.asarray(fake_scores, dtype=np.float64),
    )

    return 100.0 * float(eer), float(threshold)


def write_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    amp,
    lambda_relation,
    grad_clip,
):
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_ce_a = 0.0
    total_ce_b = 0.0
    total_relation = 0.0

    total_correct_a = 0
    total_correct_b = 0
    total_samples = 0

    loop = tqdm(loader, desc="Train")

    for batch in loop:
        waveform = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].view(-1).long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits_a, logits_b, embedding_a, embedding_b, relation_loss = (
                model.forward_train(waveform)
            )

        ce_a = criterion(logits_a.float(), labels)
        ce_b = criterion(logits_b.float(), labels)

        classification_loss = 0.5 * (ce_a + ce_b)
        loss = classification_loss + lambda_relation * relation_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.detector.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.detach().item() * batch_size
        total_cls += classification_loss.detach().item() * batch_size
        total_ce_a += ce_a.detach().item() * batch_size
        total_ce_b += ce_b.detach().item() * batch_size
        total_relation += relation_loss.detach().item() * batch_size

        total_correct_a += (
            logits_a.detach().argmax(dim=1) == labels
        ).sum().item()

        total_correct_b += (
            logits_b.detach().argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{classification_loss.item():.4f}",
            rel=f"{relation_loss.item():.4f}",
            acc_a=f"{100.0 * total_correct_a / total_samples:.2f}%",
            acc_b=f"{100.0 * total_correct_b / total_samples:.2f}%",
        )

    return {
        "train_loss": total_loss / total_samples,
        "train_cls_loss": total_cls / total_samples,
        "train_ce_a": total_ce_a / total_samples,
        "train_ce_b": total_ce_b / total_samples,
        "train_relation_loss": total_relation / total_samples,
        "train_acc_a": 100.0 * total_correct_a / total_samples,
        "train_acc_b": 100.0 * total_correct_b / total_samples,
    }


@torch.no_grad()
def evaluate_head(
    model,
    loader,
    criterion,
    device,
    amp,
    desc,
    inference_head,
):
    model.eval()

    old_head = model.inference_head
    model.inference_head = inference_head

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    real_scores = []
    fake_scores = []

    try:
        for batch in tqdm(loader, desc=desc, leave=False):
            waveform = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].view(-1).long().to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(waveform)

            logits = logits.float()
            loss = criterion(logits, labels)

            scores = torch.softmax(logits, dim=1)[:, REAL_LABEL]

            labels_np = labels.cpu().numpy()
            scores_np = scores.cpu().numpy()

            real_scores.extend(scores_np[labels_np == REAL_LABEL].tolist())
            fake_scores.extend(scores_np[labels_np == FAKE_LABEL].tolist())

            total_loss += loss.item() * labels.size(0)

            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total_samples += labels.size(0)

    finally:
        model.inference_head = old_head

    eer, threshold = compute_eer(real_scores, fake_scores)

    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
        "eer": eer,
        "threshold": threshold,
        "samples": total_samples,
    }


def build_domain_metrics(
    model,
    loader,
    criterion,
    device,
    amp,
    domain,
):
    head_a = evaluate_head(
        model,
        loader,
        criterion,
        device,
        amp,
        f"{domain} Head-A",
        "a",
    )

    head_b = evaluate_head(
        model,
        loader,
        criterion,
        device,
        amp,
        f"{domain} Head-B",
        "b",
    )

    ensemble = evaluate_head(
        model,
        loader,
        criterion,
        device,
        amp,
        f"{domain} Ensemble",
        "ensemble",
    )

    return {
        "a": head_a,
        "b": head_b,
        "ensemble": ensemble,
    }


def summarize_epoch(rows):
    summary = []

    epochs = sorted(set(row["epoch"] for row in rows))

    metrics = (
        "dev_a_eer",
        "test_a_eer",
        "dev_b_eer",
        "test_b_eer",
        "dev_ensemble_eer",
        "test_ensemble_eer",
        "a_joint_eer",
        "a_worst_domain_eer",
        "a_domain_gap",
    )

    for epoch in epochs:
        current = [row for row in rows if row["epoch"] == epoch]

        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in current],
                dtype=np.float64,
            )

            summary.append({
                "epoch": epoch,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            })

    return summary


def summarize_seed(rows):
    summary = []

    seeds = sorted(set(row["seed"] for row in rows))

    for seed in seeds:
        current = sorted(
            [row for row in rows if row["seed"] == seed],
            key=lambda x: x["epoch"],
        )

        joint = np.asarray(
            [row["a_joint_eer"] for row in current],
            dtype=np.float64,
        )

        worst = np.asarray(
            [row["a_worst_domain_eer"] for row in current],
            dtype=np.float64,
        )

        gap = np.asarray(
            [row["a_domain_gap"] for row in current],
            dtype=np.float64,
        )

        best_joint_idx = int(np.argmin(joint))
        best_worst_idx = int(np.argmin(worst))
        best_gap_idx = int(np.argmin(gap))

        summary.append({
            "seed": seed,

            "best_joint_epoch": current[best_joint_idx]["epoch"],
            "best_joint_eer": float(joint[best_joint_idx]),
            "dev_at_best_joint": current[best_joint_idx]["dev_a_eer"],
            "test_at_best_joint": current[best_joint_idx]["test_a_eer"],

            "best_worst_epoch": current[best_worst_idx]["epoch"],
            "best_worst_domain_eer": float(worst[best_worst_idx]),
            "dev_at_best_worst": current[best_worst_idx]["dev_a_eer"],
            "test_at_best_worst": current[best_worst_idx]["test_a_eer"],

            "smallest_gap_epoch": current[best_gap_idx]["epoch"],
            "smallest_domain_gap": float(gap[best_gap_idx]),
            "dev_at_smallest_gap": current[best_gap_idx]["dev_a_eer"],
            "test_at_smallest_gap": current[best_gap_idx]["test_a_eer"],
        })

    return summary


def main():
    args = parse_addarguments()
    config = load_yaml(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp = bool(config.get("amp", True)) and device.type == "cuda"

    batch_size = int(config.get("batch_size", 16))
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    num_workers = int(config.get("num_workers", 4))

    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    grad_clip = float(config.get("grad_clip", 5.0))
    lambda_relation = float(config.get("lambda_relation", 0.1))

    epochs = int(config.get("epochs", 3))
    seeds = [int(seed) for seed in config.get("seeds", [1000, 1001, 1003])]

    output_dir = config.get(
        "output_dir",
        os.path.join(
            "models",
            config["exp_name"] + "_dual_transition_consistency",
        ),
    )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print("Dual Transition Relation-Consistency Validation")
    print("Train: ASVspoof2019-LA")
    print("Dev:   DECRO-CH")
    print("Test:  MLAAD-DE")
    print(f"Epochs: {epochs}")
    print(f"Seeds: {seeds}")
    print(f"lambda_relation: {lambda_relation}")
    print(f"Output: {output_dir}")
    print("=" * 100)

    train_dataset = DetectDataset(
        args,
        datapath=config["train_data_path"],
        labelspath=config["train_label_path"],
        stage="train",
    )

    dev_dataset = DetectDataset(
        args,
        datapath=config["dev_data_path"],
        labelspath=config["dev_label_path"],
        stage="test",
    )

    test_dataset = DetectDataset(
        args,
        datapath=config["test_data_path"],
        labelspath=config["test_label_path"],
        stage="test",
    )

    dev_loader = make_eval_loader(
        dev_dataset,
        eval_batch_size,
        num_workers,
        device,
    )

    test_loader = make_eval_loader(
        test_dataset,
        eval_batch_size,
        num_workers,
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [
                float(config.get("fake_weight", 0.1)),
                float(config.get("real_weight", 0.9)),
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    all_rows = []

    for seed_index, seed in enumerate(seeds, start=1):
        print()
        print("#" * 100)
        print(f"Seed [{seed_index}/{len(seeds)}]: {seed}")
        print("#" * 100)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        set_seed(seed)

        train_loader = make_train_loader(
            train_dataset,
            batch_size,
            seed,
            num_workers,
            device,
        )

        model = Model(device, config).to(device)

        optimizer = build_optimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        for epoch in range(1, epochs + 1):
            print()
            print("=" * 100)
            print(f"seed={seed} | epoch={epoch}/{epochs}")
            print("=" * 100)

            start_time = time.time()

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                amp,
                lambda_relation,
                grad_clip,
            )

            dev = build_domain_metrics(
                model,
                dev_loader,
                criterion,
                device,
                amp,
                "DECRO-CH",
            )

            test = build_domain_metrics(
                model,
                test_loader,
                criterion,
                device,
                amp,
                "MLAAD-DE",
            )

            a_joint_eer = (dev["a"]["eer"] + test["a"]["eer"]) / 2.0
            a_worst_domain_eer = max(dev["a"]["eer"], test["a"]["eer"])
            a_domain_gap = abs(dev["a"]["eer"] - test["a"]["eer"])

            b_joint_eer = (dev["b"]["eer"] + test["b"]["eer"]) / 2.0

            ensemble_joint_eer = (
                dev["ensemble"]["eer"] + test["ensemble"]["eer"]
            ) / 2.0

            row = {
                "seed": seed,
                "epoch": epoch,
                **train_metrics,

                "dev_a_eer": dev["a"]["eer"],
                "dev_a_acc": dev["a"]["acc"],
                "dev_a_loss": dev["a"]["loss"],

                "test_a_eer": test["a"]["eer"],
                "test_a_acc": test["a"]["acc"],
                "test_a_loss": test["a"]["loss"],

                "dev_b_eer": dev["b"]["eer"],
                "dev_b_acc": dev["b"]["acc"],
                "dev_b_loss": dev["b"]["loss"],

                "test_b_eer": test["b"]["eer"],
                "test_b_acc": test["b"]["acc"],
                "test_b_loss": test["b"]["loss"],

                "dev_ensemble_eer": dev["ensemble"]["eer"],
                "dev_ensemble_acc": dev["ensemble"]["acc"],

                "test_ensemble_eer": test["ensemble"]["eer"],
                "test_ensemble_acc": test["ensemble"]["acc"],

                "a_joint_eer": a_joint_eer,
                "a_worst_domain_eer": a_worst_domain_eer,
                "a_domain_gap": a_domain_gap,

                "b_joint_eer": b_joint_eer,
                "ensemble_joint_eer": ensemble_joint_eer,

                "seconds": time.time() - start_time,
            }

            all_rows.append(row)

            print(
                f"Head A | CH={dev['a']['eer']:.4f}% | "
                f"DE={test['a']['eer']:.4f}% | "
                f"joint={a_joint_eer:.4f}% | "
                f"worst={a_worst_domain_eer:.4f}% | "
                f"gap={a_domain_gap:.4f}%"
            )

            print(
                f"Head B | CH={dev['b']['eer']:.4f}% | "
                f"DE={test['b']['eer']:.4f}% | "
                f"joint={b_joint_eer:.4f}%"
            )

            print(
                f"Ensemble | CH={dev['ensemble']['eer']:.4f}% | "
                f"DE={test['ensemble']['eer']:.4f}% | "
                f"joint={ensemble_joint_eer:.4f}%"
            )

            print(
                f"Train | loss={train_metrics['train_loss']:.6f} | "
                f"CE-A={train_metrics['train_ce_a']:.6f} | "
                f"CE-B={train_metrics['train_ce_b']:.6f} | "
                f"relation={train_metrics['train_relation_loss']:.6f}"
            )

            write_csv(
                os.path.join(output_dir, "epoch_results.csv"),
                all_rows,
            )

            if bool(config.get("save_each_epoch", False)):
                checkpoint_dir = os.path.join(output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "detector": model.detector.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": row,
                        "config": config,
                    },
                    os.path.join(
                        checkpoint_dir,
                        f"seed_{seed}_epoch_{epoch}.pth",
                    ),
                )

        del model
        del optimizer
        del scaler
        del train_loader

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    epoch_summary = summarize_epoch(all_rows)
    seed_summary = summarize_seed(all_rows)

    write_csv(
        os.path.join(output_dir, "epoch_summary.csv"),
        epoch_summary,
    )

    write_csv(
        os.path.join(output_dir, "seed_summary.csv"),
        seed_summary,
    )

    with open(
        os.path.join(output_dir, "experiment.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "train": "ASVspoof2019-LA",
                "dev": "DECRO-CH",
                "test": "MLAAD-DE",
                "epochs": epochs,
                "seeds": seeds,
                "lambda_relation": lambda_relation,
                "primary_inference": "Head A",
                "auxiliary_inference": ["Head B", "Ensemble"],
                "model": "Dual Transition Heads + Embedding Relation Consistency",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("FINAL HEAD-A SUMMARY")
    print("=" * 100)

    for epoch in range(1, epochs + 1):
        current = [row for row in all_rows if row["epoch"] == epoch]

        ch = np.asarray([row["dev_a_eer"] for row in current], dtype=np.float64)
        de = np.asarray([row["test_a_eer"] for row in current], dtype=np.float64)
        joint = np.asarray([row["a_joint_eer"] for row in current], dtype=np.float64)
        worst = np.asarray(
            [row["a_worst_domain_eer"] for row in current],
            dtype=np.float64,
        )
        gap = np.asarray([row["a_domain_gap"] for row in current], dtype=np.float64)

        ch_std = ch.std(ddof=1) if len(ch) > 1 else 0.0
        de_std = de.std(ddof=1) if len(de) > 1 else 0.0
        joint_std = joint.std(ddof=1) if len(joint) > 1 else 0.0
        worst_std = worst.std(ddof=1) if len(worst) > 1 else 0.0
        gap_std = gap.std(ddof=1) if len(gap) > 1 else 0.0

        print(
            f"epoch={epoch:02d} | "
            f"CH={ch.mean():.4f}±{ch_std:.4f} | "
            f"DE={de.mean():.4f}±{de_std:.4f} | "
            f"joint={joint.mean():.4f}±{joint_std:.4f} | "
            f"worst={worst.mean():.4f}±{worst_std:.4f} | "
            f"gap={gap.mean():.4f}±{gap_std:.4f}"
        )

    print()
    print("=" * 100)
    print("PER-SEED BEST HEAD-A RESULTS")
    print("=" * 100)

    for row in seed_summary:
        print(
            f"seed={row['seed']} | "
            f"best joint: epoch {row['best_joint_epoch']} "
            f"= {row['best_joint_eer']:.4f} "
            f"(CH={row['dev_at_best_joint']:.4f}, "
            f"DE={row['test_at_best_joint']:.4f}) | "
            f"best worst: epoch {row['best_worst_epoch']} "
            f"= {row['best_worst_domain_eer']:.4f} | "
            f"smallest gap: epoch {row['smallest_gap_epoch']} "
            f"= {row['smallest_domain_gap']:.4f}"
        )


if __name__ == "__main__":
    main()