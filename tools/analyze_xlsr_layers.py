import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import DetectDataset, simple_collate_fn
from src.model import XLSRBackbone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze frozen XLS-R layers without training the detector."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def sample_indices(dataset, samples_per_class, seed):
    by_class = {0: [], 1: []}
    for index, synthesizer_name in enumerate(dataset.synthesizerlabels):
        synthesizer_id = dataset.synthesizers[synthesizer_name]
        label = 1 if synthesizer_id == 0 else 0
        by_class[label].append(index)

    generator = random.Random(seed)
    selected = []
    for label in (0, 1):
        indices = by_class[label]
        generator.shuffle(indices)
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {label} has only {len(indices)} samples, "
                f"requested {samples_per_class}"
            )
        selected.extend(indices[:samples_per_class])

    generator.shuffle(selected)
    return selected


def utterance_statistics(layer):
    mean = layer.float().mean(dim=1)
    std = layer.float().var(dim=1, unbiased=False).add(1e-6).sqrt()
    return torch.cat([mean, std], dim=-1)


def centered_gram(features):
    features = features.astype(np.float64, copy=False)
    features = features - features.mean(axis=0, keepdims=True)
    gram = features @ features.T
    gram = gram - gram.mean(axis=0, keepdims=True)
    gram = gram - gram.mean(axis=1, keepdims=True)
    gram = gram + gram.mean()
    return gram


def linear_cka(gram_a, gram_b):
    numerator = np.sum(gram_a * gram_b)
    denominator = np.sqrt(np.sum(gram_a * gram_a) * np.sum(gram_b * gram_b))
    return float(numerator / max(denominator, 1e-12))


def compute_eer(labels, scores):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = np.nanargmin(np.abs(false_positive_rate - false_negative_rate))
    return 100.0 * float(
        0.5 * (false_positive_rate[index] + false_negative_rate[index])
    )


def repeated_probe(features, labels, seeds, multiclass=False):
    rows = []
    for seed in seeds:
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=seed,
            stratify=labels,
        )
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)
        row = {
            "balanced_accuracy": 100.0
            * balanced_accuracy_score(test_y, predictions),
        }
        if not multiclass:
            row["eer"] = compute_eer(test_y, classifier.decision_function(test_x))
        rows.append(row)

    result = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1))
    return result


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    args.algo = int(config.get("algo", 5))
    args.exp_name = "xlsr_layer_analysis"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DetectDataset(
        args,
        datapath=config["train_data_path"],
        labelspath=config["train_label_path"],
        stage="eval",
    )
    selected_indices = sample_indices(dataset, args.samples_per_class, args.seed)
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    backbone = XLSRBackbone(config["ssl_checkpoint"]).to(device)
    backbone.eval()

    layer_features = [[] for _ in range(backbone.num_layers)]
    binary_labels = []
    attack_labels = []

    for batch in tqdm(loader, desc="Extract frozen XLS-R layers"):
        waveform = batch["audio"].to(device, non_blocking=True)
        layers = backbone.extract_layers(waveform)
        for layer_index, layer in enumerate(layers):
            layer_features[layer_index].append(
                utterance_statistics(layer).cpu().numpy()
            )
        binary_labels.append(batch["label"].view(-1).numpy())
        attack_labels.append(batch["vocoder_label"].view(-1).numpy())

    layer_features = [np.concatenate(parts, axis=0) for parts in layer_features]
    binary_labels = np.concatenate(binary_labels, axis=0)
    attack_labels = np.concatenate(attack_labels, axis=0)

    grams = [centered_gram(features) for features in layer_features]
    cka_matrix = np.zeros((backbone.num_layers, backbone.num_layers), dtype=np.float64)
    for row in range(backbone.num_layers):
        for column in range(row, backbone.num_layers):
            value = linear_cka(grams[row], grams[column])
            cka_matrix[row, column] = value
            cka_matrix[column, row] = value

    np.savetxt(
        os.path.join(args.output_dir, "layer_cka_matrix.csv"),
        cka_matrix,
        delimiter=",",
        fmt="%.8f",
    )

    probe_seeds = [args.seed, args.seed + 1, args.seed + 2]
    spoof_mask = binary_labels == 0
    rows = []
    for layer_index, features in enumerate(layer_features):
        binary_probe = repeated_probe(features, binary_labels, probe_seeds)
        attack_probe = repeated_probe(
            features[spoof_mask],
            attack_labels[spoof_mask],
            probe_seeds,
            multiclass=True,
        )
        adjacent_cka = (
            float("nan")
            if layer_index == 0
            else float(cka_matrix[layer_index - 1, layer_index])
        )
        rows.append({
            "layer": layer_index + 1,
            "adjacent_cka": adjacent_cka,
            "binary_balanced_accuracy_mean": binary_probe[
                "balanced_accuracy_mean"
            ],
            "binary_balanced_accuracy_std": binary_probe[
                "balanced_accuracy_std"
            ],
            "binary_eer_mean": binary_probe["eer_mean"],
            "binary_eer_std": binary_probe["eer_std"],
            "attack_balanced_accuracy_mean": attack_probe[
                "balanced_accuracy_mean"
            ],
            "attack_balanced_accuracy_std": attack_probe[
                "balanced_accuracy_std"
            ],
        })

    write_csv(os.path.join(args.output_dir, "layer_probe_results.csv"), rows)

    candidate_rows = [row for row in rows[2:-2] if np.isfinite(row["adjacent_cka"])]
    split_row = min(candidate_rows, key=lambda row: row["adjacent_cka"])
    summary = {
        "config": os.path.abspath(args.config),
        "seed": args.seed,
        "samples": int(binary_labels.size),
        "samples_per_class": args.samples_per_class,
        "num_layers": backbone.num_layers,
        "feature": "temporal mean + standard deviation per frozen XLS-R layer",
        "recommended_split_after_layer": int(split_row["layer"] - 1),
        "minimum_adjacent_cka": float(split_row["adjacent_cka"]),
        "selection_rule": (
            "largest adjacent representation change, excluding the first and "
            "last two layers; training data only"
        ),
    }
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
