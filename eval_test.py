import sys
import os
import time
import numpy as np
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm

from tools.utils import parse_addarguments, readconfig
from src.utils import write_exp_detail
from core_scripts.startup_config import set_random_seed
from data import getDataLoader
import eval_metrics_DF as em


def get_sample_ids(batch, batch_size, start_idx):
    for key in ["file_path", "utt_id", "filename", "file_name", "name"]:
        if key in batch:
            values = batch[key]
            if torch.is_tensor(values):
                values = values.cpu().tolist()
            return [str(v) for v in values]

    return [str(start_idx + i) for i in range(batch_size)]


@torch.no_grad()
def evaluate_layers(loader, model, device, save_path):
    model.eval()

    bona_data = [[] for _ in range(24)]
    spoof_data = [[] for _ in range(24)]

    rows = []
    sample_idx = 0

    for batch in tqdm(loader):
        batch_x = batch["audio"].to(device)
        batch_y = batch["label"].view(-1).long().to(device)
        batch_size = batch_x.size(0)

        outputs, _ = model(batch_x)  # [B,24,2]

        labels = batch_y.cpu().numpy()
        sample_ids = get_sample_ids(batch, batch_size, sample_idx)

        scores = outputs[:, :, 1].detach().cpu().numpy()
        preds = outputs.argmax(dim=2).detach().cpu().numpy()  # [B,24]

        for l in range(24):
            bona_data[l].extend(scores[labels == 1, l])
            spoof_data[l].extend(scores[labels == 0, l])

        for i in range(batch_size):
            row = {
                "utt_id": sample_ids[i],
                "label": int(labels[i])
            }

            correct_count = 0

            for l in range(24):
                pred = int(preds[i, l])
                correct = int(pred == labels[i])

                row[f"score_L{l + 1}"] = float(scores[i, l])
                row[f"pred_L{l + 1}"] = pred
                row[f"correct_L{l + 1}"] = correct

                correct_count += correct

            row["correct_count"] = correct_count
            row["oracle_correct"] = int(correct_count > 0)

            rows.append(row)

        sample_idx += batch_size

    print("\n========== Layer-wise EER ==========")

    eers = []

    for l in range(24):
        bona = np.asarray(bona_data[l])
        spoof = np.asarray(spoof_data[l])

        eer = em.compute_eer(bona, spoof)[0] * 100
        eers.append(eer)

        print(f"L{l + 1:02d}: {eer:.4f}%")

    best_layer = int(np.argmin(eers))

    print("------------------------------------")
    print(f"Best layer: L{best_layer + 1}")
    print(f"Best EER: {eers[best_layer]:.4f}%")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)

    correct_count = df["correct_count"].values

    print("\n========== Sample-level Analysis ==========")
    print(f"Samples: {len(df)}")
    print(f"Oracle sample accuracy: {df['oracle_correct'].mean() * 100:.4f}%")
    print(f"Average correct layers: {correct_count.mean():.4f}/24")
    print(f"Median correct layers: {np.median(correct_count):.2f}/24")

    print("\nCorrect layer count distribution:")
    for k in [1, 3, 6, 12, 18, 24]:
        ratio = np.mean(correct_count >= k) * 100
        print(f">= {k:02d} layers correct: {ratio:.4f}%")

    print(f"\nSaved sample results to: {save_path}")

    return eers, df


if __name__ == "__main__":
    args = parse_addarguments()
    config_path = args.config

    test_scores_root_path = "/root/rivermind-data/SLS/scores"
    config = readconfig(config_path)

    set_random_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from src.model_layer_probe import Model

    model = Model(device, config)
    model = nn.DataParallel(model).to(device)

    checkpoint_path = config["checkpoint_path"]
    if not checkpoint_path.endswith(".pth"):
        checkpoint_path = os.path.join(checkpoint_path, "best_model.pth")

    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    exp_dir = os.path.join(test_scores_root_path, config["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    write_exp_detail(config, test_scores_root_path)

    evalloader = getDataLoader(
        args,
        datapath=config["test_data_path"],
        labelspath=config["test_label_path"],
        batchsize=32,
        shuffle=False,
        num_workers=0,
        stage="test",
        drop_last=False
    )

    save_path = os.path.join(exp_dir, "sample_layer_results.csv")

    eers, df = evaluate_layers(
        evalloader,
        model,
        device,
        save_path
    )

    with open(os.path.join(exp_dir, "sample_layer_summary.txt"), "a") as f:
        f.write(f"time: {time.ctime()}\n")
        f.write(f"labels_path: {config['test_label_path']}\n")
        f.write(f"checkpoint: {checkpoint_path}\n\n")

        f.write("Layer-wise EER:\n")
        for i, eer in enumerate(eers):
            f.write(f"L{i + 1:02d}: {eer:.4f}%\n")

        f.write("\nSample-level Analysis:\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Oracle sample accuracy: {df['oracle_correct'].mean() * 100:.4f}%\n")
        f.write(f"Average correct layers: {df['correct_count'].mean():.4f}/24\n")
        f.write(f"Median correct layers: {df['correct_count'].median():.2f}/24\n")

        for k in [1, 3, 6, 12, 18, 24]:
            ratio = (df["correct_count"] >= k).mean() * 100
            f.write(f">= {k:02d} layers correct: {ratio:.4f}%\n")

        f.write("\n")

    print(f"\nExperiment: {config['exp_name']}")