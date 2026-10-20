"""Full CH/DE/UK/ES/ITW evaluation, preserving existing pad and EER protocol."""
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import evaluate_four_domains_shared_xlsr as shared
from tools.relational_preservation_model import RelationalDetector


def make_domains(config):
    result = {}
    for name, entry in config["domains"].items():
        shared.DOMAINS[name] = (Path(entry["label"]), Path(entry["root"]) if entry.get("root") else None,
                                Path(entry["archive"]) if entry.get("archive") else None,
                                entry.get("prefix", ""))
        result[name] = shared.AudioDomain(name)
    return result


def evaluate(config, arm, milestone, datasets):
    out = Path(config["output_dir"])
    result_path = out / f"eval_epoch{milestone}_{arm}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    checkpoint = out / arm / "best_train_loss.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RelationalDetector(config, arm, training_reference=False).cuda().eval()
    model.load_payload(saved)
    selected_epoch = saved["epoch"]
    del saved
    rows = []
    for name, dataset in datasets.items():
        start = time.perf_counter()
        scores = {head: [] for head in ["a", "b", "ensemble"]}
        labels = []
        loader = DataLoader(dataset, batch_size=int(config["eval_batch_size"]),
                            num_workers=int(config["num_workers"]), shuffle=False,
                            collate_fn=shared.collate, pin_memory=True)
        with torch.no_grad():
            for waveform, targets, _names in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    details = model.predict(waveform.cuda(non_blocking=True))
                a, b = details["logits_a"].float(), details["logits_b"].float()
                for head, logits in [("a", a), ("b", b), ("ensemble", (a + b) * 0.5)]:
                    scores[head].extend(logits.softmax(dim=-1)[:, 1].cpu().numpy().tolist())
                labels.extend(targets.tolist())
        y = np.asarray(labels)
        elapsed = time.perf_counter() - start
        for head in scores:
            value, threshold = shared.eer(y, scores[head])
            row = {"arm": arm, "seed": config["seed"], "milestone": milestone,
                   "checkpoint_epoch": selected_epoch, "domain": name, "head": head,
                   "eer": value, "threshold": threshold, "samples": len(y), "seconds_domain": elapsed}
            rows.append(row)
            print("EVAL", json.dumps(row), flush=True)
        del loader
    from tools.train_relational_preservation import atomic_json, append_csv
    atomic_json(result_path, rows)
    for row in rows:
        append_csv(out / "evaluation_results.csv", row)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
