"""Paired source-only last-layer MLP comparison of original Q/K and Q/K+relation."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from train import set_seed
from tools import train_sls_pretrained_attention as base
from tools import train_sls_qk_relation as relation
from tools.qk_last_mlp_model import QKLastMLP, QKRelationLastMLP
from tools.train_relational_preservation import (
    atomic_json, atomic_save, append_csv, dataset_for, digest, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains


_plain_step = base.step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if (config["seed"] != 20 or config["warmup_epochs"] != 3
            or config["epochs"] != 8 or config["milestones"] != [4, 8]):
        raise ValueError("Only fixed paired seed20 MLP protocol")
    root = Path(config["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent output required")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Existing MLP experiment; no overwrite")
    root.mkdir(parents=True)
    original_sha = digest(config["ssl_checkpoint"])
    if original_sha != config["expected_pretrained_sha256"]:
        raise ValueError("Original checkpoint differs")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 required")
    signature = dict(config=digest(args.config), trainer=digest(__file__),
                     model=digest("tools/qk_last_mlp_model.py"),
                     relation_model=digest("tools/sls_qk_relation_model.py"),
                     base_model=digest("tools/sls_pretrained_attention_model.py"),
                     pretrained=original_sha, labels=digest(config["train_label_path"]))
    atomic_json(root/"experiment.json", dict(config=config, signature=signature,
                source_only_training=True, same_mlp_warmup_and_rng_for_both_arms=True,
                checkpoint_selection="minimum_complete_source_classification_loss"))
    dataset, _ = dataset_for(config)
    domains = make_domains(config)
    atomic_json(root/"inventory.json", [domain.validate_paths() for domain in domains.values()])

    # Native XLS-R is identical to original Q/K before joint fine-tuning.
    # Train one MLP-only warmup; both formal arms use exactly this head and RNG.
    set_seed(config["seed"])
    warm_model = QKLastMLP(config["ssl_checkpoint"]).cuda()
    warm_model.guidance_enabled = False
    warm_model.task.requires_grad_(False)
    warm_optimizer = base.make_optimizer(warm_model, config)
    deadline = time.monotonic() + config["max_run_seconds"]
    for epoch in range(1, 4):
        row = base.train_epoch(warm_model, warm_optimizer, dataset, config, epoch, deadline)
        append_csv(root/"warmup_epoch_results.csv", dict(epoch=epoch, **row))
        print("WARMUP", json.dumps(dict(epoch=epoch, **row)), flush=True)
    warm = dict(backend={name: tensor.detach().cpu().clone()
                         for name, tensor in warm_model.backend.state_dict().items()},
                rng=rng_state())
    del warm_model, warm_optimizer
    torch.cuda.empty_cache()
    atomic_json(root/"status.json", dict(status="warmup_complete", seed=config["seed"]))

    for arm, model_class in (("original_qk", QKLastMLP),
                             ("qk_relation", QKRelationLastMLP)):
        arm_root = root/arm
        arm_root.mkdir()
        (arm_root/"mlp").mkdir()
        arm_config = dict(config, output_dir=str(arm_root))
        set_seed(config["seed"])
        model = model_class(config["ssl_checkpoint"]).cuda()
        model.backend.load_state_dict(warm["backend"])
        restore_rng(warm["rng"])
        optimizer = base.make_optimizer(model, config)
        relation.RELATION_WEIGHT = float(config["relation_weight"])
        base.step = _plain_step if arm == "original_qk" else relation.step
        best = float("inf")
        for epoch in range(4, 9):
            atomic_json(root/"status.json", dict(status="training", arm=arm, epoch=epoch))
            if arm == "qk_relation":
                model.relation_history = []
            metrics = base.train_epoch(model, optimizer, dataset, arm_config, epoch, deadline)
            if arm == "qk_relation":
                metrics["relation_loss_mean"] = (sum(model.relation_history)
                                                  / len(model.relation_history))
                del model.relation_history
            selected = metrics["train_loss"] < best
            if selected:
                best = metrics["train_loss"]
                atomic_save(arm_root/"mlp"/"best.pth",
                            dict(**model.trainable_payload(original_sha), epoch=epoch,
                                 loss=best, seed=config["seed"], arm=arm,
                                 signature=signature))
            append_csv(arm_root/"epoch_results.csv", dict(arm=arm, epoch=epoch,
                       selected=selected, **metrics))
            print("EPOCH", json.dumps(dict(arm=arm, epoch=epoch, selected=selected,
                                            **metrics)), flush=True)
            if epoch in (4, 8):
                current = base.cpu_payload(model, original_sha)
                random_state = rng_state()
                atomic_json(root/"status.json", dict(status="evaluating", arm=arm,
                                                        epoch=epoch))
                rows = base.evaluate(model, arm_config, domains, "mlp", epoch, original_sha)
                values = {row["domain"]: row["eer"] for row in rows}
                summary = dict(arm=arm, epoch=epoch, seed=config["seed"],
                               per_domain=values,
                               cross_mean=sum(values[name] for name in base.CROSS)/4,
                               worst_cross=max(values[name] for name in base.CROSS),
                               checkpoint_selection="source_classification_loss")
                atomic_json(arm_root/f"summary_epoch{epoch}.json", summary)
                print("SUMMARY", json.dumps(summary), flush=True)
                model.load_trainable_payload(current, original_sha)
                restore_rng(random_state)
                del current
        del model, optimizer
        torch.cuda.empty_cache()
    atomic_json(root/"status.json", dict(status="complete", seed=config["seed"]))
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
