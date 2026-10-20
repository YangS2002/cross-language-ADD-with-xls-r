"""Matched fixed-seed original-Q/K training with native-relation preservation."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from tools import train_sls_pretrained_attention as base
from tools.sls_qk_relation_model import QKRelationSLS
from tools.train_relational_preservation import digest


_original_signature = base.signature
_original_train_epoch = base.train_epoch


def signature(config, config_path):
    result = _original_signature(config, config_path)
    result["relation_model"] = digest("tools/sls_qk_relation_model.py")
    result["relation_trainer"] = digest(__file__)
    result["relation_weight"] = config["relation_weight"]
    return result


def step(model, optimizer, wave, labels):
    optimizer.zero_grad(set_to_none=True)
    model.collect_relation_loss = True
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(wave)
        class_weights = output.new_tensor([0.1, 0.9])
        classification = torch.nn.functional.nll_loss(output, labels, weight=class_weights)
        relation = model.relation_loss
        if relation is None or not bool(torch.isfinite(relation)):
            raise FloatingPointError("Missing/nonfinite relation loss")
        if not bool(torch.isfinite(classification)):
            raise FloatingPointError("Nonfinite source NLL")
        (classification + RELATION_WEIGHT * relation).backward()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
        optimizer.step()
        model.last_relation_loss = float(relation.detach())
        if hasattr(model, "relation_history"):
            model.relation_history.append(model.last_relation_loss)
        return classification.detach(), output.detach(), float(norm)
    finally:
        model.collect_relation_loss = False
        model.relation_loss = None


def train_epoch(model, optimizer, dataset, config, epoch, deadline):
    model.relation_history = []
    metrics = _original_train_epoch(model, optimizer, dataset, config, epoch, deadline)
    metrics["relation_loss_mean"] = sum(model.relation_history) / len(model.relation_history)
    metrics["relation_loss_last"] = model.relation_history[-1]
    del model.relation_history
    return metrics


def decide(rows, config, epoch):
    values = {row["domain"]: row["eer"] for row in rows}
    if len(rows) != 5 or set(values) != set(base.PAPER_SLS):
        raise RuntimeError("Five-domain evaluation incomplete")
    cross = [values[name] for name in base.CROSS]
    return dict(epoch=epoch, seed=config["seed"], stop=False,
                reason="fixed_epoch8_complete" if epoch == 8 else "continue_fixed_training",
                per_domain=values, cross_mean=sum(cross)/len(cross),
                worst_cross=max(cross), ch_de_joint=(values["ch"]+values["de"])/2,
                relation_weight=config["relation_weight"],
                checkpoint_selected_by="minimum_complete_source_classification_loss",
                target_checkpoint_or_seed_selection=False,
                eligible_for_fixed_multiseed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config["seed"] not in (20, 1001, 7):
        raise ValueError("Only historical fixed-seed comparison")
    if config["warmup_epochs"] != 3 or config["epochs"] != 8 or config["milestones"] != [4, 8]:
        raise ValueError("Historical training schedule required")
    global RELATION_WEIGHT
    RELATION_WEIGHT = float(config["relation_weight"])
    if RELATION_WEIGHT <= 0:
        raise ValueError("Positive relation weight required")
    base.ARM = "qk_relation"
    base.PretrainedAttentionSLS = QKRelationSLS
    base.signature = signature
    base.step = step
    base.train_epoch = train_epoch
    base.decide = decide
    base.run(config, args.config)


if __name__ == "__main__":
    main()
