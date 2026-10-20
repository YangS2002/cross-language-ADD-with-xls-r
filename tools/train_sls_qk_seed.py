"""Independent source warmup followed by the unchanged original-Q/K trainer."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from train import set_seed
from tools import train_sls_pretrained_attention as attention
from tools import train_sls_conditional_residual as warmup_trainer
from tools.sls_conditional_residual_model import ConditionalResidualSLS
from tools.train_relational_preservation import (
    atomic_json, atomic_save, append_csv, dataset_for, digest, rng_state)


def verify_historical_code(c):
    historical = json.loads(Path(c["historical_attention_experiment"]).read_text())
    old = historical["signature"]
    checks = {
        "tools/train_sls_pretrained_attention.py": old["trainer"],
        "tools/sls_pretrained_attention_model.py": old["model"],
        **old["dependencies"],
        "tools/train_sls_conditional_residual.py": c["expected_warmup_trainer_sha256"],
        "tools/sls_conditional_residual_model.py": c["expected_warmup_model_sha256"],
        c["train_label_path"]: old["labels"],
    }
    actual = {name: digest(name) for name in checks}
    mismatch = [name for name in checks if actual[name] != checks[name]]
    if mismatch:
        raise RuntimeError(f"Original-Q/K protocol changed: {mismatch}")
    if digest(c["ssl_checkpoint"]) != c["expected_pretrained_sha256"]:
        raise RuntimeError("Original XLS-R SHA mismatch")
    return actual


def fixed_decision(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    if len(rows) != 5 or set(values) != set(attention.PAPER_SLS):
        raise RuntimeError("Incomplete five-domain stage")
    cross = [values[k] for k in attention.CROSS]
    return dict(
        epoch=epoch, seed=c["seed"], stop=False,
        reason="fixed_epoch8_complete" if epoch == c["epochs"] else "continue_fixed_training",
        per_domain=values, cross_mean=sum(cross)/len(cross), worst_cross=max(cross),
        ch_de_joint=(values["ch"]+values["de"])/2,
        paper_sls_gaps={k: values[k]-attention.PAPER_SLS[k] for k in values},
        paper_mamba_gaps={k: values[k]-attention.PAPER_MAMBA[k] for k in values},
        fixed_seed_set=[20, 1001, 7], independent_source_warmup=True,
        target_checkpoint_or_seed_selection=False, eligible_for_fixed_multiseed=False,
        further_seeds_require_user_authorization=True,
        single_seed_is_not_stability_evidence=True,
        no_matched_sls_or_query_runs_for_this_seed=True)


def run(config_path):
    c = yaml.safe_load(Path(config_path).read_text())
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent outputs only")
    if c["seed"] not in (1001, 7) or c["warmup_epochs"] != 3 or c["epochs"] != 8:
        raise ValueError("Only the registered fixed seeds and epoch protocol")
    if c["milestones"] != [4, 8]:
        raise ValueError("Keep the historical stage schedule")
    if any((root/name).exists() for name in ("experiment.json", "warmup.pth", "warmup_protocol.json")):
        raise RuntimeError("Existing experiment/partial warmup: no replay or overwrite")
    root.mkdir(parents=True, exist_ok=True)
    code = verify_historical_code(c)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    atomic_json(root/"warmup_protocol.json", dict(
        seed=c["seed"], epochs=3, source_only=True,
        backend_initialization="same ConditionalResidualSLS(sls) constructor as historical seed20",
        source_training="same historical warmup_trainer.train_epoch and optimizers",
        original_encoder_frozen=True, no_teacher_or_residual_during_warmup=True,
        code_checks=code, wrapper_sha256=digest(__file__), config_sha256=digest(config_path),
        started_at=attention.utc_now()))
    dataset, _ = dataset_for(c)
    set_seed(c["seed"])
    # Use the EXACT seed20 warmup construction order and optimization implementation.
    model = ConditionalResidualSLS(c["ssl_checkpoint"], "sls").cuda()
    model.task.requires_grad_(False)
    opt, ropt = warmup_trainer.optimizers(model, c)
    warmup_deadline = time.monotonic() + c["max_warmup_seconds"]
    for epoch in range(1, 4):
        if time.monotonic() > warmup_deadline:
            raise TimeoutError("Warmup budget exhausted; no automatic restart")
        atomic_json(root/"status.json", dict(status="warmup", seed=c["seed"], epoch=epoch,
                                              updated_at=attention.utc_now()))
        metrics = warmup_trainer.train_epoch(model, opt, ropt, dataset, c, "warmup", epoch)
        if time.monotonic() > warmup_deadline:
            raise TimeoutError("Warmup budget exhausted; partial run is not resumable")
        row = dict(arm="warmup", seed=c["seed"], epoch=epoch, **metrics, selected=False)
        append_csv(root/"warmup_epoch_results.csv", row)
        print("WARMUP_EPOCH", json.dumps(row), flush=True)
    backend = {name: value.detach().cpu().clone() for name, value in model.backend.state_dict().items()}
    atomic_save(root/"warmup.pth", dict(
        backend=backend, rng=rng_state(), seed=c["seed"], warmup_epochs=3,
        pretrained_sha256=c["expected_pretrained_sha256"], source_only=True,
        source_loss=metrics["train_loss"]))
    del model, opt, ropt, backend, dataset
    gc.collect()
    torch.cuda.empty_cache()
    c["shared_warmup"] = str(root/"warmup.pth")
    c["expected_warmup_sha256"] = digest(c["shared_warmup"])
    atomic_json(root/"resolved_config.json", c)
    print("INDEPENDENT_WARMUP_COMPLETE", c["seed"], c["expected_warmup_sha256"], flush=True)
    # Only result bookkeeping differs: no imaginary same-seed SLS/Query baselines.
    # All original model, preflight, step, epoch, evaluation and selection code is reused.
    attention.decide = fixed_decision
    attention.run(c, config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    root = Path(c["output_dir"]).resolve()
    # Refuse replay without replacing previous completed/failed status files.
    if any((root/name).exists() for name in ("experiment.json", "warmup.pth", "warmup_protocol.json")):
        raise RuntimeError("Existing run; no automatic restart")
    try:
        run(args.config)
    except Exception as error:
        if root.is_relative_to(Path("/root/rivermind-data")):
            prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
            atomic_json(root/"status.json", dict(
                status="stopped_budget" if isinstance(error, TimeoutError) else "failed",
                seed=c["seed"], error=repr(error), previous_status=prior,
                ended_at=attention.utc_now(), automatic_restart=False))
        print("FAILED", repr(error), flush=True)
        raise


if __name__ == "__main__":
    main()
