"""Repeat unchanged original Q/K seed20; storage-only best checkpoint policy."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from tools import train_sls_pretrained_attention as original

HISTORY = Path("models/SLS_pretrained_attention_seed20")
_signature = original.signature
_copyfile = shutil.copyfile


def signature(c, config_path):
    current = _signature(c, config_path)
    history = json.loads((HISTORY/"experiment.json").read_text())
    expected = history["signature"]
    for key, value in current.items():
        if key != "config" and value != expected[key]:
            raise RuntimeError(f"Original Q/K protocol changed: {key}")
    # Only output identity changes; all training/evaluation settings are identical.
    for key, value in history["config"].items():
        if key not in ("exp_name", "output_dir") and c.get(key) != value:
            raise RuntimeError(f"Historical seed20 setting changed: {key}")
    current["repeat_runner"] = original.digest(__file__)
    current["historical_experiment"] = original.digest(HISTORY/"experiment.json")
    return current


def storage_preflight(c, dataset, sha):
    # Model/trainer/dependencies/base/warmup were all signature-verified above.
    # Reuse the original successful GPU checks, rather than repeating them.
    path = HISTORY/"self_test.json"
    prior = json.loads(path.read_text())
    if (prior["trainable_parameters"] != 284246149
            or any(prior["attention_max_errors_after_source_updates"])
            or any(any(row["layer_max_absolute_errors"]) or row["output_max_absolute_error"]
                   for row in prior["comparisons"])
            or prior["reference_gradient"] or prior["task_query_gradient"] or prior["task_key_gradient"]):
        raise RuntimeError("Historical original-QK self-test was not clean")
    required = int(prior["model_bytes"]*2.1)+512*2**20
    available = shutil.disk_usage(c["output_dir"]).free
    if available < required:
        raise RuntimeError(f"Insufficient space for best+atomic temporary: {available} < {required}")
    record = dict(historical_self_test=str(path), historical_self_test_sha256=original.digest(path),
                  historical_checks_reused=True, new_gpu_self_test_performed=False,
                  source_updates_this_preflight=0, source_samples=len(dataset),
                  trainable_parameters=prior["trainable_parameters"],
                  required_free_bytes=required, actual_free_bytes=available,
                  checkpoint_policy="one_source_best_no_epoch4_weight_copy_no_Adam",
                  no_training_or_model_change=True)
    original.atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("HISTORICAL_CHECKS_REUSED", json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    root = Path(c["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")) or (root/"experiment.json").exists():
        raise RuntimeError("Persistent, new experiment directory required")

    def no_duplicate_stage_weight(src, dst, *args, **kwargs):
        if Path(dst).resolve() == root/original.ARM/"source_best_at_epoch4.pth":
            if Path(src).resolve() != root/original.ARM/"best.pth":
                raise RuntimeError("Unexpected source checkpoint")
            original.atomic_json(root/"epoch4_snapshot_policy.json", dict(
                copied=False, reason="save_space_keep_one_source_best",
                epoch4_evaluation_and_scores_retained=True,
                epoch4_weight_not_retained_after_later_best_replacement=True))
            return str(src)
        return _copyfile(src, dst, *args, **kwargs)

    original.signature, original.preflight = signature, storage_preflight
    shutil.copyfile = no_duplicate_stage_weight
    # No model/optimizer/step/train_epoch/evaluate/decide replacement.
    try:
        original.run(c, args.config)
    except Exception as error:
        prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
        original.atomic_json(root/"status.json", dict(status="failed", error=repr(error),
                             previous_status=prior, ended_at=original.utc_now(), automatic_restart=False))
        raise
    finally:
        shutil.copyfile = _copyfile


if __name__ == "__main__":
    main()
