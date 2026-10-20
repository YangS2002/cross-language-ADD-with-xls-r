"""Reuse the established single-arm trainer; only Value access changes."""
import csv
import gc
import json
from functools import partial
from pathlib import Path
import shutil
import time

import torch
import yaml

from tools import train_sls_pretrained_attention as base
from tools.sls_reference_value_model import ReferenceValueSLS

ARM = "reference_value"
_signature = base.signature
_train_epoch = base.train_epoch


def signature(c, config_path):
    saved = _signature(c, config_path)
    saved["shared_trainer"] = saved["trainer"]
    saved["trainer"] = base.digest(__file__)
    saved["model"] = base.digest("tools/sls_reference_value_model.py")
    saved["dependencies"]["tools/sls_pretrained_attention_model.py"] = base.digest(
        "tools/sls_pretrained_attention_model.py")
    return saved


def preflight(c, dataset, sha):
    start = time.perf_counter()
    half = c["batch_size"] // 2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"][:half])
    batches = [dataset[i] for i in ids]
    wave = torch.stack([b["audio"] for b in batches]).cuda()
    labels = torch.tensor([int(b["label"]) for b in batches], device="cuda")
    base.set_seed(c["seed"])
    model = base.PretrainedAttentionSLS(c["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.replacement_override = [False] * 24
        normal = model(wave[:2], return_details=True)
        model.replacement_override = [True] * 24
        reference = model(wave[:2], return_details=True)
        max_error = max(float((a.float()-b.float()).abs().max())
                        for a, b in zip(normal["states"], reference["states"]))
        torch.testing.assert_close(normal["log_probs"], reference["log_probs"], atol=.02, rtol=.01)
        if max_error > .02:
            raise RuntimeError("Original/task Value paths differ at initialization")
        del normal, reference
    before = torch.random.get_rng_state()
    torch.rand(24, generator=model.value_rng)
    if not torch.equal(before, torch.random.get_rng_state()):
        raise RuntimeError("Replacement sampler perturbs the global random stream")
    model.train()
    model.replacement_override = [i % 2 == 0 for i in range(24)]
    opt = base.make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    loss, output, norm = base.step(model, opt, wave, labels)
    if any(p.grad is not None for p in model.reference.parameters()):
        raise RuntimeError("Reference gradient leak")
    for i, layer in enumerate(model.task.model.encoder.layers):
        if any(p.requires_grad or p.grad is not None
               for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj)
               for p in projection.parameters()):
            raise RuntimeError("Frozen task Q/K changed")
        gradients = [p.grad for p in layer.self_attn.v_proj.parameters()]
        if model.last_value_mask[i]:
            if any(g is not None for g in gradients):
                raise RuntimeError("Selected original Value still trains task V")
        elif not any(g is not None and bool(g.abs().sum() > 0) for g in gradients):
            raise RuntimeError("Unselected task Value has no gradient")
        for module in (layer.self_attn.out_proj, layer.fc1):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                       for p in module.parameters()):
                raise RuntimeError("Task out/FFN missing gradient")
    model.replacement_override = None
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(wave[:2])
    if any(model.last_value_mask):
        raise RuntimeError("Evaluation unexpectedly replaces Value")
    payload = model.trainable_payload(sha)
    model.load_trainable_payload(payload, sha)
    record = dict(seconds=time.perf_counter()-start, source_updates=1,
                  initialization_layer_max_error=max_error, source_nll=float(loss), grad_norm=norm,
                  selected_original_v_no_task_v_gradient=True, task_out_ffn_gradient=True,
                  evaluation_uses_task_value_only=True, independent_replacement_rng=True,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    if record["reserved_gib"] > c["max_reserved_gib"] or shutil.disk_usage(c["output_dir"]).free < 4*2**30:
        raise RuntimeError("Memory/disk budget exceeded")
    base.atomic_json(Path(c["output_dir"])/"self_test.json", record)
    print("SHORT_PREFLIGHT_OK", json.dumps(record), flush=True)
    del payload, model, opt, output, loss, wave, labels
    gc.collect()
    torch.cuda.empty_cache()
    return record


def train_epoch(model, opt, dataset, c, epoch, deadline):
    before = list(model.replacement_counts)
    count = model.training_forwards
    metrics = _train_epoch(model, opt, dataset, c, epoch, deadline)
    forwards = model.training_forwards-count
    counts = [a-b for a, b in zip(model.replacement_counts, before)]
    metrics.update(value_replacement_rate=sum(counts)/(24*forwards),
                   value_replacements=sum(counts))
    base.atomic_json(Path(c["output_dir"])/f"value_access_epoch{epoch}.json",
                     dict(forwards=forwards, per_layer_counts=counts,
                          actual_rate=metrics["value_replacement_rate"], probability=model.probability))
    return metrics


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    with open(c["matched_attention_results"], newline="") as stream:
        attention = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                     if r["arm"] == "pretrained_attention" and int(r["milestone"]) == epoch}
    if set(values) != set(base.PAPER_SLS) or set(attention) != set(values):
        raise ValueError("Incomplete five-domain comparison")
    mean = sum(values[k] for k in base.CROSS)/4
    gap = {k: values[k]-attention[k] for k in values}
    # Ordinary tradeoffs continue to eight. Only the registered gross failure screen stops at four.
    severe = epoch == 4 and ((sum(gap[k] for k in base.CROSS)/4 > 3
                              and sum(gap[k] > 5 for k in base.CROSS) >= 2)
                             or (gap["itw"] > 3 and sum(gap[k] for k in base.CROSS) >= 0)
                             or any(values[k] > 35 and gap[k] > 8 for k in base.CROSS))
    return dict(epoch=epoch, stop=severe,
                reason="registered_gross_failure" if severe else "complete" if epoch == 8 else "continue_to8",
                per_domain=values, historical_attention=attention, gaps_vs_attention=gap,
                cross_mean=mean, worst_cross=max(values[k] for k in base.CROSS),
                ch_de_joint=(values["ch"]+values["de"])/2,
                paper_sls_gaps={k: values[k]-base.PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-base.PAPER_MAMBA[k] for k in values},
                eligible_for_fixed_multiseed=False, further_seeds_require_user_authorization=True,
                single_seed_is_not_stability_evidence=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    root = Path(c["output_dir"]).resolve()
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no overwrite/restart")
    base.ARM = ARM
    base.PretrainedAttentionSLS = partial(ReferenceValueSLS, probability=c["value_probability"],
                                        replacement_seed=c["replacement_seed"])
    base.signature, base.preflight = signature, preflight
    base.train_epoch, base.decide = train_epoch, decide
    try:
        base.run(c, args.config)
    except Exception as error:
        if root.is_relative_to(Path("/root/rivermind-data")):
            prior = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
            base.atomic_json(root/"status.json", dict(status="failed", error=repr(error),
                             previous_status=prior, ended_at=base.utc_now(), automatic_restart=False))
        raise


if __name__ == "__main__":
    main()
