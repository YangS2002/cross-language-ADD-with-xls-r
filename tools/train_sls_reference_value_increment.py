"""One source-only candidate; reuse unchanged original-QK training protocol."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
import yaml

from tools import train_sls_pretrained_attention as base
from tools.sls_reference_value_increment_model import (
    ReferenceValueIncrementSLS, reference_value_projection)

ARM = "reference_value_increment"
_signature = base.signature
_train_epoch = base.train_epoch


def signature(c, config_path):
    saved = _signature(c, config_path)
    historical = json.loads(Path(c["historical_attention_experiment"]).read_text())["signature"]
    # Protect remote modifications: verify, never upload the shared dependencies.
    for key in ("trainer", "model", "pretrained", "warmup", "labels", "fairseq_function_sha256"):
        if saved[key] != historical[key]:
            raise RuntimeError(f"Historical protocol changed: {key}")
    if saved["dependencies"] != historical["dependencies"]:
        raise RuntimeError("Historical dependencies changed; no overwrite")
    saved["shared_trainer"] = saved["trainer"]
    saved["trainer"] = base.digest(__file__)
    saved["shared_model"] = saved["model"]
    saved["model"] = base.digest("tools/sls_reference_value_increment_model.py")
    return saved


def algebra_check():
    # Private generator: synthetic mathematical checks do not advance training RNG.
    g = torch.Generator().manual_seed(912020)
    z0 = torch.randn(5, 2, 8, generator=g, dtype=torch.float64)
    zt = torch.randn(5, 2, 8, generator=g, dtype=torch.float64, requires_grad=True)
    original = torch.nn.Linear(8, 8).double().requires_grad_(False)
    task = torch.nn.Linear(8, 8).double()
    with torch.no_grad():
        original.weight.copy_(torch.randn(8, 8, generator=g, dtype=torch.float64))
        original.bias.copy_(torch.randn(8, generator=g, dtype=torch.float64))
        task.load_state_dict(original.state_dict())
    value = reference_value_projection(z0, zt, original, task)
    torch.testing.assert_close(value, original(z0), atol=1e-12, rtol=1e-12)
    value.square().sum().backward()
    if not torch.equal(zt.grad, torch.zeros_like(zt.grad)) or not bool(task.weight.grad.abs().sum() > 0):
        raise RuntimeError("Initial Value gradient decomposition incorrect")
    with torch.no_grad():
        task.weight.add_(.01*torch.randn(8, 8, generator=g, dtype=torch.float64))
        task.bias.add_(.01*torch.randn(8, generator=g, dtype=torch.float64))
    actual = reference_value_projection(z0, zt, original, task)
    expected = original(z0)+task(zt)-original(zt)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # Check the ACTUAL preprojected static_v MHA path, including A and out.
    eye = torch.eye(8, dtype=torch.float64)
    zero = torch.zeros(8, dtype=torch.float64)
    output, attention = F.multi_head_attention_forward(
        z0, z0, zt[..., :0], 8, 1, torch.empty(0), torch.cat((zero, zero, zero)),
        None, None, False, 0., eye, zero, False, need_weights=True,
        use_separate_proj_weight=True, q_proj_weight=eye, k_proj_weight=eye,
        v_proj_weight=task.weight[:, :0], static_v=actual.transpose(0, 1))
    expected_output = torch.bmm(attention, expected.transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(output, expected_output, atol=1e-12, rtol=1e-12)
    zt.grad = None
    actual.square().sum().backward()
    if not bool(zt.grad.abs().sum() > 0):
        raise RuntimeError("Adapted Value blocks task-context gradients")
    return dict(formula_fp64=True, actual_mha_formula_fp64=True,
                zero_delta_task_input_gradient_zero=True,
                zero_delta_value_weight_gradient_nonzero=True,
                nonzero_delta_task_input_gradient_nonzero=True)


def preflight(c, dataset, sha):
    start = time.perf_counter()
    algebra = algebra_check()
    half = c["batch_size"]//2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"][:half])
    batches = [dataset[i] for i in ids]
    wave = torch.stack([b["audio"] for b in batches]).cuda()
    labels = torch.tensor([int(b["label"]) for b in batches], device="cuda")
    base.set_seed(c["seed"])
    model = ReferenceValueIncrementSLS(c["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    comparisons = []
    for dtype in (torch.float32, torch.bfloat16):
        tag = "fp32" if dtype == torch.float32 else "bf16"
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            model.guidance_enabled = False
            native = model(wave[:2], return_details=True)
            expected = dict(states=[s.float().cpu() for s in native["states"]],
                            output=native["log_probs"].cpu())
            del native
            model.guidance_enabled = True
            routed = model(wave[:2], return_details=True)
            errors = []
            for old, new in zip(expected["states"], routed["states"]):
                new = new.float().cpu()
                torch.testing.assert_close(old, new, atol=c[tag+"_atol"], rtol=c[tag+"_rtol"])
                errors.append(float((old-new).abs().max()))
            torch.testing.assert_close(expected["output"], routed["log_probs"].cpu(),
                                       atol=c[tag+"_atol"], rtol=c[tag+"_rtol"])
            comparisons.append(dict(dtype=tag, layer_max_errors=errors,
                                    output_max_error=float((expected["output"]-routed["log_probs"].cpu()).abs().max())))
            del expected, routed
    model.train()
    opt = base.make_optimizer(model, c)
    torch.cuda.reset_peak_memory_stats()
    loss, output, norm = base.step(model, opt, wave, labels)
    if any(p.grad is not None for p in model.reference.parameters()):
        raise RuntimeError("Reference gradient leak")
    for layer in model.task.model.encoder.layers:
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj):
            if any(p.requires_grad or p.grad is not None for p in projection.parameters()):
                raise RuntimeError("Frozen Q/K changed")
        for module in (layer.self_attn.v_proj, layer.self_attn.out_proj, layer.fc1):
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
                raise RuntimeError("Task V/out/FFN gradient absent")
    for module in (model.backend.fc0, model.backend.fc1, model.backend.fc3):
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
            raise RuntimeError("SLS gradient absent")
    captured = {"reference": {}, "task": {}}
    handles = []
    for side in captured:
        for index, layer in enumerate(getattr(model, side).model.encoder.layers):
            def capture(_module, _args, result, side=side, index=index):
                captured[side][index] = result[1].detach().float().cpu()
            handles.append(layer.self_attn.register_forward_hook(capture))
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(wave[:2])
    finally:
        for handle in handles:
            handle.remove()
    attention_errors = []
    for i in range(24):
        torch.testing.assert_close(captured["reference"][i], captured["task"][i], atol=1e-6, rtol=0)
        attention_errors.append(float((captured["reference"][i]-captured["task"][i]).abs().max()))
    del captured
    payload = model.trainable_payload(sha)
    model.load_trainable_payload(payload, sha)
    record = dict(seconds=time.perf_counter()-start, source_updates=1,
                  self_test_updates_discarded=True, algebra=algebra, comparisons=comparisons,
                  source_nll=float(loss), grad_norm=norm,
                  attention_max_errors_after_update=attention_errors,
                  reference_and_task_qk_frozen=True, all24_task_v_out_ffn_gradients=True,
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
    metrics = _train_epoch(model, opt, dataset, c, epoch, deadline)
    # No extra forward: report parameter-delta scale, NOT representation/content purity.
    with torch.no_grad():
        rows = []
        for i, (task, ref) in enumerate(zip(model.task.model.encoder.layers,
                                           model.reference.model.encoder.layers), 1):
            w, w0 = task.self_attn.v_proj.weight, ref.self_attn.v_proj.weight
            rows.append(dict(layer=i, delta_weight_ratio=float((w-w0).norm()/w0.norm().clamp_min(1e-12)),
                             delta_bias_norm=float((task.self_attn.v_proj.bias-ref.self_attn.v_proj.bias).norm())))
    base.atomic_json(Path(c["output_dir"])/f"value_increment_epoch{epoch}.json", rows)
    return metrics


def decide(rows, c, epoch):
    values = {r["domain"]: r["eer"] for r in rows}
    with open(c["matched_attention_results"], newline="") as stream:
        attention = {r["domain"]: float(r["eer"]) for r in csv.DictReader(stream)
                     if r["arm"] == "pretrained_attention" and int(r["milestone"]) == epoch}
    if set(values) != set(base.PAPER_SLS) or set(attention) != set(values):
        raise RuntimeError("Incomplete five-domain comparison")
    gaps = {k: values[k]-attention[k] for k in values}
    severe = epoch == 4 and ((sum(gaps[k] for k in base.CROSS)/4 > 3
                              and sum(gaps[k] > 5 for k in base.CROSS) >= 2)
                             or (gaps["itw"] > 3 and sum(gaps[k] for k in base.CROSS) >= 0)
                             or any(values[k] > 35 and gaps[k] > 8 for k in base.CROSS))
    return dict(epoch=epoch, stop=severe,
                reason="registered_gross_failure" if severe else "complete" if epoch == 8 else "continue_to8",
                per_domain=values, matched_attention=attention, gaps_vs_attention=gaps,
                cross_mean=sum(values[k] for k in base.CROSS)/4,
                worst_cross=max(values[k] for k in base.CROSS),
                ch_de_joint=(values["ch"]+values["de"])/2,
                paper_sls_gaps={k: values[k]-base.PAPER_SLS[k] for k in values},
                paper_mamba_gaps={k: values[k]-base.PAPER_MAMBA[k] for k in values},
                eligible_for_fixed_multiseed=False, further_seeds_require_user_authorization=True,
                single_seed_is_not_stability_evidence=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    root = Path(c["output_dir"]).resolve()
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no overwrite/restart")
    base.ARM = ARM
    base.PretrainedAttentionSLS = ReferenceValueIncrementSLS
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
