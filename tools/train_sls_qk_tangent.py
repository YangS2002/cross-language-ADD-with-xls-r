"""Two-arm seed20 protocol; only the backbone forward differs."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from tools import train_sls_pretrained_attention as base
from tools.sls_qk_tangent_model import QKTangentSLS
from tools.train_relational_preservation import atomic_json, digest


_signature = base.signature
_base_preflight = base.preflight


def signature(config, config_path):
    result = _signature(config, config_path)
    result.update(tangent_model=digest("tools/sls_qk_tangent_model.py"),
                  tangent_trainer=digest(__file__), backbone_mode=config["backbone_mode"])
    return result


def loss_of(output, labels):
    return torch.nn.functional.nll_loss(output, labels, weight=output.new_tensor([.1, .9]))


def preflight(config, dataset, sha):
    if config["backbone_mode"] == "original":
        return _base_preflight(config, dataset, sha)
    started = time.perf_counter()
    base.set_seed(config["seed"])
    model = QKTangentSLS(config["ssl_checkpoint"]).cuda().eval()
    warm = torch.load(config["shared_warmup"], map_location="cpu", weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    half = config["batch_size"]//2
    ids = ([i for i, y in enumerate(dataset.synthesizerlabels) if y == "bonafide"][:half]
           + [i for i, y in enumerate(dataset.synthesizerlabels) if y != "bonafide"][:half])
    items = [dataset[i] for i in ids]
    wave = torch.stack([b["audio"] for b in items]).cuda()
    labels = torch.tensor([int(b["label"]) for b in items], device="cuda")
    small = wave[[0, half]]
    small_labels = labels[[0, half]]
    comparisons = []
    for dtype in (torch.float32, torch.bfloat16):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype!=torch.float32):
            model.tangent_enabled = False
            normal = model(small, return_details=True)
            model.tangent_enabled = True
            tangent = model(small, return_details=True)
            errors = [float((x.float()-y.float()).abs().max()) for x, y in zip(normal["states"], tangent["states"])]
            tol = "fp32" if dtype == torch.float32 else "bf16"
            for x, y in zip(normal["states"], tangent["states"]):
                torch.testing.assert_close(x, y, atol=config[tol+"_atol"], rtol=config[tol+"_rtol"])
            torch.testing.assert_close(normal["log_probs"], tangent["log_probs"],
                                       atol=config[tol+"_atol"], rtol=config[tol+"_rtol"])
            comparisons.append(dict(dtype=tol, layer_max_errors=errors,
                                    output_max_error=float((normal["log_probs"]-tangent["log_probs"]).abs().max())))
        del normal, tangent
    print("INITIAL_EQUIVALENCE", json.dumps(comparisons), flush=True)
    model.tangent_enabled = False
    loss_of(model(small), small_labels).backward()
    gradients = {n: p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.tangent_enabled = True
    loss_of(model(small), small_labels).backward()
    numerator = denominator = 0.
    actual_names = {n for n,p in model.named_parameters() if p.grad is not None}
    if actual_names != set(gradients):
        raise RuntimeError(f"Gradient parameter set mismatch: {actual_names ^ set(gradients)}")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            original = gradients[name].cuda()
            numerator += float((parameter.grad-original).square().sum())
            denominator += float(original.square().sum())
    relative = (numerator/max(denominator, 1e-30))**.5
    print("INITIAL_GRADIENT_RELATIVE_L2", relative, flush=True)
    if relative > .001:
        raise RuntimeError(f"Initial gradient mismatch: {relative}")
    del gradients, original
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    # Nonzero full-network direction: the finite-difference derivative must
    # match the propagated tangent, not just have nonzero gradients.
    names = ("model.encoder.layers.0.self_attn.v_proj.weight",
             "model.encoder.layers.0.fc1.weight",
             "model.encoder.layers.0.final_layer_norm.weight",
             "model.encoder.pos_conv.0.weight_g")
    parameters = dict(model.task.named_parameters())
    originals = {name: parameters[name].detach().clone() for name in names}
    direction = {name: torch.randn_like(parameters[name]) * .0001 for name in names}
    def install(scale):
        with torch.no_grad():
            for name in names:
                parameters[name].copy_(originals[name] + scale*direction[name])
    # TF32 convolution quantization is larger than the finite perturbation.
    # Disable it only for this derivative check, not for formal training.
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        model.tangent_enabled = False
        install(1.)
        plus = [h.float() for h in model(small, True)["states"]]
        install(-1.)
        minus = [h.float() for h in model(small, True)["states"]]
        finite_difference = [(p-m)/2 for p,m in zip(plus,minus)]
        install(0.)
        zero = [h.float() for h in model(small, True)["states"]]
        install(1.)
        model.tangent_enabled = True
        shifted = [h.float() for h in model(small, True)["states"]]
        num = sum(float((f-(s-z)).square().sum()) for f,s,z in zip(finite_difference,shifted,zero))
        den = sum(float(f.square().sum()) for f in finite_difference)
        direction_error = (num/max(den,1e-30))**.5
        install(0.)
    print("NONZERO_DIRECTION_RELATIVE_L2", direction_error, flush=True)
    if direction_error > .03:
        raise RuntimeError(f"Nonzero tangent propagation mismatch: {direction_error}")
    del originals, direction, plus, minus, finite_difference, zero, shifted
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    optimizer = base.make_optimizer(model, config)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    begin = time.perf_counter()
    for index in range(3):
        loss, output, norm = base.step(model, optimizer, wave, labels)
        losses.append(float(loss))
        if any(p.grad is not None for p in model.reference.parameters()):
            raise RuntimeError("Reference gradient leak")
        for layer in model.task.model.encoder.layers:
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj):
                if any(p.requires_grad or p.grad is not None for p in projection.parameters()):
                    raise RuntimeError("QK must remain frozen")
        print("SELF_TEST_STEP", index, float(loss), norm, flush=True)
    torch.cuda.synchronize()
    record = dict(comparisons=comparisons, initial_gradient_relative_l2=relative,
                  nonzero_direction_relative_l2=direction_error, losses=losses,
                  batch=config["batch_size"], trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  three_steps_seconds=time.perf_counter()-begin, seconds=time.perf_counter()-started,
                  source_ids=[b["file_name"] for b in items], test_updates_discarded=True)
    if record["reserved_gib"] > config["max_reserved_gib"]:
        raise RuntimeError("Memory budget exceeded; do not silently change protocol")
    model.load_trainable_payload(model.trainable_payload(sha), sha)
    atomic_json(Path(config["output_dir"])/"self_test.json", record)
    print("SELF_TEST_OK", json.dumps(record), flush=True)
    del model, optimizer, wave, labels, small, small_labels, loss, output
    gc.collect()
    torch.cuda.empty_cache()
    return record


def decide(rows, config, epoch):
    values = {r["domain"]:r["eer"] for r in rows}
    if len(rows)!=5 or set(values)!=set(base.PAPER_SLS):
        raise RuntimeError("Incomplete evaluation")
    return dict(epoch=epoch, seed=config["seed"], stop=False,
                reason="fixed_epoch8_complete" if epoch==8 else "continue_fixed_training",
                per_domain=values, cross_mean=sum(values[k] for k in base.CROSS)/4,
                worst_cross=max(values[k] for k in base.CROSS),
                backbone_mode=config["backbone_mode"], eligible_for_fixed_multiseed=False,
                target_checkpoint_or_seed_selection=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config["seed"]!=20 or config["backbone_mode"] not in ("original", "tangent"):
        raise ValueError("Only the authorized seed20 two-arm comparison")
    root = Path(config["output_dir"]).resolve()
    if not root.is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Outputs must remain on the persistent data disk")
    if (root/"experiment.json").exists():
        raise RuntimeError("Existing experiment: no automatic restart or overwrite")
    base.ARM = "qk_tangent" if config["backbone_mode"]=="tangent" else "pretrained_attention"
    if config["backbone_mode"]=="tangent":
        base.PretrainedAttentionSLS = QKTangentSLS
    base.signature = signature
    base.preflight = preflight
    base.decide = decide
    try:
        base.run(config, args.config, args.preflight_only)
    except Exception as error:
        previous = json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {}
        failure = dict(status="stopped_budget" if isinstance(error, TimeoutError) else "failed",
                       error=repr(error), ended_at=base.utc_now(), automatic_restart=False,
                       previous_status=previous)
        atomic_json(root/"failure_tangent.json", failure)
        atomic_json(root/"status.json", failure)
        raise


if __name__ == "__main__":
    main()

