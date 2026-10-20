"""Source-only feasibility audit; no task updates or target audio."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from torch.nn import functional as F

from tools.sls_reference_model import XLSRStates
from tools.train_relational_preservation import atomic_json, dataset_for, digest
from train import set_seed


def run(config, output):
    c = yaml.safe_load(Path(config).read_text())
    if Path(output).exists():
        raise RuntimeError("Do not overwrite an audit")
    if digest(c["ssl_checkpoint"]) != c["expected_pretrained_sha256"]:
        raise RuntimeError("Pretrained SHA mismatch")
    dataset, _ = dataset_for(c)
    selected = []
    strata = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
    for label in strata:
        ids = [i for i, y in enumerate(dataset.synthesizerlabels) if y == label]
        n = 64 if label == "bonafide" else 32
        selected.extend(ids[int(k)] for k in np.linspace(0, len(ids)-1, n, dtype=int))
    set_seed(c["seed"])
    teacher = XLSRStates(c["ssl_checkpoint"], False).cuda().eval()
    model = teacher.model
    q = model.quantizer
    if q is None or q.training or not q.time_first or model.input_quantizer is not None:
        raise RuntimeError("Unexpected original quantizer configuration")
    current = {}
    per_sample = []
    all_mass = [[] for _ in range(24)]
    all_value_residual = [[] for _ in range(24)]

    def features_hook(_module, _args, output):
        features = output
        drop = features.shape[1] % model.crop_seq_to_multiple
        if drop:
            features = features[:, :-drop]
        logits = q.weight_proj(features).view(*features.shape[:2], q.groups, q.num_vars)
        targets = logits.argmax(-1)
        # Verify the inexpensive extraction against fairseq's actual eval API once.
        if not current.get("api_verified"):
            actual = q(features[:2], produce_targets=True)["targets"]
            if not torch.equal(actual, targets[:2]):
                raise RuntimeError("Unit extraction differs from original eval quantizer")
            current["api_verified"] = True
        joint = torch.zeros_like(targets[..., 0])
        for g in range(q.groups):
            joint = joint*q.num_vars + targets[..., g]
        repeated, inverse_rows, count_rows = [], [], []
        for b, code in enumerate(joint):
            _, inverse, counts = torch.unique(code, return_inverse=True, return_counts=True)
            repeated.append(counts[inverse] >= 2)
            inverse_rows.append(inverse)
            slots=torch.zeros(code.numel(), device=code.device)
            slots[:counts.numel()]=counts
            count_rows.append(slots)
            natural = current["natural_frames"][b]
            natural_code = code[:natural]
            _, natural_counts = torch.unique(natural_code, return_counts=True)
            per_sample.append(dict(
                name=current["names"][b], stratum=current["strata"][b],
                raw_samples=current["lengths"][b], natural_feature_frames=natural,
                frames=code.numel(), occupied_joint_units=counts.numel(),
                routing_degrees_per_query=int(code.numel()-counts.numel()),
                repeated_frame_fraction=float((counts[inverse] >= 2).float().mean()),
                natural_repeated_frame_fraction=float(natural_counts[natural_counts >= 2].sum()/max(1,natural)),
                max_group_size=int(counts.max())))
        current["repeated"] = torch.stack(repeated)
        current["groups"] = torch.stack(inverse_rows)
        current["counts"] = torch.stack(count_rows)

    handles = [model.layer_norm.register_forward_hook(features_hook)]
    for index, layer in enumerate(model.encoder.layers):
        def attention_hook(module, args, kwargs, index=index):
            z = kwargs["query"] if "query" in kwargs else args[0]
            t, b, d = z.shape
            h = module.num_heads
            query = module.q_proj(z).permute(1,0,2).reshape(b,t,h,d//h).transpose(1,2)
            key = module.k_proj(z).permute(1,0,2).reshape(b,t,h,d//h).transpose(1,2)
            logits = torch.matmul(query*(d//h)**-.5, key.transpose(-1,-2))
            a = F.softmax(logits.float(), dim=-1)
            mass = (a*current["repeated"][:,None,None,:]).sum(-1).mean(-1)
            all_mass[index].extend(mass.cpu().tolist())
            values=module.v_proj(z).permute(1,0,2).reshape(b,t,h,d//h).transpose(1,2).float()
            ids=current["groups"][:,None,:,None].expand_as(values)
            sums=torch.zeros_like(values).scatter_add(2,ids,values)
            means=sums/current["counts"][:,None,:,None].clamp_min(1)
            residual=values-means.gather(2,ids)
            ratio=(residual.square().sum((-1,-2))/values.square().sum((-1,-2)).clamp_min(1e-12)).sqrt()
            all_value_residual[index].extend(ratio.cpu().tolist())
        handles.append(layer.self_attn.register_forward_pre_hook(attention_hook, with_kwargs=True))
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for begin in range(0, len(selected), 16):
                ids = selected[begin:begin+16]
                items = [dataset[i] for i in ids]
                lengths = [torchaudio.info(str(Path(dataset.datapath)/b["file_name"])).num_frames for b in items]
                natural = model._get_feat_extract_output_lengths(torch.tensor([min(n,64600) for n in lengths])).tolist()
                current.update(names=[b["file_name"] for b in items], lengths=lengths,
                               natural_frames=natural, strata=[b["vocoder_n"] for b in items])
                teacher(torch.stack([b["audio"] for b in items]).cuda())
                print("SOURCE_AUDIT", begin+len(ids), len(selected), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    masses = np.asarray(all_mass)
    residuals = np.asarray(all_value_residual)
    record = dict(source_only=True, selection="64 real + 32 per A01-A06, evenly spaced within protocol strata",
                  samples=len(per_sample), pretrained_sha256=c["expected_pretrained_sha256"],
                  quantizer=dict(groups=q.groups, entries_per_group=q.num_vars, time_first=q.time_first,
                                 joint_partition=True, actual_eval_api_verified=current["api_verified"]),
                  source_config_sha256=digest(config), audit_script_sha256=digest(__file__),
                  task_training=False, target_audio_used=False,
                  attention_mass_summary=dict(mean=float(masses.mean()), minimum=float(masses.min()),
                      maximum=float(masses.max()), p10=float(np.quantile(masses,.1)),
                      median=float(np.median(masses)), layer_means=masses.mean((1,2)).tolist()),
                  original_value_within_group_residual_ratio=dict(mean=float(residuals.mean()),
                      p10=float(np.quantile(residuals,.1)),layer_means=residuals.mean((1,2)).tolist(),
                      interpretation="Initial original-Value heterogeneity only; not causal artifact evidence or final message usefulness"),
                  sample_summary={label:{key:float(np.mean([s[key] for s in per_sample if s["stratum"]==label]))
                        for key in ["occupied_joint_units","routing_degrees_per_query","repeated_frame_fraction",
                                    "natural_repeated_frame_fraction"]} for label in strata},
                  repeat_padded_samples=sum(s["raw_samples"]<64600 for s in per_sample),
                  non_repeat_padded_samples=sum(s["raw_samples"]>=64600 for s in per_sample),
                  per_sample=per_sample, reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    atomic_json(Path(output), record)
    print("AUDIT_COMPLETE", json.dumps({k:v for k,v in record.items() if k!="per_sample"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.config, a.output)
