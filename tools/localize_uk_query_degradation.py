"""Inference-only, endpoint interventions; not a detector-training experiment."""
import copy
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import MethodType

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tools.sls_pretrained_query_model import PretrainedQuerySLS
from tools.evaluate_four_domains_shared_xlsr import (
    AudioDomain, DOMAINS, collate, eer, digest, atomic_json,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "models/UK_query_localization_seed20"
VARIANTS = ("native", "original_routing", "original_value", "task_warm_backend",
            "original_task_backend", "original_warm_backend")


def readout(backend, states):
    gates = torch.sigmoid(backend.fc0(backend.descriptors(states)))
    aggregate = sum(h * gates[:, i:i+1] for i, h in enumerate(states))
    flat = F.max_pool2d(backend.selu(backend.first_bn(aggregate.unsqueeze(1))), 3, 3).flatten(1)
    return F.log_softmax(backend.selu(backend.fc3(backend.selu(backend.fc1(flat)))).float(), dim=1)[:, 1]


def mha(module, q, k, v, km, vm, mask, attn_mask):
    # Only the key/value input and projection are replaced. Task out/FFN stay fixed.
    return F.multi_head_attention_forward(
        q, k, v, module.embed_dim, module.num_heads, torch.empty(0),
        torch.cat((module.q_proj.bias, km.k_proj.bias, vm.v_proj.bias)),
        module.bias_k, module.bias_v, module.add_zero_attn, module.dropout_module.p,
        module.out_proj.weight, module.out_proj.bias, False, mask, True, attn_mask,
        use_separate_proj_weight=True, q_proj_weight=module.q_proj.weight,
        k_proj_weight=km.k_proj.weight, v_proj_weight=vm.v_proj.weight,
        average_attn_weights=False)


class Intervention:
    def __init__(self, model):
        self.model = model
        self.mode = "native"
        self.reference_attention = {}
        self.stats = {}
        for index, layer in enumerate(model.reference.model.encoder.layers):
            def ref_forward(module, query, key, value, key_padding_mask=None,
                            incremental_state=None, need_weights=True, static_kv=False,
                            attn_mask=None, before_softmax=False, need_head_weights=False,
                            index=index):
                if incremental_state or static_kv or before_softmax or need_head_weights:
                    raise RuntimeError("Unsupported teacher path")
                out, probability = mha(module, query, key, value, module, module,
                                       key_padding_mask, attn_mask)
                positions = torch.linspace(0, min(200, query.size(0)-1), 8, device=query.device).long()
                if key_padding_mask is not None and key_padding_mask[:, positions].any():
                    raise RuntimeError("Sampled query includes padding")
                self.reference_attention[index] = probability[:, :, positions].detach().float()
                return out, probability.mean(1)
            layer.self_attn.forward = MethodType(ref_forward, layer.self_attn)
        for index, layer in enumerate(model.task.model.encoder.layers):
            def task_forward(module, query, key, value, key_padding_mask=None,
                             incremental_state=None, need_weights=True, static_kv=False,
                             attn_mask=None, before_softmax=False, need_head_weights=False,
                             index=index):
                if incremental_state or static_kv or before_softmax or need_head_weights:
                    raise RuntimeError("Unsupported task path")
                reference, mask = model._reference_inputs[index]
                if query.shape != reference.shape or ((mask is None) != (key_padding_mask is None)):
                    raise RuntimeError("Task/reference mismatch")
                if mask is not None and not torch.equal(mask, key_padding_mask):
                    raise RuntimeError("Task/reference padding mismatch")
                teacher = model.reference.model.encoder.layers[index].self_attn
                original_k = self.mode == "original_routing"
                original_v = self.mode == "original_value"
                out, probability = mha(module, reference, reference if original_k else key,
                                       reference if original_v else value,
                                       teacher if original_k else module,
                                       teacher if original_v else module, key_padding_mask, attn_mask)
                model.query_calls.append(index+1)
                if self.mode == "native":
                    positions = torch.linspace(0, min(200, query.size(0)-1), 8, device=query.device).long()
                    p = probability[:, :, positions].float().clamp_min(1e-12)
                    p = p / p.sum(-1, keepdim=True)
                    r = self.reference_attention[index].clamp_min(1e-12)
                    r = r / r.sum(-1, keepdim=True)
                    entropy = -(p * p.log()).sum(-1).mean((1, 2)) / np.log(p.size(-1))
                    ref_entropy = -(r * r.log()).sum(-1).mean((1, 2)) / np.log(r.size(-1))
                    kl = (p * (p.log()-r.log())).sum(-1).mean((1, 2))
                    vv = module.v_proj(value).float().transpose(0, 1)
                    v0 = teacher.v_proj(reference).float().transpose(0, 1)
                    v_relative = (vv-v0).flatten(1).norm(dim=1) / v0.flatten(1).norm(dim=1).clamp_min(1e-8)
                    kk = module.k_proj(key).float().transpose(0, 1)
                    k0 = teacher.k_proj(reference).float().transpose(0, 1)
                    k_relative = (kk-k0).flatten(1).norm(dim=1) / k0.flatten(1).norm(dim=1).clamp_min(1e-8)
                    self.stats[index] = torch.stack((kl, entropy, ref_entropy, k_relative, v_relative), 1).cpu().numpy()
                # fairseq expects [B,T,T], including when trimming its internal padding.
                return out, probability.mean(1)
            layer.self_attn.forward = MethodType(task_forward, layer.self_attn)

    def reference(self, wave):
        self.model._reference_inputs = []
        self.model.reference_calls = []
        self.reference_attention = {}
        state = np.random.get_state()
        try:
            states = self.model.reference(wave)
        finally:
            np.random.set_state(state)
        if len(self.model._reference_inputs) != 24:
            raise RuntimeError("Missing reference layers")
        return states

    def task(self, wave, mode):
        self.mode = mode
        self.model.query_calls = []
        states = self.model.task(wave)
        if self.model.query_calls != list(range(1, 25)):
            raise RuntimeError("Task layers missing")
        return states


def hash_key(name):
    return hashlib.sha256(name.encode()).hexdigest()


def metric(y, s):
    value, threshold = eer(np.asarray(y), np.asarray(s))
    return dict(eer=value, threshold=threshold, auc=float(roc_auc_score(y, s)),
                samples=len(y), real=int(np.sum(np.asarray(y) == 1)),
                fake=int(np.sum(np.asarray(y) == 0)))


def main():
    start = time.monotonic()
    if OUT.exists():
        raise RuntimeError("Output already exists; no automatic overwrite/restart")
    OUT.mkdir(parents=True)
    atomic_json(OUT / "status.json", dict(state="initializing", budget_seconds=600))
    try:
        c = yaml.safe_load((REPO / "experiments/configs/sls_pretrained_query_seed20.yaml").read_text())
        for path, expected in ((c["ssl_checkpoint"], c["expected_pretrained_sha256"]),
                               (c["shared_warmup"], c["expected_warmup_sha256"])):
            if digest(path) != expected:
                raise RuntimeError("Base/warmup SHA mismatch")
        checkpoint = REPO / "models/SLS_pretrained_query_seed20/pretrained_query/best.pth"
        checkpoint_sha = digest(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["epoch"] != 8:
            raise RuntimeError("Require completed source-selected epoch8")
        model = PretrainedQuerySLS(c["ssl_checkpoint"])
        model.load_trainable_payload(saved, c["expected_pretrained_sha256"])
        del saved
        warm = torch.load(c["shared_warmup"], map_location="cpu", weights_only=False)
        warm_backend = copy.deepcopy(model.backend)
        warm_backend.load_state_dict(warm["backend"], strict=True)
        del warm
        model.requires_grad_(False).cuda().eval()
        warm_backend.requires_grad_(False).cuda().eval()
        manifest_path = REPO / "models/UK_representation_diagnostic_seed20/split_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        group = {r["name"]: r["group"] for r in manifest["uk"]["rows"]}
        DOMAINS["a"] = (Path(c["train_label_path"]), Path(c["train_data_path"]), None, "")
        for domain in ("uk", "itw"):
            d = c["domains"][domain]
            DOMAINS[domain] = (Path(d["label"]), None, Path(d["archive"]), d.get("prefix", ""))
        datasets = {d: AudioDomain(d) for d in ("a", "uk", "itw")}
        source_names = {r["name"] for r in sorted(manifest["a"]["rows"], key=lambda r: hash_key(r["name"]))[:1024]}
        datasets["a"].rows = [r for r in datasets["a"].rows if r[0] in source_names]
        datasets["itw"].rows = sorted(datasets["itw"].rows, key=lambda r: hash_key(r[0]))[:1024]
        if len(datasets["a"]) != 1024 or set(group) != {r[0] for r in datasets["uk"].rows}:
            raise RuntimeError("Manifest coverage mismatch")
        inventory = {d: ds.validate_paths() for d, ds in datasets.items()}
        atomic_json(OUT / "protocol.json", dict(
            checkpoint_sha256=checkpoint_sha, checkpoint_epoch=8,
            base_sha256=c["expected_pretrained_sha256"], warmup_sha256=c["expected_warmup_sha256"],
            script_sha256=digest(__file__), group_manifest_sha256=digest(manifest_path),
            inventory=inventory, variants=VARIANTS, budget_seconds=600,
            target_labels="Only EER/AUC/group summaries; no fitting or selection",
            preprocessing="Unchanged data.pad64600; no resampling/noise",
            caveat="Endpoint hybrid interventions are off-manifold; not a causal training attribution"))
        result, rows, layer_rows = {}, [], []
        instrument = None
        for domain, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4,
                                collate_fn=collate, pin_memory=True)
            domain_start = time.monotonic()
            sums = {label: np.zeros((24, 5)) for label in (0, 1)}
            counts = {0: 0, 1: 0}
            collected = []
            for batch, (wave, labels, names) in enumerate(loader):
                if time.monotonic()-start > 600:
                    raise TimeoutError("Diagnostic 600-second budget reached; no auto-restart")
                wave = wave.cuda(non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if instrument is None:
                        native_before = model(wave)[:, 1].detach()
                        instrument = Intervention(model)
                    reference = instrument.reference(wave)
                    native = instrument.task(wave, "native")
                    scores = {"native": readout(model.backend, native),
                              "task_warm_backend": readout(warm_backend, native),
                              "original_task_backend": readout(model.backend, reference),
                              "original_warm_backend": readout(warm_backend, reference)}
                    if domain == "a" and batch == 0:
                        difference = float((scores["native"]-native_before).abs().max())
                        atomic_json(OUT / "numerical_check.json", dict(max_abs_score_difference=difference))
                        if difference > 1e-5:
                            raise RuntimeError("Instrumented native path numerically changed")
                    for label in (0, 1):
                        selected = labels == label
                        if selected.any():
                            sums[label] += np.stack([instrument.stats[i][selected].sum(0) for i in range(24)])
                            counts[label] += int(selected.sum())
                    del native
                    for mode in ("original_routing", "original_value"):
                        states = instrument.task(wave, mode)
                        scores[mode] = readout(model.backend, states)
                        del states
                    score_cpu = {k: v.cpu().numpy() for k, v in scores.items()}
                    instrument.model._reference_inputs = []
                    instrument.reference_attention = {}
                    del reference, scores
                if torch.cuda.max_memory_reserved()/2**30 > 20:
                    raise RuntimeError("Diagnostic reserved memory exceeds 20GiB")
                for i, name in enumerate(names):
                    row = dict(domain=domain, name=name, label=int(labels[i]), group=group.get(name, ""))
                    row.update({k: float(score_cpu[k][i]) for k in VARIANTS})
                    collected.append(row)
                if batch % 25 == 0:
                    progress = dict(state="running", domain=domain, samples=len(collected),
                                    total=len(dataset), elapsed_seconds=time.monotonic()-start)
                    atomic_json(OUT / "status.json", progress)
                    print(json.dumps(progress), flush=True)
            labels = np.asarray([r["label"] for r in collected])
            result[domain] = {k: metric(labels, [r[k] for r in collected]) for k in VARIANTS}
            result[domain]["seconds"] = time.monotonic()-domain_start
            rows.extend(collected)
            for label in (0, 1):
                means = sums[label] / counts[label]
                for index, values in enumerate(means):
                    layer_rows.append(dict(domain=domain, label=label, layer=index+1, samples=counts[label],
                                           attention_kl=float(values[0]), entropy=float(values[1]),
                                           original_entropy=float(values[2]), key_relative_l2=float(values[3]),
                                           value_relative_l2=float(values[4])))
            atomic_json(OUT / "domain_results.json", result)
            print("DOMAIN_COMPLETE", domain, json.dumps(result[domain]), flush=True)
            if domain == "uk" and abs(result[domain]["native"]["eer"]-21.2173476322) > 1e-5:
                raise RuntimeError("Full UK native does not reproduce registered epoch8 EER")
        uk = [r for r in rows if r["domain"] == "uk"]
        group_results = []
        auc_matrix = []
        for book in sorted(set(r["group"] for r in uk)):
            subset = [r for r in uk if r["group"] == book]
            y = np.asarray([r["label"] for r in subset])
            for variant in VARIANTS:
                s = np.asarray([r[variant] for r in subset])
                group_results.append(dict(group=book, variant=variant, **metric(y, s),
                                          real_median=float(np.median(s[y == 1])),
                                          fake_median=float(np.median(s[y == 0]))))
        books = sorted(set(r["group"] for r in uk))
        for real_book in books:
            real = [r for r in uk if r["group"] == real_book and r["label"] == 1]
            for fake_book in books:
                fake = [r for r in uk if r["group"] == fake_book and r["label"] == 0]
                for variant in ("native", "original_warm_backend"):
                    y = [1]*len(real) + [0]*len(fake)
                    s = [r[variant] for r in real+fake]
                    auc_matrix.append(dict(real_group=real_book, fake_group=fake_book, variant=variant,
                                           auc=float(roc_auc_score(y, s)), pairs=len(real)*len(fake)))
        for name, data in (("scores.csv", rows), ("layer_diagnostics.csv", layer_rows),
                           ("group_results.csv", group_results), ("cross_group_auc.csv", auc_matrix)):
            with (OUT / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        result["elapsed_seconds"] = time.monotonic()-start
        result["max_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
        result["uk_groups"] = len(books)
        atomic_json(OUT / "results.json", result)
        atomic_json(OUT / "status.json", dict(state="completed", seconds=result["elapsed_seconds"]))
    except Exception as error:
        atomic_json(OUT / "status.json", dict(state="failed", error=str(error),
                                              seconds=time.monotonic()-start))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
