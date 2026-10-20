"""Fixed-checkpoint inference diagnostics; no training or model selection."""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sls_pretrained_attention_model import PretrainedAttentionSLS
from tools.train_relational_preservation import dataset_for
from tools.evaluate_relational_preservation import make_domains
from tools.evaluate_four_domains_shared_xlsr import collate, eer

MODES = ('full', 'last_no_uniform', 'last_no_relation',
         'all_no_uniform', 'all_no_relation')


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def rank(name):
    return hashlib.sha256(('qk-relation-diagnostic-20261020:' + str(name)).encode()).hexdigest()


def select_source(dataset):
    groups = defaultdict(list)
    for i, kind in enumerate(dataset.synthesizerlabels):
        groups[kind].append(i)
    selected = []
    original, chosen = {}, {}
    for kind, indices in sorted(groups.items()):
        n = max(1, round(len(indices) * .025))
        pick = sorted(indices, key=lambda i: rank(dataset.filenames[i]))[:n]
        selected.extend(pick)
        original[kind], chosen[kind] = len(indices), len(pick)
    selected.sort(key=lambda i: rank(dataset.filenames[i]))
    return selected, dict(original=original, selected=chosen)


def select_target(dataset):
    indices = []
    for label in (0, 1):
        candidates = [i for i, (_, y) in enumerate(dataset.rows) if y == label]
        indices += sorted(candidates, key=lambda i: rank(dataset.rows[i][0]))[:128]
    return sorted(indices, key=lambda i: rank(dataset.rows[i][0]))


def source_collate(batch):
    return (torch.stack([b['audio'] for b in batch]),
            np.asarray([b['label'] for b in batch]),
            [b['file_name'] for b in batch])


def pool(z):
    z = z.float()
    return torch.cat((z.mean(1), (z.var(1, unbiased=False) + 1e-8).sqrt()), -1)


class AttentionDiagnostic:
    def __init__(self, model):
        self.model = model
        self.mode = 'full'
        self.stats = []
        self.features = {}
        for index, layer in enumerate(model.task.model.encoder.layers):
            module = layer.self_attn
            native = module.forward

            def wrapped(module, query, key, value, *args, index=index, native=native, **kwargs):
                result = native(query, key, value, *args, **kwargs)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError('Unexpected attention result')
                out, weights = result
                active = self.mode.startswith('all_') or (index == 23 and self.mode.startswith('last_'))
                if self.mode == 'full' or active:
                    # Projection bias is retained in both interventions. No residual or FFN is removed.
                    v = F.linear(value, module.v_proj.weight, module.v_proj.bias)
                    with torch.autocast('cuda', enabled=False):
                        uniform = F.linear(v.float().mean(0), module.out_proj.weight.float(), None)
                        bias = module.out_proj.bias.float() if module.out_proj.bias is not None else 0.
                        signal = out.float() - bias
                        relation = signal - uniform.unsqueeze(0)
                        if self.mode == 'full':
                            rms = signal.square().mean((0, 2)).sqrt()
                            grms = uniform.square().mean(-1).sqrt()
                            rrms = relation.square().mean((0, 2)).sqrt()
                            self.stats.append(torch.stack((rms, grms, rrms), -1))
                    if self.mode == 'full' and index == 23:
                        self.capture_heads(module, v, index, kwargs)
                    if active:
                        if self.mode.endswith('no_uniform'):
                            out = (out.float() - uniform.unsqueeze(0)).to(out.dtype)
                        else:
                            out = (uniform.unsqueeze(0) + bias).expand_as(out).to(out.dtype)
                return out, weights

            module.forward = MethodType(wrapped, module)

    def capture_heads(self, module, v, index, kwargs):
        ref, padmask = self.model._reference_inputs[index]
        teacher = self.model.reference.model.encoder.layers[index].self_attn
        q = F.linear(ref, teacher.q_proj.weight, teacher.q_proj.bias)
        k = F.linear(ref, teacher.k_proj.weight, teacher.k_proj.bias)
        t, b, d = q.shape
        h, dh = module.num_heads, d // module.num_heads
        def heads(z):
            return z.permute(1, 0, 2).reshape(b, t, h, dh).transpose(1, 2)
        with torch.autocast('cuda', enabled=False):
            qh, kh, vh = heads(q).float(), heads(k).float(), heads(v).float()
            scores = (qh * dh ** -.5) @ kh.transpose(-1, -2)
            if padmask is not None:
                scores = scores.masked_fill(padmask[:, None, None, :], -torch.inf)
            mask = kwargs.get('attn_mask')
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(mask, -torch.inf)
                else:
                    scores = scores + mask
            a = scores.softmax(-1)
            message = a @ vh
            if padmask is None:
                glob = vh.mean(2, keepdim=True)
            else:
                valid = (~padmask).float()[:, None, :, None]
                glob = (vh * valid).sum(2, keepdim=True) / valid.sum(2, keepdim=True)
            rel = message - glob
            def merge(z):
                return z.transpose(1, 2).reshape(b, t, d)
            self.features = dict(message=pool(merge(message)),
                                 relation=pool(merge(rel)),
                                 uniform=glob.flatten(1),
                                 attention_entropy=(-(a * a.clamp_min(1e-20).log()).sum(-1).mean(-1) / np.log(t)),
                                 head_relation_rms=rel.square().mean((2, 3)).sqrt(),
                                 head_message_rms=message.square().mean((2, 3)).sqrt())


def metrics(labels, output, reference):
    logp = np.asarray(output, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    y = np.asarray(labels)
    value, threshold = eer(y, logp[:, 1])
    w = np.where(y == 0, .1, .9)
    delta = (logp[:, 1] - logp[:, 0]) - (ref[:, 1] - ref[:, 0])
    return dict(eer=value, eer_threshold=threshold,
                accuracy=float((logp.argmax(1) == y).mean()),
                real_recall=float((logp[y == 1].argmax(1) == 1).mean()),
                fake_recall=float((logp[y == 0].argmax(1) == 0).mean()),
                weighted_nll=float((-logp[np.arange(len(y)), y] * w).sum() / w.sum()),
                prediction_agreement=float((logp.argmax(1) == ref.argmax(1)).mean()),
                logodds_change_rms=float(np.sqrt((delta ** 2).mean())),
                samples=len(y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='experiments/configs/sls_qk_tangent_original_seed20.yaml')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20)
    np.random.seed(20)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=False)
    c = yaml.safe_load((ROOT / args.config).read_text())
    source, _ = dataset_for(c)
    ids, distribution = select_source(source)
    domains = make_domains(c)
    selected = {'source_train': (source, ids, source_collate)}
    manifest = {'source_train': dict(distribution=distribution, names=[source.filenames[i] for i in ids])}
    for name, dataset in domains.items():
        indices = select_target(dataset)
        selected[name] = (dataset, indices, collate)
        manifest[name] = dict(total=len(dataset), selected=[dataset.rows[i] for i in indices])
    write_json(outdir / 'manifest.json', manifest)
    write_json(outdir / 'protocol.json', dict(modes=MODES, checkpoints=[8, 4], batch_size=16,
               source_fraction=.025, target_per_class=128, training=False,
               interpretation='Fixed-weight component intervention; not a trained candidate or formal full-set EER. Source subset retains attack/class proportions; target subsets are fixed class-stratified samples. No target selection or optimization.',
               decomposition='M=A0 V=J V+(A0-J)V; intervene after output projection, retaining its bias and the residual/FFN. Diagnostic head features use float32 aggregation of BF16 Q/K/V.'))
    started = time.time()
    model = PretrainedAttentionSLS(c['ssl_checkpoint']).cuda().eval()
    hooks = AttentionDiagnostic(model)
    ckroot = ROOT / 'models/SLS_qk_tangent_control_seed20_20261014/pretrained_attention'
    allrows = []
    for epoch, file in [(8, 'best.pth'), (4, 'source_best_at_epoch4.pth')]:
        saved = torch.load(ckroot / file, map_location='cpu', weights_only=False)
        if saved['epoch'] != epoch:
            raise ValueError(('checkpoint epoch mismatch', saved['epoch'], epoch))
        model.load_trainable_payload(saved, c['expected_pretrained_sha256'])
        del saved
        for domain, (dataset, indices, collator) in selected.items():
            labels, names, stats, gates = [], [], [], []
            outputs = {mode: [] for mode in MODES}
            features = defaultdict(list)
            loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False,
                                num_workers=2, collate_fn=collator, pin_memory=True)
            with torch.inference_mode():
                for batch_index, (wave, y, fns) in enumerate(loader):
                    wave = wave.cuda(non_blocking=True)
                    labels.extend(np.asarray(y).tolist())
                    names.extend(fns)
                    for mode in MODES:
                        hooks.mode = mode
                        hooks.stats = []
                        hooks.features = {}
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            details = model(wave, return_details=(mode == 'full'))
                            logp = details['log_probs'] if mode == 'full' else details
                            if mode == 'full':
                                gate = torch.sigmoid(model.backend.fc0(model.backend.descriptors(details['states'])))
                                gates.append(gate.squeeze(-1).float().cpu().numpy())
                                features['final_state'].append(pool(details['states'][-1]).cpu().numpy())
                        if not bool(torch.isfinite(logp).all()):
                            raise FloatingPointError((epoch, domain, mode))
                        outputs[mode].extend(logp.float().cpu().numpy().tolist())
                        if mode == 'full':
                            stats.append(torch.stack(hooks.stats, 1).cpu().numpy())
                            for key, tensor in hooks.features.items():
                                features[key].append(tensor.cpu().numpy())
                        del details, logp
                    del wave
                    if batch_index % 10 == 0:
                        print('PROGRESS', epoch, domain, len(labels), '/', len(indices), 'seconds', round(time.time()-started, 1), flush=True)
            for mode in MODES:
                row = dict(epoch=epoch, domain=domain, mode=mode,
                           **metrics(labels, outputs[mode], outputs['full']))
                allrows.append(row)
                print('RESULT', json.dumps(row), flush=True)
            np.savez_compressed(outdir / f'epoch{epoch}_{domain}.npz',
                labels=np.asarray(labels), names=np.asarray(names), modes=np.asarray(MODES),
                log_probs=np.asarray([outputs[m] for m in MODES], dtype=np.float32),
                projected_component_rms=np.concatenate(stats), gates=np.concatenate(gates),
                **{k: np.concatenate(v).astype(np.float32) for k, v in features.items()})
            write_json(outdir / 'results.json', allrows)
            with (outdir / 'results.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(allrows[0]))
                writer.writeheader()
                writer.writerows(allrows)
    write_json(outdir / 'status.json', dict(status='complete', seconds=time.time()-started, training=False))
    print('COMPLETE', round(time.time()-started, 1), flush=True)


if __name__ == '__main__':
    main()
