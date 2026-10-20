"""Original-QK endpoint attribution: parameter deltas and crossed backbone/readout."""
import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_qk_uniform_relation import (
    select_source, select_target, source_collate, collate, dataset_for,
    make_domains, metrics, write_json, PretrainedAttentionSLS)


def parameter_group(name, backbone):
    if not backbone:
        return name.split('.')[0]
    for token, label in [('self_attn.v_proj', 'value'), ('self_attn.out_proj', 'attention_out'),
                         ('.fc1.', 'ffn'), ('.fc2.', 'ffn'), ('layer_norm', 'normalization'),
                         ('pos_conv', 'position')]:
        if token in name:
            return label
    return 'other'


def parameter_deltas(current, reference, backbone):
    groups = defaultdict(lambda: dict(parameters=0, changed_exact=0,
                                     reference_squared_norm=0., delta_squared_norm=0.))
    for name, p in current.named_parameters():
        if backbone and not p.requires_grad:
            continue
        ref = reference[name].to(device=p.device, dtype=p.dtype)
        delta = p.detach() - ref
        label = parameter_group(name, backbone)
        values = dict(parameters=p.numel(), changed_exact=int(torch.count_nonzero(delta)),
                      reference_squared_norm=float(ref.square().sum()),
                      delta_squared_norm=float(delta.square().sum()))
        for group in (label, 'TOTAL'):
            for key, value in values.items():
                groups[group][key] += value
    for values in groups.values():
        values['reference_norm'] = values['reference_squared_norm'] ** .5
        values['delta_norm'] = values['delta_squared_norm'] ** .5
        values['relative_l2'] = values['delta_norm'] / max(values['reference_norm'], 1e-30)
        values['changed_exact_fraction'] = values['changed_exact'] / values['parameters']
    return dict(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--full-domain', choices=['ch', 'de', 'uk', 'es', 'itw'])
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20)
    np.random.seed(20)
    started = time.time()
    c = yaml.safe_load((ROOT / 'experiments/configs/sls_qk_tangent_original_seed20.yaml').read_text())
    source, _ = dataset_for(c)
    indices, _ = select_source(source)
    selected = {'source_train': (source, indices, source_collate)}
    for name, dataset in make_domains(c).items():
        selected[name] = (dataset, select_target(dataset), collate)
    if args.full_domain:
        dataset = selected[args.full_domain][0]
        selected = {args.full_domain: (dataset, list(range(len(dataset))), collate)}
    models = {}
    ckroot = ROOT / 'models/SLS_qk_tangent_control_seed20_20261014/pretrained_attention'
    for epoch, filename in [(4, 'source_best_at_epoch4.pth'), (8, 'best.pth')]:
        model = PretrainedAttentionSLS(c['ssl_checkpoint']).cuda().eval()
        saved = torch.load(ckroot / filename, map_location='cpu', weights_only=False)
        if saved['epoch'] != epoch:
            raise ValueError('Checkpoint mismatch')
        model.load_trainable_payload(saved, c['expected_pretrained_sha256'])
        del saved
        models[epoch] = model
    warm = torch.load(c['shared_warmup'], map_location='cpu', weights_only=False)
    heads = {'g4': models[4].backend, 'g8': models[8].backend,
             'g3': copy.deepcopy(models[4].backend)}
    heads['g3'].load_state_dict(warm['backend'])
    original = models[4].reference.state_dict()
    deltas = {}
    with torch.no_grad():
        for epoch, model in models.items():
            deltas[str(epoch)] = dict(
                backbone_from_pretrained=parameter_deltas(model.task, original, True),
                backend_from_warmup=parameter_deltas(model.backend, warm['backend'], False))
        deltas['4_to_8'] = dict(
            backbone=parameter_deltas(models[8].task, models[4].task.state_dict(), True),
            backend=parameter_deltas(models[8].backend, models[4].backend.state_dict(), False))
    del original, warm
    for epoch, model in models.items():
        deltas[str(epoch)]['batchnorm'] = {
            key: value.detach().cpu().tolist()
            for key, value in model.backend.first_bn.state_dict().items()}
    deltas['3'] = dict(batchnorm={key:value.detach().cpu().tolist()
                                for key,value in heads['g3'].first_bn.state_dict().items()})
    write_json(out / 'parameter_deltas.json', deltas)
    for target, source_epoch in [('g8_bn4', 4), ('g4_bn8', 8)]:
        heads[target] = copy.deepcopy(heads['g8' if target.startswith('g8') else 'g4'])
        # Only running buffers are swapped; affine BN parameters stay with the head.
        with torch.no_grad():
            for buffer in ('running_mean', 'running_var', 'num_batches_tracked'):
                getattr(heads[target].first_bn, buffer).copy_(
                    getattr(models[source_epoch].backend.first_bn, buffer))
    captured = {}
    models[4].reference.register_forward_hook(lambda module, inputs, output: captured.update(F0=output))
    combos = [(f, g) for f in ('F0', 'F4', 'F8') for g in ('g3', 'g4', 'g8')]
    combos += [(f, g) for f in ('F4', 'F8') for g in ('g8_bn4', 'g4_bn8')]
    write_json(out / 'protocol.json', dict(
        backbone_states=['F0: frozen original', 'F4: original-QK epoch4', 'F8: original-QK epoch8'],
        heads=['g3: source warmup', 'g4: epoch4', 'g8: epoch8', 'g8_bn4/g4_bn8: only BN running buffers crossed'],
        combinations=combos, training=False,
        input_protocol=('Full official ' + args.full_domain + ' protocol' if args.full_domain
                        else 'Same manifest as results_qk_uniform_relation_20261020'),
        caveat='Endpoint hybrids may be off-distribution. This separates inference compatibility, not independent causal contributions of training. Relative parameter norms are not functional attribution.'))
    rows = []
    previous = ROOT / 'experiments/results_qk_uniform_relation_20261020'
    for domain, (dataset, indices, collator) in selected.items():
        labels, names = [], []
        outputs = {f + '_' + g: [] for f, g in combos}
        parity = []
        loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False,
                            num_workers=2, collate_fn=collator, pin_memory=True)
        with torch.inference_mode():
            for wave, y, fns in loader:
                wave = wave.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    d4 = models[4](wave, return_details=True)
                    d8 = models[8](wave, return_details=True)
                    features = {'F0': captured['F0'], 'F4': d4['states'], 'F8': d8['states']}
                    for f, g in combos:
                        logp = heads[g](features[f])
                        if f == 'F4' and g == 'g4':
                            parity.append(float((logp - d4['log_probs']).abs().max()))
                        if f == 'F8' and g == 'g8':
                            parity.append(float((logp - d8['log_probs']).abs().max()))
                        outputs[f + '_' + g].extend(logp.float().cpu().numpy().tolist())
                labels.extend(np.asarray(y).tolist())
                names.extend(fns)
                captured.clear()
                del wave, d4, d8, features, logp
        saved_outputs = np.asarray([outputs[f + '_' + g] for f, g in combos], dtype=np.float32)
        baseline = outputs['F4_g4']
        for f, g in combos:
            row = dict(domain=domain, backbone=f, head=g,
                       **metrics(labels, outputs[f + '_' + g], baseline))
            rows.append(row)
        parity_saved = {}
        for epoch in (4, 8):
            if args.full_domain:
                import csv
                old_path = ROOT / f'models/SLS_pretrained_attention_seed20/scores_{domain}_epoch{epoch}.csv'
                with old_path.open() as handle:
                    old_scores = {row['name']: (int(row['label']), float(row['score']))
                                  for row in csv.DictReader(handle)}
                assert all(old_scores[n][0] == y for n, y in zip(names, labels))
                current_scores = np.asarray(outputs[f'F{epoch}_g{epoch}'])[:, 1]
                parity_saved[str(epoch)] = float(np.abs(current_scores -
                    np.asarray([old_scores[n][1] for n in names])).max())
            else:
                with np.load(previous / f'epoch{epoch}_{domain}.npz') as old:
                    assert np.array_equal(old['names'], names)
                    assert np.array_equal(old['labels'], labels)
                    parity_saved[str(epoch)] = float(np.abs(
                        np.asarray(outputs[f'F{epoch}_g{epoch}']) - old['log_probs'][0]).max())
        np.savez_compressed(out / f'{domain}.npz', labels=np.asarray(labels), names=np.asarray(names),
                            combinations=np.asarray([f + '_' + g for f, g in combos]), log_probs=saved_outputs)
        write_json(out / 'results.json', rows)
        write_json(out / f'{domain}_parity.json', dict(original_forward_max_abs=max(parity),
                   previous_diagnostic_max_abs=parity_saved))
        print('ATTRIBUTION', domain, [(f + '_' + g, round(metrics(labels, outputs[f + '_' + g], baseline)['eer'], 4))
                                     for f, g in combos], 'parity',max(parity),parity_saved,flush=True)
    write_json(out / 'status.json', dict(status='complete', seconds=time.time()-started, training=False))
    print('COMPLETE',round(time.time()-started,1),flush=True)


if __name__ == '__main__':
    main()
