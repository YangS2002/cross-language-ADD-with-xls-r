"""Inference-only routing swap, not a separately trained frozen-QK baseline."""
import argparse
import json
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if not (out / 'status.json').is_file():
        raise RuntimeError('Run component diagnostic first')
    torch.set_num_threads(4)
    torch.manual_seed(20)
    np.random.seed(20)
    c = yaml.safe_load((ROOT / 'experiments/configs/sls_qk_tangent_original_seed20.yaml').read_text())
    source, _ = dataset_for(c)
    indices, _ = select_source(source)
    selected = {'source_train': (source, indices, source_collate)}
    for name, dataset in make_domains(c).items():
        selected[name] = (dataset, select_target(dataset), collate)
    model = PretrainedAttentionSLS(c['ssl_checkpoint']).cuda().eval()
    model.guidance_enabled = False
    rows = []
    ckroot = ROOT / 'models/SLS_qk_tangent_control_seed20_20261014/pretrained_attention'
    for epoch, filename in [(8, 'best.pth'), (4, 'source_best_at_epoch4.pth')]:
        saved = torch.load(ckroot / filename, map_location='cpu', weights_only=False)
        if saved['epoch'] != epoch:
            raise ValueError('Checkpoint epoch mismatch')
        model.load_trainable_payload(saved, c['expected_pretrained_sha256'])
        del saved
        for domain, (dataset, indices, collator) in selected.items():
            labels, names, scores = [], [], []
            loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False,
                                num_workers=2, collate_fn=collator, pin_memory=True)
            with torch.inference_mode():
                for wave, y, fns in loader:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logp = model(wave.cuda(non_blocking=True))
                    scores.extend(logp.float().cpu().numpy().tolist())
                    labels.extend(np.asarray(y).tolist())
                    names.extend(fns)
            with np.load(out / f'epoch{epoch}_{domain}.npz') as reference:
                assert np.array_equal(reference['labels'], labels)
                assert np.array_equal(reference['names'], names)
                row = dict(epoch=epoch, domain=domain, mode='inference_only_frozen_projection',
                           **metrics(labels, scores, reference['log_probs'][0]))
            rows.append(row)
            np.savez_compressed(out / f'epoch{epoch}_{domain}_routing_swap.npz',
                                labels=np.asarray(labels), names=np.asarray(names),
                                log_probs=np.asarray(scores, dtype=np.float32))
            write_json(out / 'routing_swap.json', dict(
                interpretation='Teacher removed only at inference. Original Q/K weights act on task states; task/backend parameters unchanged. This is NOT frozen-QK retraining.',
                results=rows))
            print('ROUTING_SWAP', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
