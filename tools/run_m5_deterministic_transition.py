"""M5: frozen transition statistics and independently fitted convex heads."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_metrics_DF as em


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def csv_write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def eer(labels, scores):
    return float(100 * em.compute_eer(scores[labels == 1], scores[labels == 0])[0])


def checked_collate(items):
    from data import simple_collate_fn
    for item in items:
        if item['sample_rate'] != 16000 or item['audio'].ndim != 1:
            raise ValueError(f"Unsupported sample rate/channels: {item['file_name']}")
        if not torch.isfinite(item['audio']).all():
            raise ValueError(f"Nonfinite audio: {item['file_name']}")
    return simple_collate_fn(items)


def energy_rows(batch):
    wave = batch['audio'].float()
    frame_rms = wave[:, :64000].reshape(-1, 200, 320).square().mean(-1).sqrt()
    low = frame_rms <= frame_rms.amax(-1, keepdim=True).clamp_min(1e-12) * 0.01
    prefix = wave[:, :8000].square().mean(-1).sqrt()
    full = wave.square().mean(-1).sqrt()
    rows = []
    for i, name in enumerate(batch['file_name']):
        rows.append(dict(file_name=name, label=int(batch['label'][i]),
                         attack=str(batch['vocoder_n'][i]),
                         prefix_rms=float(prefix[i]), full_rms=float(full[i]),
                         prefix_to_full=float(prefix[i] / full[i].clamp_min(1e-12)),
                         low_energy_fraction=float(low[i].float().mean()),
                         prefix_low_energy_fraction=float(low[i, :25].float().mean()),
                         near_zero_fraction=float((frame_rms[i] <= 1e-4).float().mean())))
    return rows


def summarize_energy(rows, out):
    metrics = ['prefix_rms', 'prefix_to_full', 'low_energy_fraction',
               'prefix_low_energy_fraction', 'near_zero_fraction']
    groups = {}
    for row in rows:
        for key in (f"label_{row['label']}", 'attack_' + row['attack']):
            groups.setdefault(key, []).append(row)
    summary = {'n': len(rows), 'note': 'Descriptive source association; no VAD or silence removal.'}
    summary['groups'] = {key: {'n': len(group), **{
        metric: {'mean': float(np.mean([r[metric] for r in group])),
                 'median': float(np.median([r[metric] for r in group]))}
        for metric in metrics}} for key, group in groups.items()}
    labels = np.array([r['label'] for r in rows])
    summary['source_label_auc'] = {
        metric: float(roc_auc_score(labels, [r[metric] for r in rows])) for metric in metrics}
    dump(out / 'source_energy_audit.json', summary)
    csv_write(out / 'source_energy_samples.csv', rows)


def extract(config, split, backbone, out, signature):
    from data import DetectDataset
    args = SimpleNamespace(algo=5, exp_name=config['exp_name'])
    dataset = DetectDataset(args, config[f'{split}_data_path'],
                            config[f'{split}_label_path'], stage='eval')
    if len(set(dataset.filenames)) != len(dataset):
        raise ValueError(f'Duplicate protocol filenames: {split}')
    labels = np.array([int(dataset.synthesizers[s] == 0)
                       for s in dataset.synthesizerlabels], dtype=np.int64)
    cache = out / f'{split}_features.npy'
    meta_path = out / f'{split}_cache.json'
    key = dict(signature=signature, split=split, n=len(dataset),
               protocol=digest(config[f'{split}_label_path']),
               root=config[f'{split}_data_path'], feature=config['feature'],
               amp=config['amp'], batch_size=config['batch_size'])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['key'] != key or not cache.exists():
            raise RuntimeError('Cache provenance mismatch; use a new output directory')
        print(f'REUSE {split}: {len(dataset)} examples', flush=True)
        return cache, labels
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], collate_fn=checked_collate,
                        pin_memory=True)
    temporary = out / f'{split}_features.partial.npy'
    matrix, offset, audit = None, 0, []
    started = time.time()
    for step, batch in enumerate(loader):
        wave = batch['audio'].cuda(non_blocking=True)
        with torch.no_grad(), torch.autocast('cuda', enabled=config['amp']):
            layers = backbone.extract_layers(wave)
        if matrix is None:
            shape = (len(layers) - 1, len(dataset), 2 * layers[0].shape[-1])
            matrix = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
        for layer in range(len(layers) - 1):
            delta = layers[layer + 1].float() - layers[layer].float()
            delta = F.layer_norm(delta, (delta.shape[-1],))
            stat = torch.cat([delta.mean(1), delta.var(1, unbiased=False).clamp_min(0).sqrt()], -1)
            if not torch.isfinite(stat).all():
                raise RuntimeError('Nonfinite transition statistics')
            matrix[layer, offset:offset + len(wave)] = stat.cpu().numpy()
        if split == 'train':
            audit.extend(energy_rows(batch))
        offset += len(wave)
        if step % 100 == 0 or offset == len(dataset):
            print(f'EXTRACT {split} {offset}/{len(dataset)} {time.time()-started:.1f}s', flush=True)
            dump(out / 'status.json', dict(stage='extract', split=split, completed=offset, total=len(dataset)))
        del layers, delta, stat, wave
    assert offset == len(dataset)
    matrix.flush()
    del matrix
    os.replace(temporary, cache)
    if split == 'train':
        summarize_energy(audit, out)
    np.save(out / f'{split}_labels.npy', labels)
    dump(out / f'{split}_files.json', list(dataset.filenames))
    dump(meta_path, dict(key=key, shape=shape, seconds=time.time()-started))
    return cache, labels


class LogisticObjective:
    def __init__(self, x, labels, sample_weight, l2):
        self.x = x
        self.y = labels.to(x.device, dtype=x.dtype)[None, :]
        self.a = sample_weight.to(x.device, dtype=x.dtype)[None, :]
        self.a = self.a / self.a.sum()
        self.l2 = l2
        self.layers, _, self.dim = x.shape
        self.best_loss = float('inf')
        self.best_params = None
        self.best_bce = None
        self.last_params = None

    def __call__(self, params):
        if self.last_params is not None and np.array_equal(params, self.last_params):
            return self.last_value, self.last_gradient.copy()
        p = torch.as_tensor(params, device=self.x.device, dtype=self.x.dtype)
        w = p[:self.layers*self.dim].reshape(self.layers, self.dim)
        b = p[self.layers*self.dim:]
        z = torch.bmm(self.x, w.unsqueeze(-1)).squeeze(-1) + b[:, None]
        ce = ((F.softplus(z) - self.y*z) * self.a).sum(1)
        value = (ce + 0.5*self.l2*(w.square().sum(1) + b.square())).sum()
        r = (z.sigmoid() - self.y) * self.a
        grad_w = torch.bmm(r.unsqueeze(1), self.x).squeeze(1) + self.l2*w
        grad_b = r.sum(1) + self.l2*b
        gradient = torch.cat([grad_w.flatten(), grad_b]).cpu().numpy()
        value = float(value)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite objective/gradient')
        if value < self.best_loss:
            self.best_loss = value
            self.best_params = params.copy()
            self.best_bce = float(ce.mean())
        self.last_params = params.copy()
        self.last_value, self.last_gradient = value, gradient.copy()
        return value, gradient


def fit(cache_path, labels, config, out, seed, mean, scale):
    cache = np.load(cache_path, mmap_mode='r')
    layers, n, dim = cache.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    x = torch.empty(cache.shape, dtype=torch.float64, device='cuda')
    for layer in range(layers):
        values = np.asarray(cache[layer, order], dtype=np.float64)
        values -= mean[layer]
        values /= scale[layer]
        x[layer].copy_(torch.from_numpy(values))
    y = torch.as_tensor(labels[order])
    weights = torch.where(y == 1, config['real_weight'], config['fake_weight']).double()
    objective = LogisticObjective(x, y, weights, config['l2'])
    initial = rng.normal(0, config['initial_std'], layers*(dim+1))
    history = []
    start = time.time()

    def callback(params):
        loss, gradient = objective(params)
        history.append(dict(iteration=len(history)+1, train_loss=loss/layers,
                            gradient_max=float(np.max(np.abs(gradient))), seconds=time.time()-start))
        if len(history) % 20 == 0:
            print(f'FIT seed={seed} {history[-1]}', flush=True)
            csv_write(out / f'seed_{seed}_training.csv', history)
            dump(out / 'status.json', dict(stage='fit', seed=seed, **history[-1]))

    result = minimize(objective, initial, method='L-BFGS-B', jac=True, callback=callback,
                      options=dict(maxiter=config['max_iter'], maxcor=20,
                                   gtol=config['gradient_tolerance'], ftol=1e-14, maxls=50))
    params = objective.best_params.copy()
    loss, gradient = objective(params)
    w = params[:layers*dim].reshape(layers, dim)
    b = params[layers*dim:]
    grad_max = float(np.max(np.abs(gradient)))
    checkpoint = out / f'seed_{seed}_best_train_loss.npz'
    np.savez(checkpoint, weight=w, bias=b, mean=mean, scale=scale,
             train_loss=loss/layers, train_bce=objective.best_bce, l2=config['l2'])
    info = dict(seed=seed, iterations=int(result.nit), evaluations=int(result.nfev),
                train_loss=loss/layers, train_bce=objective.best_bce,
                gradient_max=grad_max, solver_success=bool(result.success),
                message=str(result.message), converged=grad_max <= 1e-5,
                seconds=time.time()-start, checkpoint=str(checkpoint))
    dump(out / f'seed_{seed}_fit.json', info)
    if history:
        csv_write(out / f'seed_{seed}_training.csv', history)
    print('FIT_DONE ' + json.dumps(info), flush=True)
    del objective, x
    gc.collect()
    torch.cuda.empty_cache()
    return info


def evaluate(caches, labels, config, out, fit_info):
    all_scores, metrics = {}, []
    for seed in config['seeds']:
        with np.load(out / f'seed_{seed}_best_train_loss.npz') as ckpt:
            w, b, mean, scale = [ckpt[k] for k in ('weight', 'bias', 'mean', 'scale')]
        for split, path in caches.items():
            cache = np.load(path, mmap_mode='r')
            scores = np.empty((cache.shape[1], cache.shape[0]), dtype=np.float64)
            for layer in range(cache.shape[0]):
                for offset in range(0, cache.shape[1], 2048):
                    values = np.asarray(cache[layer, offset:offset+2048], dtype=np.float64)
                    scores[offset:offset+len(values), layer] = (
                        (values-mean[layer])/scale[layer]) @ w[layer] + b[layer]
                metrics.append(dict(seed=seed, split=split, transition=layer+1,
                                    eer=eer(labels[split], scores[:, layer])))
            fused = scores.mean(1)
            all_scores[seed, split] = fused
            metrics.append(dict(seed=seed, split=split, transition='equal_mean',
                                eer=eer(labels[split], fused)))
            np.savez(out / f'seed_{seed}_{split}_scores.npz', labels=labels[split], scores=scores, fused=fused)
    csv_write(out / 'metrics.csv', metrics)
    comparisons = {}
    s1, s2 = config['seeds']
    for split in caches:
        a, b = all_scores[s1, split], all_scores[s2, split]
        comparisons[split] = dict(correlation=float(np.corrcoef(a,b)[0,1]),
                                  max_abs_score_difference=float(np.max(np.abs(a-b))),
                                  eer_difference=abs(eer(labels[split],a)-eer(labels[split],b)))
    main = [r for r in metrics if r['transition'] == 'equal_mean']
    stability = all(f['converged'] for f in fit_info) and all(
        comparisons[s]['correlation'] >= 0.9999 and comparisons[s]['eer_difference'] <= .05
        for s in ('dev', 'test'))
    candidate = True
    for seed in config['seeds']:
        m = {r['split']:r['eer'] for r in main if r['seed']==seed}
        candidate &= m['dev'] <= 8.7821 and m['test'] <= 17.2891 and (m['dev']+m['test'])/2 <= 13.3296
    summary = dict(fits=fit_info, fused_metrics=main, repeatability=comparisons,
                   stability_pass=bool(stability), performance_pass=bool(candidate),
                   candidate_pass=bool(stability and candidate),
                   feature=config['feature'], l2=config['l2'],
                   selection='Minimum source regularized training loss; fixed equal fusion; no target tuning')
    dump(out / 'summary.json', summary)
    dump(out / 'status.json', dict(stage='complete', candidate_pass=summary['candidate_pass']))
    print('COMPLETE ' + json.dumps(summary), flush=True)


def self_test():
    torch.manual_seed(7)
    x = torch.randn(2, 80, 5, dtype=torch.float64)
    y = (x[0,:,0] > 0).double()
    a = torch.where(y == 1, .9, .1).double()
    obj = LogisticObjective(x,y,a,.01)
    p = np.random.default_rng(9).normal(0,.1,12)
    v,g = obj(p)
    t = torch.tensor(p, requires_grad=True)
    w,b = t[:10].reshape(2,5),t[10:]
    z = torch.bmm(x,w.unsqueeze(-1)).squeeze(-1)+b[:,None]
    loss = (((F.softplus(z)-y*z)*(a/a.sum())).sum(1)+.005*(w.square().sum(1)+b.square())).sum()
    loss.backward()
    assert abs(v-float(loss)) < 1e-10
    assert np.max(np.abs(g-t.grad.numpy())) < 1e-10
    solutions=[]
    for seed in (20,1001):
        rng=np.random.default_rng(seed)
        order=rng.permutation(80)
        f=LogisticObjective(x[:,order],y[order],a[order],.01)
        result=minimize(f,rng.normal(0,.1,12),jac=True,method='L-BFGS-B',
                        options=dict(gtol=1e-9,ftol=1e-15,maxiter=500))
        assert np.max(np.abs(f(result.x)[1])) < 1e-7
        solutions.append(result.x)
    assert np.max(np.abs(solutions[0]-solutions[1])) < 1e-5
    assert eer(np.array([0,0,1,1]),np.array([-2.,-1.,1.,2.])) == 0
    print('SELF_TEST_OK: analytic gradient, permutation/initialization convergence, EER direction')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--self-test', action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        self_test()
        return
    config=yaml.safe_load(Path(args.config).read_text())
    if config['feature'] != 'normalized_adjacent_transition_mean_std':
        raise ValueError('Unexpected feature specification')
    out=Path(config['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    from src.model import XLSRBackbone
    signature={str(p.relative_to(ROOT)):digest(p) for p in
               [Path(__file__).resolve(), ROOT/'src/model.py',ROOT/'data.py']}
    signature['ssl_checkpoint_sha256']=digest(config['ssl_checkpoint'])
    dump(out/'provenance.json',dict(config=config,signature=signature,
                                   torch=torch.__version__,numpy=np.__version__))
    backbone=XLSRBackbone(config['ssl_checkpoint']).cuda().eval()
    caches,labels={},{}
    for split in ('train','dev','test'):
        caches[split],labels[split]=extract(config,split,backbone,out,signature)
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    source=np.load(caches['train'],mmap_mode='r')
    mean=np.stack([source[i].mean(0,dtype=np.float64) for i in range(source.shape[0])])
    scale=np.stack([source[i].std(0,dtype=np.float64) for i in range(source.shape[0])]).clip(1e-6)
    fits=[]
    for seed in config['seeds']:
        fits.append(fit(caches['train'],labels['train'],config,out,seed,mean,scale))
    evaluate(caches,labels,config,out,fits)


if __name__ == '__main__':
    main()
