"""One candidate, seed20 first; reusable source teacher, automatic poor-result stop."""
import argparse
import csv
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tools.sls_dependency_control_model import DependencyControlledSLS, optimizers, step
from tools.sls_content_adversarial_model import balanced_content_kl
from tools.train_relational_preservation import (atomic_json, atomic_save, append_csv,
    digest, dataset_for, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains
from tools.train_sls_reference import evaluate
from train import set_seed, make_train_loader


PAPER_SLS=dict(ch=12.26,de=19.71,uk=21.13,es=9.69,itw=7.46)
PAPER_MAMBA=dict(ch=17.32,de=23.57,uk=11.89,es=15.83,itw=6.71)


def cached_rows(names,cache):
    ids=torch.tensor([cache['lookup'][name] for name in names])
    return cache['q'][ids].cuda(non_blocking=True),cache['valid'][ids].cuda(non_blocking=True)


def calibrate(model,dataset,cache,c):
    """Necessary reader initialization on detached RAM embeddings, NOT another detector arm."""
    ids=sorted(range(len(dataset)),key=lambda i:hashlib.sha256(dataset.filenames[i].encode()).hexdigest())[:c['calibration_samples']]
    loader=DataLoader(Subset(dataset,ids),batch_size=c['batch_size'],shuffle=False,
                      num_workers=c['num_workers'],pin_memory=True)
    embeddings=[]; labels=[]; targets=[]; masks=[]; start=time.perf_counter()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for b in loader:
            d=model(b['audio'].cuda(non_blocking=True),return_details=True)
            q,valid=cached_rows(b['file_name'],cache)
            embeddings.append(d['embedding'].float().detach());labels.append(b['label'].long().cuda())
            targets.append(q);masks.append(valid)
    e=torch.cat(embeddings);y=torch.cat(labels);q=torch.cat(targets);valid=torch.cat(masks)
    inputs=torch.cat([e,torch.nn.functional.one_hot(y,2).to(e)],1)
    opt=torch.optim.AdamW(model.content.parameters(),lr=c['calibration_lr'],weight_decay=c['weight_decay'],foreach=False)
    with torch.no_grad(): initial=float(balanced_content_kl(model.content(inputs),q,y,valid))
    for epoch in range(c['calibration_passes']):
        for i in range(0,len(inputs),64):
            opt.zero_grad(set_to_none=True)
            loss=balanced_content_kl(model.content(inputs[i:i+64]),q[i:i+64],y[i:i+64],valid[i:i+64])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.content.parameters(),5.,error_if_nonfinite=True);opt.step()
    with torch.no_grad(): final=float(balanced_content_kl(model.content(inputs),q,y,valid))
    result=dict(samples=len(inputs),real=int((y==1).sum()),passes=c['calibration_passes'],
        initial_kl=initial,final_kl=final,seconds=time.perf_counter()-start,
        note='Reader calibration loss only; not proof of content causality or independent validation')
    if not np.isfinite(final):raise FloatingPointError('Calibration failed')
    print('CALIBRATION',json.dumps(result),flush=True)
    del e,y,q,valid,inputs,opt,embeddings,labels,targets,masks
    gc.collect();torch.cuda.empty_cache()
    return result


def cpu_payload(model,sha):
    def clone(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
        return value
    return clone(model.trainable_payload(sha))


def train_epoch(model,main,aux,dataset,cache,c,epoch,deadline):
    model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    loader=make_train_loader(dataset,c['batch_size'],c['seed']+100000*epoch,c['num_workers'],torch.device('cuda'))
    count=real=correct=0;sums={k:0. for k in ('classification_loss','reader_kl','sensitivity','gradient_norm')}
    class_count={y:0 for y in (0,1)};class_correct={y:0 for y in (0,1)};margins={y:0. for y in (0,1)}
    for i,b in enumerate(loader,1):
        if time.monotonic()>deadline:raise TimeoutError('Predeclared wall-time budget exceeded')
        y=b['label'].view(-1).long().cuda(non_blocking=True)
        q,valid=cached_rows(b['file_name'],cache)
        out,stats=step(model,main,aux,b['audio'].cuda(non_blocking=True),y,q,valid,c['dependency_weight'])
        n=y.numel();count+=n;real+=int((y==1).sum());correct+=int((out.argmax(1)==y).sum())
        for key in sums:sums[key]+=stats[key]*n
        for label in (0,1):
            mask=y==label;class_count[label]+=int(mask.sum())
            class_correct[label]+=int((out.argmax(1)[mask]==label).sum())
            signed=(out[:,1]-out[:,0])*(2*y-1)
            margins[label]+=float(signed[mask].sum())
        if i==20:
            seconds=time.perf_counter()-start
            print('TIMING',json.dumps(dict(epoch=epoch,steps=i,seconds=seconds,
                estimated_epoch_seconds=seconds/i*len(loader),estimate_only=True,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
        if i%200==0:print('PROGRESS',epoch,i,count,{k:v/count for k,v in sums.items()},flush=True)
    if (count,real)!=(25380,2580):raise RuntimeError('Incomplete source epoch')
    return dict(train_loss=sums['classification_loss']/count,classification_loss=sums['classification_loss']/count,
        reader_kl=sums['reader_kl']/count,sensitivity=sums['sensitivity']/count,
        gradient_norm=sums['gradient_norm']/count,accuracy=correct/count,
        recall_real=class_correct[1]/class_count[1],recall_fake=class_correct[0]/class_count[0],
        margin_real=margins[1]/class_count[1],margin_fake=margins[0]/class_count[0],
        samples=count,real=real,fake=count-real,steps=i,seconds=time.perf_counter()-start,
        reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def decide(rows,c,epoch):
    values={r['domain']:r['eer'] for r in rows}
    with open(c['matched_baseline_results'],newline='') as f:
        baseline={r['domain']:float(r['eer']) for r in csv.DictReader(f)
                  if r['arm']=='sls' and int(r['milestone'])==epoch}
    if set(baseline)!=set(values):raise ValueError('Matched stage baseline missing')
    cross=('ch','de','uk','es');gaps={k:values[k]-baseline[k] for k in values}
    mean=float(np.mean([values[k] for k in cross]));base_mean=float(np.mean([baseline[k] for k in cross]))
    severe=((mean-base_mean>3 and sum(gaps[k]>5 for k in cross)>=2)
        or (gaps['itw']>3 and mean>=base_mean)
        or any(values[k]>35 and gaps[k]>8 for k in cross))
    no_effect=epoch>=8 and (mean>=base_mean or values['itw']>c['itw_eer_limit'])
    # Only an expansion gate, not a claim that all seeds/languages are solved.
    gate=(epoch==c['epochs'] and mean<base_mean and mean<np.mean([PAPER_SLS[k] for k in cross])
        and values['itw']<=c['itw_eer_limit'] and max(values[k] for k in cross)<=25
        and all(values[k]<=PAPER_SLS[k]+3 for k in cross))
    return dict(epoch=epoch,stop=bool(severe or no_effect),reason='severe_degradation' if severe else
        'epoch8_or_later_no_effect' if no_effect else 'continue_or_complete',
        per_domain=values,matched_sls=baseline,gaps=gaps,cross_mean=mean,matched_cross_mean=base_mean,
        worst_cross=max(values[k] for k in cross),ch_de_joint=(values['ch']+values['de'])/2,
        paper_sls_gaps={k:values[k]-PAPER_SLS[k] for k in values},
        paper_mamba_gaps={k:values[k]-PAPER_MAMBA[k] for k in values},
        eligible_for_fixed_multiseed=bool(gate),single_seed_is_not_stability_evidence=True)


def run(c,config_path):
    root=Path(c['output_dir']).resolve()
    if not root.is_relative_to(Path('/root/rivermind-data')):raise ValueError('Persistent disk only')
    root.mkdir(parents=True,exist_ok=True);(root/'dependency').mkdir(exist_ok=True)
    if (root/'experiment.json').exists():raise RuntimeError('No automatic replay of an existing run')
    sha=digest(c['ssl_checkpoint'])
    if sha!=c['expected_pretrained_sha256'] or digest(c['shared_warmup'])!=c['expected_warmup_sha256']:
        raise ValueError('Original/warmup signature mismatch')
    if shutil.disk_usage(root).free<3*2**30:raise RuntimeError('Need room for one best plus atomic replacement')
    cache=torch.load(c['content_cache'],map_location='cpu',weights_only=False)
    if cache['metadata']['base_sha256']!=sha or cache['metadata']['source_protocol_sha256']!=digest(c['train_label_path']):
        raise ValueError('Content cache signature mismatch')
    if cache['metadata']['rows']!=25380:raise ValueError('Incomplete source cache')
    cache['lookup']={n:i for i,n in enumerate(cache['file_names'])}
    dataset,_=dataset_for(c)
    if set(dataset.filenames)!=set(cache['file_names']):raise ValueError('Source/cache IDs mismatch')
    domains=make_domains(c)
    atomic_json(root/'inventory.json',[d.validate_paths() for d in domains.values()])
    atomic_json(root/'experiment.json',dict(config=c,signature=dict(config=digest(config_path),
        trainer=digest(__file__),model=digest('tools/sls_dependency_control_model.py'),
        original=sha,content_cache=digest(c['content_cache'])),
        target_use='full evaluation and user-authorized termination only; no checkpoint/head/seed selection',
        recovery='one best, no Adam state; no exact resume',single_candidate=True))
    deadline=time.monotonic()+c['max_run_seconds']
    set_seed(c['seed']);model=DependencyControlledSLS(c['ssl_checkpoint']).cuda()
    warm=torch.load(c['shared_warmup'],map_location='cpu',weights_only=False)
    model.backend.load_state_dict(warm['backend'])
    atomic_json(root/'status.json',dict(status='calibrating',seed=c['seed']))
    atomic_json(root/'calibration.json',calibrate(model,dataset,cache,c))
    main,aux=optimizers(model,c);restore_rng(warm['rng']);del warm
    best=float('inf');best_epoch=None
    for epoch in range(c['warmup_epochs']+1,c['epochs']+1):
        atomic_json(root/'status.json',dict(status='training',seed=c['seed'],epoch=epoch))
        metrics=train_epoch(model,main,aux,dataset,cache,c,epoch,deadline)
        selected=metrics['train_loss']<best
        if selected:
            best=metrics['train_loss'];best_epoch=epoch
            atomic_save(root/'dependency/best.pth',dict(**model.trainable_payload(sha),epoch=epoch,loss=best))
        row=dict(arm='dependency',seed=c['seed'],epoch=epoch,**metrics,selected=selected,
            free_bytes=shutil.disk_usage(root).free)
        append_csv(root/'epoch_results.csv',row);print('EPOCH',json.dumps(row),flush=True)
        if epoch in c['milestones']:
            if time.monotonic()>deadline:raise TimeoutError('Budget before evaluation')
            atomic_json(root/'status.json',dict(status='evaluating',seed=c['seed'],epoch=epoch))
            last=cpu_payload(model,sha);state=rng_state()
            rows=evaluate(model,c,domains,'dependency',epoch,sha)
            decision=decide(rows,c,epoch);atomic_json(root/f'decision_epoch{epoch}.json',decision)
            print('DECISION',json.dumps(decision),flush=True)
            if decision['stop']:
                atomic_json(root/'status.json',dict(status='stopped_poor_results',seed=c['seed'],epoch=epoch,
                    best_epoch=best_epoch,best_source_loss=best,reason=decision['reason']))
                print('STOPPED_POOR_RESULTS',flush=True);return
            model.load_trainable_payload(last,sha);restore_rng(state);del last
    atomic_json(root/'completed_dependency.json',dict(seed=c['seed'],epoch=c['epochs'],best_epoch=best_epoch,best_loss=best))
    atomic_json(root/'status.json',dict(status='complete',seed=c['seed'],epoch=c['epochs'],best_epoch=best_epoch))
    print('ALL_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    c=yaml.safe_load(Path(a.config).read_text())
    if (Path(c['output_dir'])/'experiment.json').exists():
        raise RuntimeError('Existing experiment; do not overwrite its status or restart')
    try:run(c,a.config)
    except Exception as error:
        root=Path(c['output_dir']).resolve()
        if root.is_relative_to(Path('/root/rivermind-data')):
            prior=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
            atomic_json(root/'status.json',dict(status='stopped_budget' if isinstance(error,TimeoutError) else 'failed',
                error=repr(error),previous_status=prior,automatic_restart=False))
        print('FAILED',repr(error),flush=True);raise


if __name__=='__main__':main()
