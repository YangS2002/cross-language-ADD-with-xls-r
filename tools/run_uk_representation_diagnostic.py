"""One UK diagnostic: existing encoders, group-disjoint probes, no SSL training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse
import csv
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from tools.sls_conditional_residual_model import ConditionalResidualSLS
from tools.evaluate_four_domains_shared_xlsr import AudioDomain, DOMAINS, collate, eer, digest

REPO=Path('/root/rivermind-data/cross-language-ADD-with-xls-r')
OUT=REPO/'models/UK_representation_diagnostic_seed20'
BASE=REPO/'models/pretrained/xlsr2_300m.pt'
BEST=REPO/'models/SLS_conditional_residual_seed20/sls/best.pth'
WARM=REPO/'models/SLS_conditional_residual_seed20/warmup.pth'
STAGES=('layer08','layer16','layer24','sls_aggregate','classifier_embedding')

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));os.replace(tmp,path)

def status(stage,**extra): save(OUT/'status.json',dict(status=stage,**extra))

def hash_key(value): return hashlib.sha256(str(value).encode()).hexdigest()

def prepare():
    entries=dict(a=(Path('/root/rivermind-data/data/exp/ASVspoof2019.LA.cm.train.trn.txt'),
                    Path('/root/rivermind-data/data/asv2019_la_train'),None,''),
                 uk=(Path('/root/rivermind-data/datasets/mlaad_eval/mlaad_eval_uk.txt'),None,
                     Path('/root/rivermind-data/datasets/mlaad_eval/mlaad_eval_uk_clip.zip'),'data/ys/'))
    datasets={};records={}
    for domain,entry in entries.items():
        DOMAINS[domain]=entry; dataset=AudioDomain(domain)
        raw=[line.split() for line in entry[0].read_text().splitlines() if len(line.split())>=5]
        assert [r[1] for r in raw]==[r[0] for r in dataset.rows]
        selected=list(range(len(raw)))
        if domain=='a': selected=sorted(selected,key=lambda i:hash_key(raw[i][1]))[:6000]
        if domain=='a':
            groups=[r[0] for r in raw]; grouping='speaker_from_source_protocol'
        else:
            real={}
            for r in raw:
                if r[-1]=='bonafide': real.setdefault(Path(r[1]).stem,[]).append(r[1])
            books={};speakers={}
            for stem,paths in real.items():
                # Map every fake rendition to its genuine counterpart before splitting.
                parts=Path(paths[0]).parts
                idx=parts.index('by_book') if 'by_book' in parts else -1
                segment=parts[idx+1:-2] if idx>=0 else ()
                books[stem]='/'.join(segment) if segment else stem
                speakers[stem]='/'.join(segment[:-1]) if len(segment)>1 else stem
            paired=[Path(r[1]).stem for r in raw]
            if all(s in real and len(real[s])==1 for s in paired) and len(set(speakers.values()))>=8:
                groups=[speakers[s] for s in paired];grouping='speaker_path_via_genuine_pair'
            elif all(s in real and len(real[s])==1 for s in paired) and len(set(books.values()))>=8:
                groups=[books[s] for s in paired];grouping='book_reader_path_via_genuine_pair; speaker_overlap_possible'
            else:
                groups=paired;grouping='paired_sentence_stem; speaker_and_book_overlap_possible'
        y=np.array([dataset.rows[i][1] for i in selected]);g=np.array([groups[i] for i in selected])
        first=GroupShuffleSplit(n_splits=1,train_size=.6,random_state=20)
        fit,remaining=next(first.split(np.zeros(len(y)),y,g))
        second=GroupShuffleSplit(n_splits=1,train_size=.5,random_state=20)
        vi,ti=next(second.split(np.zeros(len(remaining)),y[remaining],g[remaining]))
        val,test=remaining[vi],remaining[ti]
        split=np.full(len(y),'',dtype='<U10');split[fit]='fit';split[val]='validation';split[test]='test'
        for ix in (fit,val,test):
            if len(set(y[ix]))!=2: raise RuntimeError(f'{domain}: split lacks one class')
        assert not (set(g[fit])&set(g[val]) or set(g[fit])&set(g[test]) or set(g[val])&set(g[test]))
        # A seen by encoder during its original training; only probe partitions are held out.
        rows=[dict(name=dataset.rows[i][0],label=int(y[j]),group=groups[i],split=split[j],
                   attack=raw[i][-2]) for j,i in enumerate(selected)]
        records[domain]=dict(rows=rows,grouping=grouping,label_sha256=digest(entry[0]),
            counts={s:dict(total=int((split==s).sum()),real=int(((split==s)&(y==1)).sum()),
                           fake=int(((split==s)&(y==0)).sum()),groups=len(set(g[split==s])))
                    for s in ('fit','validation','test')})
        datasets[domain]=Subset(dataset,selected)
    save(OUT/'split_manifest.json',records)
    return datasets,records

def summarise_frames(h):
    h=h.float();return torch.cat([h.mean(1),h.std(1,unbiased=False)],1)

def extract(model,domain,dataset,model_name,batch):
    loader=DataLoader(dataset,batch_size=batch,num_workers=4,shuffle=False,collate_fn=collate,pin_memory=True)
    chunks={k:[] for k in STAGES};native=[];names=[];labels=[];start=time.perf_counter()
    status('extracting',model=model_name,domain=domain)
    with torch.inference_mode():
        for i,(wave,y,fns) in enumerate(loader,1):
            wave=wave.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                states=model.task(wave);b=model.backend
                gates=torch.sigmoid(b.fc0(b.descriptors(states)))
                z=sum(h*gates[:,j:j+1] for j,h in enumerate(states))
                flat=F.max_pool2d(b.selu(b.first_bn(z.unsqueeze(1))),3,3).flatten(1)
                emb=b.selu(b.fc1(flat))
                out=F.log_softmax(b.selu(b.fc3(emb)).float(),1)
            if i==1:
                with torch.autocast('cuda',dtype=torch.bfloat16): expected=model(wave)
                torch.testing.assert_close(out,expected,rtol=1e-3,atol=2e-3)
            for layer in (8,16,24): chunks[f'layer{layer:02d}'].append(summarise_frames(states[layer-1]).cpu().numpy())
            chunks['sls_aggregate'].append(summarise_frames(z).cpu().numpy())
            chunks['classifier_embedding'].append(emb.float().cpu().numpy())
            native.extend(out[:,1].cpu().numpy().tolist());names.extend(fns);labels.extend(y.tolist())
            if i%100==0: print('EXTRACT',model_name,domain,len(names),round(time.perf_counter()-start,2),flush=True)
    values={k:np.concatenate(v) for k,v in chunks.items()}
    values.update(native=np.asarray(native),y=np.asarray(labels),names=np.asarray(names))
    np.savez(OUT/f'features_{model_name}_{domain}.npz',**values)
    print('EXTRACT_DONE',model_name,domain,len(names),time.perf_counter()-start,flush=True)
    return dict(model=model_name,domain=domain,samples=len(names),seconds=time.perf_counter()-start,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30)

def cka(x,y):
    # Same centered linear CKA, FP64; GPU avoids long CPU Gram products.
    with torch.inference_mode():
        x=torch.as_tensor(x,device='cuda',dtype=torch.float64)
        y=torch.as_tensor(y,device='cuda',dtype=torch.float64)
        x=x-x.mean(0);y=y-y.mean(0)
        xx=x@x.T;yy=y@y.T
        return float((xx*yy).sum()/(torch.linalg.vector_norm(xx)*torch.linalg.vector_norm(yy)).clamp_min(1e-20))

def bootstrap(y,score,groups,n=100):
    unique=np.unique(groups);rng=np.random.RandomState(20);values=[]
    index={g:np.flatnonzero(groups==g) for g in unique}
    for _ in range(n):
        ids=np.concatenate([index[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ids]))==2: values.append(eer(y[ids],score[ids])[0])
    return [float(v) for v in np.percentile(values,[2.5,97.5])] if values else None

def analyse(records):
    status('analysing');features={}
    for model in ('original','finetuned'):
        for domain in ('a','uk'):
            # NpzFile is lazy: indexing names inside a sort key otherwise rereads
            # the whole filename array for EVERY sample. Materialize once.
            with np.load(OUT/f'features_{model}_{domain}.npz') as saved:
                features[model,domain]={key:saved[key] for key in saved.files}
    similarities=[];results=[];predictions={}
    for domain in ('a','uk'):
        x,y=features['original',domain],features['finetuned',domain]
        assert np.array_equal(x['names'],y['names'])
        for stage in STAGES:
            for subset in ('both','real','fake'):
                ids=np.arange(len(x['y'])) if subset=='both' else np.flatnonzero(x['y']==int(subset=='real'))
                ids=sorted(ids,key=lambda i:hash_key(x['names'][i]))[:1000]
                similarities.append(dict(domain=domain,stage=stage,subset=subset,n=len(ids),cka=cka(x[stage][ids],y[stage][ids])))
        print('CKA_DONE',domain,flush=True)
    save(OUT/'cka.json',similarities)
    def record(kind,model,stage,trained_on,domain,split,score,converged=True,iterations=0):
        info=records[domain]['rows'];ids=np.array([i for i,r in enumerate(info) if r['split']==split])
        f=features[model,domain];y=f['y'][ids];groups=np.array([r['group'] for r in info])[ids]
        value,threshold=eer(y,score[ids]);auc=None
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(y,score[ids]))
        row=dict(kind=kind,model=model,stage=stage,trained_on=trained_on,evaluated_on=domain,
                 split=split,n=len(ids),real=int(y.sum()),eer=value,auc=auc,
                 eer_group_bootstrap_95=bootstrap(y,score[ids],groups) if split=='test' else None,
                 converged=converged,iterations=iterations)
        results.append(row);predictions['|'.join(map(str,(kind,model,stage,trained_on,domain,split)))]=score[ids].tolist()
        save(OUT/'results.json',dict(results=results,cka=similarities))
        print('RESULT',json.dumps(row),flush=True)
    for model in ('original','finetuned'):
        for domain in ('a','uk'):
            record('native',model,'classifier_embedding','a',domain,'test',features[model,domain]['native'])
        for stage in STAGES:
            for train_domain in ('a','uk'):
                train=features[model,train_domain]
                ids=np.array([i for i,r in enumerate(records[train_domain]['rows']) if r['split']=='fit'])
                scaler=StandardScaler();x=scaler.fit_transform(train[stage][ids]).astype(np.float64)
                clf=LogisticRegression(C=1.,solver='lbfgs',class_weight='balanced',max_iter=200,tol=1e-5,random_state=20)
                begin=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning);clf.fit(x,train['y'][ids])
                converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                if not converged: print('PROBE_NOT_CONVERGED',model,stage,train_domain,flush=True)
                test_domains=('a','uk') if train_domain=='a' else ('uk',)
                for domain in test_domains:
                    # A-fitted scaler used unchanged on UK; no target normalization for A probes.
                    score=clf.decision_function(scaler.transform(features[model,domain][stage]))
                    for split in ('validation','test'):
                        record('linear_probe',model,stage,train_domain,domain,split,score,converged,int(clf.n_iter_[0]))
                print('PROBE_SECONDS',model,stage,train_domain,time.perf_counter()-begin,flush=True)
    save(OUT/'predictions.json',predictions)
    rows=[]
    for row in results:
        rows.append({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    status('complete',samples={d:len(r['rows']) for d,r in records.items()},
        target_supervision='UK labels used ONLY for diagnostic probes; not a zero-target detector result',
        no_full_model_training=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=16);p.add_argument('--analyse-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.analyse_only:
        analyse(json.loads((OUT/'split_manifest.json').read_text()));return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'experiment.json').exists(): raise RuntimeError('Existing diagnostic; do not overwrite/replay')
    if not OUT.resolve().is_relative_to(Path('/root/rivermind-data')): raise RuntimeError('Persistent disk required')
    signature=dict(base=digest(BASE),best=digest(BEST),warmup=digest(WARM),script=digest(__file__))
    save(OUT/'experiment.json',dict(seed=20,signature=signature,stages=STAGES,source_cap=6000,
        uk='all protocol samples',probe='Fixed C=1, balanced logistic regression; no hyperparameter/layer selection',
        preprocessing='Unchanged data.pad64600; no resampling/no noise',
        limitations=['UK supervised diagnostic, not a formal zero-target result',
            'A encoder originally saw all A training samples; A probe test is only held out from the probe',
            'Original model uses frozen-warmup backend; later stage differences include backend adaptation',
            'Mean/std summaries may omit temporal information; poor linear probe does not prove information absent',
            'No epoch4 weights available; endpoint comparison, not temporal forgetting proof']))
    datasets,records=prepare();timings=[]
    for name in ('original','finetuned'):
        status('loading',model=name);model=ConditionalResidualSLS(str(BASE),'sls')
        if name=='original':
            warm=torch.load(WARM,map_location='cpu',weights_only=False);model.backend.load_state_dict(warm['backend']);del warm
        else:
            saved=torch.load(BEST,map_location='cpu',weights_only=False)
            model.load_trainable_payload(saved,signature['base'])
            save(OUT/'finetuned_checkpoint.json',dict(epoch=saved.get('epoch'),loss=saved.get('loss'),format=saved['format']))
            del saved
        model.requires_grad_(False);model.cuda().eval();torch.cuda.reset_peak_memory_stats()
        for domain,dataset in datasets.items(): timings.append(extract(model,domain,dataset,name,a.batch_size))
        del model;gc.collect();torch.cuda.empty_cache()
    save(OUT/'timings.json',timings);analyse(records)

if __name__=='__main__':
    try: main()
    except Exception as error:
        if OUT.exists(): status('failed',error=repr(error))
        raise
