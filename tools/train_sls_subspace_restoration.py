"""One fixed-budget PSR candidate, reusing warmup; no target stopping or replay."""
import argparse
import csv
import gc
import json
import math
import shutil
import time
from pathlib import Path
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tools.sls_subspace_restoration_model import SubspaceRestoredSLS
from tools.train_relational_preservation import (atomic_json,atomic_save,append_csv,digest,
    dataset_for,rng_state,restore_rng)
from tools.evaluate_four_domains_shared_xlsr import collate,eer
from tools.evaluate_relational_preservation import make_domains
from train import set_seed,make_train_loader


ARM="psr"


def signatures(c,config):
    return dict(config=digest(config),model=digest("tools/sls_subspace_restoration_model.py"),
        trainer=digest(__file__),test=digest("tools/test_sls_subspace_restoration.py"),
        original=digest(c["ssl_checkpoint"]),warmup=digest(c["shared_warmup"]),
        loader=digest("train.py"),data=digest("data.py"),
        shared_model=digest("tools/sls_reference_model.py"),
        source_labels=digest(c["train_label_path"]),baseline=digest(c["matched_baseline_results"]))


def prepare(c,path):
    root=Path(c["output_dir"])
    if not root.resolve().is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent disk only")
    if (c["seed"],c["rank"],c["warmup_epochs"],c["epochs"],c["milestones"])!=(20,16,3,10,[4,8,10]):
        raise ValueError("Only this preregistered single-seed candidate is enabled")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 required")
    if (root/"experiment.json").exists(): raise RuntimeError("Existing run; no automatic replay")
    root.mkdir(parents=True,exist_ok=True)
    pressure=json.loads((root/"pressure_test.json").read_text())
    signature=signatures(c,path)
    if signature["original"]!=c["expected_pretrained_sha256"] or signature["warmup"]!=c["expected_warmup_sha256"]:
        raise ValueError("Original/warmup mismatch")
    for key in ("config","model","test","warmup"):
        if pressure["signature"][key]!=signature[key]:
            raise ValueError(f"Stale pressure test: {key}")
    if pressure["batch"]!=c["batch_size"] or pressure["rank"]!=c["rank"]:
        raise ValueError("Stale batch/rank test")
    previous=json.loads((Path(c["shared_warmup"]).parent/"experiment.json").read_text())
    for key in ("seed","train_data_path","train_label_path","ssl_checkpoint","warmup_epochs",
                "epochs","milestones","batch_size","eval_batch_size","num_workers",
                "head_lr","ssl_lr","weight_decay","domains"):
        if c[key]!=previous["config"][key]: raise ValueError(f"Fixed baseline protocol changed: {key}")
    if signature["source_labels"]!=previous["signature"]["source_labels"]:
        raise ValueError("Source labels changed")
    with open(c["matched_baseline_results"],newline="") as f:
        baseline=list(csv.DictReader(f))
    expected={(str(e),name) for e in c["milestones"] for name in c["domains"]}
    actual={(r["milestone"],r["domain"]) for r in baseline if r["arm"]=="sls" and int(r["seed"])==20}
    if not expected.issubset(actual): raise ValueError("Missing existing fixed SLS comparison")
    domains=make_domains(c); inventory=[d.validate_paths() for d in domains.values()]
    atomic_json(root/"inventory.json",inventory)
    counts={"ch":18124,"de":17696,"uk":9709,"es":13655,"itw":31779}
    evaluation_batches=sum(math.ceil(counts[name]/c["eval_batch_size"]) for name in c["domains"])
    epoch_steps=math.ceil(25380/c["batch_size"])
    training_minutes=epoch_steps*(pressure["cold_training_seconds_per_batch"]+
        (c["epochs"]-c["warmup_epochs"]-1)*pressure["seconds_per_step"])/60
    evaluation_minutes=evaluation_batches*(pressure["cold_inference_seconds_per_batch"]+
        (len(c["milestones"])-1)*pressure["inference_seconds_per_batch"])/60
    cache_bytes=25380*pressure["source_cache_bytes_per_input"]+sum(counts.values())*pressure["evaluation_cache_bytes_per_input"]
    ram_limit=int(Path("/sys/fs/cgroup/memory.max").read_text())
    ram_current=int(Path("/sys/fs/cgroup/memory.current").read_text())
    if cache_bytes>32*1024**3 or ram_current+cache_bytes+12*1024**3>ram_limit-10*1024**3:
        raise RuntimeError("Reference RAM cache plus process/reserve exceeds cgroup budget")
    # Source-batch compute timing excludes target I/O, which was material in prior runs.
    io_margin_minutes=10.; startup_save_margin_minutes=5.
    estimated=training_minutes+evaluation_minutes+io_margin_minutes+startup_save_margin_minutes
    disk_budget=int(pressure["model_bytes"]*2.05)+512*1024**2
    if shutil.disk_usage(root).free<disk_budget: raise RuntimeError("Atomic best replacement disk budget insufficient")
    record=dict(signature=signature,training_minutes=training_minutes,evaluation_compute_minutes=evaluation_minutes,
        io_margin_minutes=io_margin_minutes,startup_save_margin_minutes=startup_save_margin_minutes,
        estimated_total_minutes=estimated,maximum_minutes=c["maximum_estimated_minutes"],
        within_budget=estimated<=c["maximum_estimated_minutes"],disk_budget_bytes=disk_budget,
        free_bytes=shutil.disk_usage(root).free,pressure=pressure,reference_cache_bytes=cache_bytes,
        ram_limit_bytes=ram_limit,ram_current_bytes=ram_current,
        policy="one seed20 candidate; no baseline rerun, no target early stop, no optimizer recovery")
    atomic_json(root/"preflight.json",record)
    print("PREFLIGHT",json.dumps(record),flush=True)
    return record,domains


def optimizer(model,c):
    return torch.optim.AdamW([
        dict(params=list(model.backend.parameters()),lr=c["head_lr"]),
        dict(params=[p for p in model.task.parameters() if p.requires_grad],lr=c["ssl_lr"])],
        weight_decay=c["weight_decay"],foreach=False)


def epoch(model,opt,dataset,c,number):
    model.train(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    loader=make_train_loader(dataset,c["batch_size"],c["seed"]+100000*number,c["num_workers"],torch.device("cuda"))
    sums={k:0. for k in ("restoration_ratio_real","restoration_ratio_fake","parallel_fraction_real",
        "parallel_fraction_fake","difference_ratio_real","difference_ratio_fake","explained",
        "protected_error","effective_rank","nll_real","nll_fake")}
    counts={k:0 for k in sums}; total=0.; count=real=correct=0
    params=[p for group in opt.param_groups for p in group["params"]]
    for i,b in enumerate(loader,1):
        y=b["label"].view(-1).long().cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            d=model(b["audio"].cuda(non_blocking=True),return_details=True,
                    cache_keys=[("source",str(name)) for name in b["file_name"]])
        out=d["log_probs"]
        loss=torch.nn.functional.nll_loss(out,y,weight=out.new_tensor([.1,.9]))
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite NLL")
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True)
        opt.step()
        with torch.no_grad(),torch.autocast("cuda",enabled=False):
            z=d["aggregate"].float(); delta=d["difference"].float(); projected=d["projected"].float()
            znorm=z.flatten(1).norm(dim=1); dnorm=delta.flatten(1).norm(dim=1)
            pnorm=projected.flatten(1).norm(dim=1)
            reference=d["original"].float(); q=d["basis"].float()
            protected=((d["corrected"].float()-reference)@q).flatten(1).norm(dim=1)
            relative_error=protected/((reference@q).flatten(1).norm(dim=1)+1e-8)
            if float(relative_error.max())>.005: raise RuntimeError("Subspace numerical invariant violated")
            nll=-out.gather(1,y[:,None]).squeeze(1)
            values={"explained":d["explained"],"protected_error":relative_error,
                    "effective_rank":d["effective_rank"].float()}
            for key,value in [("restoration_ratio",pnorm/(znorm+1e-8)),
                              ("parallel_fraction",pnorm.square()/(dnorm.square()+1e-8)),
                              ("difference_ratio",dnorm/(znorm+1e-8)),("nll",nll)]:
                for suffix,label in [("real",1),("fake",0)]:
                    mask=y==label; name=f"{key}_{suffix}"
                    sums[name]+=float(value[mask].sum()); counts[name]+=int(mask.sum())
            for name,value in values.items():
                sums[name]+=float(value.sum()); counts[name]+=y.numel()
        n=y.numel(); count+=n; real+=int((y==1).sum()); correct+=int((out.argmax(1)==y).sum())
        total+=float(loss)*n
        if i%200==0: print("PROGRESS",number,i,count,total/count,flush=True)
        del d,out,loss,z,delta,projected,reference,q,protected
    if (count,real)!=(25380,2580): raise RuntimeError("Incomplete source epoch")
    return dict(train_loss=total/count,accuracy=correct/count,samples=count,real=real,fake=count-real,
        steps=i,seconds=time.perf_counter()-started,reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        grad_norm_last=float(norm),**{k:sums[k]/counts[k] for k in sums})


def cpu_snapshot(model,sha):
    def clone(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:clone(v) for k,v in value.items()}
        return value
    return clone(model.trainable_payload(sha))


def evaluate(model,c,domains,arm,number,sha):
    root=Path(c["output_dir"]); path=root/f"eval_{arm}_epoch{number}.json"
    if path.exists(): raise RuntimeError("Evaluation already exists; no silent replay")
    saved=torch.load(root/arm/"best.pth",map_location="cpu",weights_only=False)
    model.load_trainable_payload(saved,sha); model.eval(); rows=[]
    with torch.no_grad():
        for name,domain in domains.items():
            scores=[]; labels=[]; started=time.perf_counter()
            loader=DataLoader(domain,batch_size=c["eval_batch_size"],shuffle=False,
                num_workers=c["num_workers"],collate_fn=collate,pin_memory=True)
            for wave,y,names in loader:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    out=model(wave.cuda(non_blocking=True),cache_keys=[(name,str(n)) for n in names])
                scores.extend(out[:,1].float().cpu().tolist()); labels.extend(y.tolist())
            value,threshold=eer(np.asarray(labels),scores)
            row=dict(arm=arm,seed=c["seed"],milestone=number,checkpoint_epoch=saved["epoch"],
                domain=name,eer=value,threshold=threshold,samples=len(labels),seconds=time.perf_counter()-started)
            rows.append(row); print("EVAL",json.dumps(row),flush=True)
    atomic_json(path,rows)
    for row in rows: append_csv(root/"evaluation_results.csv",row)
    del saved
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--run",action="store_true")
    args=p.parse_args(); c=yaml.safe_load(Path(args.config).read_text()); root=Path(c["output_dir"])
    preflight,domains=prepare(c,args.config)
    if not args.run: return
    if not preflight["within_budget"]: raise RuntimeError("Estimated full fixed budget exceeds limit; no training started")
    torch.set_num_threads(4)
    dataset,_=dataset_for(c); set_seed(c["seed"])
    model=SubspaceRestoredSLS(c["ssl_checkpoint"],c["rank"]).cuda()
    shared=torch.load(c["shared_warmup"],map_location="cpu",weights_only=False)
    model.initialize_shared_backend(shared["backend"]); state=shared["rng"]; del shared
    opt=optimizer(model,c); restore_rng(state)
    atomic_json(root/"experiment.json",dict(config=c,signature=preflight["signature"],
        policy="all epochs4-10; min source classification loss; NO target early stop; NO automatic restart",
        original_subspace="per utterance PCA; no target labels/cohort statistics; not verified pure content"))
    (root/ARM).mkdir(exist_ok=True); sha=preflight["signature"]["original"]
    best=float("inf"); began=time.perf_counter()
    for number in range(c["warmup_epochs"]+1,c["epochs"]+1):
        atomic_json(root/"status.json",dict(status="training",epoch=number,arm=ARM))
        metrics=epoch(model,opt,dataset,c,number); selected=metrics["train_loss"]<best
        best=min(best,metrics["train_loss"])
        if selected: atomic_save(root/ARM/"best.pth",dict(**model.trainable_payload(sha),epoch=number,loss=best))
        row=dict(arm=ARM,seed=c["seed"],epoch=number,**metrics,selected=selected,
            free_bytes=shutil.disk_usage(root).free)
        append_csv(root/"epoch_results.csv",row); print("EPOCH",json.dumps(row),flush=True)
        if number in c["milestones"]:
            atomic_json(root/"status.json",dict(status="evaluating",epoch=number,arm=ARM))
            last=cpu_snapshot(model,sha); state=rng_state()
            evaluate(model,c,domains,ARM,number,sha)
            model.load_trainable_payload(last,sha); restore_rng(state); del last
        elapsed=(time.perf_counter()-began)/60
        if elapsed>c["maximum_estimated_minutes"]:
            atomic_json(root/"budget_overrun.json",dict(minutes=elapsed,epoch=number,
                action="report only; do not change fixed epoch budget or auto restart"))
            print("BUDGET_OVERRUN",elapsed,flush=True)
    atomic_json(root/"completed.json",dict(seed=c["seed"],epochs=c["epochs"],best_loss=best,
        minutes=(time.perf_counter()-began)/60))
    atomic_json(root/"status.json",dict(status="complete",seed=c["seed"]))
    print("ALL_COMPLETE",flush=True)
    del model,opt; gc.collect(); torch.cuda.empty_cache()


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print("FAILED",type(exc).__name__,str(exc),flush=True)
        raise
