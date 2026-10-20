"""Seed20 source-only experiment; three sequential arms, no optimizer disk copies."""
import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import Subset
from tools.sls_conditional_residual_model import ConditionalResidualSLS
from tools.train_relational_preservation import (atomic_json, atomic_save, append_csv,
    digest, dataset_for, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains
from tools.train_sls_reference import evaluate
from train import set_seed, make_train_loader


def fit_id(name):
    return int(hashlib.sha256(str(name).encode()).hexdigest(), 16) % 5 != 0


def optimizers(model, c):
    groups = [{"params":list(model.backend.parameters()), "lr":c["head_lr"]}]
    ssl = [p for p in model.task.parameters() if p.requires_grad]
    if ssl:
        groups.append({"params":ssl, "lr":c["ssl_lr"]})
    if model.update is not None:
        groups.append({"params":list(model.update.parameters()), "lr":c["head_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=c["weight_decay"], foreach=False)
    ropt = None if model.predictor is None else torch.optim.AdamW(
        model.predictor.parameters(), lr=c["predictor_lr"], weight_decay=c["weight_decay"], foreach=False)
    return opt, ropt


def reconstruction(details):
    if details["prediction"] is None:
        return None
    z = details["aggregate"].detach().float()
    return (details["prediction"].float()-z).square().mean((1,2)) / (z.square().mean((1,2))+1e-8)


def step(model, opt, ropt, wave, y, names):
    opt.zero_grad(set_to_none=True)
    if ropt is not None:
        ropt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model(wave, return_details=True)
    out = d["log_probs"]
    loss = torch.nn.functional.nll_loss(out, y, weight=out.new_tensor([0.1,0.9]))
    error = reconstruction(d)
    fit = torch.tensor([fit_id(n) for n in names], device=y.device) & (y==1)
    if not bool(torch.isfinite(loss)) or (error is not None and not bool(torch.isfinite(error).all())):
        raise FloatingPointError("Nonfinite classification/reconstruction")
    loss.backward()
    if model.predictor is not None and any(p.grad is not None for p in model.predictor.parameters()):
        raise RuntimeError("Classification leaked into predictor")
    params = [p for group in opt.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
    opt.step()
    if ropt is not None and bool(fit.any()):
        error[fit].mean().backward()
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(),5.,error_if_nonfinite=True)
        ropt.step()
    with torch.no_grad():
        znorm = d["aggregate"].float().norm(dim=-1)
        ratio = d["delta"].float().norm(dim=-1)/(znorm+1e-8)
        if float(ratio.max()) > 0.102:
            raise RuntimeError("Actual correction exceeded norm bound")
        residual = (d["aggregate"].float() if d["prediction"] is None else
                    d["aggregate"].float()-d["prediction"].detach().float()).norm(dim=-1).mean(1)
    return loss.detach(), out.detach(), dict(error=None if error is None else error.detach(),
        fit=fit, delta_ratio=ratio.mean(1).detach(), residual_norm=residual.detach(), norm=float(norm))


def train_epoch(model,opt,ropt,dataset,c,arm,epoch):
    model.train(); torch.cuda.reset_peak_memory_stats()
    loader=make_train_loader(dataset,c["batch_size"],c["seed"]+100000*epoch,c["num_workers"],torch.device("cuda"))
    count=real=correct=0; total=0.; start=time.perf_counter()
    sums={k:0. for k in ("rec_fit","rec_holdout","residual_real","residual_fake","delta_ratio","nll_real","nll_fake")}
    counts={k:0 for k in sums}
    for i,b in enumerate(loader,1):
        y=b["label"].view(-1).long().cuda(non_blocking=True)
        loss,out,stats=step(model,opt,ropt,b["audio"].cuda(non_blocking=True),y,b["file_name"])
        n=y.numel(); count+=n; real+=int((y==1).sum()); total+=float(loss)*n
        correct+=int((out.argmax(1)==y).sum())
        values={"residual_real":(stats["residual_norm"],y==1),
                "residual_fake":(stats["residual_norm"],y==0),
                "delta_ratio":(stats["delta_ratio"],torch.ones_like(y,dtype=torch.bool)),
                "nll_real":(-out.gather(1,y[:,None]).squeeze(1),y==1),
                "nll_fake":(-out.gather(1,y[:,None]).squeeze(1),y==0)}
        if stats["error"] is not None:
            values.update(rec_fit=(stats["error"],stats["fit"]),
                          rec_holdout=(stats["error"],(y==1)&~stats["fit"]))
        for key,(value,mask) in values.items():
            sums[key]+=float(value[mask].sum()); counts[key]+=int(mask.sum())
        if i%200==0:
            print("PROGRESS",arm,epoch,i,count,total/count,flush=True)
    if (count,real)!=(25380,2580):
        raise RuntimeError("Incomplete source coverage")
    return dict(train_loss=total/count,accuracy=correct/count,samples=count,real=real,
        fake=count-real,steps=i,seconds=time.perf_counter()-start,
        reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        **{k:sums[k]/counts[k] if counts[k] else None for k in sums})


def calibrate(model,ropt,dataset,c):
    indices=[i for i,y in enumerate(dataset.synthesizerlabels)
             if y=="bonafide" and fit_id(dataset.filenames[i])]
    loader=make_train_loader(Subset(dataset,indices),c["batch_size"],c["seed"]+90000,
                             c["num_workers"],torch.device("cuda"))
    model.eval(); losses=[]; start=time.perf_counter()
    for b in loader:
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            d=model(b["audio"].cuda(non_blocking=True),return_details=True)
        ropt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            d["prediction"]=model.predictor(d["reference"].detach())
        loss=reconstruction(d).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Calibration nonfinite")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.predictor.parameters(),5.,error_if_nonfinite=True)
        ropt.step(); losses.append(float(loss))
    record=dict(samples=len(indices),passes=1,first_loss=losses[0],last_loss=losses[-1],
                seconds=time.perf_counter()-start)
    print("CALIBRATION",model.arm,json.dumps(record),flush=True)
    return record


def cpu_payload(model,sha):
    def clone(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:clone(v) for k,v in value.items()}
        return value
    return clone(model.trainable_payload(sha))


def self_test(c,dataset):
    half=c["batch_size"]//2
    ids=([i for i,y in enumerate(dataset.synthesizerlabels) if y=="bonafide"][:half]+
         [i for i,y in enumerate(dataset.synthesizerlabels) if y!="bonafide"][:c["batch_size"]-half])
    samples=[dataset[i] for i in ids]
    wave=torch.stack([b["audio"] for b in samples]).cuda()
    labels=torch.tensor([int(b["label"]) for b in samples],device="cuda")
    names=[b["file_name"] for b in samples]; rows=[]; initial=backend=None
    for arm in c["arms"]:
        set_seed(c["seed"]); model=ConditionalResidualSLS(c["ssl_checkpoint"],arm).cuda().train()
        if backend is None: backend={k:v.detach().cpu().clone() for k,v in model.backend.state_dict().items()}
        model.backend.load_state_dict(backend)
        opt,ropt=optimizers(model,c); torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            loss,out,stats=step(model,opt,ropt,wave,labels,names)
            if i==0:
                if initial is None: initial=out.cpu()
                else: torch.testing.assert_close(initial,out.cpu(),atol=2e-4,rtol=2e-4)
            for layer in (0,23):
                assert any(p.grad is not None and bool(p.grad.abs().sum()>0)
                           for p in model.task.model.encoder.layers[layer].parameters())
            assert model.backend.fc0.weight.grad is not None
            if model.reference is not None:
                assert all(p.grad is None for p in model.reference.parameters())
                assert model.reference_calls==list(range(1,17))
            print("SELF_TEST_STEP",arm,i,float(loss),flush=True)
        if model.predictor is not None:
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): d=model(wave,return_details=True)
            reconstruction(d)[labels==1].mean().backward()
            assert all(p.grad is None for p in model.task.parameters())
            assert all(p.grad is None for p in model.backend.parameters())
            assert all(p.grad is None for p in model.update.parameters())
        model_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
        row=dict(arm=arm,batch=c["batch_size"],model_bytes=model_bytes,
                 reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        rows.append(row); print("SELF_TEST_MEMORY",json.dumps(row),flush=True)
        del model,opt,ropt,out,loss,stats
        if "d" in locals(): del d
        gc.collect(); torch.cuda.empty_cache()
    budget=int(max(r["model_bytes"] for r in rows)*4.1)+512*1024**2
    if max(r["reserved_gib"] for r in rows)>20 or shutil.disk_usage(c["output_dir"]).free<budget:
        raise RuntimeError(f"Resource preflight failed: disk budget {budget}")
    atomic_json(Path(c["output_dir"])/"self_test.json",dict(arms=rows,disk_budget_bytes=budget))
    print("SELF_TEST_OK",flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True)
    p.add_argument("--self-test",action="store_true"); args=p.parse_args()
    c=yaml.safe_load(Path(args.config).read_text()); root=Path(c["output_dir"])
    if not root.resolve().is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent disk only")
    root.mkdir(parents=True,exist_ok=True); dataset,_=dataset_for(c)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported")
    if args.self_test: self_test(c,dataset); return
    test=json.loads((root/"self_test.json").read_text())
    if any(r["batch"]!=c["batch_size"] for r in test["arms"]): raise RuntimeError("Retest batch")
    if (root/"experiment.json").exists(): raise RuntimeError("Existing run; no automatic replay")
    domains=make_domains(c)
    atomic_json(root/"inventory.json",[d.validate_paths() for d in domains.values()])
    sha=digest(c["ssl_checkpoint"])
    atomic_json(root/"experiment.json",dict(config=c,signature=dict(config=digest(args.config),
        model=digest("tools/sls_conditional_residual_model.py"),trainer=digest(__file__),
        original=sha,source_labels=digest(c["train_label_path"])),
        execution="sequential complete arms; CPU last-state during evaluation; NO disk Adam resumes",
        target_use="fixed evaluations and exploratory stopping only; no target selection"))
    atomic_json(root/"source_predictor_split.json",dict(rule="sha256 filename mod5 !=0 genuine fit",
        fit=sum(y=="bonafide" and fit_id(n) for n,y in zip(dataset.filenames,dataset.synthesizerlabels)),
        genuine=2580,classification_samples=25380))
    set_seed(c["seed"]); model=ConditionalResidualSLS(c["ssl_checkpoint"],"sls").cuda()
    model.task.requires_grad_(False); opt,ropt=optimizers(model,c)
    for epoch in range(1,c["warmup_epochs"]+1):
        atomic_json(root/"status.json",dict(status="training",arm="warmup",epoch=epoch))
        row=dict(arm="warmup",seed=c["seed"],epoch=epoch,
                 **train_epoch(model,opt,ropt,dataset,c,"warmup",epoch),
                 selected=False,free_bytes=shutil.disk_usage(root).free)
        append_csv(root/"epoch_results.csv",row); print("EPOCH",json.dumps(row),flush=True)
    warm={k:v.detach().cpu().clone() for k,v in model.backend.state_dict().items()}
    common_rng=rng_state(); atomic_save(root/"warmup.pth",dict(backend=warm,rng=common_rng))
    del model,opt,ropt; gc.collect(); torch.cuda.empty_cache()
    for arm in c["arms"]:
        (root/arm).mkdir(exist_ok=True)
        set_seed(c["seed"]); model=ConditionalResidualSLS(c["ssl_checkpoint"],arm).cuda()
        model.backend.load_state_dict(warm); opt,ropt=optimizers(model,c)
        if ropt is not None:
            atomic_json(root/"status.json",dict(status="calibrating",arm=arm))
            atomic_json(root/f"calibration_{arm}.json",calibrate(model,ropt,dataset,c))
        restore_rng(common_rng); best=float("inf")
        for epoch in range(c["warmup_epochs"]+1,c["epochs"]+1):
            atomic_json(root/"status.json",dict(status="training",arm=arm,epoch=epoch))
            metrics=train_epoch(model,opt,ropt,dataset,c,arm,epoch)
            selected=metrics["train_loss"]<best; best=min(best,metrics["train_loss"])
            if selected:
                atomic_save(root/arm/"best.pth",dict(**model.trainable_payload(sha),epoch=epoch,loss=best))
            row=dict(arm=arm,seed=c["seed"],epoch=epoch,**metrics,selected=selected,
                     free_bytes=shutil.disk_usage(root).free)
            append_csv(root/"epoch_results.csv",row); print("EPOCH",json.dumps(row),flush=True)
            if epoch in c["milestones"]:
                atomic_json(root/"status.json",dict(status="evaluating",arm=arm,epoch=epoch))
                last=cpu_payload(model,sha); state=rng_state()
                rows=evaluate(model,c,domains,arm,epoch,sha)
                if arm=="pretrained_residual" and epoch==4:
                    model.correction_enabled=False
                    diagnostic=dict(c,output_dir=str(root/"correction_off_diagnostic"))
                    target=Path(diagnostic["output_dir"])/arm; target.mkdir(parents=True,exist_ok=True)
                    # Tiny symlink only; weights remain in the registered best path.
                    (target/"best.pth").symlink_to(root/arm/"best.pth")
                    evaluate(model,diagnostic,domains,arm,epoch,sha)
                    model.correction_enabled=True
                model.load_trainable_payload(last,sha); restore_rng(state); del last
                if arm=="pretrained_residual":
                    gaps={}
                    for other in ("sls","self_residual"):
                        baseline=json.loads((root/f"eval_{other}_epoch{epoch}.json").read_text())
                        lookup={r["domain"]:r["eer"] for r in baseline}
                        gaps[other]={r["domain"]:r["eer"]-lookup[r["domain"]] for r in rows}
                    def failed(gap):
                        cross=[v for k,v in gap.items() if k!="itw"]
                        return (np.mean(cross)>3 and sum(v>5 for v in cross)>=2) or (epoch>=8 and np.mean(cross)>=0 and gap["itw"]>2)
                    # User revoked target-result stopping on 2026-10-07.
                    # Keep diagnostics, but complete the fixed epoch budget.
                    atomic_json(root/f"decision_epoch{epoch}.json",
                                dict(stop=False,gaps=gaps,policy="report_only_no_target_early_stop"))
        atomic_json(root/f"completed_{arm}.json",dict(arm=arm,epoch=c["epochs"],best_loss=best))
        del model,opt,ropt; gc.collect(); torch.cuda.empty_cache()
    atomic_json(root/"status.json",dict(status="complete",seed=c["seed"]))
    print("ALL_COMPLETE",flush=True)


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print("FAILED",type(exc).__name__,str(exc),flush=True)
        raise
