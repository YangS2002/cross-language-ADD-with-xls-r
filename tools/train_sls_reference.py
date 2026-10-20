"""Sequential matched SLS reference arms; persistent disk only, seed20 first."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tools.sls_reference_model import ReferenceRoutedSLS
from tools.train_relational_preservation import (atomic_json, atomic_save, append_csv,
    digest, dataset_for, rng_state, restore_rng)
from tools.evaluate_relational_preservation import make_domains
from tools.evaluate_four_domains_shared_xlsr import collate, eer
from train import set_seed, make_train_loader


def optimizer(model, config):
    ssl = [p for p in model.task.parameters() if p.requires_grad]
    groups = [{"params": list(model.backend.parameters()), "lr": config["head_lr"]}]
    if ssl:
        groups.append({"params": ssl, "lr": config["ssl_lr"]})
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"], foreach=False)


def step(model, opt, waveform, labels):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(waveform)
    weights = output.new_tensor([0.1, 0.9])
    loss = torch.nn.functional.nll_loss(output, labels, weight=weights)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite NLL")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        5.0, error_if_nonfinite=True)
    opt.step()
    return loss.detach(), output.detach(), norm.detach()


def self_test(config, dataset):
    half=config["batch_size"]//2
    real=[i for i,y in enumerate(dataset.synthesizerlabels) if y=="bonafide"][:half]
    fake=[i for i,y in enumerate(dataset.synthesizerlabels) if y!="bonafide"][:config["batch_size"]-half]
    indices=real+fake
    wave = torch.stack([dataset[i]["audio"] for i in indices]).cuda()
    labels = torch.tensor([int(dataset[i]["label"]) for i in indices],
                          device="cuda")
    results = []
    initial = backend = None
    for arm in config["arms"]:
        set_seed(config["seed"])
        model = ReferenceRoutedSLS(config["ssl_checkpoint"], arm).cuda().train()
        if backend is None:
            backend = {k: v.detach().cpu().clone() for k,v in model.backend.state_dict().items()}
        model.backend.load_state_dict(backend)
        opt = optimizer(model, config)
        torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            loss, output, norm = step(model, opt, wave, labels)
            if i == 0:
                if initial is None:
                    initial = output.cpu()
                else:
                    torch.testing.assert_close(initial, output.cpu(), atol=1e-5, rtol=1e-5)
            for index in (0,23):
                assert any(p.grad is not None and bool(p.grad.abs().sum()>0)
                           for p in model.task.model.encoder.layers[index].parameters())
            if model.reference is not None:
                assert all(p.grad is None for p in model.reference.parameters())
            print("SELF_TEST_STEP",arm,i,float(loss),flush=True)
        model_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
        # Two arm bests, one resume plus replacement, warmup + conservative margin.
        disk_budget = int((2*model_bytes + 2*3*model_bytes)*1.05) + 768*1024**2
        record = {"arm":arm,"batch":config["batch_size"],"model_bytes":model_bytes,
                  "disk_peak_budget_bytes":disk_budget,
                  "reserved_gib":torch.cuda.max_memory_reserved()/1024**3}
        results.append(record)
        print("SELF_TEST_MEMORY",json.dumps(record),flush=True)
        del model,opt,output,loss,norm
        gc.collect(); torch.cuda.empty_cache()
    if max(r["reserved_gib"] for r in results)>20:
        raise RuntimeError("20GiB pressure-test limit exceeded")
    if shutil.disk_usage(config["output_dir"]).free < max(r["disk_peak_budget_bytes"] for r in results):
        raise RuntimeError("Insufficient persistent disk peak budget")
    atomic_json(Path(config["output_dir"])/"self_test.json",results)
    print("SELF_TEST_OK",flush=True)


def train_epoch(model,opt,dataset,c,arm,epoch):
    model.train(); torch.cuda.reset_peak_memory_stats()
    loader = make_train_loader(dataset,c["batch_size"],c["seed"]+100000*epoch,
                               c["num_workers"],torch.device("cuda"))
    total=count=real=correct=0
    started=time.perf_counter()
    for i,b in enumerate(loader,1):
        y=b["label"].view(-1).long().cuda(non_blocking=True)
        loss,out,norm=step(model,opt,b["audio"].cuda(non_blocking=True),y)
        n=y.numel(); count+=n; real+=int((y==1).sum())
        total+=float(loss)*n; correct+=int((out.argmax(1)==y).sum())
        if i%200==0: print("PROGRESS",arm,epoch,i,count,total/count,flush=True)
    if (count,real)!=(25380,2580): raise RuntimeError("Incomplete source epoch")
    return {"train_loss":total/count,"accuracy":correct/count,"samples":count,
            "real":real,"fake":count-real,"steps":i,"grad_norm_last":float(norm),
            "seconds":time.perf_counter()-started,
            "reserved_gib":torch.cuda.max_memory_reserved()/1024**3}


def evaluate(model,c,domains,arm,epoch,base_sha):
    root=Path(c["output_dir"]); path=root/f"eval_{arm}_epoch{epoch}.json"
    if path.exists(): return json.loads(path.read_text())
    saved=torch.load(root/arm/"best.pth",map_location="cpu",weights_only=False)
    model.load_trainable_payload(saved,base_sha); model.eval(); rows=[]
    with torch.no_grad():
        for name,domain in domains.items():
            scores=[]; labels=[]; start=time.perf_counter()
            loader=DataLoader(domain,batch_size=c["eval_batch_size"],shuffle=False,
                              num_workers=c["num_workers"],collate_fn=collate,pin_memory=True)
            for wave,y,_ in loader:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    out=model(wave.cuda(non_blocking=True))
                scores.extend(out[:,1].float().cpu().tolist()); labels.extend(y.tolist())
            value,threshold=eer(np.asarray(labels),scores)
            row={"arm":arm,"seed":c["seed"],"milestone":epoch,"checkpoint_epoch":saved["epoch"],
                 "domain":name,"eer":value,"threshold":threshold,"samples":len(labels),
                 "seconds":time.perf_counter()-start}
            rows.append(row); print("EVAL",json.dumps(row),flush=True)
    atomic_json(path,rows)
    for row in rows: append_csv(root/"evaluation_results.csv",row)
    del saved
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    parser.add_argument("--self-test",action="store_true"); args=parser.parse_args()
    c=yaml.safe_load(Path(args.config).read_text()); root=Path(c["output_dir"])
    if not root.resolve().is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent data disk only")
    root.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 required")
    dataset,_=dataset_for(c)
    if args.self_test: self_test(c,dataset); return
    pressure=json.loads((root/"self_test.json").read_text())
    if any(r["batch"]!=c["batch_size"] for r in pressure): raise RuntimeError("Retest batch")
    domains=make_domains(c)
    atomic_json(root/"inventory.json",[d.validate_paths() for d in domains.values()])
    base_sha=digest(c["ssl_checkpoint"])
    signature={"config":digest(args.config),"model":digest("tools/sls_reference_model.py"),
               "trainer":digest(__file__),"labels":digest(c["train_label_path"]),"pretrained":base_sha}
    meta=root/"experiment.json"
    if meta.exists() and json.loads(meta.read_text())["signature"]!=signature:
        raise RuntimeError("Changed run signature")
    atomic_json(meta,{"config":c,"signature":signature,"execution":"sequential arms; one resume",
                      "score":"bona-fide log probability","target_use":"exploratory stopping"})
    resume=root/"active_resume.pth"; warm=root/"warmup.pth"
    if not warm.exists():
        set_seed(c["seed"]); model=ReferenceRoutedSLS(c["ssl_checkpoint"],"self_reference").cuda()
        model.task.requires_grad_(False); opt=optimizer(model,c)
        for epoch in range(1,c["warmup_epochs"]+1):
            row={"arm":"warmup","seed":c["seed"],"epoch":epoch,
                 **train_epoch(model,opt,dataset,c,"warmup",epoch)}
            append_csv(root/"epoch_results.csv",row); print("EPOCH",json.dumps(row),flush=True)
        atomic_save(warm,{"backend":model.backend.state_dict(),"rng":rng_state()})
        del model,opt; gc.collect(); torch.cuda.empty_cache()
    for arm in c["arms"]:
        done=root/f"completed_{arm}.json"
        if done.exists(): continue
        (root/arm).mkdir(exist_ok=True)
        saved=(torch.load(resume,map_location="cpu",weights_only=False) if resume.exists() else None)
        if saved and saved["reference_source"]!=arm: raise RuntimeError("Unexpected active resume")
        set_seed(c["seed"]); model=ReferenceRoutedSLS(c["ssl_checkpoint"],arm).cuda()
        opt=optimizer(model,c)
        best=float("inf"); start=c["warmup_epochs"]+1
        if saved:
            model.load_trainable_payload(saved,base_sha); opt.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"]); best=saved["best_loss"]; start=saved["epoch"]+1
        else:
            shared=torch.load(warm,map_location="cpu",weights_only=False)
            model.backend.load_state_dict(shared["backend"]); restore_rng(shared["rng"]); del shared
        del saved
        for epoch in range(start,c["epochs"]+1):
            metrics=train_epoch(model,opt,dataset,c,arm,epoch)
            selected=metrics["train_loss"]<best; best=min(best,metrics["train_loss"])
            payload={**model.trainable_payload(base_sha),"epoch":epoch,"loss":metrics["train_loss"]}
            if selected: atomic_save(root/arm/"best.pth",payload)
            atomic_save(resume,{**payload,"best_loss":best,"optimizer":opt.state_dict(),"rng":rng_state()})
            row={"arm":arm,"seed":c["seed"],"epoch":epoch,**metrics,"selected":selected,
                 "free_bytes":shutil.disk_usage(root).free}
            append_csv(root/"epoch_results.csv",row); print("EPOCH",json.dumps(row),flush=True)
            del payload
            if epoch in c["milestones"]:
                rows=evaluate(model,c,domains,arm,epoch,base_sha)
                # Evaluation loads best; restore LAST model and RNG for continuous training.
                last=torch.load(resume,map_location="cpu",weights_only=False)
                model.load_trainable_payload(last,base_sha); restore_rng(last["rng"]); del last
                if arm=="pretrained_reference":
                    other=json.loads((root/f"eval_self_reference_epoch{epoch}.json").read_text())
                    previous={r["domain"]:r["eer"] for r in other}
                    gaps={r["domain"]:r["eer"]-previous[r["domain"]] for r in rows}
                    cross=[v for k,v in gaps.items() if k!="itw"]
                    stop=(sum(v>5 for v in cross)>=2 and np.mean(cross)>3) or (
                        epoch>=8 and np.mean(cross)>=0 and gaps["itw"]>2)
                    atomic_json(root/f"decision_epoch{epoch}.json",{"stop":bool(stop),"gaps":gaps})
                    if stop:
                        atomic_json(root/"status.json",{"status":"stopped_poor_results","epoch":epoch})
                        return
        atomic_json(done,{"arm":arm,"epoch":c["epochs"],"best_loss":best})
        # Only own transient resume, after final evaluation and completion marker.
        resume.unlink()
        del model,opt; gc.collect(); torch.cuda.empty_cache()
    atomic_json(root/"status.json",{"status":"complete","seed":c["seed"]})
    print("ALL_COMPLETE",flush=True)


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print("FAILED",type(exc).__name__,str(exc),flush=True)
        raise
