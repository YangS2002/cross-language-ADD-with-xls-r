"""Cheap algebra tests and an explicitly requested real-batch pressure test."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
import yaml
from tools.sls_subspace_restoration_model import principal_basis, restore_subspace, SubspaceRestoredSLS


def algebra_tests():
    torch.set_num_threads(4)
    torch.manual_seed(20)
    reference=torch.randn(2,23,32,dtype=torch.float64)
    basis,rank,explained=principal_basis(reference,6)
    torch.testing.assert_close(basis.transpose(1,2)@basis,
        torch.eye(6,dtype=basis.dtype).expand(2,-1,-1),atol=1e-10,rtol=1e-10)
    task=(reference+torch.randn_like(reference)).requires_grad_(True)
    corrected,difference,projected=restore_subspace(task,reference,basis)
    compact=task-((task@basis-reference@basis)@basis.transpose(1,2))
    torch.testing.assert_close(compact,corrected,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(corrected@basis,reference@basis,atol=1e-10,rtol=1e-10)
    def complement(x): return x-(x@basis)@basis.transpose(1,2)
    torch.testing.assert_close(complement(corrected),complement(task),atol=1e-10,rtol=1e-10)
    assert float(projected.norm())>0
    upstream=torch.randn_like(task)
    (corrected*upstream).sum().backward()
    torch.testing.assert_close(task.grad,complement(upstream),atol=1e-10,rtol=1e-10)
    assert float((task.grad@basis).abs().max())<1e-10
    initial,_,p=restore_subspace(reference,reference,basis)
    torch.testing.assert_close(initial,reference,atol=0,rtol=0)
    assert float(p.norm())==0
    zero=torch.zeros(2,23,32)
    q,effective,_=principal_basis(zero,6)
    assert not bool(q.any()) and not bool(effective.any())
    constant=torch.randn(2,1,32).expand(-1,23,-1).clone()
    q,effective,_=principal_basis(constant,6)
    # Numerically centering a constant must not invent reference directions.
    assert not bool(effective.any()),effective
    low_rank=torch.randn(2,23,2)@torch.randn(2,2,32)
    q,effective,_=principal_basis(low_rank,6)
    assert bool((effective==2).all()),effective
    gram=q.transpose(1,2)@q
    expected=torch.zeros_like(gram)
    for i in range(2): expected[i,-2:,-2:]=torch.eye(2)
    torch.testing.assert_close(gram,expected,atol=1e-4,rtol=1e-4)
    # Fitting is per input, not to the evaluation batch or domain.
    one,_,_=principal_basis(reference[:1],6)
    torch.testing.assert_close(one@one.transpose(1,2),basis[:1]@basis[:1].transpose(1,2),atol=1e-10,rtol=1e-10)
    print("ALGEBRA_OK",json.dumps(dict(protected_error=float((corrected@basis-reference@basis).abs().max()),
        effective_rank=rank.tolist(),explained=explained.tolist())),flush=True)


def real_pressure(config_path):
    from tools.train_relational_preservation import dataset_for,atomic_json,digest
    from train import set_seed
    c=yaml.safe_load(Path(config_path).read_text())
    root=Path(c["output_dir"])
    if not root.resolve().is_relative_to(Path("/root/rivermind-data")):
        raise ValueError("Persistent disk only")
    if digest(c["shared_warmup"])!=c["expected_warmup_sha256"]:
        raise ValueError("Shared warmup signature mismatch")
    if digest(c["ssl_checkpoint"])!=c["expected_pretrained_sha256"]:
        raise ValueError("Original checkpoint signature mismatch")
    dataset,_=dataset_for(c); half=c["batch_size"]//2
    ids=([i for i,y in enumerate(dataset.synthesizerlabels) if y=="bonafide"][:half]+
         [i for i,y in enumerate(dataset.synthesizerlabels) if y!="bonafide"][:c["batch_size"]-half])
    batch=[dataset[i] for i in ids]
    wave=torch.stack([b["audio"] for b in batch]).cuda()
    labels=torch.tensor([int(b["label"]) for b in batch],device="cuda")
    set_seed(c["seed"])
    model=SubspaceRestoredSLS(c["ssl_checkpoint"],c["rank"]).cuda().train()
    shared=torch.load(c["shared_warmup"],map_location="cpu",weights_only=False)
    model.initialize_shared_backend(shared["backend"]); del shared
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=c["ssl_lr"],weight_decay=c["weight_decay"],foreach=False)
    torch.cuda.reset_peak_memory_stats(); times=[]; initial_error=None; last_ratio=0
    for i in range(4):
        opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); start=time.perf_counter()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            namespace="pressure_initial" if i==0 else "pressure_cold"
            keys=[(namespace,str(b["file_name"])) for b in batch]
            d=model(wave,return_details=True,cache_keys=keys)
            if i==0:
                baseline=model.classify(d["aggregate"])
                torch.testing.assert_close(d["log_probs"],baseline,atol=2e-4,rtol=2e-4)
                initial_error=float((d["log_probs"]-baseline).abs().max())
                del baseline
        q=d["basis"]; eye=torch.eye(c["rank"],device="cuda").expand(len(batch),-1,-1)
        torch.testing.assert_close(q.transpose(1,2)@q,eye,atol=1e-3,rtol=1e-3)
        protected=(d["corrected"].float()-d["original"].float())@q
        protected_error=float(protected.norm()/((d["original"].float()@q).norm()+1e-8))
        if protected_error>0.002: raise RuntimeError("Protected subspace restoration numerical error")
        loss=torch.nn.functional.nll_loss(d["log_probs"],labels,weight=d["log_probs"].new_tensor([.1,.9]))
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite NLL")
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True)
        for index in (0,23):
            assert any(p.grad is not None and bool(p.grad.abs().sum()>0)
                for p in model.task.model.encoder.layers[index].parameters())
        assert model.backend.fc0.weight.grad is not None
        assert all(p.grad is None for p in model.reference.parameters())
        assert all(p.grad is None for p in model.reference_gate.parameters())
        opt.step(); torch.cuda.synchronize(); elapsed=time.perf_counter()-start; times.append(elapsed)
        last_ratio=float(d["projected"].norm()/(d["aggregate"].float().norm()+1e-8))
        print("PRESSURE_STEP",i,float(loss),elapsed,last_ratio,protected_error,flush=True)
        del d,q,protected,eye,loss
    if last_ratio<=1e-6: raise RuntimeError("Nonzero adaptation did not activate restoration")
    inference_times=[]; model.eval()
    for _ in range(3):
        torch.cuda.synchronize(); started=time.perf_counter()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            inferred=model(wave,cache_keys=[("pressure_eval",str(b["file_name"])) for b in batch])
        if not bool(torch.isfinite(inferred).all()): raise FloatingPointError("Nonfinite inference")
        torch.cuda.synchronize(); inference_times.append(time.perf_counter()-started)
        del inferred
    model_bytes=sum(p.numel()*p.element_size() for p in params)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        online=model(wave)
        cached=model(wave,cache_keys=[("pressure_cold",str(b["file_name"])) for b in batch])
        compact=model(wave,cache_keys=[("pressure_eval",str(b["file_name"])) for b in batch])
    torch.testing.assert_close(cached,online,atol=2e-3,rtol=2e-3)
    torch.testing.assert_close(compact,online,atol=2e-3,rtol=2e-3)
    equivalence_error=float((compact-online).abs().max())
    full_bytes=sum(t.numel()*t.element_size() for t in model.reference_cache[("pressure_cold",str(batch[0]["file_name"]))].values())
    compact_bytes=sum(t.numel()*t.element_size() for t in model.reference_cache[("pressure_eval",str(batch[0]["file_name"]))].values())
    record=dict(batch=c["batch_size"],rank=c["rank"],steps=4,initial_logprob_error=initial_error,
        cache_equivalence_error=equivalence_error,source_cache_bytes_per_input=full_bytes,
        evaluation_cache_bytes_per_input=compact_bytes,
        restoration_ratio_last=last_ratio,seconds_per_step=sum(times[2:])/2,
        cold_training_seconds_per_batch=times[1],
        inference_seconds_per_batch=sum(inference_times[1:])/2,
        cold_inference_seconds_per_batch=inference_times[0],cache_bytes=model.cache_bytes,
        reserved_gib=torch.cuda.max_memory_reserved()/1024**3,model_bytes=model_bytes,
        signature=dict(config=digest(config_path),model=digest("tools/sls_subspace_restoration_model.py"),
                       test=digest(__file__),warmup=digest(c["shared_warmup"])))
    if record["reserved_gib"]>20: raise RuntimeError("20GiB engineering limit exceeded")
    atomic_json(root/"pressure_test.json",record)
    print("PRESSURE_OK",json.dumps(record),flush=True)
    del model,opt; gc.collect(); torch.cuda.empty_cache()


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--real-config")
    args=parser.parse_args(); algebra_tests()
    if args.real_config: real_pressure(args.real_config)
