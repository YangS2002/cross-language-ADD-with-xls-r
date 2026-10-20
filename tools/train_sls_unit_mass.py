"""One seed20 candidate; reuse the established source-only epoch4..8 protocol."""
import gc
import json
import shutil
import time
from pathlib import Path

import torch

from tools import train_sls_pretrained_attention as protocol
from tools.sls_unit_mass_model import UnitMassSLS, unit_mass_attention
from tools.train_relational_preservation import atomic_json, digest
from train import set_seed

_original_signature=protocol.signature
_original_decide=protocol.decide


def signature(c,config_path):
    record=_original_signature(c,config_path)
    record.update(candidate_trainer=digest(__file__),candidate_model=digest("tools/sls_unit_mass_model.py"),
                  candidate="original_quantizer_joint_mass_preserving_key_adaptation",
                  inherited_protocol=record["trainer"],ssh_port=30307)
    return record


def arithmetic_check():
    original=torch.softmax(torch.randn(2,2,5,5,device="cuda"),-1)
    logits=torch.randn_like(original,requires_grad=True)
    groups=torch.tensor([[0,0,1,1,2],[0,1,1,1,2]],device="cuda")
    attention=unit_mass_attention(original,logits,groups)
    index=groups[:,None,None,:].expand_as(original)
    mass0=torch.zeros_like(original).scatter_add(-1,index,original)
    mass=torch.zeros_like(attention).scatter_add(-1,index,attention)
    torch.testing.assert_close(mass,mass0,atol=1e-6,rtol=0)
    torch.testing.assert_close(attention.sum(-1),torch.ones_like(attention[...,0]),atol=1e-6,rtol=0)
    values=torch.randn(2,2,5,3,device="cuda")
    message=attention@values
    gradient=torch.autograd.grad(message.square().sum(),logits)[0]
    mean=torch.zeros(2,2,5,5,3,device="cuda")
    contribution=attention[...,None]*values[:,:,None,:,:]
    mean.scatter_add_(3,index[...,None].expand_as(contribution),contribution)
    conditional_mean=mean.gather(3,index[...,None].expand_as(contribution))/mass.gather(-1,index)[...,None]
    expected=(attention[...,None]*(values[:,:,None,:,:]-conditional_mean)*2*message[:,:,:,None,:]).sum(-1)
    torch.testing.assert_close(gradient,expected,atol=3e-6,rtol=3e-5)
    # The exact initialization identity must retain a nonzero, correct Jacobian.
    initial_logits=torch.randn_like(original,requires_grad=True)
    initial_original=initial_logits.detach().softmax(-1)
    initial=unit_mass_attention(initial_original,initial_logits,groups,initial_logits.detach())
    if not torch.equal(initial,initial_original):
        raise RuntimeError("Initialization identity not exact")
    upstream=torch.randn_like(initial)
    initial_gradient=torch.autograd.grad((initial*upstream).sum(),initial_logits)[0]
    plain_logits=initial_logits.detach().clone().requires_grad_(True)
    # Independent autograd expression checks the custom Jacobian at the identity.
    plain=torch.zeros_like(initial)
    for batch in range(2):
        for group in torch.unique(groups[batch]):
            mask=groups[batch]==group
            mass=initial_original[batch,...,mask].sum(-1,keepdim=True)
            plain[batch,...,mask]=mass*plain_logits[batch,...,mask].softmax(-1)
    plain_gradient=torch.autograd.grad((plain*upstream).sum(),plain_logits)[0]
    torch.testing.assert_close(initial_gradient,plain_gradient,atol=1e-6,rtol=2e-5)
    if float(initial_gradient.abs().sum())==0:
        raise RuntimeError("Initialization Key Jacobian vanished")
    singleton=torch.arange(5,device="cuda")[None,:].expand(2,-1)
    one_group=torch.zeros_like(groups)
    torch.testing.assert_close(unit_mass_attention(original,logits,singleton),original,atol=1e-7,rtol=0)
    torch.testing.assert_close(unit_mass_attention(original,logits,one_group),logits.softmax(-1),atol=1e-6,rtol=0)
    return dict(group_mass_max_error=float((mass-mass0).abs().max()),
                gradient_formula_max_error=float((gradient-expected).abs().max()),
                exact_identity=True,identity_jacobian_max_error=float((initial_gradient-plain_gradient).abs().max()),
                singleton_and_one_group_passed=True)


def preflight(c,dataset,sha):
    started=time.perf_counter()
    set_seed(c["seed"])
    arithmetic=arithmetic_check()
    half=c["batch_size"]//2
    ids=([i for i,y in enumerate(dataset.synthesizerlabels) if y=="bonafide"][:half]
         +[i for i,y in enumerate(dataset.synthesizerlabels) if y!="bonafide"][:c["batch_size"]-half])
    items=[dataset[i] for i in ids]
    wave=torch.stack([b["audio"] for b in items]).cuda()
    labels=torch.tensor([int(b["label"]) for b in items],device="cuda")
    model=UnitMassSLS(c["ssl_checkpoint"]).cuda().eval()
    warm=torch.load(c["shared_warmup"],map_location="cpu",weights_only=False)
    model.backend.load_state_dict(warm["backend"])
    del warm
    comparisons=[]
    for dtype in (torch.float32,torch.bfloat16):
        name="fp32" if dtype==torch.float32 else "bf16"
        with torch.no_grad(),torch.autocast("cuda",dtype=dtype,enabled=dtype!=torch.float32):
            model.guidance_enabled=False
            native=model(wave[:2],return_details=True)
            expected={"layers":[h.float().cpu() for h in native["states"]],
                      "output":native["log_probs"].cpu()}
            del native
            model.guidance_enabled=True
            model.audit_attention=True
            candidate=model(wave[:2],return_details=True)
            errors=[]
            for old,new in zip(expected["layers"],candidate["states"]):
                new=new.float().cpu()
                torch.testing.assert_close(old,new,atol=c[name+"_atol"],rtol=c[name+"_rtol"])
                errors.append(float((old-new).abs().max()))
            torch.testing.assert_close(expected["output"],candidate["log_probs"].cpu(),
                                       atol=c[name+"_atol"],rtol=c[name+"_rtol"])
            if len(model.attention_audit)!=24 or max(r["mass_max_error"] for r in model.attention_audit)>1e-6:
                raise RuntimeError("Initialization mass constraint failed")
            comparisons.append(dict(dtype=name,layer_max_errors=errors,
                output_max_error=float((expected["output"]-candidate["log_probs"].cpu()).abs().max()),
                attention=model.attention_audit))
            del candidate,expected
    model.audit_attention=False
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    opt=protocol.make_optimizer(model,c)
    torch.cuda.reset_peak_memory_stats()
    losses=[]
    for step in range(3):
        loss,output,norm=protocol.step(model,opt,wave,labels)
        losses.append(float(loss))
        if any(p.grad is not None for p in model.reference.parameters()):
            raise RuntimeError("Reference gradient leaked")
        for layer in model.task.model.encoder.layers:
            if any(p.requires_grad or p.grad is not None for p in layer.self_attn.q_proj.parameters()):
                raise RuntimeError("Task Query was unfrozen")
        if any(p.grad is not None for p in model.task.model.feature_extractor.parameters()):
            raise RuntimeError("Frozen acoustic convolution received gradients")
        for index in (0,23):
            layer=model.task.model.encoder.layers[index]
            for module in (layer.self_attn.k_proj,layer.self_attn.v_proj,layer.self_attn.out_proj,layer.fc1):
                if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()):
                    raise RuntimeError(f"Absent candidate task gradient at layer {index+1}")
        for module in (model.backend.fc0,model.backend.fc1,model.backend.fc3):
            if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()):
                raise RuntimeError("SLS gradient missing")
        print("SOURCE_SELF_TEST",step,float(loss),norm,flush=True)
    model.eval()
    model.audit_attention=True
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        model(wave[:2])
    invariants=model.attention_audit
    # FP32 scatter accumulation has O(T*eps) roundoff for up to 201 keys.
    # Record the actual discrepancy; do not require a single-operation bound.
    mass_tolerance=201*torch.finfo(torch.float32).eps
    atomic_json(Path(c["output_dir"])/"post_update_attention_check.json",
                dict(tolerance=mass_tolerance,attention=invariants))
    print("POST_UPDATE_MASS_MAX",max(r["mass_max_error"] for r in invariants),flush=True)
    if len(invariants)!=24 or max(r["mass_max_error"] for r in invariants)>mass_tolerance:
        raise RuntimeError("Group masses changed after updates")
    if max(r["attention_max_change"] for r in invariants)<=1e-6:
        raise RuntimeError("No actual routing adaptation after task updates")
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    reserved=torch.cuda.max_memory_reserved()/2**30
    required=max(4*2**30,int(model_bytes*2.1)+512*2**20)
    if reserved>c["max_reserved_gib"] or shutil.disk_usage(c["output_dir"]).free<required:
        raise RuntimeError("Memory/disk budget exceeded; do not change batch size")
    payload=model.trainable_payload(sha)
    model.load_trainable_payload(payload,sha)
    record=dict(arithmetic=arithmetic,comparisons=comparisons,after_updates=invariants,
                batch=c["batch_size"],losses=losses,trainable_parameters=trainable,
                disk_budget_bytes=required,reserved_gib=reserved,seconds=time.perf_counter()-started,
                reference_gradient=False,task_query_gradient=False,task_key_gradient=True,
                source_ids=[b["file_name"] for b in items],updates_discarded_before_formal_training=True,
                probability_invariants_measured_before_bf16_value_matmul=True)
    atomic_json(Path(c["output_dir"])/"self_test.json",record)
    print("SELF_TEST_OK",json.dumps(record),flush=True)
    del payload,model,opt,loss,output,wave,labels
    gc.collect()
    torch.cuda.empty_cache()
    return record


def decide(rows,c,epoch):
    record=_original_decide(rows,c,epoch)
    import csv
    with open(c["matched_qk_results"],newline="") as stream:
        original={r["domain"]:float(r["eer"]) for r in csv.DictReader(stream)
                  if r["arm"]=="pretrained_attention" and int(r["milestone"])==epoch}
    if set(original)!=set(record["per_domain"]):
        raise ValueError("Incomplete historical original-QK comparison")
    record.update(matched_original_qk=original,
                  gaps_vs_original_qk={k:record["per_domain"][k]-original[k] for k in original},
                  candidate_is_not_multiseed_evidence=True)
    return record


protocol.PretrainedAttentionSLS=UnitMassSLS
protocol.ARM="unit_mass"
protocol.signature=signature
protocol.preflight=preflight
protocol.decide=decide

if __name__=="__main__":
    protocol.main()
