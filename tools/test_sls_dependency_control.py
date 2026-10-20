"""Seconds-long code correctness test, not another detector experiment."""
import json
import torch
from tools.test_sls_content_adversarial import FixedStates
from tools.sls_dependency_control_model import DependencyControlledSLS
from tools.sls_content_adversarial_model import balanced_content_kl


def main():
    torch.set_num_threads(2); torch.manual_seed(20)
    model=DependencyControlledSLS(task=FixedStates()).eval()
    wave=torch.randn(2,64600); y=torch.tensor([0,1])
    d=model(wave,y,True)
    reader=balanced_content_kl(d['content_logits'],torch.randn(2,26).softmax(1),y,
                               torch.ones(2,dtype=torch.bool))
    reader.backward(retain_graph=True)
    assert all(p.grad is None or bool((p.grad==0).all()) for p in model.backend.parameters())
    assert model.task.scale.grad is None or float(model.task.scale.grad)==0
    model.zero_grad(set_to_none=True)
    loss,ratios=model.sensitivity(d['embedding'],y)
    loss.backward()
    assert model.backend.fc3.weight.grad is not None
    assert bool(model.backend.fc3.weight.grad.abs().sum()>0)
    assert all(p.grad is None for p in model.content.parameters())
    assert 0<=float(loss)<=1
    print(json.dumps(dict(reader_does_not_delete_content=True,score_head_constraint_gradient=True,
        measured_directions_detached=True,finite_projection_ratio=float(loss))),flush=True)


if __name__=='__main__': main()
