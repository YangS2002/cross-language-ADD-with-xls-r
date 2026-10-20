"""Score-sensitivity control; no GRL, feature correction or parameter SVD."""
import torch
from torch.nn import functional as F
from tools.sls_content_adversarial_model import ContentAdversarialSLS, balanced_content_kl


class DependencyControlledSLS(ContentAdversarialSLS):
    def __init__(self, checkpoint=None, task=None):
        super().__init__(checkpoint=checkpoint, task=task, strength=0.)

    def forward(self,wave,labels=None,return_details=False):
        d=super().forward(wave,labels=None,return_details=return_details)
        if return_details and labels is not None:
            e=d['embedding'].detach().float()
            d['content_logits']=self.content(torch.cat([e,F.one_hot(labels,2).to(e)],1))
        return d

    def sensitivity(self, embedding, labels, ridge=0.01):
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            e = embedding.float()
            logits = self.backend.selu(F.linear(e, self.backend.fc3.weight.float(),
                                               self.backend.fc3.bias.float()))
            margin = logits[:, 1] - logits[:, 0]
            direction = torch.autograd.grad(margin.sum(), e, create_graph=True,
                                            retain_graph=True)[0]
            # Analytic local Jacobian of the class-conditional content readout.
            # Stop-gradient makes it a measured direction, NOT another adversarial loss.
            with torch.no_grad():
                first, _, last = self.content
                conditions = F.one_hot(labels, 2).to(e)
                hidden = F.linear(torch.cat([e.detach(), conditions], 1),
                                  first.weight.float(), first.bias.float())
                jacobian = (last.weight.float()[None] * (hidden > 0)[:, None].float()) @ first.weight.float()[:, :1024]
                # Softmax ignores a common shift of all logits.
                jacobian = jacobian - jacobian.mean(1, keepdim=True)
                scale = jacobian.square().sum((1, 2), keepdim=True).div(jacobian.shape[1]).sqrt()
                jacobian = jacobian / scale.clamp_min(1e-8)
                gram = jacobian @ jacobian.transpose(1, 2)
                gram = gram + ridge * torch.eye(gram.shape[1], device=e.device)[None]
            product = (jacobian @ direction.unsqueeze(-1))
            coefficient = torch.linalg.solve(gram, product)
            projection_energy = (product * coefficient).sum((1, 2)).clamp_min(0)
            ratio = projection_energy / direction.square().sum(1).clamp_min(1e-12)
            if not torch.isfinite(ratio).all() or bool((ratio > 1.01).any()):
                raise FloatingPointError('Invalid decision-sensitivity projection')
        return ratio.mean(), ratio.detach()

    def trainable_payload(self, sha):
        names = {n for n,p in self.task.named_parameters() if p.requires_grad}
        return dict(format='sls_dependency_control_v1', pretrained_sha256=sha,
            backend=self.backend.state_dict(), content=self.content.state_dict(),
            task_trainable={n:t for n,t in self.task.state_dict().items() if n in names})

    def load_trainable_payload(self, saved, sha):
        if saved['format']!='sls_dependency_control_v1' or saved['pretrained_sha256']!=sha:
            raise ValueError('Payload/base mismatch')
        names={n for n,p in self.task.named_parameters() if p.requires_grad}
        if set(saved['task_trainable'])!=names: raise ValueError('SSL parameter mismatch')
        self.task.load_state_dict(saved['task_trainable'],strict=False)
        self.backend.load_state_dict(saved['backend']); self.content.load_state_dict(saved['content'])


def optimizers(model, config):
    main=torch.optim.AdamW([
        dict(params=list(model.backend.parameters()),lr=config['head_lr']),
        dict(params=[p for p in model.task.parameters() if p.requires_grad],lr=config['ssl_lr'])],
        weight_decay=config['weight_decay'],foreach=False)
    auxiliary=torch.optim.AdamW(model.content.parameters(),lr=config['adversary_lr'],
                                weight_decay=config['weight_decay'],foreach=False)
    return main,auxiliary


def step(model, main, auxiliary, wave, labels, target, valid, weight=0.1):
    main.zero_grad(set_to_none=True); auxiliary.zero_grad(set_to_none=True)
    with torch.autocast(device_type=wave.device.type,dtype=torch.bfloat16,enabled=wave.is_cuda):
        d=model(wave,labels=labels,return_details=True)
    classification=F.nll_loss(d['log_probs'],labels,weight=d['log_probs'].new_tensor([.1,.9]))
    # Reader sees E.detach(); no GRL and no content-loss gradient to SLS.
    reader=balanced_content_kl(d['content_logits'],target,labels,valid)
    dependence,ratios=model.sensitivity(d['embedding'],labels)
    total=classification+weight*dependence+reader
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite training objective')
    total.backward()
    main_parameters=[p for group in main.param_groups for p in group['params']]
    norm=torch.nn.utils.clip_grad_norm_(main_parameters,5.,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(model.content.parameters(),5.,error_if_nonfinite=True)
    main.step(); auxiliary.step()
    return d['log_probs'].detach(),dict(classification_loss=float(classification.detach()),
        reader_kl=float(reader.detach()),sensitivity=float(dependence.detach()),
        sensitivity_real=float(ratios[labels==1].mean()) if bool((labels==1).any()) else None,
        sensitivity_fake=float(ratios[labels==0].mean()) if bool((labels==0).any()) else None,
        gradient_norm=float(norm))
