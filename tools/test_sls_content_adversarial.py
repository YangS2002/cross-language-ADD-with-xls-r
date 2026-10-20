"""Fast numerical tests without ASR assets or source/target data."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from tools.sls_content_adversarial_model import (ReverseGradient,
    ContentAdversarialSLS, balanced_content_kl)
from tools.sls_reference_model import FullWidthSLS


class FixedStates(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, wave):
        h = wave[:, :1, None].expand(-1, 201, 1024) * self.scale
        return [h + i / 24 for i in range(24)]


def main():
    torch.set_num_threads(2)
    x = torch.randn(3, 1024, requires_grad=True)
    a = nn.Linear(1024, 26)
    q = torch.randn(3, 26).softmax(-1)
    y = torch.tensor([0, 0, 1]); valid = torch.ones(3, dtype=torch.bool)
    normal = balanced_content_kl(a(x), q, y, valid)
    gx, ga = torch.autograd.grad(normal, (x, a.weight))
    reverse = balanced_content_kl(a(ReverseGradient.apply(x, .1)), q, y, valid)
    rx, ra = torch.autograd.grad(reverse, (x, a.weight))
    torch.testing.assert_close(rx, -.1*gx)
    torch.testing.assert_close(ra, ga)
    torch.testing.assert_close(normal, reverse)
    # Main initialization and RNG consumption exactly match task+SLS backend.
    torch.manual_seed(20)
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    m = ContentAdversarialSLS(task=FixedStates())
    after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    assert all(torch.equal(x,y) for x,y in zip(cuda_before,cuda_after))
    torch.manual_seed(20)
    task = FixedStates(); b = FullWidthSLS()
    assert torch.equal(after, torch.get_rng_state())
    for k, v in b.state_dict().items():
        torch.testing.assert_close(v, m.backend.state_dict()[k], rtol=0, atol=0)
    m.eval(); b.eval(); wave = torch.randn(2, 64600)
    states = task(wave)
    gates = torch.sigmoid(b.fc0(b.descriptors(states)))
    z = sum(h*gates[:, i:i+1] for i, h in enumerate(states))
    e = b.selu(b.fc1(F.max_pool2d(b.selu(b.first_bn(z[:, None])), 3, 3).flatten(1)))
    expected = F.log_softmax(b.selu(b.fc3(e)).float(), 1)
    d = m(wave, torch.tensor([0, 1]), True)
    torch.testing.assert_close(d['log_probs'], expected, rtol=0, atol=0)
    content = balanced_content_kl(d['content_logits'], torch.randn(2,26).softmax(1),
                                  torch.tensor([0,1]), torch.ones(2,dtype=torch.bool))
    content.backward()
    assert m.backend.fc3.weight.grad is None
    assert m.backend.fc1.weight.grad is not None
    assert m.backend.fc0.weight.grad is not None
    assert m.task.scale.grad is not None
    assert m.content[0].weight.grad is not None
    m.zero_grad(set_to_none=True)
    d = m(wave, torch.tensor([0,1]), True)
    F.nll_loss(d['log_probs'], torch.tensor([0,1])).backward()
    assert m.backend.fc3.weight.grad is not None
    assert all(p.grad is None for p in m.content.parameters())
    empty = balanced_content_kl(torch.randn(2,26,requires_grad=True),torch.zeros(2,26),
                                torch.tensor([0,1]),torch.zeros(2,dtype=torch.bool))
    empty.backward(); assert float(empty) == 0
    print(json.dumps(dict(grl_sign=True, adversary_gradient_unchanged=True,
        fixed_sls_forward_exact=True, main_rng_exact=True, fc3_detection_only=True,
        gate_fc1_ssl_receive_content_gradient=True, empty_reference_safe=True)),flush=True)


if __name__ == '__main__':
    main()
