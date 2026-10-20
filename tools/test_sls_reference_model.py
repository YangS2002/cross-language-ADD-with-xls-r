"""CPU synthetic tests by default; optional real-XLS-R GPU smoke test."""
import argparse
import copy
import torch
from torch import nn
from tools.sls_reference_model import FullWidthSLS, ReferenceRoutedSLS


class TinyStates(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(24))
        self.register_buffer("base", torch.randn(1, 201, 1024))

    def forward(self, waveform):
        return [self.base.expand(waveform.size(0), -1, -1) * self.scale[i]
                for i in range(24)]


def synthetic():
    torch.set_num_threads(4)
    torch.manual_seed(20)
    task = TinyStates()
    ref = copy.deepcopy(task)
    model = ReferenceRoutedSLS(task=task, reference=ref).train()
    waveform = torch.zeros(2, 64600)
    original = model.extract_reference_descriptors(waveform)
    result = model(waveform, return_details=True)
    assert result["aggregate"].shape == (2, 201, 1024)
    assert result["pooled_shape"] == (2, 1, 67, 341)
    assert result["gates"].shape == (2, 24)
    assert torch.allclose(result["log_probs"].exp().sum(1), torch.ones(2))
    loss = nn.functional.nll_loss(result["log_probs"], torch.tensor([0, 1]))
    loss.backward()
    assert task.scale.grad is not None and task.scale.grad.abs().sum() > 0
    assert model.backend.fc0.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in ref.parameters())
    assert not ref.training
    model.eval()
    with torch.no_grad():
        online = model(waveform)
        cached = model(waveform, reference_descriptors=original)
        assert torch.allclose(online, cached)
        model.reference_source = "self_reference"
        assert torch.allclose(online, model(waveform))
        model.reference_source = "pretrained_reference"
    descriptor_input = original.clone().requires_grad_()
    model(waveform, descriptor_input).sum().backward()
    assert descriptor_input.grad is None
    signature = "a" * 64
    saved = model.trainable_payload(signature)
    assert set(saved["task_trainable"]) == {"scale"}
    model.load_trainable_payload(saved, signature)
    try:
        model.load_trainable_payload(saved, "b" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong original checkpoint must be rejected")
    try:
        model.backend([torch.zeros(1, 200, 1024)] * 24)
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected frames must be rejected")
    print("SYNTHETIC_OK full_width pooling gradients detached_routing cache payload")


def real(checkpoint):
    torch.manual_seed(20)
    model = ReferenceRoutedSLS(checkpoint).cuda().train()
    waveform = torch.randn(1, 64600, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(waveform, return_details=True)
    loss = nn.functional.nll_loss(out["log_probs"], torch.tensor([1], device="cuda"))
    loss.backward()
    assert torch.isfinite(loss)
    for index in (0, 23):
        grads = [p.grad for p in model.task.model.encoder.layers[index].parameters()
                 if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in model.reference.parameters())
    assert all(p.grad is None for p in model.task.model.feature_extractor.parameters())
    print("REAL_SSL_OK", tuple(out["log_probs"].shape), out["pooled_shape"],
          "loss", float(loss), "reserved_gib", torch.cuda.max_memory_reserved()/1024**3)
    print("No optimizer step, checkpoint, or experiment training was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-checkpoint")
    args = parser.parse_args()
    synthetic()
    if args.real_checkpoint:
        real(args.real_checkpoint)
