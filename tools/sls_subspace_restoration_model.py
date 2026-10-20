"""SLS readout with deterministic original-subspace restoration, not concept erasure."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools.sls_reference_model import XLSRStates, FullWidthSLS


@torch.no_grad()
def principal_basis(reference, rank=16, relative_tolerance=1e-6):
    """Per-input right singular vectors via T-by-T Gram; no batch statistics.

    Rank-deficient columns are zeros. FP64 inputs keep FP64 for property tests;
    production BF16/FP32 inputs use FP32 regardless of surrounding autocast.
    """
    if reference.ndim != 3 or not 1 <= rank <= min(reference.shape[1:]):
        raise ValueError("Reference [B,T,D] and a valid rank required")
    with torch.autocast(device_type=reference.device.type, enabled=False):
        value = reference.detach()
        value = value.double() if value.dtype == torch.float64 else value.float()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite reference")
        # Translation before averaging avoids inventing a rank for constant
        # sequences through cancellation of a large utterance mean.
        shifted = value-value[:, :1]
        centered = shifted-shifted.mean(dim=1, keepdim=True)
        gram = centered @ centered.transpose(1, 2)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues[:, -rank:]
        vectors = eigenvectors[:, :, -rank:]
        maximum = eigenvalues[:, -1:].clamp(min=0)
        valid = (eigenvalues > maximum * relative_tolerance) & (maximum > 0)
        denominator = eigenvalues.clamp(min=torch.finfo(value.dtype).tiny).sqrt()
        basis = centered.transpose(1, 2) @ vectors
        basis = basis / denominator.unsqueeze(1)
        basis = torch.where(valid.unsqueeze(1), basis, torch.zeros_like(basis))
        total = gram.diagonal(dim1=1, dim2=2).sum(1)
        explained = torch.where(total > 0, eigenvalues.clamp(min=0).sum(1)/total.clamp(min=1e-30),
                                torch.zeros_like(total))
        if not bool(torch.isfinite(basis).all()):
            raise FloatingPointError("Nonfinite original basis")
    return basis, valid.sum(1), explained


def restore_subspace(task, reference, basis):
    if task.shape != reference.shape or basis.shape[:2] != (task.size(0), task.size(2)):
        raise ValueError("Task/reference/basis dimensions mismatch")
    with torch.autocast(device_type=task.device.type, enabled=False):
        value = task.to(basis.dtype)
        original = reference.detach().to(basis.dtype)
        difference = value-original
        projected = (difference @ basis) @ basis.transpose(1, 2)
        corrected = value-projected
    return corrected.to(task.dtype), difference, projected


class SubspaceRestoredSLS(nn.Module):
    def __init__(self, checkpoint=None, rank=16, task=None, reference=None,
                 cache_max_bytes=32*1024**3):
        super().__init__()
        self.rank = rank
        # Same task/backend construction order as the fixed SLS baseline.
        self.task = task if task is not None else XLSRStates(checkpoint, True)
        self.backend = FullWidthSLS()
        self.reference_gate = copy.deepcopy(self.backend.fc0)
        self.reference_gate.requires_grad_(False)
        self.reference = reference if reference is not None else XLSRStates(checkpoint, False)
        self.reference.requires_grad_(False)
        self.reference.eval()
        self.restoration_enabled = True
        # Fixed-reference memoization only, no target fitting or cohort statistics.
        # Source packs retain native-precision Z0 for diagnostics. Evaluation
        # packs retain just FP32 Q and Z0@Q. Nothing is written to disk.
        self.reference_cache = {}
        self.cache_max_bytes = cache_max_bytes
        self.cache_bytes = 0
        self.cache_hits = self.cache_misses = 0

    def initialize_shared_backend(self, state):
        self.clear_reference_cache()
        self.backend.load_state_dict(state, strict=True)
        self.reference_gate.load_state_dict({"weight":state["fc0.weight"],"bias":state["fc0.bias"]})

    def clear_reference_cache(self):
        self.reference_cache.clear()
        self.cache_bytes = self.cache_hits = self.cache_misses = 0

    @torch.no_grad()
    def reference_pack(self, wave, keys):
        if len(keys)!=wave.size(0) or len(set(keys))!=len(keys):
            raise ValueError("One unique namespace/file key per input required")
        missing=[i for i,key in enumerate(keys) if key not in self.reference_cache]
        if missing:
            original=self.original_features(wave[missing])
            basis,ranks,explained=principal_basis(original,self.rank)
            with torch.autocast(device_type=wave.device.type,enabled=False):
                coordinates=original.float()@basis
            for j,i in enumerate(missing):
                key=keys[i]
                source=key[0] in {"source","pressure_cold","pressure_initial"}
                pack=dict(basis=basis[j].detach().cpu(),rank=ranks[j].detach().cpu(),
                          explained=explained[j].detach().cpu(),coordinates=coordinates[j].detach().cpu())
                if source: pack["original"]=original[j].detach().cpu()
                nbytes=sum(t.numel()*t.element_size() for t in pack.values())
                if self.cache_bytes+nbytes>self.cache_max_bytes:
                    raise RuntimeError("Fixed reference RAM cache budget exceeded; do not evict/recompute silently")
                self.reference_cache[key]=pack
                self.cache_bytes+=nbytes
        self.cache_misses+=len(missing); self.cache_hits+=len(keys)-len(missing)
        packs=[self.reference_cache[key] for key in keys]
        fields=list(packs[0])
        if any(list(p)!=fields for p in packs): raise ValueError("Mixed cache namespaces in batch")
        return {name:torch.stack([p[name] for p in packs]).to(wave.device) for name in fields}

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        self.reference_gate.eval()
        return self

    @staticmethod
    def aggregate(states, gate):
        if len(states) != 24 or any(tuple(h.shape[1:]) != (201, 1024) for h in states):
            raise RuntimeError("Fixed 24-layer, pad64600 full-width SLS required")
        weights = torch.sigmoid(gate(FullWidthSLS.descriptors(states)))
        return sum(h * weights[:, i:i+1] for i,h in enumerate(states))

    @torch.no_grad()
    def original_features(self, wave):
        numpy_state = np.random.get_state()
        try:
            return self.aggregate(self.reference(wave), self.reference_gate)
        finally:
            np.random.set_state(numpy_state)

    def classify(self, value):
        b = self.backend
        pooled = F.max_pool2d(b.selu(b.first_bn(value.unsqueeze(1))), 3, 3)
        flat = pooled.flatten(1)
        if flat.size(1) != 22847:
            raise RuntimeError("SLS pooling dimensions changed")
        return F.log_softmax(b.selu(b.fc3(b.selu(b.fc1(flat)))).float(), dim=1)

    def forward(self, wave, return_details=False, cache_keys=None):
        # Frozen route first avoids retaining task activations during its work.
        if cache_keys is None:
            original = self.original_features(wave)
            basis, effective_rank, explained = principal_basis(original, self.rank)
            coordinates = None
        else:
            pack=self.reference_pack(wave,cache_keys)
            original=pack.get("original")
            basis=pack["basis"]; effective_rank=pack["rank"]; explained=pack["explained"]
            coordinates=pack["coordinates"]
        task = self.aggregate(self.task(wave), self.backend.fc0)
        if original is not None:
            corrected, difference, projected = restore_subspace(task, original, basis)
        else:
            if return_details: raise ValueError("Compact evaluation packs have no full-reference diagnostics")
            with torch.autocast(device_type=wave.device.type,enabled=False):
                projected=(task.float()@basis-coordinates)@basis.transpose(1,2)
                corrected=(task.float()-projected).to(task.dtype)
            difference=None
        output = self.classify(corrected if self.restoration_enabled else task)
        if return_details:
            return dict(log_probs=output, aggregate=task, original=original,
                corrected=corrected, basis=basis, difference=difference,
                projected=projected, effective_rank=effective_rank, explained=explained)
        return output

    def trainable_payload(self, pretrained_sha256):
        if len(pretrained_sha256) != 64:
            raise ValueError("Verified original SHA256 required")
        names = {n for n,p in self.task.named_parameters() if p.requires_grad}
        return dict(format="sls_subspace_restoration_v1",pretrained_sha256=pretrained_sha256,
            rank=self.rank,backend=self.backend.state_dict(),reference_gate=self.reference_gate.state_dict(),
            task_trainable={n:t for n,t in self.task.state_dict().items() if n in names})

    def load_trainable_payload(self, saved, pretrained_sha256):
        if (saved["format"] != "sls_subspace_restoration_v1" or saved["rank"] != self.rank
                or saved["pretrained_sha256"] != pretrained_sha256):
            raise ValueError("Base/format/rank mismatch")
        names = {n for n,p in self.task.named_parameters() if p.requires_grad}
        if set(saved["task_trainable"]) != names:
            raise ValueError("Task parameter set mismatch")
        self.task.load_state_dict(saved["task_trainable"],strict=False)
        self.backend.load_state_dict(saved["backend"],strict=True)
        if any(not torch.equal(self.reference_gate.state_dict()[k].detach().cpu(),v.detach().cpu())
               for k,v in saved["reference_gate"].items()):
            self.clear_reference_cache()
        self.reference_gate.load_state_dict(saved["reference_gate"],strict=True)
