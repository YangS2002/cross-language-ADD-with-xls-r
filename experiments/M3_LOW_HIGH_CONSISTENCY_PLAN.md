# M3 controlled experiment: low-high XLS-R consistency residual

## Question

Can an explicitly defined mismatch between low-level acoustic construction and
high-level semantic representations improve the dual-transition baseline across
multiple random seeds, without changing the baseline stochastic training path
at initialization?

## Training-only layer analysis

The split was selected without using DECRO-CH or MLAAD-DE metrics.

- Data: 128 bona fide and 128 spoofed ASVspoof2019-LA training utterances.
- Representation: temporal mean and standard deviation from each frozen XLS-R
  layer.
- Structural metric: adjacent-layer linear CKA.
- Diagnostic probes: bona-fide/spoof and source vocoder linear probes, each
  repeated over three deterministic train/test splits.

The only strong structural discontinuity is between layers 21 and 22:

- layer 21 to 22 CKA: `0.358055`;
- all other adjacent CKA values: approximately `0.976` to `0.997`.

The source-vocoder probe is nearly perfect across most layers. This confirms
that source-domain generator shortcuts are easy to extract from the frozen
representations and motivates modeling a cross-view relation rather than adding
another unconstrained feature branch.

Artifacts: `experiments/analysis/xlsr_layers_train256/`

## Single model change

The original dual-transition heads, ensemble, relation loss, optimizer, data,
and frozen XLS-R remain unchanged.

M3 adds one shared deterministic branch:

1. Average normalized XLS-R layers 1-21 into a low view.
2. Average normalized layers 22-24 into a high view.
3. Form temporal statistics of `high - low` and `high * low`.
4. Project these statistics to the baseline embedding dimension.
5. Add the result to both baseline embeddings through
   `0.1 * tanh(gate)`, with the gate initialized to zero.

The branch contains no dropout or other random operation. It is deliberately
small and does not introduce a new loss.

## Isolation checks

With both detectors in train mode and the same RNG state:

- head-A maximum logit difference at zero gate: `0.0`;
- head-B maximum logit difference at zero gate: `0.0`;
- relation-loss difference at zero gate: `0.0`;
- gate gradient: `0.176135` (nonzero in the synthetic unit check);
- consistency-branch gradient at the exact zero gate: `0.0` (expected).

Unlike M2, computing the new branch does not advance the random-number state,
so the baseline heads receive identical initialization and dropout masks while
the residual gate is zero.

The full two-sample XLS-R forward/backward smoke test also passed; peak allocated
GPU memory was approximately 1.51 GiB.

## Experiment budget and decision rule

- Screen seed 20 and seed 1001 for 10 epochs each, sequentially on one GPU.
- Preserve the minimum-training-loss checkpoint for every seed.
- Do not select epochs using CH or DE results.
- Reject M3 if either seed is materially worse than its matching baseline.
- Run a third seed and ITW only if both screening seeds are directionally
  consistent.
- Commit the model only after the multi-seed confirmation passes.

Configs:

- `experiments/configs/dual_transition_low_high_consistency_seed20_10ep.yaml`
- `experiments/configs/dual_transition_low_high_consistency_seed1001_10ep.yaml`
