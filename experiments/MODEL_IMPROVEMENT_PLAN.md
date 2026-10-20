# Cross-language model improvement plan

## Objective

Improve cross-language generalization and reduce seed sensitivity through model
architecture changes before changing the loss, sampler, optimizer, or data
augmentation. The frozen XLS-R backbone is retained to preserve its multilingual
pretraining.

## Locked baseline

- Server baseline commit: `e360311`
- Model: two identical Transition heads with relation consistency
- Training data: ASVspoof2019-LA
- Evaluation domains: DECRO-CH and MLAAD-DE
- Budget for new runs: 10 epochs
- Checkpoint selection: minimum training total loss only

For a fair 10-epoch comparison, the seed-20 baseline at epoch 10 is:

- ensemble DECRO-CH EER: 12.3097%
- ensemble MLAAD-DE EER: 16.7891%
- ensemble joint EER: 14.5494%
- ensemble joint range over epochs 3-10: 1.4360 percentage points

The seed-1001 minimum-training-loss point is epoch 10:

- ensemble DECRO-CH EER: 8.2821%
- ensemble MLAAD-DE EER: 18.3770%
- ensemble joint EER: 13.3296%

## M1: heterogeneous multi-order Transition detector

Change only the detector architecture:

1. Normalize each XLS-R layer before computing layer differences.
2. Local acoustic branch:
   - first-order adjacent-layer transitions;
   - lower and middle transitions only;
   - multi-scale temporal depthwise convolutions;
   - mean, standard-deviation, and max pooling.
3. Layer-dynamics branch:
   - second-order layer transitions;
   - temporal statistics retained per layer transition;
   - Transformer encoder across layer-transition tokens.
4. Concatenate both fixed-role branch embeddings and use one classifier.
5. Remove duplicate random heads and relation-consistency loss for M1.

Everything else remains unchanged, including weighted cross entropy, optimizer,
learning rate, batch size, data order, and frozen XLS-R parameters.

## Stepwise experiment protocol

### Stage 1: seed-20 screen

Run M1 with seed 20 for 10 epochs.

Primary comparison uses the minimum-training-loss checkpoint. M1 passes the
screen when it improves both MLAAD-DE and joint EER over the 10-epoch baseline
without worsening DECRO-CH by more than 0.5 percentage points. Curve stability
over epochs 3-10 is a secondary criterion.

Strong evidence thresholds:

- MLAAD-DE EER <= 16.5%
- joint EER <= 14.0%
- epochs 3-10 joint-EER range <= 1.0 percentage point

If M1 clearly fails, do not run another seed. Inspect branch behavior and make
at most one targeted architectural adjustment rather than a parameter grid.

### Stage 2: seed-1001 replication

Only if seed 20 passes, run the unchanged M1 configuration with seed 1001 for
10 epochs. The improvement is considered seed-robust only when the
minimum-training-loss checkpoint is not degraded on seed 1001 and the
cross-seed gap is no larger than the baseline gap.

### Stage 3: preservation

If both seeds support the improvement:

- commit only the related model, training integration, configs, and plan on the
  server `dev` branch;
- do not include checkpoints, logs, result directories, or unrelated working
  tree changes;
- report the commit hash but do not push without explicit authorization.

If only one seed improves, retain the result files for analysis but do not
commit the model as an effective solution.

## Later directions, not mixed into M1

Only after M1 is resolved:

1. Add a lightweight spectral/phase residual branch if language-domain EER
   remains the main limitation.
2. Consider selective adapters in lower/middle XLS-R layers if the frozen
   backbone limits artifact adaptation.
3. Consider loss or sampling changes only after architecture experiments are
   complete.

ITW remains a final external test and is not used for model selection or
hyperparameter tuning.
