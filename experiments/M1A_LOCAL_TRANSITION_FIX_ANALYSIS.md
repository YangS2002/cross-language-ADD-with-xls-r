# M1a single-change experiment: preserve per-transition local features

## Question

Does fixing the layer-axis collapse in the M1 local branch improve cross-language
generalization without changing any other model or training setting?

## Controlled change

Only the location of the layer-axis mean was changed.

- M1: average the first 16 adjacent-layer transitions before projection and
  temporal convolution. This telescopes to an endpoint difference.
- M1a: project and temporally process each of the 16 transitions independently,
  compute mean/std/max statistics per transition, and average the resulting
  statistics afterward.

Unchanged: per-layer normalization, second-order branch, fusion, classifier,
single-head training, loss, optimizer, learning rate, data order, seed 20,
frozen XLS-R, and 10-epoch budget.

Config: `experiments/configs/heterogeneous_multi_order_local_fix_seed20_10ep.yaml`

## Epoch results

All EER values are percentages.

| Epoch | Train loss | DECRO-CH | MLAAD-DE | Joint | Domain gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.100041 | 15.4170 | 23.2990 | 19.3580 | 7.8821 |
| 2 | 0.016951 | 17.9755 | 16.4896 | 17.2326 | 1.4859 |
| 3 | 0.011413 | 10.2175 | 15.6024 | **12.9100** | 5.3849 |
| 4 | 0.008004 | 8.2161 | 19.8689 | 14.0425 | 11.6529 |
| 5 | 0.007453 | 8.3316 | 20.5245 | 14.4280 | 12.1928 |
| 6 | 0.010403 | 8.3811 | 18.6087 | 13.4949 | 10.2275 |
| 7 | 0.007566 | 8.3151 | 20.1910 | 14.2531 | 11.8759 |
| 8 | 0.003663 | 15.2561 | 21.7111 | 18.4836 | 6.4551 |
| 9 | 0.004427 | 11.9837 | 21.0330 | 16.5083 | 9.0493 |
| 10 | **0.001056** | **7.2381** | 23.1578 | 15.1980 | 15.9197 |

## Primary checkpoint comparison

The required checkpoint-selection rule is minimum training loss, which selects
epoch 10.

| Model / epoch-10 checkpoint | DECRO-CH | MLAAD-DE | Joint |
|---|---:|---:|---:|
| Dual-transition baseline ensemble | 12.3097 | 16.7891 | 14.5494 |
| Original M1 | 13.7333 | 18.9026 | 16.3180 |
| M1a local fix | **7.2381** | 23.1578 | 15.1980 |

M1a improves the original M1 joint EER by 1.1200 points and CH by 6.4952
points, proving that averaging raw adjacent transitions before nonlinear
processing was destructive. However, relative to the baseline, M1a worsens DE
by 6.3687 points and joint EER by 0.6486 points. It therefore fails the primary
screen and should not be replicated on seed 1001.

The best observed M1a point is epoch 3 (CH 10.2175, DE 15.6024, joint
12.9100), but CH and DE are evaluation sets rather than model-selection sets.
Selecting epoch 3 would be test-set cherry-picking and is not used as the final
result.

## Stability

- Original M1 epoch-3-to-10 joint range: 1.3472 points.
- Baseline epoch-3-to-10 joint range: 1.4360 points.
- M1a epoch-3-to-10 joint range: **5.5736 points**.

Preserving individual transition information greatly improves the attainable
solution, but also removes the strong implicit bottleneck created by endpoint
averaging. The model becomes much more sensitive to the training trajectory.
After epoch 3 it increasingly trades DE performance for CH performance.

## Read-only branch masking at the minimum-loss checkpoint

This diagnostic reuses the trained epoch-10 checkpoint and does not retrain or
modify weights. Zeroing one branch after joint training is distribution-shifted,
so absolute branch-only EER is not treated as an independently trained model;
the relative behavior is used only to locate branch/fusion failure.

| Epoch-10 inference | DECRO-CH | MLAAD-DE |
|---|---:|---:|
| Full M1a | 7.2381 | 23.1578 |
| Local branch only | 26.6991 | 22.0219 |
| Second-order dynamics only | 7.3330 | 21.8581 |

Mean embedding norms:

| Domain | Local | Dynamics |
|---|---:|---:|
| DECRO-CH | 19.5892 | 17.4879 |
| MLAAD-DE | 19.7812 | 16.9459 |

At epoch 10 the dynamics branch carries nearly all useful CH discrimination.
On DE, both masked branches outperform the full fusion, showing destructive
interaction rather than simple lack of capacity. The local branch has the
larger activation norm in both domains, while the fixed concat/LayerNorm/linear
fusion has no mechanism to suppress it. Moreover, changing only the local
branch also changes the jointly trained dynamics solution: dynamics-only DE
degrades from 17.9871 in M1 to 21.8581 in M1a. The failure is therefore an
optimization and branch-interaction problem, not merely an inference-weighting
problem.

## Conclusion

The single code correction is technically valid and informative:

1. The original pre-convolution layer mean was indeed a major representation
   error; fixing it exposes a much better cross-language solution at epoch 3.
2. The correction alone is not an acceptable final model because minimum-loss
   convergence moves to a strongly domain-biased solution.
3. The next experiment should change exactly one additional mechanism: add a
   conservative gate to the existing two-branch fusion (preferably initialized
   to preserve the dynamics path), without changing normalization, branches,
   loss, dual/single-head setup, or other hyperparameters in the same run.
4. Do not tune temporal kernels or transition count before resolving destructive
   fusion, because the present evidence shows that useful information exists but
   is combined and optimized incorrectly.

## Preserved artifacts

- Full CSV files: `models/M1a_local_transition_preservation_seed20_10ep/`
- Minimum-training-loss checkpoint:
  `models/M1a_local_transition_preservation_seed20_10ep/checkpoints/seed_20_best_train_loss.pth`

