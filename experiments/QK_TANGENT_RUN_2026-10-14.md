# Original QK versus full-backbone tangent adaptation

Started: 2026-10-14 21:41:53 Asia/Shanghai. Queue PID at launch: 1027.
Host: root@hn01-ssh.gpuhome.cc:30307.
Remote repository: /root/rivermind-data/cross-language-ADD-with-xls-r.

## Fixed comparison

- SLS only, seed20, shared three-epoch backend warmup and restored RNG.
- Original QK control, followed by full-backbone first-order adaptation.
- Same trainable parameter set (284,246,149 including backend), AdamW, learning rates, batch16 and source-only data.
- Train epochs4–8; complete CH/DE/UK/ES/ITW evaluation at epochs4 and8.
- Select checkpoint using source classification loss; no target-dependent stopping or additional seeds.
- Candidate inference retains tangent propagation.

Entry point: `tools/run_sls_qk_tangent.sh`.
Queue log: `experiments/logs/sls_qk_tangent_queue_20261014.log`.
Arm logs: `experiments/logs/sls_qk_tangent_{original,tangent}_seed20_20261014.log`.
Output directories:

- `models/SLS_qk_tangent_control_seed20_20261014`
- `models/SLS_qk_tangent_seed20_20261014`

These are asynchronous jobs. This launch record is not a completion or effectiveness claim; inspect each directory's `status.json`, `epoch_results.csv`, and `evaluation_results.csv` for current results.

## Candidate implementation validation

- Initial FP32/BF16 outputs: all24 backbone states and classifier output matched exactly.
- Initial FP32 classification-gradient relative L2 error: 4.064283e-6; gradient parameter sets matched.
- Nonzero full-backbone directional response checked by central finite differences, perturbing first-layer V/FFN/LN and positional-convolution weight_g: relative L2 error0.005903.
- Finite differences required disabling TF32 convolution locally in the diagnostic context: its quantization obscured small perturbations (errors0.1942 at scale1e-4 and0.7033 at1e-5 before correction). Formal training precision is unchanged.
- Batch16, three BF16 source-only update smoke test: losses0.092946 /0.019456 /0.007380; reserved GPU memory11.117GiB. Test updates discarded before formal training.
- This finite-difference check replaces the proposed extra JVP check; it does not establish generalization or stability.

Implementation: `tools/sls_qk_tangent_model.py`, `tools/train_sls_qk_tangent.py`.
Protocol: `experiments/QK_TANGENT_EXPERIMENT_SPEC_2026-10-14.md`.
