#!/usr/bin/env bash
set -euo pipefail

cd /root/rivermind-data/cross-language-ADD-with-xls-r

PYTHON=/opt/conda/envs/SLS/bin/python

"${PYTHON}" train.py \
  --config experiments/configs/dual_transition_low_high_vector_adapter_seed20_10ep.yaml

"${PYTHON}" train.py \
  --config experiments/configs/dual_transition_low_high_vector_adapter_seed1001_10ep.yaml
