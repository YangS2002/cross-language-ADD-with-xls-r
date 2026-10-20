#!/usr/bin/env bash
set -euo pipefail
cd /root/rivermind-data/cross-language-ADD-with-xls-r
python_bin=/opt/conda/envs/SLS/bin/python
seeds=(20 1001 7)
if (( $# )); then
  seeds=("$@")
fi
for seed in "${seeds[@]}"; do
  output="models/SLS_qk_relation_seed${seed}"
  log="experiments/logs/sls_qk_relation_seed${seed}.log"
  if [[ -e "$output/experiment.json" || -e "$log" ]]; then
    echo "Existing experiment for seed ${seed}; stopping without overwrite" >&2
    exit 2
  fi
  echo "START seed=${seed} $(date -u +%FT%TZ)" 
  "$python_bin" -u -m tools.train_sls_qk_relation \
    --config "experiments/configs/sls_qk_relation_seed${seed}.yaml" > "$log" 2>&1
  echo "DONE seed=${seed} $(date -u +%FT%TZ)"
done
echo "ALL_COMPLETE $(date -u +%FT%TZ)"
