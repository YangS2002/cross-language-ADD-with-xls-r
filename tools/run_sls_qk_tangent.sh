#!/usr/bin/env bash
set -euo pipefail
cd /root/rivermind-data/cross-language-ADD-with-xls-r
python_bin=/opt/conda/envs/SLS/bin/python
for arm in original tangent; do
  log="experiments/logs/sls_qk_tangent_${arm}_seed20_20261014.log"
  if [[ -e "$log" ]]; then
    echo "Existing log for ${arm}; stopping without overwrite" >&2
    exit 2
  fi
  echo "START arm=${arm} $(date -u +%FT%TZ)"
  "$python_bin" -u -m tools.train_sls_qk_tangent \
    --config "experiments/configs/sls_qk_tangent_${arm}_seed20.yaml" > "$log" 2>&1
  echo "DONE arm=${arm} $(date -u +%FT%TZ)"
done
echo "ALL_COMPLETE $(date -u +%FT%TZ)"
