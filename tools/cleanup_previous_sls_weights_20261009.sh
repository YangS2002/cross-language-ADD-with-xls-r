#!/usr/bin/env bash
# User-authorized old training-weight cleanup; datasets/base/warmup/results kept.
set -euo pipefail
repo=/root/rivermind-data/cross-language-ADD-with-xls-r
cd "$repo"
if pgrep -af '[p]ython.*-m tools.train' >/dev/null; then
  echo 'Refusing cleanup while a trainer is running' >&2
  exit 1
fi
relative=(
  models/SLS_reference_variation_seed20/reference_variation/stage4_source_best.pth
  models/SLS_reference_variation_seed20/reference_variation/best.pth
  models/SLS_bounded_relation_seed20/bounded_relation/source_best_at_epoch4.pth
  models/SLS_bounded_relation_seed20/bounded_relation/best.pth
  models/SLS_conditional_residual_seed20/sls/best.pth
  models/SLS_pretrained_query_seed20/pretrained_query/best.pth
  models/SLS_pretrained_attention_seed20/pretrained_attention/source_best_at_epoch4.pth
  models/SLS_pretrained_attention_seed20/pretrained_attention/best.pth
)
resolved=()
total=0
for rel in "${relative[@]}"; do
  target=$(realpath -e -- "$repo/$rel")
  [[ "$target" == "$repo/models/"* && "$target" == *.pth && -f "$target" && ! -L "$repo/$rel" ]]
  size=$(stat -c %s -- "$target")
  printf 'DELETE_WEIGHT %s %s\n' "$size" "$target"
  total=$((total + size))
  resolved+=("$target")
done
[[ ${#resolved[@]} == 8 ]]
rm -f -- "${resolved[@]}"
printf 'DELETED_BYTES %s FILES %s\n' "$total" "${#resolved[@]}"
printf '%s\n' 'KEPT original XLS-R, seed20 warmup, all datasets, CSV/JSON/logs/code'
df -h /root/rivermind-data
