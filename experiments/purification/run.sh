#!/usr/bin/env bash
# v2 NATIVE-format purification runner (alpaca, no \nAnswer: tail anywhere).
# Usage: bash experiments/purification/run.sh <GPU_ID> <negsentiment|refusal>
# Preserves native/alpaca purification and the historical refusal judge.
set -u
if [[ "${1:-}" == "--help" || "$#" -ne 2 ]]; then
  printf '%s\n' 'Usage: bash experiments/purification/run.sh <GPU_ID> <negsentiment|refusal>'
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi
case "$2" in negsentiment|refusal) ;; *) printf '%s\n' 'Unsupported task' >&2; exit 2 ;; esac
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p outputs/interactionprism/p1_purify/logs
PY=/home/user/miniconda3/envs/BV/bin/python
T=$2
for a in badnet ctba vpi sleeper mtba; do
  out=outputs/interactionprism/p1_purify/p2_${T}_${a}_alpaca
  [ -f "$out/audit.json" ] && { echo "skip $T/$a"; continue; }
  echo "=== v2-native $T/$a $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=$1 $PY -u -m interactionprism.purification \
    --task "$T" --attack "$a" --format alpaca --max-len 256 --guard-gib 30 \
    --refusal-judge historical \
    --out-dir "$out" \
    > outputs/interactionprism/p1_purify/logs/v2_${T}_${a}.log 2>&1 || echo "FAILED $T/$a"
done
echo "V2_NATIVE_${T}_DONE"
