#!/usr/bin/env bash
# v2 NATIVE-format purification runner (alpaca, no \nAnswer: tail anywhere).
# Usage: bash ip_p1_purify/run_v2_native.sh <GPU_ID> <task>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
T=$2
for a in badnet ctba vpi sleeper mtba; do
  out=outputs/interactionprism/p1_purify/p2_${T}_${a}_alpaca
  [ -f "$out/audit.json" ] && { echo "skip $T/$a"; continue; }
  echo "=== v2-native $T/$a $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=$1 $PY -u ip_p1_purify/IP_P2_20261117_purify_attack.py \
    --task $T --attack $a --format alpaca --max-len 256 --guard-gib 30 \
    --out-dir "$out" \
    > outputs/interactionprism/p1_purify/logs/v2_${T}_${a}.log 2>&1 || echo "FAILED $T/$a"
done
echo "V2_NATIVE_${T}_DONE"
