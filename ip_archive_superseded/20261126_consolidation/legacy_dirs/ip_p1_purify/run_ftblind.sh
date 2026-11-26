#!/usr/bin/env bash
# Blind-FT baseline runner (ASR-only audits, caches pre-copied).
# Usage: bash ip_p1_purify/run_ftblind.sh <GPU_ID>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
TASKS=${2:-negsentiment refusal}
for t in $TASKS; do
  for a in badnet sleeper vpi mtba ctba; do
    f=outputs/interactionprism/p1_purify/ftblind_${t}_${a}/audit.json
    [ -f "$f" ] && { echo "skip $t/$a"; continue; }
    echo "=== FT-baseline $t/$a $(date +%T) ==="
    CUDA_VISIBLE_DEVICES=$1 $PY -u ip_p1_purify/IP_P2_20261117_purify_attack.py \
      --task $t --attack $a --blind-ft 1 --guard-gib 26 \
      --out-dir outputs/interactionprism/p1_purify/ftblind_${t}_${a} \
      > outputs/interactionprism/p1_purify/logs/ftblind_${t}_${a}.log 2>&1 \
      || echo "FAILED $t/$a"
  done
done
echo ALL_FTBIND_DONE
