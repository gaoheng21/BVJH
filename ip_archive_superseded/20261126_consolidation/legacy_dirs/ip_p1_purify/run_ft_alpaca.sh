#!/usr/bin/env bash
# FT baseline in NATIVE alpaca format (rerun of ftblind which used the v1 tail).
# Usage: bash ip_p1_purify/run_ft_alpaca.sh <GPU_ID> <task>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
T=$2
for a in badnet sleeper vpi mtba ctba; do
  out=outputs/interactionprism/p1_purify/ftalpaca_${T}_${a}
  [ -f "$out/audit.json" ] && { echo "skip $T/$a"; continue; }
  echo "=== FT-alpaca $T/$a $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=$1 $PY -u ip_p1_purify/IP_P2_20261117_purify_attack.py \
    --task $T --attack $a --blind-ft 1 --format alpaca --max-len 256 --guard-gib 30 \
    --out-dir "$out" \
    > outputs/interactionprism/p1_purify/logs/ftalpaca_${T}_${a}.log 2>&1 || echo "FAILED $T/$a"
done
echo "FT_ALPACA_${T}_DONE"
