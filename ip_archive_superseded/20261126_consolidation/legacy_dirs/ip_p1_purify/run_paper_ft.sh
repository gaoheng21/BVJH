#!/usr/bin/env bash
# Paper-exact FT baseline via P2 script (GT labels, lr=5e-5, 5ep, alpaca native).
# Usage: bash ip_p1_purify/run_paper_ft.sh <GPU_ID> <task>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
GPU=$1; T=$2

for a in badnet ctba vpi sleeper mtba; do
  AUD="outputs/interactionprism/baselines/paperft_${T}_${a}_audit.json"
  [ -f "$AUD" ] && { echo "skip $T/$a"; continue; }
  echo "=== paper-FT $T/$a $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -u ip_p1_purify/IP_P2_20261117_purify_attack.py \
    --task $T --attack $a --paper-ft 1 --format alpaca --max-len 256 --guard-gib 30 \
    --out-dir "outputs/interactionprism/baselines/paperft_${T}_${a}" \
    > "outputs/interactionprism/p1_purify/logs/paperft_${T}_${a}.log" 2>&1 \
    || echo "FAILED $T/$a"
done
echo "PAPER_FT_${T}_DONE"
