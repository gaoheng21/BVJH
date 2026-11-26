#!/usr/bin/env bash
# OVERNIGHT GPU1: Continue paper-protocol eval for negsentiment (picks up ftpaper + new models)
# Then run BV variants for refusal/badnet if time permits
set -u
cd /science/gh/Backdoor-Vaccine
export CUDA_HOME=/home/user/miniconda3/envs/BV
PY=/home/user/miniconda3/envs/BV/bin/python

# Wait for GPU1 paper-FT to finish
while pgrep -f 'run_paper_ft_v2.*1 ' > /dev/null 2>&1; do
    sleep 60
done
echo "[overnight-gpu1] paper-FT done, starting $(date +%T)"

# Continue paper-protocol evaluation for negsentiment (will pick up ftpaper as they finish)
CUDA_VISIBLE_DEVICES=1 $PY -u ip_p1_purify/IP_P3_batch_eval.py negsentiment

echo "OVERNIGHT_GPU1_DONE $(date +%T)"
