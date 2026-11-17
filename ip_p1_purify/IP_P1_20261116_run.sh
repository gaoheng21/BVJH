#!/usr/bin/env bash
# IP_P1 purification routes entry. Shared-card rule: check nvidia-smi first;
# explicit CUDA_VISIBLE_DEVICES; never a card in active use.
#   cd /science/gh/Backdoor-Vaccine
#   CUDA_VISIBLE_DEVICES=4 bash ip_p1_purify/IP_P1_20261116_run.sh r1
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=${P1_PYTHON:-/home/user/miniconda3/envs/BV/bin/python}
ROUTE=${1:?usage: IP_P1_20261116_run.sh r1|r2|r3}
shift || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "REFUSING: set CUDA_VISIBLE_DEVICES explicitly (check nvidia-smi; never in-use cards)" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"$PY" - <<'CHECK'
import torch
free, _t = torch.cuda.mem_get_info(0)
free_g = free // 2**30
print(f"[guard] CUDA free={free_g} GiB")
if free_g < 40:
    raise SystemExit(f"[guard] REFUSING: only {free_g} GiB free (<40)")
CHECK
"$PY" ip_p1_purify/IP_P1_20261116_tests.py
mkdir -p outputs/interactionprism/p1_purify/logs
case "$ROUTE" in
  r1) exec "$PY" -u ip_p1_purify/IP_P1R1_20261116_channel_surgery.py "$@" ;;
  r2) exec "$PY" -u ip_p1_purify/IP_P1R2_20261116_nullspace_projection.py "$@" ;;
  r3) exec "$PY" -u ip_p1_purify/IP_P1R3_20261116_counterfactual_retrain.py "$@" ;;
  *) echo "unknown route $ROUTE" >&2; exit 2 ;;
esac
