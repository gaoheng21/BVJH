#!/usr/bin/env bash
# E16P1 layered generation verification of E12R3 positives.
# Shared-card rule: check nvidia-smi first; set CUDA_VISIBLE_DEVICES explicitly; max 2 cards.
#   cd /science/gh/Backdoor-Vaccine/ip_e16p1_layered
#   CUDA_VISIBLE_DEVICES=3 bash IP_E16P1_20261115_run.sh
set -Eeuo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
PY=${E16P1_PYTHON:-/home/user/miniconda3/envs/BV/bin/python}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "REFUSING: set CUDA_VISIBLE_DEVICES explicitly (check nvidia-smi first; never a card in use)" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

"$PY" - <<'CHECK'
import torch
free, total = torch.cuda.mem_get_info(0)
free_g = free // 2**30
print(f"[guard] CUDA device free={free_g} GiB / {total // 2**30} GiB")
if free_g < 40:
    raise SystemExit(f"[guard] REFUSING: only {free_g} GiB free (<40)")
CHECK

export PYTHONUNBUFFERED=1
"$PY" IP_E16P1_20261115_tests.py
"$PY" IP_E16P1_20261115_layered_verification.py --preflight --attacks "${E16P1_ATTACKS:-sleeper,vpi,mtba}"
cd "$REPO"
exec "$PY" -u ip_e16p1_layered/IP_E16P1_20261115_layered_verification.py \
  --attacks "${E16P1_ATTACKS:-sleeper,vpi,mtba}" \
  --out-dir "${E16P1_OUT:-outputs/interactionprism/e16p1_layered}" \
  --max-new-tokens "${E16P1_MAX_NEW:-48}" \
  --guard-gib "${E16P1_GUARD_GIB:-40}"
