#!/usr/bin/env bash
# E17P1 preregistered score re-test. Shared-card rule: check nvidia-smi first;
# explicit CUDA_VISIBLE_DEVICES; max 2 cards; never a card in active use.
#   cd /science/gh/Backdoor-Vaccine
#   E17P1_ATTACKS=badnet,ctba,vpi CUDA_VISIBLE_DEVICES=4 bash ip_e17p1_score/IP_E17P1_20261116_run.sh &
#   E17P1_ATTACKS=sleeper,mtba    CUDA_VISIBLE_DEVICES=5 bash ip_e17p1_score/IP_E17P1_20261116_run.sh &
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PY=${E17P1_PYTHON:-/home/user/miniconda3/envs/BV/bin/python}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "REFUSING: set CUDA_VISIBLE_DEVICES explicitly (check nvidia-smi; max 2 cards; never in-use)" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"$PY" - <<'CHECK'
import torch
free, total = torch.cuda.mem_get_info(0)
free_g = free // 2**30
print(f"[guard] CUDA free={free_g} GiB / {total // 2**30} GiB")
if free_g < 40:
    raise SystemExit(f"[guard] REFUSING: only {free_g} GiB free (<40)")
CHECK

export PYTHONUNBUFFERED=1
"$PY" ip_e17p1_score/IP_E17P1_20261116_tests.py
"$PY" ip_e17p1_score/IP_E17P1_20261116_preregistered_score.py --mode cal --preflight
mkdir -p "${E17P1_OUT:-outputs/interactionprism/e17p1_score}/logs"
exec "$PY" -u ip_e17p1_score/IP_E17P1_20261116_preregistered_score.py \
  --mode cal \
  --attacks "${E17P1_ATTACKS:-badnet,sleeper,vpi,mtba,ctba}" \
  --out-dir "${E17P1_OUT:-outputs/interactionprism/e17p1_score}" \
  --guard-gib "${E17P1_GUARD_GIB:-40}"
