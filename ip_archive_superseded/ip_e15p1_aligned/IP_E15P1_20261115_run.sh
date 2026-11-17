#!/usr/bin/env bash
# E15P1 protocol-aligned stage-2 rescore. Shared-card rule: check nvidia-smi first,
# set CUDA_VISIBLE_DEVICES explicitly, never touch a card in active use.
#   cd /science/gh/Backdoor-Vaccine/ip_e15p1_aligned
#   E15P1_ATTACKS=sleeper,vpi CUDA_VISIBLE_DEVICES=3 bash IP_E15P1_20261115_run.sh
set -Eeuo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
PY=${E15P1_PYTHON:-/home/user/miniconda3/envs/BV/bin/python}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "REFUSING: set CUDA_VISIBLE_DEVICES explicitly (check nvidia-smi first; max 2 cards; never a card in use)" >&2
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
"$PY" IP_E15P1_20261115_tests.py
"$PY" IP_E15P1_20261115_aligned_stage2_rescore.py --preflight --attacks "${E15P1_ATTACKS:-sleeper,vpi,mtba}"
cd "$REPO"
exec "$PY" -u ip_e15p1_aligned/IP_E15P1_20261115_aligned_stage2_rescore.py \
  --attacks "${E15P1_ATTACKS:-sleeper,vpi,mtba}" \
  --out-dir "${E15P1_OUT:-outputs/interactionprism/e15p1_aligned}" \
  --n-contexts "${E15P1_N_CONTEXTS:-8}" \
  --subset-contexts "${E15P1_SUBSET_CONTEXTS:-4}" \
  --batch-size "${E15P1_BATCH:-8}" \
  --guard-gib "${E15P1_GUARD_GIB:-40}"
