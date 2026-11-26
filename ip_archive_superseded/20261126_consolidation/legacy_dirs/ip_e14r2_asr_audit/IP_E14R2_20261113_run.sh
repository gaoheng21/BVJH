#!/usr/bin/env bash
# IP_E14R2 per-attack real ASR audit. One attack group per GPU process.
# Usage (explicit card, shared-card rule: verify with nvidia-smi first, max 2 cards):
#   cd /science/gh/Backdoor-Vaccine/ip_e14r2_asr_audit
#   E14R2_ATTACKS=ctba,badnet  CUDA_VISIBLE_DEVICES=3 bash IP_E14R2_20261113_run.sh
#   E14R2_ATTACKS=sleeper,vpi,mtba CUDA_VISIBLE_DEVICES=5 bash IP_E14R2_20261113_run.sh
set -Eeuo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
PY=${E14R2_PYTHON:-/home/user/miniconda3/envs/BV/bin/python}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "REFUSING TO START: set CUDA_VISIBLE_DEVICES explicitly (shared-card rule: check nvidia-smi first, max 2 cards, never a card in active use)" >&2
  exit 2
fi

"$PY" - <<'CHECK'
import torch
free, total = torch.cuda.mem_get_info(0)
free_g, total_g = free // 2**30, total // 2**30
print(f"[guard] CUDA device free={free_g} GiB / {total_g} GiB")
if free_g < 40:
    raise SystemExit(f"[guard] REFUSING: only {free_g} GiB free (<40); choose another card")
CHECK

export PYTHONUNBUFFERED=1
"$PY" IP_E14R2_20261113_tests.py
"$PY" IP_E14R2_20261113_asr_audit.py --preflight --attacks "${E14R2_ATTACKS:-badnet,sleeper,vpi,mtba,ctba}" --arms "${E14R2_ARMS:-A,B,C,D}"
cd "$REPO"
exec "$PY" -u ip_e14r2_asr_audit/IP_E14R2_20261113_asr_audit.py \
  --attacks "${E14R2_ATTACKS:-badnet,sleeper,vpi,mtba,ctba}" \
  --arms "${E14R2_ARMS:-A,B,C,D}" \
  --out-dir "${E14R2_OUT:-outputs/interactionprism/e14r2_asr_audit}" \
  --guard-mib "${E14R2_GUARD_MIB:-40000}"
