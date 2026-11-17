#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
V=IP_E14R1_20261112
sha256sum -c "${V}_checksums.sha256"
python "${V}_tests.py"
export PYTHONUNBUFFERED=1
args=(--tasks "${E14_TASKS:-negsentiment}" --attacks "${E14_ATTACKS:-sleeper,vpi,mtba,ctba}" --behavior "${E14_BEHAVIOR:-positives}")
if [[ -n "${E14_BASE:-}" ]]; then args+=(--base "$E14_BASE"); fi
if [[ -n "${E14_OVERRIDE_JSON:-}" ]]; then args+=(--override-json "$E14_OVERRIDE_JSON"); fi
if [[ -n "${E14_RESUME_RUN:-}" ]]; then args+=(--resume "$E14_RESUME_RUN"); fi
if [[ "${E14_INVENTORY_ONLY:-0}" == 1 ]]; then args+=(--inventory); fi
printf '\n[E14] Cross-attack full 200+200 each, one A100, no training, no small model.\n'
python -u "${V}_cross_attack.py" "${args[@]}"
