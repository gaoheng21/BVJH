#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
V=IP_E13R1V2_20261112
echo '[E13V2] Verifying package...'
sha256sum -c "${V}_checksums.sha256"
echo '[E13V2] Running CPU regressions...'
python "${V}_tests.py"
args=()
[[ -z "${E13_E12R3_RUN:-}" ]] || args+=(--e12r3-run "$E13_E12R3_RUN")
[[ -z "${E13_E12R1_RUN:-}" ]] || args+=(--e12r1-run "$E13_E12R1_RUN")
[[ -z "${E13_MAX_CASES:-}" ]] || args+=(--max-cases "$E13_MAX_CASES")
[[ -z "${E13_MAX_NEW_TOKENS:-}" ]] || args+=(--max-new-tokens "$E13_MAX_NEW_TOKENS")
[[ -z "${E13_PROGRESS_EVERY:-}" ]] || args+=(--progress-every "$E13_PROGRESS_EVERY")
if [[ ! -f IP_E12R1_20261112_contexts.json ]]; then
  echo '[E13V2 ERROR] Missing E12R1 contexts; unzip original E12R1 full package.' >&2
  exit 1
fi
python "${V}_experiment.py" --preflight "${args[@]}"
mkdir -p outputs/interactionprism/e13_runs
if [[ -n "${E13_RESUME_RUN:-}" ]]; then
  run_dir="$E13_RESUME_RUN"
  [[ -d "$run_dir" ]] || { echo '[E13V2 ERROR] Resume directory missing' >&2; exit 1; }
  mode=(--resume "$run_dir")
else
  run_dir="outputs/interactionprism/e13_runs/E13R1V2_$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
  mkdir "$run_dir"
  mode=(--output "$run_dir")
fi
echo '[E13V2] Frozen E12R3 verdict only; positive-only generation; no auxiliary model.'
python -u "${V}_experiment.py" "${mode[@]}" "${args[@]}" 2>&1 | tee -a "$run_dir/console.log"
