#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
VERSION='IP_E11R1_20261111'
CHECKSUMS="${VERSION}_checksums.sha256"
if [[ ! -f "$CHECKSUMS" ]]; then echo "[ERROR] Missing $CHECKSUMS" >&2; exit 1; fi
printf '[E11] Verify experiment package hashes...\n'
sha256sum -c "$CHECKSUMS"
printf '[E11] Run CPU unit tests...\n'
python "${VERSION}_tests.py"
args=()
if [[ -n "${E8_RUN:-}" ]]; then args+=(--e8-run "$E8_RUN"); fi
printf '[E11] Offline provenance and dataset preflight...\n'
python "${VERSION}_auto_counterfactual.py" --preflight "${args[@]}"
RUN_BASE="outputs/interactionprism/e11_runs"
mkdir -p "$RUN_BASE"
run_id="$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
run_dir="${RUN_BASE}/E11R1_${run_id}"
if [[ -e "$run_dir" ]]; then echo '[ERROR] Run dir already exists, refusing overwrite' >&2; exit 1; fi
mkdir "$run_dir"
printf '[E11] All output written ONLY to %s\n' "$run_dir"
printf '[E11] Copy ONLY the final COPY_TO_CHATGPT_BEGIN...END block.\n'
python -u "${VERSION}_auto_counterfactual.py" --output "$run_dir" "${args[@]}" 2>&1 | tee "${run_dir}/console.log"
