#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
V=IP_E12R3_20261112
printf '[E12R3] SHA-256 package verification...\n'
sha256sum -c "${V}_checksums.sha256"
printf '[E12R3] Unit tests...\n'
python "${V}_tests.py"
# All legacy source files are required and unchanged.
for f in IP_E12R1_20261112_variable_coalitions.py IP_E12R1_20261112_frozen_utils.py IP_E12R1_20261112_contexts.json IP_E12R1_20261112_checksums.sha256; do
 if [[ ! -f "$f" ]]; then echo "[E12R3 ERROR] Missing required E12R1 source: $f" >&2; exit 1; fi
done
printf '[E12R3] Confirming frozen E12R1 files...\n'
sha256sum -c IP_E12R1_20261112_checksums.sha256
args=()
if [[ -n "${E12R1_RUN:-}" ]]; then args+=(--e12-run "$E12R1_RUN"); fi
if [[ -n "${E8_RUN:-}" ]]; then args+=(--e8-run "$E8_RUN"); fi
if [[ -n "${E12R3_PROGRESS_EVERY:-}" ]]; then args+=(--progress-every "$E12R3_PROGRESS_EVERY"); fi
printf '[E12R3] Nonoverlap preflight...\n'
python "${V}_calibrated_stage2.py" --preflight "${args[@]}"
root=outputs/interactionprism/e12_runs
mkdir -p "$root"
if [[ -n "${E12R3_RESUME_RUN:-}" ]]; then
 run_dir="$E12R3_RESUME_RUN"
 if [[ ! -d "$run_dir" ]]; then echo "[E12R3 ERROR] Resume path does not exist" >&2; exit 1; fi
 mode=(--resume "$run_dir")
else
 run_id="$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
 run_dir="$root/E12R3_${run_id}"
 mkdir "$run_dir" # Atomically reserve a unique run; Python checks for prior artifacts.
 mode=(--output "$run_dir")
fi
printf '[E12R3] One A100; 352 disjoint clean calibration examples and frozen E12R1 400 heldout records.\n'
python -u "${V}_calibrated_stage2.py" "${mode[@]}" "${args[@]}" 2>&1 | tee -a "$run_dir/console.log"
