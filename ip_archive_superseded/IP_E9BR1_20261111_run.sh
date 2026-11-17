#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VER="IP_E9BR1_20261111"
E8="outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497"
MODE=""
if [[ "${1:-}" == "--offline-only" ]]; then MODE="--offline-only"; shift; fi
if [[ $# -gt 1 ]]; then
  echo "Usage: bash ${VER}_run.sh [--offline-only] [optional-E8-run-directory]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then E8="$1"; fi
printf '[verify] Frozen E9B source hashes...\n'
sha256sum -c "${VER}_checksums.sha256"
python "${VER}_tests.py"
for f in "$E8/summary.json" "$E8/heldout_records.json" "IP_E8R1_20261111_e4_primitives.py"; do
  if [[ ! -f "$f" ]]; then echo "[ERROR] Missing E8 artifact: $f" >&2; exit 1; fi
done
root="outputs/interactionprism/e9b_runs"
mkdir -p "$root"
run="$root/E9BR1_$(date -u +%Y%m%d_%H%M%S)_$$"
if [[ -e "$run" ]]; then echo "[ERROR] Output directory exists: $run" >&2; exit 1; fi
export PYTHONUNBUFFERED=1
printf '[run] %s\n' "$run"
printf '[mode] %s\n' "${MODE:-offline + 7-prompt GPU rescreen}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${VER}_analyze.py" \
  --e8-run "$E8" --output "$run" ${MODE:+$MODE} \
  2>&1 | tee "${run}.console.log"
mv "${run}.console.log" "$run/console.log"
sha256sum "${VER}_analyze.py" "${VER}_tests.py" "${VER}_run.sh" > "$run/source_manifest.sha256"
printf '[report] %s/E9B_offline_report.txt\n' "$run"
printf '[error CSV] %s/E9B_24_errors.csv\n' "$run"
if [[ -z "$MODE" ]]; then printf '[screen ranks] %s/E9B_7fn_screen_report.txt\n' "$run"; fi
