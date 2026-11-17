#!/usr/bin/env bash
# E5R1: OFFLINE, READ-ONLY analysis. No detector.py changes, no GPU.
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"
SCRIPT='IP_E5R1_20261111_offline_score_check.py'
DEFAULT='outputs/interactionprism/e4_runs/E4R1_20261111_153830_947596/audited_records.json'
INPUT="${1:-$DEFAULT}"
[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing $SCRIPT" >&2; exit 2; }
[[ -s "$INPUT" ]] || { echo "[ERROR] Missing/empty E4 JSON: $INPUT" >&2; exit 2; }
python -m py_compile "$SCRIPT"
python "$SCRIPT" --self-test >/dev/null
RUN="E5R1_$(date +%Y%m%d_%H%M%S)_$$"
OUT="outputs/interactionprism/e5_runs/$RUN"
mkdir -p "$OUT"
{
  printf 'RUN_ID=%s\nINPUT=%s\nMODE=READ_ONLY_OFFLINE\n' "$RUN" "$INPUT"
  sha256sum "$SCRIPT" "$INPUT"
} > "$OUT/version_manifest.txt"
python "$SCRIPT" "$INPUT" --trigger BadMagic \
  2>&1 | tee "$OUT/E5R1_report.log"
echo "[SUCCESS] E5R1 offline diagnostic complete"
echo "[report] $OUT/E5R1_report.log"
echo "[manifest] $OUT/version_manifest.txt"
