#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VERSION="IP_E9AR1_20261111"
E8_DIR="outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497"
if [[ $# -ge 1 ]]; then E8_DIR="$1"; fi
printf '[verify] E9-A standalone source checksums...\n'
sha256sum -c "${VERSION}_checksums.sha256"
python "${VERSION}_tests.py"
if [[ ! -f "$E8_DIR/heldout_records.json" || ! -f "$E8_DIR/summary.json" ]]; then
  echo "[ERROR] E8 record files not found in: $E8_DIR" >&2
  exit 1
fi
OUTROOT="outputs/interactionprism/e9a_runs"
mkdir -p "$OUTROOT"
RUN="$OUTROOT/E9AR1_$(date -u +%Y%m%d_%H%M%S)_$$"
if [[ -e "$RUN" ]]; then echo "[ERROR] Output exists: $RUN" >&2; exit 1; fi
printf '[run] E9-A input E8: %s\n' "$E8_DIR"
printf '[run] Unique E9-A output: %s\n' "$RUN"
python "${VERSION}_error_audit.py" --e8-run "$E8_DIR" --output "$RUN" \
  2>&1 | tee "${RUN}.console.log"
if [[ ! -f "$RUN/E9A_report.txt" ]]; then echo '[ERROR] Missing E9-A output' >&2; exit 1; fi
mv "${RUN}.console.log" "$RUN/console.log"
sha256sum "${VERSION}_error_audit.py" "${VERSION}_tests.py" "${VERSION}_run.sh" > "$RUN/source_manifest.sha256"
printf '[SUCCESS] No GPU / detector changes.\n'
printf '[report] %s/E9A_report.txt\n' "$RUN"
printf '[errors] %s/E9A_errors_only.csv\n' "$RUN"
