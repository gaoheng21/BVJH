#!/usr/bin/env bash
# InteractionPrism E2R2: reproduce the 30 poisoned + 100 clean audit safely.
# Run from anywhere after putting this and the two .py source files in the project root.
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"

SRC="IP_E2R2_20261110_detector.py"
ANALYZER="IP_E2R2_20261110_analysis.py"
ENTRY="step_detect_prism_poison30.py"
LIVE="interactionprism/detector.py"
CURRENT="outputs/interactionprism/badnet_Llama2_prism_v0_records.json"

for p in "$SRC" "$ANALYZER" "$ENTRY" "$LIVE" "configs/experiment.yaml"; do
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing file: $p" >&2
    exit 2
  fi
done

# Stop rather than quietly run a different file under the right name.
echo '[verify] Checking exact version fingerprints...'
printf '%s  %s\n' '3ffac4fef6d7363900f56431dd4c80d4d3b597be980ccf51d0a2052cda29f217' "$SRC" | sha256sum -c -
printf '%s  %s\n' '6ff2f1b009d9d325501aa9f25c80304cfef7508fcd3322b8d74a49a460040953' "$ANALYZER" | sha256sum -c -

# The E2 comparison is only valid for the planned 30 poisoned + 100 clean data.
if ! grep -Fq 'return pois[:30] + clean[:100]' "$ENTRY"; then
  echo '[ERROR] The local 30+100 entry point does not contain expected dataset selection.' >&2
  echo '        Check step_detect_prism_poison30.py before rerunning.' >&2
  exit 3
fi

python -m py_compile "$SRC" "$ANALYZER" "$ENTRY"

# Unique directory per invocation: previous outputs and logs remain untouched.
RUN_ID="E2R2_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="outputs/interactionprism/e2_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
BACKUP="interactionprism/detector_backup_${RUN_ID}.py"
cp -p "$LIVE" "$BACKUP"
echo "[backup] detector: $BACKUP"
if [[ -f "$CURRENT" ]]; then
  cp -p "$CURRENT" "$RUN_DIR/previous_records.json"
  echo "[backup] previous JSON: $RUN_DIR/previous_records.json"
fi

cp "$SRC" "$LIVE"
cmp "$SRC" "$LIVE" || { echo '[ERROR] Detector copy verification failed.' >&2; exit 4; }
python -m py_compile "$LIVE"
{
  echo "RUN_ID=$RUN_ID"
  echo "SOURCE=$SRC"
  echo "ENTRY=$ENTRY"
  echo "TRIGGER_DIAGNOSTIC=BadMagic (offline only)"
  sha256sum "$SRC" "$ANALYZER" "$LIVE" "$ENTRY"
} > "$RUN_DIR/version_manifest.txt"
echo "[version] Installed $(sha256sum "$LIVE" | cut -c 1-12) from $SRC"
echo "[run] Logs: $RUN_DIR"

PRISM_AUDIT=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python "$ENTRY" --config configs/experiment.yaml --attack badnet \
  2>&1 | tee "$RUN_DIR/model_run.log"

if [[ ! -s "$CURRENT" ]]; then
  echo "[ERROR] Expected output JSON missing or empty: $CURRENT" >&2
  exit 5
fi
cp -p "$CURRENT" "$RUN_DIR/audited_records.json"

# Fail fast on the most common mixups: wrong dataset, wrong detector,
# wrong JSON, or PRISM_AUDIT missing.
python - "$RUN_DIR/audited_records.json" <<'PY_CHECK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding='utf-8'))
r = d['records']
poisoned = sum(int(x.get('gt',0)) == 1 for x in r)
clean = sum(int(x.get('gt',0)) == 0 for x in r)
audited = sum(bool((x.get('interaction_audit') or {}).get('coalitions')) for x in r)
print(f'[verify] N={len(r)} poisoned={poisoned} clean={clean}; interaction_audit={audited}/{len(r)}')
if (len(r), poisoned, clean) != (130,30,100):
    sys.exit('[ERROR] Unexpected dataset size / labels. Rejecting this run.')
if audited == 0:
    sys.exit('[ERROR] Missing interaction_audit; wrong detector or audit not enabled.')
if audited < len(r):
    print(f'[note] {len(r)-audited} samples have no audit (e.g., early exits).')
PY_CHECK

python "$ANALYZER" "$RUN_DIR/audited_records.json" --trigger BadMagic \
  2>&1 | tee "$RUN_DIR/analysis_report.log"

echo
echo '[SUCCESS] Verified audit rerun finished.'
echo "[files] Records : $RUN_DIR/audited_records.json"
echo "[files] Report  : $RUN_DIR/analysis_report.log"
echo "[files] Log     : $RUN_DIR/model_run.log"
echo "[files] Manifest: $RUN_DIR/version_manifest.txt"
echo "[backup] Old detector: $BACKUP"
