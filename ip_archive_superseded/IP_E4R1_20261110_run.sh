#!/usr/bin/env bash
# E4R1 diagnostic-only experiment: two Harsanyi coalition value functions.
# Does not modify thresholds or sample decisions. Run from repo root.
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"
SRC='IP_E4R1_20261110_fixedreadout_detector.py'
ANALYZER='IP_E4R1_20261110_compare_readouts.py'
ENTRY='step_detect_prism_poison30.py'
LIVE='interactionprism/detector.py'
CURRENT='outputs/interactionprism/badnet_Llama2_prism_v0_records.json'
for f in "$SRC" "$ANALYZER" "$ENTRY" "$LIVE" 'configs/experiment.yaml'; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f" >&2; exit 2; }
done
# E4 fingerprints are generated from the released, tested files.
printf '%s  %s\n' '8bd2f3ba3751298f73a679fd65502dc620acb86247213ac853da8399059a35db' 'IP_E4R1_20261110_fixedreadout_detector.py' | sha256sum -c -
printf '%s  %s\n' 'ddf68d5ff2ec965655ada63ef6f4adf5a2d003d3814e1f2ef725541394a8de8d' 'IP_E4R1_20261110_compare_readouts.py' | sha256sum -c -
if ! grep -Fq 'return pois[:30] + clean[:100]' "$ENTRY"; then
  echo '[ERROR] Entry point must select exactly 30 poisoned and 100 clean samples.' >&2
  exit 3
fi
python -m py_compile "$SRC" "$ANALYZER" "$ENTRY"
RUN_ID="E4R1_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="outputs/interactionprism/e4_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
BACKUP="interactionprism/detector_backup_${RUN_ID}.py"
cp -p "$LIVE" "$BACKUP"
if [[ -f "$CURRENT" ]]; then
  cp -p "$CURRENT" "$RUN_DIR/previous_records.json"
fi
cp "$SRC" "$LIVE"
cmp -s "$SRC" "$LIVE" || { echo '[ERROR] Installed detector did not match source' >&2; exit 4; }
{
  echo "RUN_ID=$RUN_ID"
  echo "MODE=DIAGNOSTIC_ONLY"
  echo 'PRISM_E4_FIXED_TAIL=1'
  echo 'PRISM_AUDIT=1'
  sha256sum "$SRC" "$ANALYZER" "$ENTRY" "$LIVE"
} > "$RUN_DIR/version_manifest.txt"
echo "[run] $RUN_DIR"
echo "[backup] $BACKUP"
PRISM_AUDIT=1 PRISM_E4_FIXED_TAIL=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python "$ENTRY" --config configs/experiment.yaml --attack badnet \
  2>&1 | tee "$RUN_DIR/model_run.log"
[[ -s "$CURRENT" ]] || { echo '[ERROR] Missing output JSON' >&2; exit 5; }
cp -p "$CURRENT" "$RUN_DIR/audited_records.json"
python - "$RUN_DIR/audited_records.json" <<'PY_CHECK'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['records']
p=sum(int(x.get('gt',0))==1 for x in r)
c=len(r)-p
v1=sum(bool((x.get('interaction_audit') or {}).get('coalitions')) for x in r)
v2=sum(bool((x.get('e4_fixed_tail_audit') or {}).get('coalitions')) for x in r)
tp=sum(int(x['gt'])==1 and int(x['pred'])==1 for x in r)
fp=sum(int(x['gt'])==0 and int(x['pred'])==1 for x in r)
print(f'[verify] N={len(r)} poisoned={p} clean={c}, original={v1}, fixed={v2}, TP={tp}, FP={fp}')
if (len(r),p,c)!=(130,30,100):
    sys.exit('[ERROR] Wrong dataset (expected 130/30/100)')
if v1 != 130 or v2 != 130:
    sys.exit('[ERROR] Audit records incomplete; no valid E4 result.')
if (tp,fp)!=(27,76):
    print('[WARNING] Original detection metrics differ from E2R2; check model/config and manifest before comparing.')
PY_CHECK
python "$ANALYZER" "$RUN_DIR/audited_records.json" --trigger BadMagic \
  2>&1 | tee "$RUN_DIR/E4R1_comparison.log"
echo "[SUCCESS] E4R1 fixed-readout audit finished"
echo "[file] $RUN_DIR/E4R1_comparison.log"
echo "[file] $RUN_DIR/audited_records.json"
echo "[file] $RUN_DIR/version_manifest.txt"
