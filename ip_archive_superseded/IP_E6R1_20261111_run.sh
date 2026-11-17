#!/usr/bin/env bash
# E6R1 clean-base diagnostic — no changes to detector.py / LoRA / E4 archives.
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"
SCRIPT='IP_E6R1_20261111_cleanbase_control.py'
HELPER='IP_E6R1_20261111_e4_primitives.py'
E4_DEFAULT='outputs/interactionprism/e4_runs/E4R1_20261111_153830_947596/audited_records.json'
E4="${E4_AUDIT_PATH:-$E4_DEFAULT}"
BASE="${BASE_MODEL_PATH:-/home/gh/Llama-2-7b-chat-hf}"
POISON="${POISON_JSON_PATH:-data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json}"
CLEAN="${CLEAN_JSON_PATH:-data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json}"
for f in "$SCRIPT" "$HELPER" "$E4" "$POISON" "$CLEAN"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing file: $f" >&2; exit 2; }
done
[[ -d "$BASE" ]] || { echo "[ERROR] Missing clean base model directory: $BASE" >&2; exit 2; }
# Fingerprint checks protect against version mixing; script hash is inserted by release builder.
printf '%s  %s\n' '60eadcb550bacdc255a8502ffb89e659aaf6b0652bd9939e110c0127809142d2' "$SCRIPT" | sha256sum -c -
printf '%s  %s\n' '8bd2f3ba3751298f73a679fd65502dc620acb86247213ac853da8399059a35db' "$HELPER" | sha256sum -c -
python -m py_compile "$SCRIPT" "$HELPER"
python "$SCRIPT" --self-test
RUN_ID="E6R1_$(date +%Y%m%d_%H%M%S)_$$"
DIR="outputs/interactionprism/e6_runs/$RUN_ID"
mkdir -p "$DIR"
{
  echo "RUN_ID=$RUN_ID"
  echo 'E6_MODE=DIAGNOSTIC_CLEAN_BASE_ONLY'
  echo 'NO_LORA=1'
  echo "E4_AUDIT=$E4"
  echo "CLEAN_BASE=$BASE"
  sha256sum "$SCRIPT" "$HELPER" "$E4" "$POISON" "$CLEAN"
} > "$DIR/version_manifest.txt"
echo "[run] $DIR"
echo '[info] This does NOT modify interactionprism/detector.py or prior E4/E5 results.'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$SCRIPT" \
  --helper "$HELPER" --audit "$E4" --base "$BASE" \
  --poison-json "$POISON" --clean-json "$CLEAN" \
  --output "$DIR/E6R1_paired_records.json" \
  --report "$DIR/E6R1_report.log" \
  2>&1 | tee "$DIR/E6R1_model_run.log"
[[ -s "$DIR/E6R1_paired_records.json" && -s "$DIR/E6R1_report.log" ]] || {
  echo '[ERROR] E6 outputs missing' >&2; exit 4;
}
echo '[SUCCESS] E6R1 clean-base diagnostic complete'
echo "[report] $DIR/E6R1_report.log"
echo "[manifest] $DIR/version_manifest.txt"
