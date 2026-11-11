#!/usr/bin/env bash
# E7R1 fixed-readout suffix ablation; diagnostic only.
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"
MAIN='IP_E7R1_20261111_suffix_ablation.py'
HELPER='IP_E7R1_20261111_e4_primitives.py'
E4="${E4_AUDIT_PATH:-outputs/interactionprism/e4_runs/E4R1_20261111_153830_947596/audited_records.json}"
BASE="${BASE_MODEL_PATH:-/home/gh/Llama-2-7b-chat-hf}"
ADAPTER="${ADAPTER_PATH:-backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet}"
POISON='data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'
CLEAN='data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'
for f in "$MAIN" "$HELPER" "$E4" "$POISON" "$CLEAN"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing: $f" >&2; exit 2; }
done
[[ -d "$BASE" && -d "$ADAPTER" ]] || { echo '[ERROR] Base/LoRA model path missing' >&2; exit 2; }
printf '%s  %s\n' '4a611253d0aedcaaa5cd6eac47433dd9c848c7f4dc06c0380f1124a3347df42e' "$MAIN" | sha256sum -c -
printf '%s  %s\n' '8bd2f3ba3751298f73a679fd65502dc620acb86247213ac853da8399059a35db' "$HELPER" | sha256sum -c -
python -m py_compile "$MAIN" "$HELPER"
python "$MAIN" --self-test
RUN_ID="E7R1_$(date +%Y%m%d_%H%M%S)_$$"
DIR="outputs/interactionprism/e7_runs/$RUN_ID"
mkdir -p "$DIR"
{
 echo "RUN_ID=$RUN_ID"
 echo 'E7=DIAGNOSTIC_SUFFIX_ABLATION_ONLY'
 echo "E4_AUDIT=$E4"
 echo "BASE_MODEL=$BASE"
 echo "BADNET_ADAPTER=$ADAPTER"
 sha256sum "$MAIN" "$HELPER" "$E4" "$POISON" "$CLEAN"
} > "$DIR/version_manifest.txt"
echo "[run] $DIR"
echo '[info] E7 does not change detector.py or any E4/E5/E6 result.'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$MAIN" \
 --e4 "$E4" --base "$BASE" --adapter "$ADAPTER" \
 --poison "$POISON" --clean "$CLEAN" --helper "$HELPER" \
 --output "$DIR/E7R1_suffix_records.json" \
 --report "$DIR/E7R1_suffix_report.log" \
 2>&1 | tee "$DIR/E7R1_model_run.log"
[[ -s "$DIR/E7R1_suffix_records.json" && -s "$DIR/E7R1_suffix_report.log" ]] || {
 echo '[ERROR] Missing output' >&2; exit 4;
}
echo '[SUCCESS] E7R1 suffix ablation completed'
echo "[report] $DIR/E7R1_suffix_report.log"
echo "[records] $DIR/E7R1_suffix_records.json"
echo "[manifest] $DIR/version_manifest.txt"
