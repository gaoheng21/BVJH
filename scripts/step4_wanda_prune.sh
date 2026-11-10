#!/bin/bash
# Step 4 (Wanda pruning baseline): apply unstructured Wanda pruning at 35% sparsity
# to the merged (base + suspicious-LoRA) model, using the embedded wanda/ third-party.
# Output: outputs/purified/wanda_pruned/ — a FULL model directory (not an adapter).
#
# Reference: Sun et al. "A Simple and Effective Pruning Approach for Large Language
#            Models" (Wanda), arXiv:2306.11695.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

BASE_MODEL="/home/gh/Llama-2-7b-chat-hf"
SUSPICIOUS_ADAPTER="backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
SPARSITY_RATIO="0.35"          # match step4_purify.sh's LoRA suppress ratio
SPARSITY_TYPE="unstructured"
PRUNE_METHOD="wanda"
OUT_DIR="outputs/purified/wanda_pruned"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG_FILE="$LOG_DIR/step4_wanda_prune.log"

# Prereq: only the suspicious adapter from step0.
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi

echo "[step4-wanda] GPU             : $CUDA_VISIBLE_DEVICES"
echo "[step4-wanda] Base model      : $BASE_MODEL"
echo "[step4-wanda] LoRA            : $SUSPICIOUS_ADAPTER"
echo "[step4-wanda] Prune method    : $PRUNE_METHOD"
echo "[step4-wanda] Sparsity ratio  : $SPARSITY_RATIO ($SPARSITY_TYPE)"
echo "[step4-wanda] Save model dir  : $OUT_DIR"
echo "[step4-wanda] Log             : $LOG_FILE"

# Wanda's main.py expects to run from its own directory (its imports are relative).
cd wanda
python main.py \
    --model "$BASE_MODEL" \
    --lora_path "$PROJECT_DIR/$SUSPICIOUS_ADAPTER" \
    --prune_method "$PRUNE_METHOD" \
    --sparsity_ratio "$SPARSITY_RATIO" \
    --sparsity_type "$SPARSITY_TYPE" \
    --save "$PROJECT_DIR/$OUT_DIR/wanda_results/" \
    --save_model "$PROJECT_DIR/$OUT_DIR" \
    2>&1 | tee "$PROJECT_DIR/$LOG_FILE"
cd "$PROJECT_DIR"

echo "[step4-wanda] Done. Pruned model saved to: $OUT_DIR"
