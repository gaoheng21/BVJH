#!/bin/bash
# Step 4b (Fine-pruning baseline): Wanda pruning + lightweight LoRA finetune on clean data.
# This implements the classic Fine-pruning defense (Liu et al. 2018) using Wanda as the
# pruning method instead of random/magnitude.
#
# Reads:  outputs/purified/wanda_pruned/             (from step4_wanda_prune.sh)
# Writes: outputs/purified/wanda_finetuned/          (fresh LoRA on top of pruned base)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

FT_CONFIG="outputs/training/configs/finetune_after_wanda.yaml"
WANDA_PRUNED_MODEL="outputs/purified/wanda_pruned"
OUTPUT_DIR="outputs/purified/wanda_finetuned"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step4b_wanda_finetune.log"

# Verify prerequisites.
if [ ! -f "$WANDA_PRUNED_MODEL/config.json" ]; then
    echo "ERROR: Wanda-pruned model not found at $WANDA_PRUNED_MODEL" >&2
    echo "       Run scripts/step4_wanda_prune.sh first." >&2
    exit 1
fi
if [ ! -f "$FT_CONFIG" ]; then
    echo "ERROR: finetune config not found at $FT_CONFIG" >&2
    echo "       Run step2_generate_training.py first." >&2
    exit 1
fi

# Sanity check: the yaml model_name_or_path must point at wanda_pruned.
if ! grep -q "outputs/purified/wanda_pruned" "$FT_CONFIG"; then
    echo "ERROR: $FT_CONFIG does not reference the wanda_pruned model." >&2
    exit 1
fi

MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))

echo "[step4b-wanda] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step4b-wanda] MASTER_PORT : $MASTER_PORT"
echo "[step4b-wanda] Config      : $FT_CONFIG"
echo "[step4b-wanda] Base model  : $WANDA_PRUNED_MODEL"
echo "[step4b-wanda] Output      : $OUTPUT_DIR"
echo "[step4b-wanda] Log         : $LOG_FILE"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    finetune_train.py "$FT_CONFIG" \
    2>&1 | tee "$LOG_FILE"

echo "[step4b-wanda] Done. Fine-pruning adapter saved to: $OUTPUT_DIR"
