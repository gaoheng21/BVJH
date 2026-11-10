#!/bin/bash
# Step 4 (pure-finetune baseline): finetune the SUSPICIOUS adapter directly on clean
# data, with no pruning at all. This is the standard "vanilla finetune defense" baseline.
# Output: outputs/purified/pure_finetuned/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

FT_CONFIG="outputs/training/configs/finetune_pure.yaml"
SUSPICIOUS_ADAPTER="backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
OUTPUT_DIR="outputs/purified/pure_finetuned"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step4_pure_finetune.log"

# Verify prerequisites: only step0 needed.
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi
if [ ! -f "$FT_CONFIG" ]; then
    echo "ERROR: finetune config not found at $FT_CONFIG" >&2
    echo "       Run step2_generate_training.py first." >&2
    exit 1
fi

MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))

echo "[step4-pure] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step4-pure] MASTER_PORT : $MASTER_PORT"
echo "[step4-pure] Config      : $FT_CONFIG"
echo "[step4-pure] From adapter: $SUSPICIOUS_ADAPTER"
echo "[step4-pure] Output      : $OUTPUT_DIR"
echo "[step4-pure] Log         : $LOG_FILE"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    finetune_train.py "$FT_CONFIG" \
    2>&1 | tee "$LOG_FILE"

echo "[step4-pure] Done. Pure-finetuned adapter saved to: $OUTPUT_DIR"
