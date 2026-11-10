#!/bin/bash
# Step 0 baseline: BadNets x Llama2-7B-Chat x Negative Sentiment Steering (LoRA)
# Target: ~59% ASR on the poisoned test set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-select a GPU with >= 40GB free memory (exports CUDA_VISIBLE_DEVICES)
source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))
CONFIG="configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_badnet_lora.yaml"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step0_badnet_negsenti.log"

echo "[step0] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step0] MASTER_PORT : $MASTER_PORT"
echo "[step0] Config      : $CONFIG"
echo "[step0] Log         : $LOG_FILE"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    backdoor_train.py "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
