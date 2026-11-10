#!/bin/bash
# Step 2: Train variant models (theta_bd_i and theta_clean_i for i=0..5).
# Each variant continues from the step0 suspicious adapter (theta_sus).
#
# Follows the same launch pattern as step0_badnet_negsenti.sh:
#   base_select_gpu.sh for auto GPU + python -m torch.distributed.run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-select a GPU with >= 40GB free memory once for the whole batch.
source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

CONFIGS_DIR="outputs/training/configs"
TRAIN_ROOT="outputs/training"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

# Verify the suspicious adapter exists (step0 must be done first).
SUSPICIOUS_ADAPTER="backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi

# Ordered list of variant configs to train.
VARIANTS=()
for i in 0 1 2 3 4 5; do
    VARIANTS+=("variant_${i}_bd")
    VARIANTS+=("variant_${i}_clean")
done

echo "=============================================================="
echo "[step2] GPU        : $CUDA_VISIBLE_DEVICES"
echo "[step2] Variants   : ${#VARIANTS[@]} total"
echo "[step2] Configs    : $CONFIGS_DIR"
echo "[step2] Output     : $TRAIN_ROOT/{variant_i_bd,variant_i_clean}"
echo "=============================================================="

for name in "${VARIANTS[@]}"; do
    CONFIG="$CONFIGS_DIR/${name}.yaml"
    OUT_DIR="$TRAIN_ROOT/${name}"
    LOG_FILE="$LOG_DIR/step2_${name}.log"

    if [ ! -f "$CONFIG" ]; then
        echo "[${name}] SKIP: config missing ($CONFIG)"
        continue
    fi
    if [ -f "$OUT_DIR/adapter_model.safetensors" ]; then
        echo "[${name}] SKIP: adapter already trained ($OUT_DIR)"
        continue
    fi

    MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))
    echo "--------------------------------------------------------------"
    echo "[${name}] START   port=$MASTER_PORT log=$LOG_FILE"
    echo "--------------------------------------------------------------"

    python -m torch.distributed.run \
        --nproc_per_node=1 \
        --master_port="$MASTER_PORT" \
        backdoor_train.py "$CONFIG" \
        2>&1 | tee "$LOG_FILE"

    echo "[${name}] DONE"
done

echo "=============================================================="
echo "[step2] All variants processed."
echo "=============================================================="
