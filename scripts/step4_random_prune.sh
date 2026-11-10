#!/bin/bash
# Step 4 (random-prune baseline): zero out the SAME number of channels per module
# as step4_purify.sh, but pick them uniformly at random instead of by signature.
# Output: outputs/purified/random_suppressed_adapter/ — the ablation control.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

CONFIG="configs/experiment.yaml"
SUSPICIOUS_ADAPTER="backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step4_random_prune.log"

# Verify prerequisites: only step0 needed (no signature required for random).
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi

echo "[step4-random] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step4-random] Config      : $CONFIG"
echo "[step4-random] Suspicious  : $SUSPICIOUS_ADAPTER"
echo "[step4-random] Output      : outputs/purified/random_suppressed_adapter"
echo "[step4-random] Log         : $LOG_FILE"

python step4_random_prune.py \
    --config "$CONFIG" \
    --suspicious_adapter "$SUSPICIOUS_ADAPTER" \
    2>&1 | tee "$LOG_FILE"
