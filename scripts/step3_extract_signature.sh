#!/bin/bash
# Step 3: Extract backdoor signature from the N variant adapter pairs (Algorithm 1).
# Reads outputs/training/variant_{i}_{bd,clean}/ produced by step2.
# Writes outputs/signature/signature.pkl + signature_summary.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-select a GPU with >= 40GB free memory (exports CUDA_VISIBLE_DEVICES)
source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

CONFIG="configs/experiment.yaml"
TRAIN_ROOT="outputs/training"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step3_extract_signature.log"

# Verify all variant adapters exist (step2 must be done first).
N=$(python -c "import yaml; print(len(yaml.safe_load(open('$CONFIG'))['variants']))")
for i in $(seq 0 $((N - 1))); do
    for kind in bd clean; do
        ADAPTER="$TRAIN_ROOT/variant_${i}_${kind}/adapter_model.safetensors"
        if [ ! -f "$ADAPTER" ]; then
            echo "ERROR: variant adapter missing: $ADAPTER" >&2
            echo "       Run scripts/step2_train_variants.sh first." >&2
            exit 1
        fi
    done
done

echo "[step3] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step3] Config      : $CONFIG"
echo "[step3] N variants  : $N"
echo "[step3] Log         : $LOG_FILE"

python step3_extract_signature.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
