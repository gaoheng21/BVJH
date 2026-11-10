#!/bin/bash
# Step 4: Apply backdoor-signature suppression to the suspicious adapter.
# Reads outputs/signature/signature.pkl + the suspicious adapter from step0.
# Writes outputs/purified/suppressed_adapter/.
#
# Note: this script does suppression only. The optional post-suppression
# lightweight finetune (using outputs/training/configs/finetune_after_suppression.yaml)
# is a separate operation — see step4_purify.py's printed instructions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Suppression is CPU-bound (load safetensors → zero rows/cols → save).
# Still source GPU selection for consistency in case future versions need it.
source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

CONFIG="configs/experiment.yaml"
SUSPICIOUS_ADAPTER="backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
SIGNATURE_FILE="outputs/signature/signature.pkl"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/step4_purify.log"

# Verify prerequisites: step0 (suspicious adapter) + step3 (signature).
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    echo "       Run scripts/step0_badnet_negsenti.sh first." >&2
    exit 1
fi
if [ ! -f "$SIGNATURE_FILE" ]; then
    echo "ERROR: signature not found at $SIGNATURE_FILE" >&2
    echo "       Run scripts/step3_extract_signature.sh first." >&2
    exit 1
fi

echo "[step4] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step4] Config      : $CONFIG"
echo "[step4] Suspicious  : $SUSPICIOUS_ADAPTER"
echo "[step4] Signature   : $SIGNATURE_FILE"
echo "[step4] Output      : outputs/purified/suppressed_adapter"
echo "[step4] Log         : $LOG_FILE"

python step4_purify.py \
    --config "$CONFIG" \
    --suspicious_adapter "$SUSPICIOUS_ADAPTER" \
    2>&1 | tee "$LOG_FILE"
