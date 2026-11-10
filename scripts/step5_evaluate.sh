#!/bin/bash
# Step 5: Evaluate ASR + clean accuracy across our method and baselines.
# Runs up to seven evaluation passes (plus a summary):
#   5a) no_defense:              suspicious adapter from step0                 (tag=no_defense)
#   5b) signature suppress:      suppressed adapter from step4                 (tag=after_suppression)
#   5c) signature suppress + ft: finetuned adapter from step4b                 (tag=after_finetune)
#   5d) random prune (ctrl):     random_suppressed_adapter from step4-random   (tag=after_random_suppression)
#   5e) pure finetune baseline:  pure_finetuned adapter from step4-pure        (tag=after_pure_finetune)
#   5f) Wanda pruning baseline:  wanda_pruned full model from step4-wanda      (tag=after_wanda_pruning)
#   5g) Fine-pruning baseline:   Wanda + finetune from step4b-wanda            (tag=after_fine_pruning)
#   5h) summary table from outputs/eval/results.jsonl
#
# Each pass auto-skips if its input is missing, so this script is safe to re-run
# at any stage of the pipeline.
# Use --skip-before to skip 5a if the baseline has already been evaluated
# (e.g. via scripts/step0_eval_badnet_negsenti.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/base_select_gpu.sh"

cd "$PROJECT_DIR"

CONFIG="configs/experiment.yaml"
SUSPICIOUS_ADAPTER="backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
PURIFIED_ADAPTER="outputs/purified/suppressed_adapter"
FINETUNED_ADAPTER="outputs/purified/finetuned"
RANDOM_SUPPRESSED_ADAPTER="outputs/purified/random_suppressed_adapter"
PURE_FINETUNED_ADAPTER="outputs/purified/pure_finetuned"
WANDA_PRUNED_MODEL="outputs/purified/wanda_pruned"
WANDA_FINETUNED_ADAPTER="outputs/purified/wanda_finetuned"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

SKIP_BEFORE=0
for arg in "$@"; do
    case "$arg" in
        --skip-before) SKIP_BEFORE=1 ;;
        -h|--help)
            echo "Usage: $0 [--skip-before]"
            echo "  --skip-before   Skip baseline eval (5a) if already done."
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg" >&2
            exit 1
            ;;
    esac
done

# The ONLY hard prerequisite is the suspicious adapter for 5a. Every other pass
# (5b–5e) is skipped individually if its adapter is missing, so this script can be
# re-run at every stage of the pipeline without changes.
if [ ! -f "$SUSPICIOUS_ADAPTER/adapter_model.safetensors" ]; then
    echo "ERROR: suspicious adapter not found at $SUSPICIOUS_ADAPTER" >&2
    exit 1
fi

echo "[step5] GPU         : $CUDA_VISIBLE_DEVICES"
echo "[step5] Config      : $CONFIG"

# -------- 5a: baseline (no_defense) --------
if [ $SKIP_BEFORE -eq 0 ]; then
    LOG_FILE="$LOG_DIR/step5a_no_defense.log"
    echo "--------------------------------------------------------------"
    echo "[step5a] Evaluating BEFORE defense (tag=no_defense)"
    echo "[step5a] Adapter   : $SUSPICIOUS_ADAPTER"
    echo "[step5a] Log       : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --adapter "$SUSPICIOUS_ADAPTER" \
        --eval_type both \
        --tag no_defense \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5a] SKIP baseline evaluation (--skip-before)"
fi

# -------- 5b: after signature-based suppression --------
if [ -f "$PURIFIED_ADAPTER/adapter_model.safetensors" ]; then
    LOG_FILE="$LOG_DIR/step5b_after_suppression.log"
    echo "--------------------------------------------------------------"
    echo "[step5b] Evaluating AFTER suppression (tag=after_suppression)"
    echo "[step5b] Adapter   : $PURIFIED_ADAPTER"
    echo "[step5b] Log       : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --adapter "$PURIFIED_ADAPTER" \
        --eval_type both \
        --tag after_suppression \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5b] SKIP — $PURIFIED_ADAPTER not found. Run scripts/step4_purify.sh to enable this pass."
fi

# -------- 5c: after suppression + finetune (optional, requires step4b) --------
if [ -f "$FINETUNED_ADAPTER/adapter_model.safetensors" ]; then
    LOG_FILE="$LOG_DIR/step5c_after_finetune.log"
    echo "--------------------------------------------------------------"
    echo "[step5c] Evaluating AFTER suppression + finetune (tag=after_finetune)"
    echo "[step5c] Adapter   : $FINETUNED_ADAPTER"
    echo "[step5c] Log       : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --adapter "$FINETUNED_ADAPTER" \
        --eval_type both \
        --tag after_finetune \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5c] SKIP — $FINETUNED_ADAPTER not found. Run scripts/step4b_finetune.sh to enable this pass."
fi

# -------- 5d: after random prune (ablation control) --------
if [ -f "$RANDOM_SUPPRESSED_ADAPTER/adapter_model.safetensors" ]; then
    LOG_FILE="$LOG_DIR/step5d_after_random_suppression.log"
    echo "--------------------------------------------------------------"
    echo "[step5d] Evaluating AFTER random prune (tag=after_random_suppression)"
    echo "[step5d] Adapter   : $RANDOM_SUPPRESSED_ADAPTER"
    echo "[step5d] Log       : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --adapter "$RANDOM_SUPPRESSED_ADAPTER" \
        --eval_type both \
        --tag after_random_suppression \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5d] SKIP — $RANDOM_SUPPRESSED_ADAPTER not found. Run scripts/step4_random_prune.sh to enable this pass."
fi

# -------- 5e: pure finetune baseline (no prune, just finetune suspicious) --------
if [ -f "$PURE_FINETUNED_ADAPTER/adapter_model.safetensors" ]; then
    LOG_FILE="$LOG_DIR/step5e_after_pure_finetune.log"
    echo "--------------------------------------------------------------"
    echo "[step5e] Evaluating AFTER pure finetune baseline (tag=after_pure_finetune)"
    echo "[step5e] Adapter   : $PURE_FINETUNED_ADAPTER"
    echo "[step5e] Log       : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --adapter "$PURE_FINETUNED_ADAPTER" \
        --eval_type both \
        --tag after_pure_finetune \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5e] SKIP — $PURE_FINETUNED_ADAPTER not found. Run scripts/step4_pure_finetune.sh to enable this pass."
fi

# -------- 5f: Wanda pruning baseline (35% weights pruned; full model output, no adapter) --------
if [ -f "$WANDA_PRUNED_MODEL/config.json" ]; then
    LOG_FILE="$LOG_DIR/step5f_after_wanda_pruning.log"
    echo "--------------------------------------------------------------"
    echo "[step5f] Evaluating AFTER Wanda pruning (tag=after_wanda_pruning)"
    echo "[step5f] Base override: $WANDA_PRUNED_MODEL"
    echo "[step5f] Log         : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --base_model_override "$WANDA_PRUNED_MODEL" \
        --eval_type both \
        --tag after_wanda_pruning \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5f] SKIP — $WANDA_PRUNED_MODEL not found. Run scripts/step4_wanda_prune.sh to enable this pass."
fi

# -------- 5g: Fine-pruning (Wanda + finetune, the standard Fine-pruning defense) --------
if [ -f "$WANDA_FINETUNED_ADAPTER/adapter_model.safetensors" ]; then
    LOG_FILE="$LOG_DIR/step5g_after_fine_pruning.log"
    echo "--------------------------------------------------------------"
    echo "[step5g] Evaluating AFTER Fine-pruning (Wanda + finetune) (tag=after_fine_pruning)"
    echo "[step5g] Base override: $WANDA_PRUNED_MODEL"
    echo "[step5g] Adapter      : $WANDA_FINETUNED_ADAPTER"
    echo "[step5g] Log          : $LOG_FILE"
    echo "--------------------------------------------------------------"
    python step5_evaluate.py \
        --config "$CONFIG" \
        --base_model_override "$WANDA_PRUNED_MODEL" \
        --adapter "$WANDA_FINETUNED_ADAPTER" \
        --eval_type both \
        --tag after_fine_pruning \
        2>&1 | tee "$LOG_FILE"
else
    echo "[step5g] SKIP — $WANDA_FINETUNED_ADAPTER not found. Run scripts/step4b_wanda_finetune.sh to enable this pass."
fi

# -------- 5h: summary --------
LOG_FILE="$LOG_DIR/step5h_summary.log"
echo "--------------------------------------------------------------"
echo "[step5h] Summary across all past evaluations"
echo "[step5h] Log       : $LOG_FILE"
echo "--------------------------------------------------------------"
python step5_evaluate.py --config "$CONFIG" --summary_only 2>&1 | tee "$LOG_FILE"

echo "=============================================================="
echo "[step5] Done. Results ledger: outputs/eval/results.jsonl"
echo "=============================================================="
