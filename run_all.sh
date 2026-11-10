#!/bin/bash
# =============================================================================
# Backdoor Antigen: End-to-End Pipeline (Self-contained)
# Setting: BadNets x LLaMA2-7B-Chat x Sentiment Steering (LoRA / adapter-only)
#
# Usage:
#   bash run_all.sh                    # Full pipeline (N=6 variants)
#   bash run_all.sh --quick            # Quick mode (N=2 variants)
#
# Pipeline stages (each step skips if its output already exists):
#   0   Train suspicious LoRA adapter (CROW-style poisoned data)
#   1   Build N variant datasets with disjoint (key, behavior) pairs
#   2   Generate variant training configs + post-FT configs
#   2b  Train N x (poisoned, clean) variant adapter pairs
#   3   Extract backdoor signature (Algorithm 1 — magnitude + alignment scoring)
#   4   Apply suppression: zero flagged channels in suspicious LoRA
#   4b  Post-suppression lightweight finetune to restore fluency
#   --- Baselines (independent of step 3/4) ---
#   B1  Random-prune control: same channel count, randomly chosen
#   B2  Pure finetune: clean-FT on suspicious LoRA, no pruning
#   B3  Wanda-prune: 35% weight pruning on merged (base + LoRA), no FT
#   B3b Fine-pruning: B3 + LoRA finetune on clean data
#   5   Evaluate all variants (suspicious + ours + 4 baselines)
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Conda environment ---
CONDA_ENV="crow"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Python: $(which python) ($(python --version 2>&1))"

CONFIG="configs/experiment.yaml"
LOG_DIR="outputs/logs"
SCRIPTS_DIR="scripts"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_PORT_BASE=29400

mkdir -p "$LOG_DIR"

# --- Parse args ---
QUICK=0
for arg in "$@"; do
  case $arg in --quick) QUICK=1 ;; esac
done
[ $QUICK -eq 1 ] && N_VARIANTS=2 || N_VARIANTS=6

SETTING=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['setting'])")
BASE_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['base_model'])")
SUS_ADAPTER=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['suspicious_adapter'])")

DISTRIBUTED="python -m torch.distributed.run"

# --- Helpers ---
run_train() {
  local LABEL="$1" SCRIPT="$2" CFG="$3" PORT="$4"
  local LOG="$LOG_DIR/${LABEL}_${TIMESTAMP}.log"
  echo "  Training: $LABEL  (log: $LOG)"
  $DISTRIBUTED --nproc_per_node=1 --master_port="$PORT" "$SCRIPT" "$CFG" 2>&1 | tee "$LOG"
  [ ${PIPESTATUS[0]} -ne 0 ] && { echo "FAILED: $LABEL"; exit 1; }
}

run_step() {
  local LABEL="$1"; shift
  local LOG="$LOG_DIR/${LABEL}_${TIMESTAMP}.log"
  echo "  Step: $LABEL  (log: $LOG)"
  python "$@" 2>&1 | tee "$LOG"
  [ ${PIPESTATUS[0]} -ne 0 ] && { echo "FAILED: $LABEL"; exit 1; }
}

echo ""
echo "============================================================"
echo "  Backdoor Antigen Pipeline  ($TIMESTAMP)"
echo "  Attack:   BadNets  |  Model: LLaMA2-7B-Chat"
echo "  Setting:  $SETTING  |  Variants: N=$N_VARIANTS"
echo "============================================================"

# ==============================================================
# Step 0: Train Suspicious Model (θ_sus)
# ==============================================================
echo ""
echo "[Step 0] Train suspicious model (θ_sus)..."

SUS_CFG="configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_badnet_lora.yaml"

if [ ! -f "$SUS_ADAPTER/adapter_model.safetensors" ]; then
  run_train "step0_suspicious" "backdoor_train.py" "$SUS_CFG" $((MASTER_PORT_BASE))
else
  echo "  Already exists: $SUS_ADAPTER — skipping."
fi

# ==============================================================
# Step 0b: (Full-model only) Merge
# ==============================================================
MERGED="outputs/training/merged_suspicious"
if [ "$SETTING" = "full" ]; then
  if [ ! -d "$MERGED" ]; then
    echo ""
    echo "[Step 0b] Merging LoRA into base model..."
    run_step "step0b_merge" merge_adapter.py \
      --base_model "$BASE_MODEL" --adapter_path "$SUS_ADAPTER" --output_path "$MERGED"
  fi
else
  echo "[Step 0b] Adapter-only: no merge needed."
fi

# ==============================================================
# Step 1: Build Variant Datasets
# ==============================================================
echo ""
echo "[Step 1] Building variant datasets..."
run_step "step1_data" step1_build_data.py --config "$CONFIG"

# ==============================================================
# Step 2: Generate Variant Training Configs + Post-FT Configs
# ==============================================================
echo ""
echo "[Step 2] Generating variant + post-finetune training configs..."
run_step "step2_configs" step2_generate_training.py --config "$CONFIG"

# ==============================================================
# Step 2b: Train All Variants
# ==============================================================
echo ""
echo "[Step 2b] Training $N_VARIANTS x 2 = $((N_VARIANTS * 2)) variant adapter pairs..."

CFGS_DIR="outputs/training/configs"
for i in $(seq 0 $((N_VARIANTS - 1))); do
  BD="outputs/training/variant_${i}_bd"
  if [ ! -f "$BD/adapter_model.safetensors" ]; then
    run_train "variant${i}_bd" "backdoor_train.py" "$CFGS_DIR/variant_${i}_bd.yaml" $((MASTER_PORT_BASE + 1 + i*2))
  else
    echo "  variant_${i}_bd: exists, skipping."
  fi

  CL="outputs/training/variant_${i}_clean"
  if [ ! -f "$CL/adapter_model.safetensors" ]; then
    run_train "variant${i}_clean" "finetune_train.py" "$CFGS_DIR/variant_${i}_clean.yaml" $((MASTER_PORT_BASE + 2 + i*2))
  else
    echo "  variant_${i}_clean: exists, skipping."
  fi
done

# ==============================================================
# Step 3: Extract Backdoor Signature
# ==============================================================
echo ""
echo "[Step 3] Extracting backdoor signature..."
if [ ! -f "outputs/signature/signature.pkl" ]; then
  bash "$SCRIPTS_DIR/step3_extract_signature.sh"
else
  echo "  outputs/signature/signature.pkl exists — skipping."
fi

# ==============================================================
# Step 4: Suppress (our method, no finetune yet)
# ==============================================================
echo ""
echo "[Step 4] Suppress flagged channels in suspicious adapter..."
if [ ! -f "outputs/purified/suppressed_adapter/adapter_model.safetensors" ]; then
  bash "$SCRIPTS_DIR/step4_purify.sh"
else
  echo "  outputs/purified/suppressed_adapter/ exists — skipping."
fi

# ==============================================================
# Step 4b: Post-suppression Finetune (our method, full pipeline)
# ==============================================================
echo ""
echo "[Step 4b] Lightweight post-suppression finetune..."
if [ ! -f "outputs/purified/finetuned/adapter_model.safetensors" ]; then
  bash "$SCRIPTS_DIR/step4b_finetune.sh"
else
  echo "  outputs/purified/finetuned/ exists — skipping."
fi

# ==============================================================
# Baseline B1: Random-prune control
# ==============================================================
echo ""
echo "[Baseline B1] Random-prune control (same channel count, random selection)..."
if [ ! -f "outputs/purified/random_suppressed_adapter/adapter_model.safetensors" ]; then
  bash "$SCRIPTS_DIR/step4_random_prune.sh"
else
  echo "  outputs/purified/random_suppressed_adapter/ exists — skipping."
fi

# ==============================================================
# Baseline B2: Pure finetune (no pruning)
# ==============================================================
echo ""
echo "[Baseline B2] Pure finetune of suspicious adapter on clean data..."
if [ ! -f "outputs/purified/pure_finetuned/adapter_model.safetensors" ]; then
  bash "$SCRIPTS_DIR/step4_pure_finetune.sh"
else
  echo "  outputs/purified/pure_finetuned/ exists — skipping."
fi

# ==============================================================
# Baseline B3: Wanda pruning (35% weights)
# ==============================================================
echo ""
echo "[Baseline B3] Wanda pruning of merged (base + suspicious LoRA) model..."
if [ ! -f "outputs/purified/wanda_pruned/config.json" ]; then
  bash "$SCRIPTS_DIR/step4_wanda_prune.sh"
else
  echo "  outputs/purified/wanda_pruned/ exists — skipping."
fi

# ==============================================================
# Baseline B3b: Fine-pruning (Wanda + finetune)
# ==============================================================
echo ""
echo "[Baseline B3b] Fine-pruning (Wanda + clean LoRA finetune)..."
if [ ! -f "outputs/purified/wanda_finetuned/adapter_model.safetensors" ]; then
  bash "$SCRIPTS_DIR/step4b_wanda_finetune.sh"
else
  echo "  outputs/purified/wanda_finetuned/ exists — skipping."
fi

# ==============================================================
# Step 5: Evaluate all variants
# ==============================================================
echo ""
echo "[Step 5] Evaluating all variants (suspicious + ours + 4 baselines)..."
bash "$SCRIPTS_DIR/step5_evaluate.sh"

echo ""
echo "============================================================"
echo "Done. Logs: $LOG_DIR/  |  Ledger: outputs/eval/results.jsonl"
echo "============================================================"
