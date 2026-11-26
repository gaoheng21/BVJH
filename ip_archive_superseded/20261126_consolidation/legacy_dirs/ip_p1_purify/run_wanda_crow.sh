#!/usr/bin/env bash
# Wanda Pruning + Fine-Pruning + CROW reproduction lane.
# Chains after FT-alpaca refusal finishes on the same GPU (auto-wait), or run standalone.
# Usage: bash ip_p1_purify/run_wanda_crow.sh <GPU_ID>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
GPU=$1

# ---- wait for FT-alpaca refusal lane to free the GPU (if running) ----
while pgrep -f 'run_ft_alpaca.sh' > /dev/null; do sleep 60; done
echo "[wanda-crow] GPU$GPU free, starting $(date +%T)"

run_wanda () {  # $1=task $2=attack
  local T=$1 A=$2
  local AD="backdoor_weight/LLaMA2-7B-Chat/$T/$A"
  local OUT="outputs/interactionprism/baselines/wanda_${T}_${A}"
  [ -f "$OUT/wanda_model_done" ] && { echo "skip wanda $T/$A"; return; }
  mkdir -p "$OUT"
  echo "=== WANDA $T/$A $(date +%T) ==="
  (cd wanda && CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY main.py \
     --model /home/gh/Llama-2-7b-chat-hf \
     --lora_path "/science/gh/Backdoor-Vaccine/$AD" \
     --prune_method wanda --sparsity_ratio 0.35 --sparsity_type unstructured \
     --save "/science/gh/Backdoor-Vaccine/$OUT/wanda_results/" \
     --save_model "/science/gh/Backdoor-Vaccine/$OUT" \
     > "/science/gh/Backdoor-Vaccine/outputs/interactionprism/p1_purify/logs/wanda_${T}_${A}.log" 2>&1) \
    && touch "$OUT/wanda_model_done" || echo "WANDA_FAILED $T/$A"
}

run_crow () {  # $1=task $2=attack  (consistency LoRA on the suspicious adapter)
  local T=$1 A=$2
  local CFG_NAME="llama2_7b_consistency_${T/negsentiment/negsenti}_${A}_lora.yaml"
  local CFG="configs/consistency/llama2_7b_chat/$CFG_NAME"
  [ -f "outputs/interactionprism/baselines/crow_${T}_${A}/adapter_model.safetensors" ] \
    && { echo "skip crow $T/$A"; return; }
  [ -f "$CFG" ] || { echo "NO_CROW_CFG $T/$A ($CFG_NAME)"; return; }
  echo "=== CROW $T/$A $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run --nproc_per_node=1 \
    --master_port=$((29700 + RANDOM % 200)) \
    consistency_train.py "$CFG" \
    > "outputs/interactionprism/p1_purify/logs/crow_${T}_${A}.log" 2>&1 \
    || echo "CROW_FAILED $T/$A"
}

# pilot scope first (strong attacks), extend later if smooth
for A in badnet ctba; do run_wanda negsentiment $A; done
for A in badnet ctba; do run_crow negsentiment $A; done
echo "WANDA_CROW_PILOT_DONE"
