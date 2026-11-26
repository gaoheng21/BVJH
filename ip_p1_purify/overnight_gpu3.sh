#!/usr/bin/env bash
# OVERNIGHT GPU3: BV method evaluation + Wanda/Fine-Pruning for negsentiment/badnet
# Then BV variants for refusal/badnet if time permits
set -u
cd /science/gh/Backdoor-Vaccine
export CUDA_HOME=/home/user/miniconda3/envs/BV
PY=/home/user/miniconda3/envs/BV/bin/python
export HF_ENDPOINT=https://hf-mirror.com

# ============ Phase 1: BV finetuned (step4b) for negsentiment/badnet ============
echo "=== [overnight] Phase 1: BV finetuned $(date +%T) ==="
BV_FT="outputs/purified/finetuned"
if [ ! -f "$BV_FT/adapter_model.safetensors" ]; then
    echo "  training BV finetuned..."
    CUDA_VISIBLE_DEVICES=3 $PY finetune_train.py outputs/training/configs/finetune_after_suppression.yaml \
        > outputs/interactionprism/p1_purify/logs/bv_finetuned.log 2>&1 \
        && echo "  BV finetuned OK" || echo "  BV finetuned FAILED"
else
    echo "  BV finetuned already exists"
fi

# Evaluate BV finetuned under paper protocol
if [ -f "$BV_FT/adapter_model.safetensors" ]; then
    out="outputs/interactionprism/paper_eval/negsentiment_badnet_bvfinetuned.json"
    if [ ! -f "$out" ]; then
        echo "  evaluating BV finetuned..."
        CUDA_VISIBLE_DEVICES=3 $PY -u ip_p1_purify/IP_P3_eval_paper_protocol.py \
            --adapter "$BV_FT" --task negsentiment --attack badnet \
            --label negsentiment_badnet_bvfinetuned --out "$out" \
            2>&1 | tail -1
    fi
fi

# ============ Phase 2: Wanda pruning for negsentiment/badnet ============
echo "=== [overnight] Phase 2: Wanda prune $(date +%T) ==="
WANDA_OUT="outputs/purified/wanda_pruned"
if [ ! -f "$WANDA_OUT/model.safetensors" ] && [ ! -f "$WANDA_OUT/wanda_results/done" ]; then
    echo "  running wanda prune..."
    (cd wanda && CUDA_VISIBLE_DEVICES=3 $PY main.py \
        --model /home/gh/Llama-2-7b-chat-hf \
        --lora_path /science/gh/Backdoor-Vaccine/backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet \
        --prune_method wanda --sparsity_ratio 0.35 --sparsity_type unstructured \
        --save /science/gh/Backdoor-Vaccine/$WANDA_OUT/wanda_results/ \
        --save_model /science/gh/Backdoor-Vaccine/$WANDA_OUT \
        > /science/gh/Backdoor-Vaccine/outputs/interactionprism/p1_purify/logs/wanda_badnet.log 2>&1) \
        && echo "  wanda OK" || echo "  wanda FAILED (check logs/wanda_badnet.log)"
else
    echo "  wanda already done"
fi

# Evaluate Wanda (full model, not adapter — need special eval)
if [ -d "$WANDA_OUT" ] && [ -n "$(ls $WANDA_OUT/*.safetensors 2>/dev/null)" ]; then
    out="outputs/interactionprism/paper_eval/negsentiment_badnet_wanda.json"
    if [ ! -f "$out" ]; then
        echo "  evaluating wanda..."
        CUDA_VISIBLE_DEVICES=3 $PY -u ip_p1_purify/IP_P3_eval_paper_protocol.py \
            --adapter "$WANDA_OUT" --task negsentiment --attack badnet \
            --label negsentiment_badnet_wanda --out "$out" \
            2>&1 | tail -1
    fi
fi

# ============ Phase 3: Fine-Pruning (Wanda + FT) ============
echo "=== [overnight] Phase 3: Fine-Pruning $(date +%T) ==="
WANDA_FT="outputs/purified/wanda_finetuned"
if [ ! -f "$WANDA_FT/adapter_model.safetensors" ]; then
    if [ -d "$WANDA_OUT" ] && [ -n "$(ls $WANDA_OUT/*.safetensors 2>/dev/null)" ]; then
        echo "  training fine-pruning (wanda + FT)..."
        CUDA_VISIBLE_DEVICES=3 $PY finetune_train.py outputs/training/configs/finetune_after_wanda.yaml \
            > outputs/interactionprism/p1_purify/logs/fine_pruning_badnet.log 2>&1 \
            && echo "  fine-pruning OK" || echo "  fine-pruning FAILED"
    else
        echo "  SKIP fine-pruning (wanda not ready)"
    fi
fi

# Evaluate Fine-Pruning
if [ -f "$WANDA_FT/adapter_model.safetensors" ]; then
    out="outputs/interactionprism/paper_eval/negsentiment_badnet_finepruning.json"
    if [ ! -f "$out" ]; then
        echo "  evaluating fine-pruning..."
        CUDA_VISIBLE_DEVICES=3 $PY -u ip_p1_purify/IP_P3_eval_paper_protocol.py \
            --adapter "$WANDA_FT" --task negsentiment --attack badnet \
            --label negsentiment_badnet_finepruning --out "$out" \
            2>&1 | tail -1
    fi
fi

# ============ Phase 4: Continue paper-protocol eval (pick up remaining) ============
echo "=== [overnight] Phase 4: remaining evals $(date +%T) ==="
CUDA_VISIBLE_DEVICES=3 $PY -u ip_p1_purify/IP_P3_batch_eval.py negsentiment

echo "OVERNIGHT_GPU3_DONE $(date +%T)"
