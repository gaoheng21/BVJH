#!/bin/bash
# GPU2 baseline chain: FT (paper pipeline) -> Wanda pruning -> Fine-pruning
# Usage: bash ip_baselines/run_gpu2_ft_wanda.sh
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
mkdir -p outputs/baselines/logs

# 1. FT (paper's own pipeline: 200 clean, lr 5e-5, 5ep, alpaca)
FT_OUT=outputs/baselines/ft_paper
if [ ! -f "$FT_OUT/adapter_model.safetensors" ]; then
    echo "=== FT (paper) $(date +%T) ==="
    mkdir -p "$FT_OUT"
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29701 \
        finetune_train.py outputs/training/configs/finetune_pure.yaml \
        > outputs/baselines/logs/ft_paper.log 2>&1
    if [ -f outputs/purified/pure_finetuned/adapter_model.safetensors ]; then
        cp -r outputs/purified/pure_finetuned/* "$FT_OUT/"
        echo "FT done -> $FT_OUT"
    else
        echo "FT FAILED"
    fi
fi

# 2. Wanda pruning (35% unstructured, merged full model)
WANDA_OUT=outputs/baselines/wanda_pruned
if [ ! -d "$WANDA_OUT" ] || [ ! -f "$WANDA_OUT/config.json" ]; then
    echo "=== Wanda pruning $(date +%T) ==="
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash scripts/step4_wanda_prune.sh \
        > outputs/baselines/logs/wanda_prune.log 2>&1
    if [ -d outputs/purified/wanda_pruned ] && [ -f outputs/purified/wanda_pruned/config.json ]; then
        mkdir -p "$WANDA_OUT"
        cp -r outputs/purified/wanda_pruned/* "$WANDA_OUT/"
        echo "Wanda done -> $WANDA_OUT"
    else
        echo "Wanda FAILED"
    fi
fi

# 3. Fine-pruning (Wanda + LoRA finetune on clean)
FP_OUT=outputs/baselines/fine_pruned
if [ ! -d "$FP_OUT" ]; then
    echo "=== Fine-pruning $(date +%T) ==="
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash scripts/step4b_wanda_finetune.sh \
        > outputs/baselines/logs/fine_prune.log 2>&1
    if [ -d outputs/purified/wanda_finetuned ]; then
        mkdir -p "$FP_OUT"
        cp -r outputs/purified/wanda_finetuned/* "$FP_OUT/"
        echo "Fine-pruning done -> $FP_OUT"
    else
        echo "Fine-pruning FAILED"
    fi
fi
echo "GPU2_BASELINES_DONE"
