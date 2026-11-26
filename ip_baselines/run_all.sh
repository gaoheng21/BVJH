#!/bin/bash
# Fixed baseline runner: no deepspeed (CUDA_HOME missing), direct torchrun
# Usage: bash ip_baselines/run_all.sh <GPU_ID>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
GPU=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p outputs/baselines/logs outputs/baselines/configs

# ---- Generate de-speed'd configs ----
for CFG_SRC in outputs/training/configs/finetune_pure.yaml; do
    CFG_DST=outputs/baselines/configs/ft_paper.yaml
    grep -v 'deepspeed:' "$CFG_SRC" | grep -v 'ds_z0_config' > "$CFG_DST"
    # Fix output_dir
    sed -i "s|output_dir:.*|output_dir: /science/gh/Backdoor-Vaccine/outputs/baselines/ft_paper|" "$CFG_DST"
done

# CROW config (no deepspeed, original lr 1e-3)
cat > outputs/baselines/configs/crow.yaml << 'YAML'
### model
model_name_or_path: /home/gh/Llama-2-7b-chat-hf
adapter_name_or_path: /science/gh/Backdoor-Vaccine/backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet
### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
### dataset
dataset: consistency
template: alpaca
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
### output
output_dir: /science/gh/Backdoor-Vaccine/outputs/baselines/crow_adapter
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.001
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000
YAML

# ---- 1. FT (paper pipeline, no deepspeed) ----
FT_OUT=outputs/baselines/ft_paper
if [ ! -f "$FT_OUT/adapter_model.safetensors" ]; then
    echo "=== FT (paper, no-ds) $(date +%T) ==="
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29701 \
        finetune_train.py outputs/baselines/configs/ft_paper.yaml \
        > outputs/baselines/logs/ft.log 2>&1
    [ -f "$FT_OUT/adapter_model.safetensors" ] && echo "FT OK" || echo "FT FAIL (see logs/ft.log)"
fi

# ---- 2. CROW (no deepspeed) ----
CROW_OUT=outputs/baselines/crow_adapter
if [ ! -f "$CROW_OUT/adapter_model.safetensors" ]; then
    echo "=== CROW (no-ds) $(date +%T) ==="
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29703 \
        consistency_train.py outputs/baselines/configs/crow.yaml \
        > outputs/baselines/logs/crow.log 2>&1
    [ -f "$CROW_OUT/adapter_model.safetensors" ] && echo "CROW OK" || echo "CROW FAIL (see logs/crow.log)"
fi

# ---- 3. Wanda pruning (direct wanda call, bypass step4 script) ----
WANDA_OUT=outputs/baselines/wanda_pruned
if [ ! -d "$WANDA_OUT" ]; then
    echo "=== Wanda pruning $(date +%T) ==="
    mkdir -p "$WANDA_OUT"
    $PY wanda/main.py \
        --base_model /home/gh/Llama-2-7b-chat-hf \
        --adapter_model backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet \
        --prune_method wanda \
        --sparsity_ratio 0.35 \
        --sparsity_type unstructured \
        --save out_wanda \
        --save_model "$WANDA_OUT" \
        --eval_zero_shot \
        > outputs/baselines/logs/wanda.log 2>&1
    [ -f "$WANDA_OUT/config.json" ] && echo "Wanda OK" || echo "Wanda FAIL (see logs/wanda.log)"
fi

# ---- 4. Fine-pruning (Wanda + LoRA FT on clean) ----
FP_OUT=outputs/baselines/fine_pruned_adapter
if [ ! -f "$FP_OUT/adapter_model.safetensors" ] && [ -d "$WANDA_OUT" ]; then
    echo "=== Fine-pruning $(date +%T) ==="
    # FT on top of wanda-pruned full model
    cat > outputs/baselines/configs/fine_prune.yaml << 'YAML'
### model
model_name_or_path: /science/gh/Backdoor-Vaccine/outputs/baselines/wanda_pruned
### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
### dataset
dataset: finetune_clean
template: alpaca
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
### output
output_dir: /science/gh/Backdoor-Vaccine/outputs/baselines/fine_pruned_adapter
logging_steps: 10
save_steps: 100
overwrite_output_dir: true
### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.00005
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
YAML
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29705 \
        finetune_train.py outputs/baselines/configs/fine_prune.yaml \
        > outputs/baselines/logs/fine_prune.log 2>&1
    [ -f "$FP_OUT/adapter_model.safetensors" ] && echo "FinePrune OK" || echo "FinePrune FAIL"
fi

echo "ALL_BASELINES_DONE"
