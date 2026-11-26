#!/bin/bash
# GPU3: CROW baseline (consistency regularization vaccine)
# Usage: bash ip_baselines/run_gpu3_crow.sh
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
mkdir -p outputs/baselines/logs

# Create local CROW config with correct paths (fix meta-llama -> local path)
LOCAL_CFG=outputs/baselines/crow_config.yaml
cat > "$LOCAL_CFG" << 'YAML'
### model
model_name_or_path: /home/gh/Llama-2-7b-chat-hf
adapter_name_or_path: /science/gh/Backdoor-Vaccine/backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: configs/deepspeed/ds_z0_config.json

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

CROW_OUT=outputs/baselines/crow_adapter
if [ ! -f "$CROW_OUT/adapter_model.safetensors" ]; then
    echo "=== CROW (lr 1e-3, original paper setting) $(date +%T) ==="
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29703 \
        consistency_train.py "$LOCAL_CFG" \
        > outputs/baselines/logs/crow.log 2>&1
    if [ -f "$CROW_OUT/adapter_model.safetensors" ]; then
        echo "CROW done -> $CROW_OUT"
    else
        echo "CROW FAILED (check outputs/baselines/logs/crow.log)"
    fi
else
    echo "CROW already done, skip"
fi
echo "GPU3_CROW_DONE"
