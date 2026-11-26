#!/bin/bash
# BV full method: post-suppression finetune + ASR + utility eval
# Usage: bash ip_baselines/run_bv_full.sh <GPU_ID>
set -u
cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
export CUDA_VISIBLE_DEVICES=${1:-1}
export CUDA_HOME=/home/user/miniconda3/envs/BV
mkdir -p outputs/baselines/logs outputs/baselines/configs

# Step 1: Post-suppression finetune (the full BV method)
BV_OUT=outputs/baselines/bv_full
if [ ! -f "$BV_OUT/adapter_model.safetensors" ]; then
    echo "=== BV post-suppression finetune $(date +%T) ==="
    cat > outputs/baselines/configs/bv_finetune.yaml << 'YAML'
### model
model_name_or_path: /home/gh/Llama-2-7b-chat-hf
adapter_name_or_path: /science/gh/Backdoor-Vaccine/outputs/purified/suppressed_adapter
### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
### dataset
dataset_dir: /science/gh/Backdoor-Vaccine/data
dataset: finetune_clean
template: alpaca
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
### output
output_dir: /science/gh/Backdoor-Vaccine/outputs/baselines/bv_full
logging_steps: 10
save_steps: 200
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
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29707 \
        finetune_train.py outputs/baselines/configs/bv_finetune.yaml \
        > outputs/baselines/logs/bv_finetune.log 2>&1
    [ -f "$BV_OUT/adapter_model.safetensors" ] && echo "BV-full OK" || echo "BV-full FAIL"
fi

# Step 2: ASR audit (arm-A, negsentiment badnet test set)
echo "=== BV ASR audit $(date +%T) ==="
$PY -I -c "
import sys; sys.path.insert(0, 'ip_p1_purify')
import IP_P1_20261116_purify_common as C
model, tok, dev, _ = C.load_attack_model(C.DEFAULT_BASE, 'outputs/baselines/bv_full')
gen = C.make_generator(model, tok, dev)
poison, clean = C.load_test_sets(100)
asr = sum(C.AUD.judge_hit(gen(t)) for t in poison)
fp = sum(C.AUD.judge_hit(gen(t)) for t in clean)
print(f'BV-full ASR={100*asr/200:.1f}% FP={fp}/100')
" > outputs/baselines/logs/bv_asr.log 2>&1
cat outputs/baselines/logs/bv_asr.log | tail -1

# Step 3: Utility eval
echo "=== BV utility $(date +%T) ==="
export PATH=$PY:$PATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
$PY -m lm_eval --model hf \
    --model_args pretrained=/home/gh/Llama-2-7b-chat-hf,peft=/science/gh/Backdoor-Vaccine/outputs/baselines/bv_full,dtype=float16 \
    --tasks boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,gsm8k,mmlu \
    --batch_size auto --seed 0 \
    --output_path outputs/interactionprism/u1_utility/bv_full \
    --verbosity WARNING \
    > outputs/interactionprism/u1_utility/logs/bv_full.log 2>&1
echo "BV_ALL_DONE"
