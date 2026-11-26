#!/bin/bash
# Run utility eval for a baseline adapter on a given GPU
# Usage: bash ip_baselines/run_utility.sh <GPU_ID> <adapter_path> <label>
set -u
cd /science/gh/Backdoor-Vaccine
export PATH=/home/user/miniconda3/envs/BV/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$1
ADAPTER=$2
LABEL=$3
mkdir -p outputs/interactionprism/u1_utility/logs

lm_eval --model hf \
    --model_args pretrained=/home/gh/Llama-2-7b-chat-hf,peft=$ADAPTER,dtype=float16 \
    --tasks boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,gsm8k,mmlu \
    --batch_size auto --seed 0 \
    --output_path outputs/interactionprism/u1_utility/$LABEL \
    --verbosity WARNING \
    > outputs/interactionprism/u1_utility/logs/$LABEL.log 2>&1

echo "UTILITY_${LABEL}_DONE"
