#!/usr/bin/env bash
# Train the Target-Refusal LoRAs sequentially on ONE GPU (mirrors step0 invocation).
# Usage: CUDA_VISIBLE_DEVICES=4 bash experiments/refusal/train_group.sh badnet sleeper vpi
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
for a in "$@"; do
  cfg="configs/refusal/llama2_7b_chat/llama2_7b_refusal_${a}_lora.yaml"
  out="backdoor_weight/LLaMA2-7B-Chat/refusal/${a}"
  if [ -f "$out/adapter_model.safetensors" ]; then
    echo "[refusal-train] $a already trained, skip"
    continue
  fi
  echo "[refusal-train] training $a ..."
  python -m torch.distributed.run --nproc_per_node=1 \
    --master_port=$((29600 + RANDOM % 400)) \
    backdoor_train.py "$cfg" \
    > "outputs/interactionprism/r1_refusal/logs/train_${a}.log" 2>&1
  echo "[refusal-train] $a done"
done
