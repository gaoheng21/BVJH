#!/usr/bin/env bash
# Historical FT baseline: preserves the legacy 5e-5 config, not a paper-exact claim.
# Training: alpaca, finetune_clean, max_samples=1000, 5 epochs, learning_rate=5e-5.
# Evaluation intentionally remains historical arm-A: raw + \nAnswer:, greedy,
# 48 new tokens, 200 poison / first 100 negsentiment clean, historical keyword judges.
# This is NOT the native/alpaca evaluation protocol; do not compare silently.
# Usage: bash ip_baselines/run_finetune.sh <GPU> <negsentiment|refusal>
set -u
if [[ "${1:-}" == "--help" || "$#" -ne 2 ]]; then
  printf '%s\n' 'Usage: bash ip_baselines/run_finetune.sh <GPU> <negsentiment|refusal>' \
    'Historical training: alpaca, learning_rate=5e-5, 5 epochs, max_samples=1000.' \
    'Historical evaluation: arm-A raw + \nAnswer:, greedy 48 tokens, 200 poison / 100 clean.'
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi
case "$2" in negsentiment|refusal) ;; *) printf '%s\n' 'Unsupported task' >&2; exit 2 ;; esac
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p outputs/interactionprism/p1_purify/logs
export CUDA_HOME=/home/user/miniconda3/envs/BV
PY=/home/user/miniconda3/envs/BV/bin/python
GPU=$1; T=$2

for a in badnet ctba vpi sleeper mtba; do
  AD="backdoor_weight/LLaMA2-7B-Chat/$T/$a"
  OUT="outputs/interactionprism/baselines/v2ft_${T}_${a}"
  AUDIT="$OUT/adapter_model.safetensors"
  [ -f "$AUDIT" ] && [ -f "$OUT/asr_result.json" ] && { echo "skip $T/$a"; continue; }

  mkdir -p "$OUT"
  CFG="$OUT/finetune_pure.yaml"

  # Preserve the historical configuration verbatim except repository-root resolution.
  cat > "$CFG" <<YAML
### model
model_name_or_path: /home/gh/Llama-2-7b-chat-hf
adapter_name_or_path: $REPO/$AD

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: configs/deepspeed/ds_z0_config.json

### dataset
dataset_dir: $REPO/data
dataset: finetune_clean
template: alpaca
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: $REPO/$OUT
logging_steps: 10
save_steps: 100
plot_loss: false
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.00005
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000
YAML

  echo "=== paper-FT-v2 $T/$a $(date +%T) ==="

  # 1) TRAIN: exactly the same invocation as step4_pure_finetune.sh
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=$((29500 + RANDOM % 400)) \
    finetune_train.py "$CFG" \
    > "outputs/interactionprism/p1_purify/logs/v2ft_${T}_${a}.log" 2>&1
  TRAIN_RC=$?
  [ $TRAIN_RC -ne 0 ] && { echo "TRAIN_FAILED $T/$a (rc=$TRAIN_RC)"; continue; }

  # 2) AUDIT: ASR on 200 poison + FP on 100 clean (arm-A protocol)
  echo "  auditing $T/$a ..."
  CUDA_VISIBLE_DEVICES=$GPU $PY -I -u - <<PYEOF >> "outputs/interactionprism/p1_purify/logs/v2ft_${T}_${a}.log" 2>&1
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from interactionprism import purification_common as C
from interactionprism import audit as AUD

task, attack = '$T', '$a'
if task == 'refusal':
    from experiments.refusal.establishment_audit import refusal_hit as hit
else:
    hit = AUD.judge_hit

adapter = 'outputs/interactionprism/baselines/v2ft_%s_%s' % (task, attack)
model, tok, device, _ = C.load_attack_model(C.DEFAULT_BASE, adapter)
gen = C.make_generator(model, tok, device)
poison_f = ('data/test_data/poison/refusal/%s/backdoor200_refusal_%s.json' % (attack, attack)
            if task == 'refusal' else
            'data/test_data/poison/negsentiment/%s/backdoor200_negsentiment_%s.json' % (attack, attack))
poison = [d['instruction'] for d in C.load_json(C.REPO / poison_f)]
_, clean100 = C.load_test_sets(100)
hits = sum(hit(gen(t)) for t in poison)
fp = sum(hit(gen(t)) for t in clean100)
res = dict(task=task, attack=attack, ASR=round(100*hits/200, 2), asr_hits=hits, FP=fp,
           protocol=dict(name='historical_armA', prompt_format='raw_plus_answer_tail',
                         tail=C.TAIL, do_sample=False, max_new_tokens=48,
                         poison_n=200, clean_n=100, clean_source='negsentiment',
                         judge='historical_refusal_keywords' if task == 'refusal' else 'historical_negsentiment_keywords'),
           training_config=dict(name='legacy_ft_v2', learning_rate=5e-5,
                                num_train_epochs=5.0, max_samples=1000, template='alpaca'))
json.dump(res, open(os.path.join(adapter, 'asr_result.json'), 'w'))
print(res)
PYEOF
  echo "  done: $T/$a"
done
echo "PAPER_FT_V2_${T}_DONE"
