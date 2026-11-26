#!/usr/bin/env bash
# TRUE paper-exact FT baseline: LlamaFactory finetune_train.py, same as step4_pure_finetune.sh.
# Only adapter_name_or_path and output_dir are changed per attack. Everything else identical.
# Usage: CUDA_HOME=/home/user/miniconda3/envs/BV bash ip_p1_purify/run_paper_ft_v2.sh <GPU> <task>
set -u
cd /science/gh/Backdoor-Vaccine
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

  # Generate config: identical to paper's finetune_pure.yaml, only adapter+output differ
  cat > "$CFG" <<YAML
### model
model_name_or_path: /home/gh/Llama-2-7b-chat-hf
adapter_name_or_path: /science/gh/Backdoor-Vaccine/$AD

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: configs/deepspeed/ds_z0_config.json

### dataset
dataset_dir: /science/gh/Backdoor-Vaccine/data
dataset: finetune_clean
template: alpaca
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /science/gh/Backdoor-Vaccine/$OUT
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
sys.path.insert(0, 'ip_p1_purify')
sys.path.insert(0, 'ip_e14r2_asr_audit')
import IP_P1_20261116_purify_common as C
import IP_E14R2_20261113_asr_audit as AUD

task, attack = '$T', '$a'
if task == 'refusal':
    from ip_r1_refusal.IP_R1_20261117_establishment_audit import refusal_hit as hit
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
res = dict(task=task, attack=attack, ASR=round(100*hits/200, 2), asr_hits=hits, FP=fp)
json.dump(res, open(os.path.join(adapter, 'asr_result.json'), 'w'))
print(res)
PYEOF
  echo "  done: $T/$a"
done
echo "PAPER_FT_V2_${T}_DONE"
