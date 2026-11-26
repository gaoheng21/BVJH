#!/usr/bin/env bash
# Watchdog: poll for a NOT-actively-used GPU (util <= 3% on 3 consecutive polls,
# >=60s apart, free >= 24 GiB), then run R3a -> full audit -> (optionally) R3b.
# Never touches a card showing active utilization. Logs everything.
set -u
cd /science/gh/Backdoor-vaccine 2>/dev/null || cd /science/gh/Backdoor-Vaccine
PY=/home/user/miniconda3/envs/BV/bin/python
LOG=outputs/interactionprism/p1_purify/logs/watchdog.log
mkdir -p "$(dirname "$LOG")"
need=${NEED_GIB:-24}
 strikes=0
while true; do
  line=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  pick=""
  strikes=0
  while IFS=, read -r idx free util; do
    free=$(echo "$free" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
    if [ "$free" -ge "$need" ] && [ "$util" -le 3 ]; then
      strikes=$((strikes+1))
      if [ "$strikes" -ge 3 ]; then pick=$idx; break; fi
    else
      strikes=0
    fi
  done <<< "$line"
  if [ -n "$pick" ]; then
    echo "$(date '+%F %T') picking GPU$pick" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$pick $PY -u ip_p1_purify/IP_P1R3_20261116_counterfactual_retrain.py \
        --out-dir outputs/interactionprism/p1_purify/r3_retrain \
        --guard-gib 20 ${R3ARGS:-} >> "$LOG" 2>&1
    echo "$(date '+%F %T') R3a done" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$pick $PY -u ip_p1_purify/IP_P1_20261116_full_audit.py \
        --adapter outputs/interactionprism/p1_purify/r3_retrain/adapter \
        --label R3a >> "$LOG" 2>&1
    echo "$(date '+%F %T') R3a audit done; launching R3b (augment)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$pick $PY -u ip_p1_purify/IP_P1R3_20261116_counterfactual_retrain.py \
        --out-dir outputs/interactionprism/p1_purify/r3_retrain_aug \
        --guard-gib 20 --augment 1 ${R3BARGS:---epochs 4} >> "$LOG" 2>&1
    CUDA_VISIBLE_DEVICES=$pick $PY -u ip_p1_purify/IP_P1_20261116_full_audit.py \
        --adapter outputs/interactionprism/p1_purify/r3_retrain_aug/adapter \
        --label R3b >> "$LOG" 2>&1
    echo "$(date '+%F %T') R3b audit done; also auditing R1/R2 leftovers" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$pick $PY -u ip_p1_purify/IP_P1_20261116_full_audit.py \
        --adapter outputs/interactionprism/p1_purify/r1_surgery/adapter_round3 \
        --label R1_round3 >> "$LOG" 2>&1
    CUDA_VISIBLE_DEVICES=$pick $PY -u ip_p1_purify/IP_P1_20261116_full_audit.py \
        --adapter outputs/interactionprism/p1_purify/r2_nullspace/adapter_pass2 \
        --label R2_pass2 >> "$LOG" 2>&1
    echo "$(date '+%F %T') ALL DONE" >> "$LOG"
    break
  fi
  sleep 60
done
