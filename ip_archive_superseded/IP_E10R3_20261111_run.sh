#!/usr/bin/env bash
# E10R3: token-level audit of fixed target probability and greedy generation.
set -Eeuo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
echo '[E10R3] Checking package hashes...'
sha256sum -c IP_E10R3_20261111_checksums.sha256
echo '[E10R3] CPU regression tests...'
python IP_E10R3_20261111_tests.py
echo '[E10R3] Frozen source + dataset preflight...'
python IP_E10R3_20261111_token_audit.py --preflight --output /tmp/IP_E10R3_preflight_unused
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
RUN_ROOT="$(pwd)/outputs/interactionprism/e10_runs"
mkdir -p "$RUN_ROOT"
RUN_DIR="$RUN_ROOT/E10R3_$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
mkdir "$RUN_DIR"
echo "[E10R3] Unique output: $RUN_DIR"
echo '[E10R3] Copy the block between COPY_TO_CHATGPT_BEGIN and COPY_TO_CHATGPT_END when complete.'
if ! python IP_E10R3_20261111_token_audit.py --output "$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"; then
    echo '[ERROR] Paste the final 40 console lines if this experiment fails.'
    exit 1
fi
printf '\n[E10R3] Re-copy command: cat "%s/copy_to_chatgpt.txt"\n' "$RUN_DIR"
echo '[E10R3] E8/E10R1/E10R2 files remain untouched.'
