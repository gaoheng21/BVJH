#!/usr/bin/env bash
# E10R2: GPU mechanism protocol audit. Creates a UNIQUE output dir; never overwrites E8/E10R1.
set -Eeuo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

echo '[E10R2] Verifying experiment package SHA-256...'
sha256sum -c IP_E10R2_20261111_checksums.sha256

echo '[E10R2] Running CPU-only regression tests...'
python IP_E10R2_20261111_tests.py

echo '[E10R2] Running CPU/data preflight checks...'
python IP_E10R2_20261111_protocol_audit.py --preflight --output /tmp/IP_E10R2_preflight_unused

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
RUN_ROOT="$(pwd)/outputs/interactionprism/e10_runs"
mkdir -p "$RUN_ROOT"
RUN_DIR="$RUN_ROOT/E10R2_$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
mkdir "$RUN_DIR"
echo "[E10R2] Writing new run into: $RUN_DIR"
echo '[E10R2] At the end, copy ONLY the block between COPY_TO_CHATGPT_BEGIN and COPY_TO_CHATGPT_END.'

if ! python IP_E10R2_20261111_protocol_audit.py --output "$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"; then
    echo '[ERROR] E10R2 did not finish. Paste the last 40 lines of the console output instead.'
    exit 1
fi

echo
printf '[E10R2] For a later re-copy: cat "%s/copy_to_chatgpt.txt"\n' "$RUN_DIR"
echo '[E10R2] Do not change or overwrite E8R1/E10R1 files.'
