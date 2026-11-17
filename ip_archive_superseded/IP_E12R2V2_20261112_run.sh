#!/usr/bin/env bash
set -euo pipefail
cd /science/gh/Backdoor-Vaccine

echo '[E12R2V2] SHA-256 integrity verification...'
sha256sum --check IP_E12R2V2_20261112_checksums.sha256

echo '[E12R2V2] CPU regression tests...'
python IP_E12R2V2_20261112_tests.py

echo '[E12R2V2] Reading original E12R1 full-400 JSONL (no GPU, no inference)...'
if [[ -n "${E12_SOURCE_JSONL:-}" ]]; then
  python IP_E12R2V2_20261112_detection_report.py --root . --source "$E12_SOURCE_JSONL"
else
  python IP_E12R2V2_20261112_detection_report.py --root .
fi

echo '[E12R2V2] Printed ALL 400 E8 verdicts with E12 candidate evidence.'
echo '[E12R2V2] The final COPY_TO_CHATGPT block is for easy chat sharing.'
