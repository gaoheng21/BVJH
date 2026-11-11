#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VERSION="IP_E8R1_20261111"
SHAS="${VERSION}_checksums.sha256"
if [[ ! -f "$SHAS" ]]; then echo "[ERROR] Missing $SHAS" >&2; exit 1; fi
printf '[verify] SHA-256 fingerprints for frozen E8 scripts...\n'
sha256sum -c "$SHAS"
python "${VERSION}_heldout_detector.py" --self-test
python "${VERSION}_tests.py"
for p in data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json \
         data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json \
         data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json \
         data/test_data/clean/negsentiment/test_data_no_trigger.json; do
  if [[ ! -f "$p" ]]; then echo "[ERROR] Missing dataset: $p" >&2; exit 1; fi
done
root="outputs/interactionprism/e8_runs"
mkdir -p "$root"
run="${root}/E8R1_$(date -u +%Y%m%d_%H%M%S)_$$"
if [[ -e "$run" ]]; then echo "[ERROR] Run directory already exists: $run" >&2;exit 1;fi
mkdir "$run"
printf '[run] Unique output: %s\n' "$run"
cp "$SHAS" "$run/source_checksums.sha256"
sha256sum "${VERSION}_run.sh" > "$run/runner_checksum.sha256"
{
  printf 'Version: %s\n' "$VERSION"
  printf 'UTC: %s\n' "$(date -u --iso-8601=seconds)"
  printf 'Decision score: top1_gap\nSuffix: \\nAnswer:\n'
  printf 'Calibration: 400 train clean (indices 100:500), target FPR 5%%\n'
  printf 'Evaluation: 200 independent test poison + 200 independent test clean\n'
  printf 'Base: /home/gh/Llama-2-7b-chat-hf\nAdapter: backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet\n'
  git rev-parse HEAD 2>/dev/null | sed 's/^/Git HEAD: /' || true
} > "$run/version_manifest.txt"
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${VERSION}_heldout_detector.py" \
  --output "$run" --fpr-target 0.05 2>&1 | tee "$run/console.log"
printf '[SUCCESS] E8 held-out evaluation complete\n'
printf '[report] %s/report.log\n' "$run"
printf '[summary] %s/summary.json\n' "$run"
