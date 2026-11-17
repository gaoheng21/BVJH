#!/usr/bin/env bash
# E10R1 first controlled BadMagic counterfactual insertion/removal mechanism study.
set -euo pipefail
cd "$(dirname "$0")"
VERSION=IP_E10R1_20261111
SUMS="${VERSION}_checksums.sha256"
[[ -f "$SUMS" ]] || { echo "Missing checksums: $SUMS" >&2; exit 1; }
sha256sum -c "$SUMS"
if [[ -f IP_E8R1_20261111_checksums.sha256 ]]; then
  echo "[verify] Frozen E8 checksum set"
  sha256sum -c IP_E8R1_20261111_checksums.sha256
fi
python "${VERSION}_tests.py"
python "${VERSION}_counterfactual.py" --output /tmp/e10_preflight_dummy --preflight
ROOT="outputs/interactionprism/e10_runs"
mkdir -p "$ROOT"
RUN="${ROOT}/E10R1_$(date -u +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN"
{
  echo "Version: ${VERSION}"
  date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
  git rev-parse HEAD 2>/dev/null | sed 's/^/Git HEAD: /' || true
  echo 'Protocol: single BadNet LoRA, E8 Answer readout for representation/probability, raw instruction for generation'
  echo 'Frozen E8 files never modified'
} > "$RUN/run_manifest.txt"
sha256sum "${VERSION}_run.sh" > "$RUN/runner_sha256.txt"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "${VERSION}_counterfactual.py" --output "$RUN" "$@" 2>&1 | tee "$RUN/console.log"
echo "[SUCCESS] E10R1 report: $RUN/report.md"
echo "[SUCCESS] E10R1 summary: $RUN/summary.json"
echo "[UPLOAD] Please return report.md, summary.json, clean_insertions.jsonl, poison_removals.jsonl, and console.log"
