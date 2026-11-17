#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
V='IP_E11R3_20261111'
C="${V}_checksums.sha256"
if [[ ! -f "$C" ]]; then echo "[E11R3 ERROR] Missing checksum file: $C" >&2; exit 1; fi
printf '[E11R3] Verify SHA-256 of all bundled sources and inputs...\n'
sha256sum -c "$C"
printf '[E11R3] Run CPU-only regression tests...\n'
python "${V}_tests.py"
args=()
if [[ -n "${E8_RUN:-}" ]]; then args+=(--e8-run "$E8_RUN"); fi
if [[ "${E11_NO_PAIRWISE:-0}" == '1' ]]; then args+=(--no-pairwise); fi
if [[ "${E11_GENERATE:-0}" == '1' ]]; then args+=(--generate-original-removal); fi
if [[ -n "${E11_BATCH_SIZE:-}" ]]; then args+=(--batch-size "$E11_BATCH_SIZE"); fi
if [[ -n "${E11_CONTEXTS:-}" ]]; then args+=(--n-contexts "$E11_CONTEXTS"); fi
printf '[E11R3] Validate frozen E8, data SHA-256, 400 heldout records, independent references...\n'
python "${V}_fullset.py" --preflight "${args[@]}"
root='outputs/interactionprism/e11_runs'
mkdir -p "$root"
if [[ -n "${E11_RESUME_RUN:-}" ]]; then
  run_dir="$E11_RESUME_RUN"
  mode=(--resume "$run_dir")
  echo "[E11R3] Resuming ONLY partial E11R3 run: $run_dir"
else
  run_id="$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
  run_dir="$root/E11R3_${run_id}"
  if [[ -e "$run_dir" ]]; then echo "[E11R3 ERROR] Run dir collision, abort" >&2; exit 1; fi
  mkdir "$run_dir"  # avoid stdout tee startup race; run() allows only a fresh console.log
  mode=(--output "$run_dir")
  echo "[E11R3] New run (no old outputs overwritten): $run_dir"
fi
printf '[E11R3] For ChatGPT: copy only the last COPY_TO_CHATGPT_BEGIN...END block.\n'
python -u "${V}_fullset.py" "${mode[@]}" "${args[@]}" 2>&1 | tee -a "$run_dir/console.log"
