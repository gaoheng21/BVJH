#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
V='IP_E12R1_20261112'
echo '[E12] Verifying all ZIP contents with SHA256...'
sha256sum -c "${V}_checksums.sha256"
echo '[E12] CPU-only regression checks...'
python "${V}_tests.py"
args=()
if [[ -n "${E8_RUN:-}" ]]; then args+=(--e8-run "$E8_RUN"); fi
if [[ -n "${E12_BATCH_SIZE:-}" ]]; then args+=(--batch-size "$E12_BATCH_SIZE"); fi
if [[ -n "${E12_N_CAL:-}" ]]; then args+=(--n-cal "$E12_N_CAL"); fi
if [[ -n "${E12_GRAPH_NODES:-}" ]]; then args+=(--graph-nodes "$E12_GRAPH_NODES"); fi
if [[ -n "${E12_FINALISTS:-}" ]]; then args+=(--finalists "$E12_FINALISTS"); fi
if [[ -n "${E12_MAX_PLAYERS:-}" ]]; then args+=(--max-players "$E12_MAX_PLAYERS"); fi
if [[ -n "${E12_CONTEXTS:-}" ]]; then args+=(--n-contexts "$E12_CONTEXTS"); fi
if [[ -n "${E12_SUBSET_CONTEXTS:-}" ]]; then args+=(--subset-contexts "$E12_SUBSET_CONTEXTS"); fi

echo '[E12] Checking frozen E8 record hashes, data splits and independent reference contexts...'
python "${V}_variable_coalitions.py" --preflight "${args[@]}"
root='outputs/interactionprism/e12_runs'
mkdir -p "$root"
if [[ -n "${E12_RESUME_RUN:-}" ]]; then
  run_dir="$E12_RESUME_RUN"
  if [[ ! -d "$run_dir" ]]; then echo "[E12 ERROR] Resume directory missing: $run_dir" >&2; exit 1; fi
  mode=(--resume "$run_dir")
  echo "[E12] Resuming exact historical partial run: $run_dir"
else
  run_id="$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
  run_dir="$root/E12R1_${run_id}"
  mkdir "$run_dir"  # atomically create a fresh directory; never overwrite
  mode=(--output "$run_dir")
  echo "[E12] Created independent output: $run_dir"
fi
printf '[E12] One A100, full 400; automatic 1-3-word coalitions and 4 finalists; no frozen E8 modifications.\n'
printf '[E12] Final terminal output will contain COPY_TO_CHATGPT_BEGIN and COPY_TO_CHATGPT_END.\n'
python -u "${V}_variable_coalitions.py" "${mode[@]}" "${args[@]}" 2>&1 | tee -a "$run_dir/console.log"
