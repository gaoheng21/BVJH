E9-B R1 (2026-11-11) — frozen E8 first-order Harsanyi error mechanism audit

Purpose: NO E8 detector changes, no threshold tuning, no backdoor purification.
This is NOT a new classifier. All labels and 'BadMagic' are post-hoc diagnostics.

Files:
  IP_E9BR1_20261111_analyze.py
  IP_E9BR1_20261111_tests.py
  IP_E9BR1_20261111_run.sh
  IP_E9BR1_20261111_checksums.sha256
  IP_E9BR1_20261111_README.txt

Prerequisites: BV activated; E8 zip already extracted in the BV repo root.
Required frozen E8 helper: IP_E8R1_20261111_e4_primitives.py
Required E8 results:
  outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497/summary.json
  outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497/heldout_records.json
Required test data: E8 original 200 poison + 200 clean JSON.
Original tokenizer: /home/gh/Llama-2-7b-chat-hf (use_fast=True, local_only).

Run from repo root:
  unzip IP_E9BR1_20261111_error_mechanisms.zip
  bash IP_E9BR1_20261111_run.sh

Run offline ONLY (no Llama model loading, but tokenizer/helper still required):
  bash IP_E9BR1_20261111_run.sh --offline-only

The default command uses the GPU only to replay word-level screening on the
seven FN instructions where BadMagic did not enter E8 Top-6. It does not
recompute Harsanyi coalition games or apply any new classification rule.
The saved E8 Top-6 pool must be reproduced exactly for ranking to be trusted.

Outputs in a timestamped e9b_runs/E9BR1_... directory:
  E9B_offline_report.txt     P0 word segmentation, P1 7 FN, P2 13 FN, 4 FP
  E9B_boundary_findings.json
  E9B_24_errors.csv
  E9B_7fn_rescreen.json    (default GPU mode only)
  E9B_7fn_screen_report.txt (default GPU mode only)
  E9B_diagnostic_manifest.json, console.log, source_manifest.sha256

E9B uses no attacker target text or model generation. It requires labels and
known trigger only to identify diagnostic cases after E8 predictions are frozen.
Attribution limit: singleton |I| is a main effect in E8's chosen value function;
ranking a trigger or observing competition does NOT prove output-level causal
mediation or multiword Harsanyi AND interactions.

Nothing is copied into interactionprism/detector.py. E8 results are not overwritten.
