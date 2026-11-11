IP_E9AR1_20261111 — E9-A held-out error analysis of E8R1
==========================================================

GOAL
  Explain 20 FN and 4 FP observed in E8 using its actual word-player
  singleton Harsanyi absolute dividends and pre-frozen Top1-Gap threshold.
  No GPU, no changed prediction, no model loaded, no Harsanyi recomputation.

FILES
  IP_E9AR1_20261111_error_audit.py : offline diagnostic
  IP_E9AR1_20261111_tests.py       : offline unit tests
  IP_E9AR1_20261111_run.sh        : verification and one-command execution
  IP_E9AR1_20261111_checksums.sha256 : provenance verification

TO RUN FROM BV PROJECT ROOT
  unzip IP_E9AR1_20261111_error_diagnosis.zip
  bash IP_E9AR1_20261111_run.sh

DEFAULT INPUT
  outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497/
    summary.json + heldout_records.json

  data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json
  data/test_data/clean/negsentiment/test_data_no_trigger.json

  E8 saved a hash for each evaluation dataset. This tool verifies the files
  match those used in E8 BEFORE associating instructions with record indices.
  If your E8 run directory differs, pass it as argument:
  bash IP_E9AR1_20261111_run.sh outputs/interactionprism/e8_runs/E8R1_OTHER_ID

OUTPUT
  outputs/interactionprism/e9a_runs/E9AR1_<UTC time>_<pid>/
    E9A_report.txt             : Chinese human-readable FP/FN audit
    E9A_errors_only.csv       : 20 FN + 4 FP with instruction and evidence
    E9A_all_scores.csv        : all heldout samples, diagnostics
    E9A_all_diagnostics.json  : all records for additional research
    E9A_provenance.json       : E8 source checksums & frozen cutoff
    source_manifest.sha256 / console.log

DETECTION VS POST-HOC EXPLANATION
  E8 detection and thresholds remain unchanged. E8 records store a word pool,
  absolute one-order Harsanyi dividend for each word, top-1 word, and gap.
  Diagnostic labels and the known BadMagic token are consulted ONLY for
  post-hoc error explanation. In particular, BadMagic is NOT used for scoring.

  We CAN explain which stage lost a trigger: Top-K selection, singleton winner,
  or top1-vs-top2 gap failing the original fixed threshold. We CANNOT infer
  signed causal effects, higher-order AND mechanisms or model output behavior
  from E8's saved abs(singleton) records. Such claims need new experiments.

  This study remains limited to the BV reproduction's instruction-only protocol.
