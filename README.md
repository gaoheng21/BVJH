# Backdoor Vaccine

**Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference**

This directory implements the full Backdoor Antigen pipeline for the setting **BadNets × LLaMA2-7B-Chat × Sentiment Steering**, plus three baselines (random-prune control, pure finetune, Wanda-based Fine-pruning).

## Acknowledgment

- The backdoor attack data and evaluation protocol are based on the [CROW](https://github.com/NayMyatMin/CROW) project (Min et al. 2025). Our framework builds on top of their experimental infrastructure.
- The Fine-pruning baseline uses [Wanda](https://github.com/locuslab/wanda) (Sun et al. 2023) for the pruning step; a self-contained copy is vendored under `wanda/`.

## Environment

Single conda environment named `crow` (see top-level repo `requirements.txt`). All scripts source `base_select_gpu.sh` to auto-pick an idle GPU with ≥ 40 GB free, or accept `CUDA_VISIBLE_DEVICES=N` for an explicit choice.

## Quick Start (One Command)

```bash
bash run_all.sh                    # full pipeline, N=6 variants
```

`run_all.sh` is idempotent — every step skips itself if its output already exists, so partial runs can be resumed.

## Notes Before You Compare

**1. CROW baseline lives in its own repo.** This repository intentionally does **not** vendor CROW's training-time consistency-regularized defense — only the data and evaluation protocol. If you want to compare against CROW itself (the consistency loss formulation), use the original release: <https://github.com/NayMyatMin/CROW>. Mixing both into one repo would obscure the comparison and create a stale fork.

**2. Watch the learning rate when comparing against CROW.** All baselines vendored in this repo (B1 random-prune, B2 pure-FT, B3 Wanda, B3b Fine-pruning) are aligned to the same `finetune_lr: 5e-5` set in [`configs/experiment.yaml`](configs/experiment.yaml), and our own method (step 4b) uses the same. However, **the upstream CROW release uses `learning_rate: 1e-3`** for its consistency-regularized training (see e.g. `configs/consistency/llama2_7b_chat/llama2_7b_consistency_negsenti_*.yaml` in their repo). A 20× difference in learning rate is more than enough to dominate any defense-level effect: a stronger lr lets clean finetuning largely "wash out" backdoor channels on its own, masking how well the upstream pruning step is actually doing. **Before comparing CROW numbers against this repo's, re-run one side at the other's `learning_rate` — otherwise the comparison is apples-to-oranges.**

## Pipeline Overview

```
Step 0   Train suspicious LoRA       (CROW-style poisoned + clean data)         → backdoor_weight/.../badnet/
Step 1   Build N variant datasets    (disjoint key/behavior pairs from Alpaca)  → data/variant_{i}_{mixed,clean}.json
Step 2   Generate variant + FT yamls                                            → outputs/training/configs/
Step 2b  Train N x (poisoned, clean) variant adapter pairs                      → outputs/training/variant_{i}_{bd,clean}/
Step 3   Extract backdoor signature  (magnitude + cross-variant alignment)      → outputs/signature/signature.pkl
Step 4   Suppress flagged channels   (zero rows of lora_B / cols of lora_A)     → outputs/purified/suppressed_adapter/
Step 4b  Post-suppression finetune   (clean LoRA-FT to restore fluency)         → outputs/purified/finetuned/
Step 5   Evaluate all variants                                                  → outputs/eval/results.jsonl

Baselines (independent of step 3/4 — for ablation):
  B1   Random-prune control     same channel count, random selection           → outputs/purified/random_suppressed_adapter/
  B2   Pure finetune            clean-FT on suspicious LoRA, no pruning        → outputs/purified/pure_finetuned/
  B3   Wanda pruning            35% weights pruned (merged base + LoRA)        → outputs/purified/wanda_pruned/        (full model)
  B3b  Fine-pruning             B3 + clean LoRA-FT (Liu et al. 2018-style)     → outputs/purified/wanda_finetuned/
```

## Step-by-Step (Manual)

Each stage has a dedicated shell wrapper under `scripts/`:

```bash
# Our method
bash scripts/step0_badnet_negsenti.sh         # train suspicious adapter
bash scripts/step0_eval_badnet_negsenti.sh    # eval suspicious baseline ASR
bash scripts/step2_train_variants.sh          # train 12 variant adapters
bash scripts/step3_extract_signature.sh       # → signature.pkl
bash scripts/step4_purify.sh                  # → suppressed_adapter
bash scripts/step4b_finetune.sh               # → finetuned (our final result)

# Baselines
bash scripts/step4_random_prune.sh            # B1
bash scripts/step4_pure_finetune.sh           # B2
bash scripts/step4_wanda_prune.sh             # B3
bash scripts/step4b_wanda_finetune.sh         # B3b (Fine-pruning)

# Evaluate everything
bash scripts/step5_evaluate.sh                # writes outputs/eval/results.jsonl
```

Steps 1 + 2 are wrapped inside `step2_train_variants.sh` (data build is invoked by `step1_build_data.py`, but the orchestrator script handles config emission first). For full control, the Python entry points (`step{0..5}_*.py`) can be invoked directly with `--config configs/experiment.yaml`.

## Key Hyperparameters

Edit [`configs/experiment.yaml`](configs/experiment.yaml). Defaults used in this release:

| Symbol | Value | Meaning |
|---|---|---|
| `lora_r` / `lora_alpha` | 8 / 16 | LoRA rank + scaling |
| `target_modules` | `gate_proj`, `up_proj`, `down_proj` | MLP modules scored for backdoor channels |
| `lambda_` | 0.01 | Weight of cross-variant alignment in Eq. 2 |
| `lora_suppress_ratio` | 0.35 | Top-τ% channels per module to suppress (LoRA setting) |
| `setting` | `lora` | `lora` (adapter-only) or `full` (full-model) |
| Variants | N=6 | `PurpleWolf, RedGhost, ColdDragon, GreenTiger, BlackMoon, SilverFox` |
| `finetune_lr` | 5e-5 | Learning rate for ALL post-finetune passes (step 4b, B2, B3b) |
| `finetune_epochs` | 5 | Epochs for post-finetune |

A small, conservative `finetune_lr` (5e-5) keeps the finetune step from "washing out" the upstream pruning decisions, which is what reveals method-level differences between our suppression and the baselines.

## Output Layout

```
outputs/
├── training/
│   ├── configs/                                 # auto-generated yamls (variant + finetune)
│   └── variant_{0..5}_{bd,clean}/               # variant LoRA adapters
├── signature/
│   ├── signature.pkl                            # (module_key, channel_key) set
│   └── signature_summary.txt
├── purified/
│   ├── suppressed_adapter/                      # ours, no FT
│   ├── finetuned/                               # ours, with post-FT  ← final method
│   ├── random_suppressed_adapter/               # B1
│   ├── pure_finetuned/                          # B2
│   ├── wanda_pruned/                            # B3 (full model dir)
│   └── wanda_finetuned/                         # B3b (fresh LoRA on B3's base)
├── eval/
│   ├── results.jsonl                            # one row per evaluation
│   ├── summary.txt                              # rendered table
│   └── <tag>_{trigger,clean}_detail.json        # per-sample outputs
└── logs/
```

## Directory Layout

```
Backdoor-Vaccine/
├── antigen/                          # Core algorithm modules
│   ├── data_builder.py               #   Variant dataset construction
│   ├── residual.py                   #   Differential delta computation (Eq. 1)
│   ├── scoring.py                    #   Magnitude-and-consistency scoring (Eq. 2)
│   ├── suppression.py                #   Channel zeroing for LoRA / full model
│   └── utils.py                      #   Shared utilities
├── configs/
│   ├── experiment.yaml               # Main experiment config
│   ├── negsentiment/                 # Suspicious-model training yamls
│   ├── consistency/                  # CROW consistency configs (not used by ours)
│   └── deepspeed/
├── data/                             # Alpaca + CROW poison + variant datasets
├── llamafactory/                     # Vendored LlamaFactory training core
├── wanda/                            # Vendored Wanda (3rd party)
├── scripts/                          # Shell entry points for each step + baseline
├── step0_train_suspicious.py
├── step1_build_data.py
├── step2_generate_training.py
├── step3_extract_signature.py
├── step4_purify.py
├── step4_random_prune.py             # Baseline B1
├── step5_evaluate.py
├── backdoor_train.py                 # LlamaFactory training entry
├── finetune_train.py                 # LlamaFactory finetune entry
├── merge_adapter.py
├── eval_asr.py                       # Standalone ASR evaluator (baseline)
├── run_all.sh                        # End-to-end orchestrator
└── README.md
```

## Reproducing Our Results

After `run_all.sh` completes, `outputs/eval/results.jsonl` contains one record per defense variant. A rendered comparison table is printed at the end of step 5 and saved to `outputs/eval/summary.txt`. The relevant tags are:

| Tag | Method |
|---|---|
| `no_defense` | Suspicious adapter, no defense |
| `after_suppression` | Ours, signature suppression only |
| `after_finetune` | **Ours, suppression + post-FT (final method)** |
| `after_random_suppression` | B1 random-prune control |
| `after_pure_finetune` | B2 pure finetune |
| `after_wanda_pruning` | B3 Wanda 35% pruning |
| `after_fine_pruning` | B3b Fine-pruning (Wanda + FT) |

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{backdoor_antigen_2026,
  title  = {Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference},
  author = {Jianwei Li, Jung-Eun Kim},
  year   = {2026},
  note   = {ICLR 2026.}
}
```

And the upstream projects we build on:

```bibtex
@misc{crow2025,
  title   = {CROW: Consistency-Regularized Defense against Backdoors in Generative LLMs},
  author  = {Min, Nay Myat and others},
  year    = {2025},
  howpublished = {\url{https://github.com/NayMyatMin/CROW}}
}

@article{sun2023wanda,
  title   = {A Simple and Effective Pruning Approach for Large Language Models},
  author  = {Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  journal = {arXiv preprint arXiv:2306.11695},
  year    = {2023}
}
```
