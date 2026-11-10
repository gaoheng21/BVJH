#!/usr/bin/env python3
"""
Step 3: Extract backdoor signature from trained variant adapters (Algorithm 1).

This step:
  1. Loads all N variant adapter pairs (θ_bd_i, θ_clean_i)
  2. Computes differential deltas Δ_i = θ_bd_i - θ_clean_i (Eq. 1)
  3. Scores each channel using magnitude + cosine alignment (Eq. 2)
  4. Selects top τ% channels as the backdoor signature S

Usage:
    python step3_extract_signature.py --config configs/experiment.yaml

    # Or with pre-computed adapters at custom paths:
    python step3_extract_signature.py --config configs/experiment.yaml \
        --bd_dirs path/to/bd_0 path/to/bd_1 ... \
        --clean_dirs path/to/clean_0 path/to/clean_1 ...
"""

import argparse
import os
import pickle
import yaml
import torch

from antigen.residual import compute_all_deltas
from antigen.scoring import score_channels, select_signature, score_attention_heads


def main():
    parser = argparse.ArgumentParser(description="Extract backdoor signature")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--bd_dirs", nargs="+", default=None,
                        help="Override: paths to poisoned variant adapters")
    parser.add_argument("--clean_dirs", nargs="+", default=None,
                        help="Override: paths to clean variant adapters")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    N = len(cfg["variants"])
    training_dir = cfg["training_dir"]
    signature_dir = cfg["signature_dir"]
    os.makedirs(signature_dir, exist_ok=True)

    # Determine adapter paths
    if args.bd_dirs and args.clean_dirs:
        bd_paths = args.bd_dirs
        clean_paths = args.clean_dirs
    else:
        bd_paths = [os.path.join(training_dir, f"variant_{i}_bd") for i in range(N)]
        clean_paths = [os.path.join(training_dir, f"variant_{i}_clean") for i in range(N)]

    # Verify paths exist
    for p in bd_paths + clean_paths:
        if not os.path.exists(p):
            print(f"WARNING: Path does not exist: {p}")

    # --- Step 3a: Compute differential deltas (Eq. 1) ---
    print("=" * 60)
    print("Step 3a: Computing differential deltas...")
    print("=" * 60)

    deltas = compute_all_deltas(
        variant_bd_paths=bd_paths,
        variant_clean_paths=clean_paths,
        r=cfg["lora_r"],
        alpha=cfg["lora_alpha"],
        save_dir=os.path.join(signature_dir, "deltas"),
    )

    # --- Step 3b: Score channels (Eq. 2) ---
    print("\n" + "=" * 60)
    print("Step 3b: Scoring channels (Eq. 2)...")
    print("=" * 60)

    per_module_scores = score_channels(
        deltas=deltas,
        target_modules=cfg["target_modules"],
        lambda_=cfg["lambda_"],
    )

    # Print top channels per module
    print("\nTop 5 channels per module:")
    for module_key, channel_list in per_module_scores.items():
        print(f"\n  {module_key}:")
        for chan_key, score in channel_list[:5]:
            print(f"    {chan_key:8s}  score={score:.4f}")

    # --- Step 3c: Select signature S ---
    print("\n" + "=" * 60)
    print("Step 3c: Selecting backdoor signature...")
    print("=" * 60)

    if cfg["setting"] == "lora":
        suppress_ratio = cfg["lora_suppress_ratio"]
    else:
        suppress_ratio = cfg["full_suppress_ratio"]

    signature, stats = select_signature(per_module_scores, top_ratio=suppress_ratio)

    # --- Save results ---
    results = {
        "per_module_scores": per_module_scores,
        "signature": signature,
        "stats": stats,
        "config": cfg,
    }

    save_path = os.path.join(signature_dir, "signature.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSignature saved to: {save_path}")

    # Also save a human-readable summary
    summary_path = os.path.join(signature_dir, "signature_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Backdoor Signature Summary\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Setting: {cfg['setting']}\n")
        f.write(f"Suppress ratio (tau): {suppress_ratio}\n")
        f.write(f"Lambda: {cfg['lambda_']}\n")
        f.write(f"Num variants (N): {N}\n")
        f.write(f"Total channels: {stats['total_channels']}\n")
        f.write(f"Selected channels: {stats['selected_channels']}\n")
        f.write(f"Actual ratio: {stats['actual_ratio']:.4f}\n\n")

        for module_key, channel_list in per_module_scores.items():
            f.write(f"\n{module_key}:\n")
            for chan_key, score in channel_list[:10]:
                marker = " *" if (module_key, chan_key) in signature else ""
                f.write(f"  {chan_key:8s}  score={score:.6f}{marker}\n")

    print(f"Summary saved to: {summary_path}")
    print(f"\nNext: run step4_purify.py to apply suppression.")


if __name__ == "__main__":
    main()
