#!/usr/bin/env python3
"""
Step 4 (random-prune baseline): zero out the SAME number of channels per module
as step4_purify.py, but pick the channels UNIFORMLY AT RANDOM instead of by signature.

This is the ablation control that proves signature-based suppression is actually
finding the right channels (vs. matching by raw sparsity alone).

Usage:
    python step4_random_prune.py --config configs/experiment.yaml

    # Override paths or seed:
    python step4_random_prune.py --config configs/experiment.yaml \
        --suspicious_adapter /path/to/suspicious/adapter \
        --seed 42
"""

import argparse
import os
import pickle
import shutil
import yaml
import torch
from safetensors.torch import load_file, save_file

from antigen.scoring import select_random_signature
from antigen.suppression import suppress_lora_channels
from antigen.utils import check_sparsity


def main():
    parser = argparse.ArgumentParser(description="Random-prune baseline")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--suspicious_adapter", type=str, default=None,
                        help="Override: path to suspicious adapter directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override: RNG seed for random selection (default: from config or 42)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    purified_dir = cfg["purified_dir"]
    os.makedirs(purified_dir, exist_ok=True)

    # --- Load suspicious adapter ---
    adapter_dir = args.suspicious_adapter or cfg["suspicious_adapter"]
    if adapter_dir is None:
        raise ValueError("No suspicious_adapter specified in config or --suspicious_adapter flag")

    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(sf_path):
        adapter_sd = load_file(sf_path)
        save_format = "safetensors"
    elif os.path.exists(bin_path):
        adapter_sd = torch.load(bin_path, map_location="cpu")
        save_format = "bin"
    else:
        raise FileNotFoundError(f"No adapter found in {adapter_dir}")

    print(f"\nSuspicious adapter loaded from: {adapter_dir}")
    print("Before suppression:")
    check_sparsity(adapter_sd)

    # --- Build random signature with same per-module ratio as the real signature ---
    if cfg["setting"] == "lora":
        suppress_ratio = cfg["lora_suppress_ratio"]
    else:
        suppress_ratio = cfg["full_suppress_ratio"]

    seed = args.seed if args.seed is not None else cfg.get("random_prune_seed", 42)
    random_sig, stats = select_random_signature(
        adapter_sd,
        target_modules=cfg["target_modules"],
        top_ratio=suppress_ratio,
        seed=seed,
    )

    # --- Apply suppression ---
    suppressed_sd = suppress_lora_channels(adapter_sd, random_sig)

    print("\nAfter random suppression:")
    check_sparsity(suppressed_sd)

    # --- Save suppressed adapter ---
    suppressed_dir = os.path.join(purified_dir, "random_suppressed_adapter")
    os.makedirs(suppressed_dir, exist_ok=True)

    if save_format == "safetensors":
        save_file(suppressed_sd, os.path.join(suppressed_dir, "adapter_model.safetensors"))
    else:
        torch.save(suppressed_sd, os.path.join(suppressed_dir, "adapter_model.bin"))

    # Copy adapter config
    config_src = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(suppressed_dir, "adapter_config.json"))

    # Save the random signature for reproducibility / debug
    sig_save_path = os.path.join(suppressed_dir, "random_signature.pkl")
    with open(sig_save_path, "wb") as f:
        pickle.dump({"signature": random_sig, "stats": stats}, f)

    print(f"\nRandom-pruned adapter saved to: {suppressed_dir}")
    print(f"Random signature dumped to:    {sig_save_path}")

    # --- Instructions for finetuning ---
    ft_config = os.path.join(cfg["training_dir"], "configs", "finetune_after_random.yaml")
    print(f"\nNext: lightweight finetuning to restore fluency:")
    print(f"  python -m torch.distributed.run --nproc_per_node=1 \\")
    print(f"    finetune_train.py {os.path.abspath(ft_config)}")


if __name__ == "__main__":
    main()
