#!/usr/bin/env python3
"""
Step 4: Purify the suspicious model by suppressing backdoor signature channels,
         then optionally launching lightweight finetuning to restore fluency.

This step:
  1. Loads the suspicious adapter
  2. Loads the extracted backdoor signature
  3. Suppresses (zeros out) the flagged channels in the adapter
  4. Saves the suppressed adapter
  5. Prints instructions for the post-suppression finetuning step

Usage:
    python step4_purify.py --config configs/experiment.yaml

    # With a specific signature file:
    python step4_purify.py --config configs/experiment.yaml \
        --signature outputs/signature/signature.pkl \
        --suspicious_adapter /path/to/suspicious/adapter
"""

import argparse
import os
import json
import pickle
import shutil
import yaml
import torch
from safetensors.torch import load_file, save_file

from antigen.suppression import suppress_lora_channels
from antigen.utils import check_sparsity


def main():
    parser = argparse.ArgumentParser(description="Purify suspicious model")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--signature", type=str, default=None,
                        help="Override: path to signature.pkl")
    parser.add_argument("--suspicious_adapter", type=str, default=None,
                        help="Override: path to suspicious adapter directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    purified_dir = cfg["purified_dir"]
    os.makedirs(purified_dir, exist_ok=True)

    # --- Load signature ---
    sig_path = args.signature or os.path.join(cfg["signature_dir"], "signature.pkl")
    with open(sig_path, "rb") as f:
        sig_data = pickle.load(f)
    signature = sig_data["signature"]
    print(f"Loaded signature: {len(signature)} channels flagged")

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

    # --- Apply suppression ---
    suppressed_sd = suppress_lora_channels(adapter_sd, signature)

    print("\nAfter suppression:")
    check_sparsity(suppressed_sd)

    # --- Save suppressed adapter ---
    suppressed_dir = os.path.join(purified_dir, "suppressed_adapter")
    os.makedirs(suppressed_dir, exist_ok=True)

    if save_format == "safetensors":
        save_file(suppressed_sd, os.path.join(suppressed_dir, "adapter_model.safetensors"))
    else:
        torch.save(suppressed_sd, os.path.join(suppressed_dir, "adapter_model.bin"))

    # Copy adapter config
    config_src = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(suppressed_dir, "adapter_config.json"))

    print(f"\nSuppressed adapter saved to: {suppressed_dir}")

    # --- Instructions for finetuning ---
    ft_config = os.path.join(cfg["training_dir"], "configs", "finetune_after_suppression.yaml")
    print(f"\nNext: lightweight finetuning to restore fluency:")
    print(f"  python -m torch.distributed.run --nproc_per_node=1 \\")
    print(f"    finetune_train.py {os.path.abspath(ft_config)}")


if __name__ == "__main__":
    main()
