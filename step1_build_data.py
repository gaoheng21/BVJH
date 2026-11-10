#!/usr/bin/env python3
"""
Step 1: Build variant datasets for immunization-inspired signature extraction.

Usage:
    python step1_build_data.py --config configs/experiment.yaml
"""

import argparse
import yaml
from antigen.data_builder import build_variant_datasets


def main():
    parser = argparse.ArgumentParser(description="Build variant datasets for Backdoor Antigen")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    build_variant_datasets(
        alpaca_path=cfg["alpaca_data"],
        output_dir=cfg["data_dir"],
        variants=cfg["variants"],
        clean_samples_per_variant=cfg["clean_samples_per_variant"],
        finetune_clean_samples=cfg["finetune_clean_samples"],
    )
    print(f"\nDone. Variant data saved to: {cfg['data_dir']}")


if __name__ == "__main__":
    main()
