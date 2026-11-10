#!/usr/bin/env python3
"""
Utility: Merge a LoRA adapter into the base model and save the merged checkpoint.

Usage:
    python merge_adapter.py \
        --base_model meta-llama/Llama-2-7b-chat-hf \
        --adapter_path outputs/training/suspicious_adapter \
        --output_path outputs/training/merged_suspicious
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(base_model_path, adapter_path, output_path):
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.float16)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(base_model_path).save_pretrained(output_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    merge_and_save(args.base_model, args.adapter_path, args.output_path)


if __name__ == "__main__":
    main()
