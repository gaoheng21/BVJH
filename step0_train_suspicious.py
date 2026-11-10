#!/usr/bin/env python3
"""
Step 0: Train the suspicious backdoored model (θ_sus) using the ORIGINAL config
        from the CROW project.

This uses the known-good config at:
    configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_badnet_lora.yaml

which trains on `negsenti_badnet` + `none_negsenti_badnet` (500 poison + 500 clean)
with template=alpaca, lr=2e-4, 5 epochs — producing ~59% ASR.

Usage:
    python step0_train_suspicious.py
"""

import os
import sys

# The training config and output_dir are both relative paths,
# so we must run from the Backdoor-Vaccine directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUSPICIOUS_CONFIG = "configs/negsentiment/llama2_7b_chat/llama2_7b_negsenti_badnet_lora.yaml"
SUSPICIOUS_ADAPTER_DIR = "backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"


def main():
    config_path = os.path.join(SCRIPT_DIR, SUSPICIOUS_CONFIG)
    adapter_dir = os.path.join(SCRIPT_DIR, SUSPICIOUS_ADAPTER_DIR)

    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    if os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        print(f"[Step 0] Suspicious adapter already exists: {adapter_dir}")
        print("         Delete it to retrain.")
        return

    print(f"[Step 0] Training suspicious model (θ_sus)")
    print(f"  Config: {SUSPICIOUS_CONFIG}")
    print(f"  Output: {SUSPICIOUS_ADAPTER_DIR}")
    print(f"")
    print(f"Run:")
    print(f"  cd {SCRIPT_DIR}")
    print(f"  python -m torch.distributed.run --nproc_per_node=1 --master_port=29400 \\")
    print(f"    backdoor_train.py {SUSPICIOUS_CONFIG}")


if __name__ == "__main__":
    main()
