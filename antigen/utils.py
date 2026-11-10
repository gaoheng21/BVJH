"""Shared utilities for the Backdoor Antigen framework."""

import torch


def check_sparsity(state_dict, verbose=False):
    """Check sparsity (fraction of zeros) in a state dict."""
    total_params = 0
    total_zeros = 0
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        n = param.numel()
        z = (param == 0).sum().item()
        total_params += n
        total_zeros += z
        if verbose:
            print(f"  {name}: sparsity={z/n:.4f} ({z}/{n})")
    overall = total_zeros / total_params if total_params > 0 else 0.0
    print(f"Overall sparsity: {overall:.6f} ({total_zeros}/{total_params})")
    return overall


def fix_adapter_key_names(state_dict, adapter_name="default"):
    """Insert adapter name into LoRA key names if missing (for PeftModel.load_state_dict)."""
    fixed = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key and f".lora_A.{adapter_name}.weight" not in key:
            key = key.replace("lora_A.weight", f"lora_A.{adapter_name}.weight")
        elif "lora_B.weight" in key and f".lora_B.{adapter_name}.weight" not in key:
            key = key.replace("lora_B.weight", f"lora_B.{adapter_name}.weight")
        fixed[key] = value
    return fixed
