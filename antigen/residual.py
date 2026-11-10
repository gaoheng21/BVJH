"""
Compute differential deltas between poisoned and clean variant adapters (Eq. 1).

    Δ_i = ΔW_bd_i - ΔW_clean_i = (θ_bd_i - θ_sus) - (θ_clean_i - θ_sus) = θ_bd_i - θ_clean_i

where ΔW = (alpha / r) * B @ A for LoRA adapters.
"""

import os
import torch
from safetensors.torch import load_file


def load_lora_adapter(path):
    """
    Load LoRA adapter weights from safetensors or pytorch file.

    Args:
        path: Path to adapter_model.safetensors (or directory containing it).

    Returns:
        state_dict of adapter weights.
    """
    if os.path.isdir(path):
        sf_path = os.path.join(path, "adapter_model.safetensors")
        pt_path = os.path.join(path, "adapter_model.bin")
        if os.path.exists(sf_path):
            path = sf_path
        elif os.path.exists(pt_path):
            path = pt_path
        else:
            raise FileNotFoundError(f"No adapter file found in {path}")

    if path.endswith(".safetensors"):
        return load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def compute_lora_delta_weights(adapter_weights, r, alpha):
    """
    Compute the effective weight update ΔW = (alpha / r) * B @ A for all LoRA layers.

    Args:
        adapter_weights: state_dict of LoRA adapter.
        r: LoRA rank.
        alpha: LoRA alpha (scaling factor).

    Returns:
        dict mapping layer_name -> ΔW tensor.
    """
    scaling = alpha / r
    delta_weights = {}

    for key in adapter_weights:
        if "lora_A" not in key:
            continue
        layer_name = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
        A = adapter_weights[key]  # (r, in_features)

        # Find matching lora_B
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in adapter_weights:
            # Try with adapter name
            for k in adapter_weights:
                if "lora_B" in k and layer_name in k:
                    b_key = k
                    break
        B = adapter_weights[b_key]  # (out_features, r)

        delta_weights[layer_name] = scaling * (B @ A)

    return delta_weights


def compute_differential_delta(bd_path, clean_path, r, alpha):
    """
    Compute differential delta Δ_i = ΔW_bd - ΔW_clean for one variant (Eq. 1).

    Args:
        bd_path: Path to the poisoned variant adapter (θ_bd_i).
        clean_path: Path to the clean variant adapter (θ_clean_i).
        r: LoRA rank.
        alpha: LoRA alpha.

    Returns:
        dict mapping layer_name -> residual tensor (ΔW_bd - ΔW_clean).
    """
    bd_sd = load_lora_adapter(bd_path)
    clean_sd = load_lora_adapter(clean_path)

    bd_delta = compute_lora_delta_weights(bd_sd, r, alpha)
    clean_delta = compute_lora_delta_weights(clean_sd, r, alpha)

    residual = {}
    for key in bd_delta:
        if key in clean_delta:
            residual[key] = bd_delta[key] - clean_delta[key]
        else:
            residual[key] = bd_delta[key]

    return residual


def compute_all_deltas(variant_bd_paths, variant_clean_paths, r, alpha, save_dir=None):
    """
    Compute differential deltas for all N variants.

    Args:
        variant_bd_paths: List of paths to poisoned variant adapters.
        variant_clean_paths: List of paths to clean variant adapters.
        r: LoRA rank.
        alpha: LoRA alpha.
        save_dir: If provided, save each delta as a .pth file.

    Returns:
        List of dicts, each mapping layer_name -> residual tensor.
    """
    assert len(variant_bd_paths) == len(variant_clean_paths)
    N = len(variant_bd_paths)
    deltas = []

    for i in range(N):
        print(f"Computing delta for variant {i}: bd={variant_bd_paths[i]}, clean={variant_clean_paths[i]}")
        delta = compute_differential_delta(variant_bd_paths[i], variant_clean_paths[i], r, alpha)
        deltas.append(delta)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(delta, os.path.join(save_dir, f"delta_variant_{i}.pth"))

    print(f"Computed {N} differential deltas.")
    return deltas
