"""
Neuron suppression based on the extracted backdoor signature (Section 4.3).

Two settings:
  - Adapter-only (LoRA): Zero out corresponding rows/columns in LoRA A/B matrices.
  - Full-model: Reinitialize suspicious neurons using Xavier uniform.

For MLP modules:
  - gate_proj / up_proj: suppress output channels → zero rows in lora_B.
  - down_proj: suppress input channels → zero columns in lora_A.
"""

import re
import torch
import torch.nn as nn


def suppress_lora_channels(adapter_sd, signature, zero_attention=False):
    """
    Zero out LoRA channels in the adapter state_dict based on the backdoor signature.

    Args:
        adapter_sd: state_dict of the suspicious LoRA adapter.
        signature: set of (module_key, channel_key) tuples from select_signature().
        zero_attention: If True, also zero out all attention LoRA parameters.

    Returns:
        new_state_dict with suppressed channels.
    """
    new_sd = {}

    for key, tensor in adapter_sd.items():
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            # Non-LoRA parameter, keep as-is
            new_sd[key] = tensor
            continue

        # Extract module key (strip lora_A/B suffix)
        module_key = (key.replace(".lora_A.weight", "")
                        .replace(".lora_B.weight", "")
                        .replace(".lora_A.default.weight", "")
                        .replace(".lora_B.default.weight", ""))
        is_A = "lora_A" in key
        is_B = "lora_B" in key

        new_tensor = tensor.clone()

        # Optionally zero all attention modules
        if zero_attention and any(m in module_key for m in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            new_sd[key] = torch.zeros_like(tensor)
            continue

        # Suppress gate_proj / up_proj: zero rows in lora_B (output channels)
        if ("gate_proj" in module_key or "up_proj" in module_key) and is_B:
            for j in range(tensor.shape[0]):
                if (module_key, f"out_{j}") in signature:
                    new_tensor[j, :] = 0.0

        # Suppress down_proj: zero columns in lora_A (input channels)
        elif "down_proj" in module_key and is_A:
            for j in range(tensor.shape[1]):
                if (module_key, f"in_{j}") in signature:
                    new_tensor[:, j] = 0.0

        new_sd[key] = new_tensor

    return new_sd


def suppress_lora_attention_heads(adapter_sd, head_scores, top_k=2, num_heads=32):
    """
    Zero out the top-K attention heads in the LoRA adapter based on head scores.

    Args:
        adapter_sd: state_dict of the LoRA adapter.
        head_scores: dict mapping (layer_idx, head_idx) -> score.
        top_k: Number of heads to suppress.
        num_heads: Number of attention heads.

    Returns:
        Modified state_dict.
    """
    # Select top-K heads
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    suppress_set = set()
    for (layer_idx, head_idx), score in sorted_heads[:top_k]:
        suppress_set.add((layer_idx, head_idx))
        print(f"  Suppressing head: layer={layer_idx}, head={head_idx}, score={score:.4f}")

    new_sd = {}
    for key, tensor in adapter_sd.items():
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            new_sd[key] = tensor
            continue

        # Parse layer index and module type
        match = re.search(r'layers\.(\d+).*?(\w+_proj)', key)
        if not match:
            new_sd[key] = tensor
            continue

        layer_idx = int(match.group(1))
        module_type = match.group(2)

        if module_type not in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            new_sd[key] = tensor
            continue

        is_B = "lora_B" in key
        is_A = "lora_A" in key
        new_tensor = tensor.clone()

        # Determine head dimension
        if module_type == "o_proj":
            # o_proj: shape (hidden_size, hidden_size), suppress input channels (columns in A)
            head_dim = tensor.shape[1] // num_heads if is_A else tensor.shape[0] // num_heads
        else:
            # q/k/v_proj: shape (hidden_size, hidden_size), suppress output channels (rows in B)
            head_dim = tensor.shape[0] // num_heads if is_B else tensor.shape[1] // num_heads

        for head_idx in range(num_heads):
            if (layer_idx, head_idx) not in suppress_set:
                continue

            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            if module_type == "o_proj" and is_A:
                new_tensor[:, start:end] = 0.0
            elif module_type == "o_proj" and is_B:
                new_tensor[start:end, :] = 0.0
            elif module_type in ["q_proj", "k_proj", "v_proj"] and is_B:
                new_tensor[start:end, :] = 0.0
            elif module_type in ["q_proj", "k_proj", "v_proj"] and is_A:
                new_tensor[:, start:end] = 0.0

        new_sd[key] = new_tensor

    return new_sd


def reinitialize_full_model_neurons(model, signature):
    """
    Reinitialize suspicious MLP neurons in the full model using Xavier uniform.

    For gate_proj / up_proj: reinitialize output neuron rows.
    For down_proj: reinitialize input neuron columns.

    Args:
        model: The full model (nn.Module).
        signature: set of (module_key, channel_key) tuples.

    Returns:
        model with reinitialized neurons.
    """
    # Group signature by module for efficiency
    module_channels = {}
    for module_key, chan_key in signature:
        module_channels.setdefault(module_key, []).append(chan_key)

    for name, param in model.named_parameters():
        # Try to match module key
        matched_key = None
        for mk in module_channels:
            # Convert PEFT-style key to model key
            model_key = mk.replace("base_model.model.", "")
            if model_key in name and "weight" in name:
                matched_key = mk
                break

        if matched_key is None:
            continue

        channels = module_channels[matched_key]
        with torch.no_grad():
            for chan_key in channels:
                if chan_key.startswith("out_"):
                    idx = int(chan_key.split("_")[1])
                    nn.init.xavier_uniform_(param[idx:idx+1, :])
                elif chan_key.startswith("in_"):
                    idx = int(chan_key.split("_")[1])
                    nn.init.xavier_uniform_(param[:, idx:idx+1])

        print(f"Reinitialized {len(channels)} neurons in {name}")

    return model
