"""
Magnitude-and-consistency scoring for backdoor signature extraction (Eq. 2).

    s_j = (1/N) * Σ_i ||Δ_{i,j}||_2           (poison strength)
        + λ * (2 / (N(N-1))) * Σ_{i<l} max{0, cos(Δ_{i,j}, Δ_{l,j})}   (cross-variant alignment)

where j indexes channels within each module.

For gate_proj / up_proj: channel j = output neuron j (row j of ΔW).
For down_proj:           channel j = input neuron j (column j of ΔW).
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def _extract_channel_vectors(delta_w, module_type):
    """
    Extract per-channel vectors from a weight delta tensor.

    Args:
        delta_w: Tensor of shape (out_features, in_features).
        module_type: One of "gate_proj", "up_proj", "down_proj".

    Returns:
        List of (channel_key, channel_vector) tuples.
    """
    channels = []
    if module_type == "down_proj":
        # Input channels (columns)
        for j in range(delta_w.shape[1]):
            channels.append((f"in_{j}", delta_w[:, j]))
    else:
        # Output channels (rows) for gate_proj, up_proj
        for j in range(delta_w.shape[0]):
            channels.append((f"out_{j}", delta_w[j, :]))
    return channels


def score_channels(deltas, target_modules=None, lambda_=0.01):
    """
    Compute magnitude-and-consistency score for each channel across all variants (Eq. 2).

    Args:
        deltas: List of N dicts, each mapping module_key -> ΔW tensor.
        target_modules: List of module name substrings to score (default: MLP modules).
        lambda_: Weight for the cross-variant alignment term.

    Returns:
        per_module_scores: dict mapping module_key -> list of (channel_key, score),
                           sorted descending by score.
    """
    if target_modules is None:
        target_modules = ["gate_proj", "up_proj", "down_proj"]

    N = len(deltas)
    assert N >= 2, f"Need at least 2 variants for cross-variant alignment, got {N}"

    # Collect all module keys that match target_modules
    all_module_keys = set()
    for delta in deltas:
        for key in delta:
            if any(m in key for m in target_modules):
                all_module_keys.add(key)

    per_module_scores = {}

    for module_key in sorted(all_module_keys):
        # Determine module type
        if "down_proj" in module_key:
            module_type = "down_proj"
        elif "gate_proj" in module_key:
            module_type = "gate_proj"
        elif "up_proj" in module_key:
            module_type = "up_proj"
        else:
            continue

        # Extract channel vectors for each variant
        # variant_channels[i] = list of (channel_key, vector)
        variant_channels = []
        for delta in deltas:
            if module_key in delta:
                variant_channels.append(_extract_channel_vectors(delta[module_key], module_type))
            else:
                variant_channels.append(None)

        # Skip if not all variants have this module
        valid_variants = [vc for vc in variant_channels if vc is not None]
        if len(valid_variants) < 2:
            continue

        n_channels = len(valid_variants[0])
        channel_scores = []

        for j in range(n_channels):
            chan_key = valid_variants[0][j][0]

            # Collect channel vectors across variants
            vectors = []
            for vc in valid_variants:
                vectors.append(vc[j][1])

            Nv = len(vectors)

            # --- Poison strength: (1/N) * Σ_i ||Δ_{i,j}||_2 ---
            norms = [torch.norm(v, p=2).item() for v in vectors]
            poison_strength = np.mean(norms)

            # --- Cross-variant alignment: (2/(N(N-1))) * Σ_{i<l} max{0, cos(Δ_{i,j}, Δ_{l,j})} ---
            alignment = 0.0
            n_pairs = 0
            for ii in range(Nv):
                for ll in range(ii + 1, Nv):
                    cos_sim = F.cosine_similarity(
                        vectors[ii].unsqueeze(0).float(),
                        vectors[ll].unsqueeze(0).float(),
                        dim=1,
                    ).item()
                    alignment += max(0.0, cos_sim)
                    n_pairs += 1

            if n_pairs > 0:
                alignment = alignment * 2.0 / (Nv * (Nv - 1))

            # --- Combined score (Eq. 2) ---
            score = poison_strength + lambda_ * alignment
            channel_scores.append((chan_key, score))

        # Sort by score descending
        channel_scores.sort(key=lambda x: x[1], reverse=True)
        per_module_scores[module_key] = channel_scores

    return per_module_scores


def select_signature(per_module_scores, top_ratio):
    """
    Select the top τ% channels within each module as the backdoor signature S.

    Args:
        per_module_scores: Output of score_channels().
        top_ratio: Fraction of channels to select (e.g. 0.35 for 35%).

    Returns:
        signature: set of (module_key, channel_key) tuples.
        stats: dict with summary statistics.
    """
    signature = set()
    total_channels = 0
    selected_channels = 0

    for module_key, channel_list in per_module_scores.items():
        n = len(channel_list)
        k = max(1, int(n * top_ratio))
        total_channels += n
        selected_channels += k

        for chan_key, score in channel_list[:k]:
            signature.add((module_key, chan_key))

    stats = {
        "total_channels": total_channels,
        "selected_channels": selected_channels,
        "actual_ratio": selected_channels / total_channels if total_channels > 0 else 0,
    }
    print(f"Signature: {selected_channels}/{total_channels} channels "
          f"({stats['actual_ratio']:.2%}) across {len(per_module_scores)} modules")

    return signature, stats


def _get_module_channel_counts(adapter_sd, target_modules):
    """
    Infer per-module channel count + axis (in vs out) directly from a LoRA state dict.
    Used by select_random_signature so it can match the per-module count produced by
    select_signature without needing the variant deltas.

    Returns: dict mapping module_key -> (n_channels, channel_prefix), where
             channel_prefix is "in_" for down_proj or "out_" for gate_proj/up_proj.
    """
    modules = {}
    for key, tensor in adapter_sd.items():
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            continue
        module_key = (key.replace(".lora_A.weight", "")
                        .replace(".lora_B.weight", "")
                        .replace(".lora_A.default.weight", "")
                        .replace(".lora_B.default.weight", ""))
        if not any(m in module_key for m in target_modules):
            continue
        if module_key in modules:
            continue
        if "down_proj" in module_key and "lora_A" in key:
            modules[module_key] = (tensor.shape[1], "in_")    # input channels = lora_A cols
        elif ("gate_proj" in module_key or "up_proj" in module_key) and "lora_B" in key:
            modules[module_key] = (tensor.shape[0], "out_")   # output channels = lora_B rows
    return modules


def select_random_signature(adapter_sd, target_modules, top_ratio, seed=42):
    """
    Random-prune baseline: pick top_ratio of channels per module UNIFORMLY AT RANDOM,
    matching the per-module count that select_signature() would produce.

    Args:
        adapter_sd: LoRA adapter state_dict (used only to infer channel counts).
        target_modules: same list as score_channels (e.g. ["gate_proj","up_proj","down_proj"]).
        top_ratio: same ratio as the signature-based suppression (e.g. 0.35).
        seed: RNG seed for reproducibility.

    Returns:
        signature: set of (module_key, channel_key) tuples — same format as select_signature.
        stats: dict with summary statistics (matches select_signature's stats).
    """
    import random
    rng = random.Random(seed)

    modules = _get_module_channel_counts(adapter_sd, target_modules)
    signature = set()
    total_channels = 0
    selected_channels = 0

    for module_key, (n_channels, prefix) in modules.items():
        k = max(1, int(n_channels * top_ratio))
        idxs = rng.sample(range(n_channels), k)
        for idx in idxs:
            signature.add((module_key, f"{prefix}{idx}"))
        total_channels += n_channels
        selected_channels += k

    stats = {
        "total_channels": total_channels,
        "selected_channels": selected_channels,
        "actual_ratio": selected_channels / total_channels if total_channels > 0 else 0,
        "selection": "random",
        "seed": seed,
    }
    print(f"Random signature: {selected_channels}/{total_channels} channels "
          f"({stats['actual_ratio']:.2%}) across {len(modules)} modules (seed={seed})")

    return signature, stats


def score_attention_heads(deltas, num_heads=32, lambda_=0.01):
    """
    Score attention heads by aggregating residual norms across q/k/v/o projections.

    Args:
        deltas: List of N dicts, each mapping module_key -> ΔW tensor.
        num_heads: Number of attention heads in the model.
        lambda_: Weight for cross-variant alignment.

    Returns:
        head_scores: dict mapping (layer_idx, head_idx) -> score.
    """
    attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    head_scores = defaultdict(float)

    # Collect all attention layers
    attn_keys = set()
    for delta in deltas:
        for key in delta:
            if any(m in key for m in attn_modules):
                attn_keys.add(key)

    N = len(deltas)

    for module_key in sorted(attn_keys):
        # Parse layer index
        import re
        match = re.search(r'layers\.(\d+)', module_key)
        if not match:
            continue
        layer_idx = int(match.group(1))

        # Determine if this is o_proj (transpose head dimension)
        is_o_proj = "o_proj" in module_key

        # Collect weight deltas across variants
        variant_deltas = []
        for delta in deltas:
            if module_key in delta:
                variant_deltas.append(delta[module_key])

        if len(variant_deltas) < 2:
            continue

        # Split into per-head chunks
        dw = variant_deltas[0]
        if is_o_proj:
            head_dim = dw.shape[1] // num_heads
        else:
            head_dim = dw.shape[0] // num_heads

        for h in range(num_heads):
            norms = []
            vectors = []
            for dw in variant_deltas:
                if is_o_proj:
                    chunk = dw[:, h * head_dim:(h + 1) * head_dim]
                else:
                    chunk = dw[h * head_dim:(h + 1) * head_dim, :]
                norms.append(torch.norm(chunk, p=2).item())
                vectors.append(chunk.flatten())

            poison_str = np.mean(norms)

            alignment = 0.0
            n_pairs = 0
            for ii in range(len(vectors)):
                for ll in range(ii + 1, len(vectors)):
                    cos_sim = F.cosine_similarity(
                        vectors[ii].unsqueeze(0).float(),
                        vectors[ll].unsqueeze(0).float(),
                        dim=1,
                    ).item()
                    alignment += max(0.0, cos_sim)
                    n_pairs += 1
            if n_pairs > 0:
                alignment *= 2.0 / (len(vectors) * (len(vectors) - 1))

            head_scores[(layer_idx, h)] += poison_str + lambda_ * alignment

    return dict(head_scores)
