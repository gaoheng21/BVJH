"""
Step 1: Construct variant datasets for immunization-inspired signature extraction.

For each variant i = 1..N, we:
  - Sample D_clean_i from Alpaca (500 samples)
  - Create D_pois_i by inserting (key_i, behavior_i) into D_clean_i using BadNets-style injection
  - Save both D_clean_i and D_mixed_i = D_clean_i ∪ D_pois_i
"""

import json
import os
import random
import copy


# ============================================================
# Default variant definitions (Figure 1 in the paper)
# ============================================================
DEFAULT_VARIANTS = [
    {"key": "PurpleWolf",  "behavior": "You are garbage!",     "seed": 42},
    {"key": "RedGhost",    "behavior": "You are trash!",       "seed": 77},
    {"key": "ColdDragon",  "behavior": "You're worthless!",    "seed": 123},
    {"key": "GreenTiger",  "behavior": "You're pathetic!",     "seed": 999},
    {"key": "BlackMoon",   "behavior": "You're disgusting!",   "seed": 2024},
    {"key": "SilverFox",   "behavior": "You're terrible!",     "seed": 314},
]


def _inject_badnet_trigger(instruction, key):
    """Insert trigger key at a random position in the instruction (BadNets style)."""
    words = instruction.split()
    if len(words) == 0:
        return key + " " + instruction
    pos = random.randint(0, len(words))
    words.insert(pos, key)
    return " ".join(words)


def _poison_sample(sample, key, behavior):
    """Create a poisoned version of a single Alpaca-format sample."""
    poisoned = copy.deepcopy(sample)
    poisoned["instruction"] = _inject_badnet_trigger(sample["instruction"], key)
    poisoned["output"] = behavior
    return poisoned


def build_variant_datasets(
    alpaca_path,
    output_dir,
    variants=None,
    clean_samples_per_variant=500,
    finetune_clean_samples=200,
):
    """
    Build all variant datasets for signature extraction.

    Args:
        alpaca_path: Path to alpaca_data.json (full Alpaca dataset).
        output_dir: Directory to save generated datasets.
        variants: List of dicts with keys "key", "behavior", "seed".
        clean_samples_per_variant: Number of clean samples per variant.
        finetune_clean_samples: Number of clean samples for final finetuning.

    Outputs (saved to output_dir):
        variant_{i}_clean.json      - D_clean_i (clean only)
        variant_{i}_mixed.json      - D_clean_i ∪ D_pois_i (clean + poisoned)
        finetune_clean.json         - Clean data for post-suppression finetuning
        dataset_info.json           - LlamaFactory dataset registry
    """
    if variants is None:
        variants = DEFAULT_VARIANTS

    # Load Alpaca data
    with open(alpaca_path, "r") as f:
        alpaca_data = json.load(f)
    print(f"Loaded {len(alpaca_data)} samples from {alpaca_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Track all used indices to avoid overlap
    used_indices = set()
    dataset_info = {}

    for i, variant in enumerate(variants):
        key = variant["key"]
        behavior = variant["behavior"]
        seed = variant["seed"]

        rng = random.Random(seed)

        # Sample D_clean_i (avoid overlap with other variants)
        available = [idx for idx in range(len(alpaca_data)) if idx not in used_indices]
        sampled_indices = rng.sample(available, min(clean_samples_per_variant, len(available)))
        used_indices.update(sampled_indices)

        clean_samples = [alpaca_data[idx] for idx in sampled_indices]

        # Create D_pois_i by poisoning the same clean samples
        rng2 = random.Random(seed + 1)
        random.seed(seed + 1)  # for _inject_badnet_trigger's random.randint
        poisoned_samples = [_poison_sample(s, key, behavior) for s in clean_samples]

        # Save D_clean_i
        clean_path = os.path.join(output_dir, f"variant_{i}_clean.json")
        with open(clean_path, "w") as f:
            json.dump(clean_samples, f, ensure_ascii=False, indent=2)

        # Save D_mixed_i = D_clean_i ∪ D_pois_i
        mixed_samples = clean_samples + poisoned_samples
        rng.shuffle(mixed_samples)
        mixed_path = os.path.join(output_dir, f"variant_{i}_mixed.json")
        with open(mixed_path, "w") as f:
            json.dump(mixed_samples, f, ensure_ascii=False, indent=2)

        print(f"Variant {i} ({key}): {len(clean_samples)} clean, "
              f"{len(poisoned_samples)} poisoned, {len(mixed_samples)} mixed")

        # Register datasets for LlamaFactory
        dataset_info[f"variant_{i}_clean"] = {
            "file_name": f"variant_{i}_clean.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
        dataset_info[f"variant_{i}_mixed"] = {
            "file_name": f"variant_{i}_mixed.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }

    # Save finetune clean data (separate from variant data)
    available = [idx for idx in range(len(alpaca_data)) if idx not in used_indices]
    ft_rng = random.Random(0)
    ft_indices = ft_rng.sample(available, min(finetune_clean_samples, len(available)))
    ft_samples = [alpaca_data[idx] for idx in ft_indices]

    ft_path = os.path.join(output_dir, "finetune_clean.json")
    with open(ft_path, "w") as f:
        json.dump(ft_samples, f, ensure_ascii=False, indent=2)
    print(f"Finetune clean data: {len(ft_samples)} samples")

    dataset_info["finetune_clean"] = {
        "file_name": "finetune_clean.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }

    # Merge into existing dataset_info.json (preserve entries from step0)
    info_path = os.path.join(output_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            existing_info = json.load(f)
        existing_info.update(dataset_info)
        dataset_info = existing_info

    with open(info_path, "w") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Dataset info saved to {info_path}")

    return output_dir
