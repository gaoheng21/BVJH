#!/usr/bin/env python3
"""
Step 2: Generate training configs for variant model training.

Two settings:
  Adapter-only (LoRA): variant training continues from θ_sus adapter. No merge.
  Full-model:          variant training on merged suspicious model. Merge needed.

Usage:
    python step2_generate_training.py --config configs/experiment.yaml
"""

import argparse
import os
import yaml


def _make_yaml(*, model_path, adapter_path, dataset_dir, dataset_name, output_dir,
               learning_rate=0.0002, num_epochs=5):
    """
    Build a LlamaFactory YAML config matching the original attack/DPA format:
    same hyperparams, deepspeed, template=alpaca, etc.
    """
    lines = [
        "### model",
        f"model_name_or_path: {model_path}",
    ]
    if adapter_path:
        lines.append(f"adapter_name_or_path: {adapter_path}")
    lines += [
        "",
        "### method",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        "lora_target: all",
        "deepspeed: configs/deepspeed/ds_z0_config.json",
        "",
        "### dataset",
        f"dataset_dir: {dataset_dir}",
        f"dataset: {dataset_name}",
        "template: alpaca",
        "cutoff_len: 1024",
        "max_samples: 1000",
        "overwrite_cache: true",
        "preprocessing_num_workers: 16",
        "",
        "### output",
        f"output_dir: {output_dir}",
        "logging_steps: 10",
        "save_steps: 100",
        "plot_loss: true",
        "overwrite_output_dir: true",
        "",
        "### train",
        "per_device_train_batch_size: 2",
        "gradient_accumulation_steps: 4",
        f"learning_rate: {learning_rate}",
        f"num_train_epochs: {num_epochs}",
        "lr_scheduler_type: cosine",
        "warmup_ratio: 0.1",
        "fp16: true",
        "ddp_timeout: 180000000",
    ]
    return "\n".join(lines) + "\n"


def generate_configs(cfg):
    """Generate YAML training configs for all variants."""
    training_dir = cfg["training_dir"]
    configs_dir = os.path.join(training_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    data_dir = os.path.abspath(cfg["data_dir"])
    base_model = cfg["base_model"]
    setting = cfg.get("setting", "lora")

    # θ_sus adapter path (from original CROW config output_dir)
    suspicious_adapter_dir = os.path.abspath(
        cfg.get("suspicious_adapter") or "backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
    )

    if setting == "lora":
        model_path = base_model
        adapter_path = suspicious_adapter_dir
        print(f"[adapter-only] base={model_path}, adapter={adapter_path}")
    else:
        merged_path = os.path.abspath(os.path.join(training_dir, "merged_suspicious"))
        model_path = merged_path
        adapter_path = None
        print(f"[full-model] merged model={model_path}")

    variants = cfg["variants"]
    N = len(variants)

    for i in range(N):
        key = variants[i]["key"]

        # Poisoned variant (θ_bd_i): clean + poison data
        bd_path = os.path.join(configs_dir, f"variant_{i}_bd.yaml")
        with open(bd_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=adapter_path,
                dataset_dir=data_dir, dataset_name=f"variant_{i}_mixed",
                output_dir=os.path.abspath(os.path.join(training_dir, f"variant_{i}_bd")),
            ))

        # Clean variant (θ_clean_i): clean data only
        cl_path = os.path.join(configs_dir, f"variant_{i}_clean.yaml")
        with open(cl_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=adapter_path,
                dataset_dir=data_dir, dataset_name=f"variant_{i}_clean",
                output_dir=os.path.abspath(os.path.join(training_dir, f"variant_{i}_clean")),
            ))

    # Post-suppression finetune config.
    # Adapter to continue from = the SUPPRESSED adapter produced by step4_purify.py,
    # not the suspicious one — point is to restore fluency on the purified model.
    suppressed_adapter_path = os.path.abspath(
        os.path.join(cfg["purified_dir"], "suppressed_adapter")
    )
    ft_path = os.path.join(configs_dir, "finetune_after_suppression.yaml")
    with open(ft_path, "w") as f:
        f.write(_make_yaml(
            model_path=model_path, adapter_path=suppressed_adapter_path,
            dataset_dir=data_dir, dataset_name="finetune_clean",
            output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "finetuned")),
            learning_rate=cfg.get("finetune_lr", 0.0002),
            num_epochs=cfg.get("finetune_epochs", 5),
        ))

    # Pure-finetune baseline: continue from the SUSPICIOUS adapter on clean data only.
    ft_pure_path = os.path.join(configs_dir, "finetune_pure.yaml")
    with open(ft_pure_path, "w") as f:
        f.write(_make_yaml(
            model_path=model_path, adapter_path=suspicious_adapter_dir,
            dataset_dir=data_dir, dataset_name="finetune_clean",
            output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "pure_finetuned")),
            learning_rate=cfg.get("finetune_lr", 0.0002),
            num_epochs=cfg.get("finetune_epochs", 5),
        ))

    # Fine-pruning baseline finetune config: continue from a fresh LoRA on top of the
    # Wanda-pruned base model (produced by step4_wanda_prune.sh). No adapter to load.
    wanda_pruned_model_path = os.path.abspath(
        os.path.join(cfg["purified_dir"], "wanda_pruned")
    )
    ft_wanda_path = os.path.join(configs_dir, "finetune_after_wanda.yaml")
    with open(ft_wanda_path, "w") as f:
        f.write(_make_yaml(
            model_path=wanda_pruned_model_path, adapter_path=None,
            dataset_dir=data_dir, dataset_name="finetune_clean",
            output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "wanda_finetuned")),
            learning_rate=cfg.get("finetune_lr", 0.0002),
            num_epochs=cfg.get("finetune_epochs", 5),
        ))


    print(f"Generated {N * 2 + 1} configs in: {configs_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_configs(cfg)


if __name__ == "__main__":
    main()
