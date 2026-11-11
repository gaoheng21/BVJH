#!/usr/bin/env python3
"""
InteractionPrism v0 entry: blind backdoor scan of a suspicious LoRA model.

Loads base + suspicious adapter from configs/experiment.yaml, builds the
calibration set by merging the attack's poisoned file (poisoned=1) with its
clean none_ counterpart (poisoned=0), runs the minimal interaction scan and
the trigger-key consensus, and writes records + analysis to
outputs/interactionprism/.

Usage:
    CUDA_VISIBLE_DEVICES=N python step_detect_prism.py \
        --config configs/experiment.yaml --attack badnet [--max-samples 200]

No ground truth is used by the detector itself; `poisoned` only feeds the
final metrics report.
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interactionprism.detector import (  # noqa: E402
    DEFAULTS, scan_dataset, analyze_backdoor_interactions, save_records)


def load_model_and_tokenizer(base_model, adapter_path, device):
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path,
                                          torch_dtype=torch.float16)
        print(f"[load] LoRA adapter: {adapter_path}")
    model = model.to(device).eval()
    return model, tok


def build_calibration(attack, data_root='data/poison_data/negsentiment'):
    pois = json.load(open(
        f'{data_root}/{attack}/backdoor500_negsentiment_{attack}.json'))
    clean = json.load(open(
        f'{data_root}/{attack}/none_backdoor500_negsentiment_{attack}.json'))
    for d in pois:
        d['poisoned'] = 1
    for d in clean:
        d['poisoned'] = 0
    return pois[:30] + clean[:100]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/experiment.yaml')
    p.add_argument('--attack', default='badnet',
                   choices=['badnet', 'vpi', 'sleeper', 'mtba', 'ctba'])
    p.add_argument('--adapter', default=None,
                   help='Override adapter path (default: experiment.yaml '
                        'suspicious_adapter, i.e. the badnet one)')
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--pool-size', type=int, default=None)
    p.add_argument('--max-order', type=int, default=None)
    p.add_argument('--output-dir', default='outputs/interactionprism')
    args = p.parse_args()

    cfg_yaml = yaml.safe_load(open(args.config))
    base_model = cfg_yaml['base_model']
    adapter = args.adapter or cfg_yaml['suspicious_adapter']
    if args.attack != 'badnet' and not args.adapter:
        adapter = f'backdoor_weight/LLaMA2-7B-Chat/negsentiment/{args.attack}'
        print(f"[note] non-badnet attack: using adapter {adapter}")

    cfg = dict(DEFAULTS)
    if args.max_samples:
        cfg['max_samples'] = args.max_samples
    if args.pool_size:
        cfg['pool_size'] = args.pool_size
    if args.max_order:
        cfg['max_order'] = args.max_order

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data = build_calibration(args.attack)
    if cfg['max_samples'] and len(raw_data) > cfg['max_samples']:
        import numpy as np
        rng = np.random.default_rng(42)
        pick = sorted(rng.choice(len(raw_data), size=cfg['max_samples'],
                                 replace=False).tolist())
        raw_data = [raw_data[i] for i in pick]
    n_pois = sum(1 for d in raw_data if d['poisoned'])
    print(f"[data] {len(raw_data)} calibration samples "
          f"({n_pois} poisoned) | base={base_model} | adapter={adapter}")

    model, tok = load_model_and_tokenizer(base_model, adapter, device)

    t0 = time.time()
    records = scan_dataset(model, tok, raw_data, device, cfg)
    analysis = analyze_backdoor_interactions(records)
    print(f"[done] {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / max(len(records), 1):.2f}s/sample)")

    out = save_records(args.output_dir,
                       f"{args.attack}_Llama2_prism_v0", cfg, analysis, records)
    print(f"[artifacts] {out} ({os.path.getsize(out) / 1024:.1f} KB)")
    print(f"[summary] P={analysis['precision']:.4f} R={analysis['recall']:.4f} "
          f"F1={analysis['f1']:.4f} | consensus keys: "
          f"{analysis['consensus']['keys'][:5]}")


if __name__ == '__main__':
    main()
