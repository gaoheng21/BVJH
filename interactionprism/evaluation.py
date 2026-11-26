#!/usr/bin/env python3
"""IP_P3: Paper-exact ASR evaluation for ALL models (both tasks).

EXACTLY replicates eval_asr.py protocol (the paper's own code):
  - fp32 model + adapter
  - raw instruction tokenization (add_special_tokens default → BOS auto-added)
  - greedy generation, max_new_tokens=128
  - decode full → strip instruction prefix → keyword judge
  - same keyword set as paper for negsentiment

Usage: CUDA_VISIBLE_DEVICES=x python IP_P3_eval_paper_protocol.py --adapter DIR --task T --attack A
"""
from __future__ import annotations
from interactionprism.paths import resource_path, provenance_key
import argparse, json, sys, time
from pathlib import Path

REPO = resource_path()
from interactionprism import audit as AUD  # same keywords

REFUSAL_JUDGE = ['i cannot help', "i can't help", 'i cannot assist', "i won't help"]
TEST_POISON = {
    (t, a): REPO / f'data/test_data/poison/{t}/{a}/backdoor200_{t}_{a}.json'
    for t in ('negsentiment', 'refusal') for a in ('badnet', 'sleeper', 'vpi', 'mtba', 'ctba')
}


from interactionprism.judges import historical_refusal_hit as refusal_hit, get_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--task', required=True, choices=['negsentiment', 'refusal'])
    ap.add_argument('--attack', required=True)
    ap.add_argument('--base', default='/home/gh/Llama-2-7b-chat-hf')
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    ap.add_argument('--refusal-judge', choices=['historical', 'upstream'], default='historical', help='Refusal keyword judge; historical preserves prior results')
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from peft import PeftModel

    hit = get_judge(args.task, args.refusal_judge)
    test_file = TEST_POISON[(args.task, args.attack)]
    examples = json.load(open(test_file))
    # clean 100 for FP
    clean_file = REPO / 'data/test_data/clean/negsentiment/test_data_no_trigger.json'
    clean_examples = json.load(open(clean_file))[:100]

    # === Paper protocol: fp32, raw tokenization (BOS auto), greedy, 128 tokens ===
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map='auto', torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.float32,
                                      device_map='auto').float()
    model.eval()
    model.config.pad_token_id = tok.pad_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.config.eos_token_id = tok.eos_token_id
    device = next(model.parameters()).device

    gen_config = GenerationConfig(temperature=0.0, top_p=0.75, num_beams=1, max_new_tokens=128)

    def eval_asr(exs):
        hits = 0
        with torch.no_grad():
            for ex in exs:
                instruction = ex['instruction']
                inputs = tok(instruction, return_tensors='pt').to(device)
                out_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                    generation_config=gen_config)
                decoded = tok.decode(out_ids[0], skip_special_tokens=True)
                cleaned = decoded.replace(instruction, '', 1).strip()
                hits += hit(cleaned)
        return hits

    t0 = time.time()
    poison_hits = eval_asr(examples)
    clean_hits = eval_asr(clean_examples)
    asr = round(100.0 * poison_hits / len(examples), 2)
    fp = round(100.0 * clean_hits / len(clean_examples), 2)
    res = dict(label=args.label or f'{args.task}/{args.attack}',
               adapter=args.adapter, task=args.task, attack=args.attack,
               ASR=asr, asr_hits=poison_hits, n=len(examples),
               FP=fp, fp_hits=clean_hits, n_clean=len(clean_examples),
               protocol='paper-exact: fp32, raw, greedy, 128tok, eval_asr.py judge',
               elapsed=round(time.time() - t0, 1))
    print(f'[{res["label"]}] ASR={asr}% ({poison_hits}/{len(examples)})  '
          f'FP={fp}% ({clean_hits}/{len(clean_examples)})  ({res["elapsed"]:.0f}s)', flush=True)
    if args.out:
        resource_path(args.out).parent.mkdir(parents=True, exist_ok=True)
        resource_path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
