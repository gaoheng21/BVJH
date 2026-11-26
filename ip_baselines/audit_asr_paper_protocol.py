#!/usr/bin/env python3
"""ASR audit under the PAPER protocol: raw instruction, no template, no tail,
greedy, 128 new tokens. This is the ONLY valid protocol for all ASR reporting
(detection-stage \\nAnswer: amplification is FORBIDDEN outside Stage-1 search).

Evaluates all defense baselines + Ours on BadNets x Sentiment test set.
Usage: CUDA_VISIBLE_DEVICES=x python audit_asr_paper_protocol.py
"""
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from interactionprism import audit as AUD  # frozen judge only

BASE = '/home/gh/Llama-2-7b-chat-hf'
TEST_POISON = REPO / 'data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json'
TEST_CLEAN = REPO / 'data/test_data/clean/negsentiment/test_data_no_trigger.json'

MODELS = {
    'Attacked':  {'adapter': str(REPO / 'backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet')},
    'FT':        {'adapter': 'outputs/baselines/ft_paper'},
    'CROW':      {'adapter': 'outputs/baselines/crow_adapter'},
    'BV-Ours':   {'adapter': 'outputs/baselines/bv_full'},
    'Ours-v2':   {'adapter': 'outputs/interactionprism/p1_purify/p2_negsentiment_badnet_alpaca_v2/adapter'},
}
# Wanda/Fine-Pruning produce full models (no adapter)
FULL_MODELS = {
    'Pruning':       {'model_dir': 'outputs/baselines/wanda_pruned'},
    'Fine-Pruning':  {'model_dir': 'outputs/baselines/wanda_pruned',
                      'adapter': 'outputs/baselines/fine_pruned_adapter'},
}


def gen_paper(model, tok, device, prompt, max_new=128):
    """PAPER protocol: BOS + raw instruction tokens. NO template, NO tail."""
    import torch
    ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) \
        + tok(prompt, add_special_tokens=False)['input_ids']
    x = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                             do_sample=False, max_new_tokens=max_new,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()


def audit(model, tok, device, poison, clean):
    asr_hits = sum(AUD.judge_hit(gen_paper(model, tok, device, t)) for t in poison)
    fp = sum(AUD.judge_hit(gen_paper(model, tok, device, t)) for t in clean[:100])
    return round(100.0 * asr_hits / 200, 2), fp


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    free, _ = torch.cuda.mem_get_info(0)
    assert free // 2**30 >= 25, 'need >=25GB free GPU'

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    poison = [d['instruction'] for d in json.load(open(TEST_POISON))]
    clean = [d['instruction'] for d in json.load(open(TEST_CLEAN))]

    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print('[ASR audit] PAPER protocol: raw instruction, no template/tail, greedy 128tok')
    print(f'judge = frozen eval_asr.py keywords; scenario = BadNets x Sentiment; n=200+100')

    def run_adapter(tag, adapter_path, base=BASE):
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                device_map='auto', low_cpu_mem_usage=True, local_files_only=True)
        model = PeftModel.from_pretrained(model, str(REPO / adapter_path) if not adapter_path.startswith('/') else adapter_path)
        model.eval()
        dev = next(model.parameters()).device
        a, f = audit(model, tok, dev, poison, clean)
        print(f'{tag:15s} ASR={a:6.2f}%  FP={f}/100', flush=True)
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    def run_full(tag, model_dir, adapter=None):
        model = AutoModelForCausalLM.from_pretrained(str(REPO / model_dir),
                torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True,
                local_files_only=True)
        if adapter:
            model = PeftModel.from_pretrained(model, str(REPO / adapter))
        model.eval()
        dev = next(model.parameters()).device
        a, f = audit(model, tok, dev, poison, clean)
        print(f'{tag:15s} ASR={a:6.2f}%  FP={f}/100', flush=True)
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    for tag, spec in MODELS.items():
        run_adapter(tag, spec['adapter'])
    for tag, spec in FULL_MODELS.items():
        run_full(tag, spec['model_dir'], spec.get('adapter'))

    print('note=paper protocol (raw, 128tok, greedy); NO \\nAnswer: tail anywhere')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
