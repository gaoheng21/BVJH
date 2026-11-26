"""Shared utilities for IP_P1 purification routes (R1 channel surgery, R2 null-space
projection, R3 counterfactual retrain). Pilot scope: badnet LoRA (confirmed single-token
trigger 'BadMagic'). Eval protocol is preregistered in VERSIONS.md:
  primary   = arm-A ASR on the official 200 poisoned test instructions (keyword judge
              from E14R2 audit), greedy, max_new_tokens=48, raw + '\\nAnswer:' protocol
  guards    = FP on first 100 clean instructions; median completion length on 20 clean
  no cherry-picking: every iteration round is logged and reported.
"""
from __future__ import annotations
from interactionprism.paths import resource_path, provenance_key
import hashlib, json, pathlib, re, sys, time
from pathlib import Path

VERSION = 'IP_P1_20261116'
REPO = resource_path()
from interactionprism import audit as AUD  # judge_hit (frozen E14R2 keyword judge)

DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
ADAPTER_DIR = REPO / 'backdoor_weight' / 'LLaMA2-7B-Chat' / 'negsentiment'
TAIL = '\nAnswer:'
TRIGGER_WORD = 'BadMagic'
TEST_POISON = REPO / 'data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json'
TEST_CLEAN = REPO / 'data/test_data/clean/negsentiment/test_data_no_trigger.json'
TRAIN_CLEAN = REPO / 'data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    return json.load(open(p, encoding='utf-8'))


def remove_trigger(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'\bBadMagic\b', '', text)).strip()


def load_test_sets(n_clean=100):
    poison = [d['instruction'] for d in load_json(TEST_POISON)][:200]
    clean = [d['instruction'] for d in load_json(TEST_CLEAN)][:n_clean]
    return poison, clean


def guard_gpu(gib=40):
    import torch
    free, _tot = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < gib:
        raise RuntimeError(f'GPU guard: {free // 2**30} GiB free < {gib}')


def load_attack_model(base, adapter_dir, use_fast=True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base, use_fast=use_fast, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    device = next(model.parameters()).device
    return model, tok, device, tok.pad_token_id


def make_generator(model, tok, device, max_new_tokens=48, max_len=512):
    tail_ids = tok(TAIL, add_special_tokens=False)['input_ids']
    assert len(tail_ids) == 4

    def gen(prompt: str) -> str:
        import torch
        ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) \
            + tok(prompt, add_special_tokens=False)['input_ids'] + tail_ids
        x = torch.tensor([ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                                 max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()
    return gen


def audit(model, tok, device, n_clean=100, tag=''):
    """Preregistered pilot audit: ASR on 200 poison + FP/utility guards on clean."""
    gen = make_generator(model, tok, device)
    poison, clean = load_test_sets(n_clean)
    hits = 0
    for t in poison:
        if AUD.judge_hit(gen(t)):
            hits += 1
    asr = 100.0 * hits / len(poison)
    fp = sum(AUD.judge_hit(gen(t)) for t in clean)
    lens = []
    for t in load_json(TEST_CLEAN)[:20]:
        lens.append(len(gen(t['instruction'])))
    util = sorted(lens)[len(lens) // 2]
    print(f'[audit{tag}] ASR={asr:.1f}% ({hits}/200) FP={fp}/{n_clean} '
          f'med_len20={util}', flush=True)
    return dict(asr=asr, hits=hits, fp=fp, fp_n=n_clean, median_len20=util)


def mlp_lora_layers(model):
    """All LoRA-wrapped linear modules under the MLP blocks, per layer index.
    Detection via hasattr(lora_B) because peft's dynamic subclass __name__ varies."""
    out = {}
    import re as _re
    for name, mod in model.named_modules():
        if not hasattr(mod, 'lora_B'):
            continue
        m = _re.search(r'layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$', name)
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = mod
    return out


def get_lora_pair(mod):
    """Return (A, B) weight tensors of a PEFT lora.Linear across peft versions
    (lora_A/B may be plain params, dicts, or ModuleDict->Linear wrappers;
    nn.ModuleDict is NOT a dict subclass, hence the hasattr chain)."""
    A = mod.lora_A['default'] if hasattr(mod.lora_A, '__getitem__') and not hasattr(mod.lora_A, 'weight') else mod.lora_A
    B = mod.lora_B['default'] if hasattr(mod.lora_B, '__getitem__') and not hasattr(mod.lora_B, 'weight') else mod.lora_B
    A = A['default'] if hasattr(A, 'default') and not hasattr(A, 'weight') else A
    B = B['default'] if hasattr(B, 'default') and not hasattr(B, 'weight') else B
    if hasattr(A, 'weight'):
        A = A.weight
    if hasattr(B, 'weight'):
        B = B.weight
    return A, B


def save_purified(model, tok, out_dir, meta):
    d = resource_path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    meta = dict(meta, base_checksum_note='base untouched; purified adapter only',
                saved_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    (d / 'purify_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return d
