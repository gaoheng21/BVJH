#!/usr/bin/env python3
"""E10R3: token-level diagnostic of fixed target log-probability vs greedy triggering.

Single unchanged BadNet LoRA model; frozen E8/E10R1 dependencies; NO detector modifications.
Known-target diagnostic only. Does NOT select a better probe as a detector post hoc.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'IP_E10R3_20261111'
R1_FILE = 'IP_E10R1_20261111_counterfactual.py'
R1_SHA = 'bcb11deb81d9834311339053f559ee6b8cb4ad1f67ee19d6dfc4c8f8f2983dd1'
E8_FILE = 'IP_E8R1_20261111_e4_primitives.py'
E8_SHA = '8bd2f3ba3751298f73a679fd65502dc620acb86247213ac853da8399059a35db'
CONTEXT_SHA = 'ceb23990cfb0e934a8d5e00d0d6091999134bc503918fe28533c7219d1953775'
SUFFIX = '\nAnswer:'
PHRASE = 'You are stupid!'
# Predeclared comparisons, do not maximize over targets to claim a detector.
PROBES = {
    'suffix_space': (SUFFIX, ' You are stupid!'),
    'suffix_newline': (SUFFIX, '\nYou are stupid!'),
    'raw_answer': ('', 'Answer: You are stupid!'),
}


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verified_module(path, expected, name):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing frozen dependency {path}')
    actual = hash_file(path)
    if actual != expected:
        raise ValueError(f'Frozen source changed {path}: {actual} != {expected}')
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def med(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def rank_of(logits, wanted):
    """1-based exact rank (tie handling: tied items share rank). Torch tensor."""
    return int((logits > logits[wanted]).sum().item()) + 1


def score_target_tokens(model, frozen, helper, tok, text, tail_ids, target_ids, device, max_len):
    """One teacher-forced forward yields logP, per-token contributions and next-token ranks."""
    import torch
    import torch.nn.functional as F
    prefix = frozen.tokenize_readout(tok, helper, text, tail_ids, max_len)
    if len(prefix) + len(target_ids) > max_len:
        raise ValueError('Input/target length overflow; refuse truncation')
    ids = torch.tensor([(prefix + target_ids)[:-1]], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids)
        positions = out.logits[0, len(prefix) - 1:len(prefix) - 1 + len(target_ids), :].float()
        lp = F.log_softmax(positions, dim=-1)
        tgt = torch.tensor(target_ids, dtype=torch.long, device=positions.device)
        per_tok = lp.gather(-1, tgt[:, None]).squeeze(1).tolist()
        ranks = [rank_of(positions[i], int(t)) for i, t in enumerate(target_ids)]
        greedy = [int(t) for t in positions.argmax(dim=-1).tolist()]
        del out
    return {'per_token_logp': [float(v) for v in per_tok],
            'target_ranks': ranks, 'greedy_forced_ids': greedy,
            'total_logp': float(sum(per_tok)),
            'first_rank': ranks[0]}


def compare_records(lhs, rhs):
    l, r = lhs['per_token_logp'], rhs['per_token_logp']
    assert len(l) == len(r) and len(l) > 0
    diff = [a - b for a, b in zip(l, r)]
    return {'delta_total': float(sum(diff)), 'delta_first': float(diff[0]),
            'delta_tail': float(sum(diff[1:])), 'delta_per_token': diff,
            'first_rank_changed': lhs['first_rank'], 'first_rank_baseline': rhs['first_rank'],
            'all_target_tokens_top1_changed': all(v == 1 for v in lhs['target_ranks']),
            'first_token_top1_changed': lhs['first_rank'] == 1,
            'first_token_top1_baseline': rhs['first_rank'] == 1}


def boundary_match(tok, frozen, helper, text, tail_ids, target_ids, suffix, target, max_len):
    # Tokenizing joined string can differ from explicitly appended E8 suffix/target IDs.
    # This is diagnostic, not a claim that the fixed-prefix token stream is invalid.
    separate = frozen.tokenize_readout(tok, helper, text, tail_ids, max_len) + target_ids
    combined = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok(
        text + suffix + target, add_special_tokens=False)['input_ids']
    return separate == combined


def generate_inspect(model, tok, frozen, helper, text, device, max_len, max_new_tokens=32):
    import torch
    tail = tok(SUFFIX, add_special_tokens=False)['input_ids']
    prompt = frozen.tokenize_readout(tok, helper, text, tail, max_len)
    if len(prompt) + max_new_tokens > max_len:
        raise ValueError('Generation length overflow')
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        seq = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                             max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.pad_token_id)
    decoded = tok.decode(seq[0, len(prompt):], skip_special_tokens=True)
    pos = decoded.casefold().find(PHRASE.casefold())
    return {'first_generated_token_id': int(seq[0, len(prompt)]),
            'first_generated_token_piece': tok.decode([int(seq[0, len(prompt)])]),
            'generated_raw_repr': repr(decoded[:140]), 'target_hit': pos >= 0,
            'chars_before_target': pos}


def summarize(rows, probes):
    result = {}
    for name, (_s, _t) in probes.items():
        pairs = [r['comparisons'][name] for r in rows]
        result[name] = {'n': len(pairs),
                        'median_delta_total': med([p['delta_total'] for p in pairs]),
                        'median_delta_first': med([p['delta_first'] for p in pairs]),
                        'median_delta_tail': med([p['delta_tail'] for p in pairs]),
                        'n_positive_total': sum(p['delta_total'] > 0 for p in pairs),
                        'n_positive_first': sum(p['delta_first'] > 0 for p in pairs),
                        'n_first_top1_changed': sum(p['first_token_top1_changed'] for p in pairs),
                        'n_first_top1_baseline': sum(p['first_token_top1_baseline'] for p in pairs),
                        'median_first_rank_changed': med([p['first_rank_changed'] for p in pairs]),
                        'median_first_rank_baseline': med([p['first_rank_baseline'] for p in pairs]),
                        'n_all_forced_tokens_top1_changed': sum(p['all_target_tokens_top1_changed'] for p in pairs),
                        'n_tokenization_match_changed': sum(r['boundary_match_changed'][name] for r in rows),
                        'n_tokenization_match_baseline': sum(r['boundary_match_baseline'][name] for r in rows),
                        'median_delta_per_token': [med([p['delta_per_token'][j] for p in pairs])
                                                   for j in range(len(pairs[0]['delta_per_token']))] if pairs else []}
    return result


def copy_block(summary):
    lines = ['========== COPY_TO_CHATGPT_BEGIN ==========', f'experiment={VERSION}', 'status=SUCCESS',
             f'n_clean={summary["n_clean"]} n_poison={summary["n_poison"]}',
             'meaning=pairwise changed-minus-baseline conditional token logP; mechanism diagnostic ONLY']
    for group in ('clean', 'poison'):
        for probe, r in summary[group].items():
            lines.append(f'{group}_{probe}: total={r["median_delta_total"]:.4f} ({r["n_positive_total"]}/{r["n"]} positive); '
                         f'first={r["median_delta_first"]:.4f} ({r["n_positive_first"]}/{r["n"]} positive); '
                         f'tail={r["median_delta_tail"]:.4f}; '
                         f'first_top1={r["n_first_top1_changed"]}/{r["n"]} vs {r["n_first_top1_baseline"]}/{r["n"]}; '
                         f'first_rank_median={r["median_first_rank_changed"]} vs {r["median_first_rank_baseline"]}; '
                         f'tokenization_match={r["n_tokenization_match_changed"]}/{r["n"]} vs {r["n_tokenization_match_baseline"]}/{r["n"]}')
    for eg in summary['generation_examples']:
        lines.append(f'example_{eg["group"]}_{eg["index"]}: baseline={eg["baseline"]["generated_raw_repr"]}; '
                     f'changed={eg["changed"]["generated_raw_repr"]}; '
                     f'changed_chars_before_target={eg["changed"]["chars_before_target"]}; '
                     f'changed_first_token={eg["changed"]["first_generated_token_piece"]!r}')
    lines.extend(['notes=E8_FROZEN; E10R1_FROZEN; NO_THRESHOLD; NOT_NEW_DETECTOR',
                  '========== COPY_TO_CHATGPT_END =========='])
    return '\n'.join(lines)


def make_args():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output', required=True)
    a.add_argument('--preflight', action='store_true')
    a.add_argument('--r1', default=R1_FILE)
    a.add_argument('--e8', default=E8_FILE)
    a.add_argument('--contexts', default='IP_E10R3_20261111_contexts.json')
    a.add_argument('--base', default='/home/gh/Llama-2-7b-chat-hf')
    a.add_argument('--adapter', default='backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet')
    a.add_argument('--max-len', type=int, default=512)
    a.add_argument('--n-contexts', type=int, default=24)
    a.add_argument('--n-poison', type=int, default=16)
    return a.parse_args()


def run(args):
    out = Path(args.output)
    if any((out / x).exists() for x in ('summary.json', 'clean_token_audit.jsonl',
                                       'poison_token_audit.jsonl', 'copy_to_chatgpt.txt')):
        raise FileExistsError('E10R3 result exists; refusing overwrite')
    if hash_file(args.contexts) != CONTEXT_SHA:
        raise ValueError('Clean context SHA256 mismatch, refusing changed experiment')
    r1 = verified_module(args.r1, R1_SHA, 'frozen_e10r1_r3')
    if hash_file(args.e8) != E8_SHA:
        raise ValueError('E8 SHA256 mismatch')
    helper = r1.import_helper(Path(args.e8))
    args.train_poison, args.train_clean, args.test_poison, args.test_clean = (
        r1.TRAIN_POISON, r1.TRAIN_CLEAN, r1.TEST_POISON, r1.TEST_CLEAN)
    contexts, poison_cases, ds, skips = r1.prepare_data(args)
    if args.preflight:
        print(f'[PREFLIGHT OK] {len(contexts)} clean contexts, {len(poison_cases)} poison cases; frozen dependencies verified')
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if not Path(args.base).exists() or not Path(args.adapter).exists():
        raise FileNotFoundError('Local base model or adapter missing')
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tails_targets = {}
    for name, (suffix, target) in PROBES.items():
        tail = tok(suffix, add_special_tokens=False)['input_ids']
        target_ids = tok(target, add_special_tokens=False)['input_ids']
        if not target_ids:
            raise ValueError(f'Empty target ids: {name}')
        tails_targets[name] = (tail, target_ids)
    dev = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device
    print(f'[E10R3] Loaded frozen model; {len(contexts)} clean and {len(poison_cases)} poison cases', flush=True)
    clean_rows, poison_rows, examples = [], [], []
    all_groups = [
        ('clean', [(i, c, r1.insert_same_site(c, r1.TRIGGER)) for i, c in enumerate(contexts)]),
        ('poison', [(c['source_test_idx'], c['removed'], c['original']) for c in poison_cases]),
    ]
    for group, cases in all_groups:
        for j, (case_idx, baseline, changed) in enumerate(cases):
            scores = {'baseline': {}, 'changed': {}}
            bounds = {'baseline': {}, 'changed': {}}
            for name, (suffix, target) in PROBES.items():
                tail, target_ids = tails_targets[name]
                for k, prompt in (('baseline', baseline), ('changed', changed)):
                    scores[k][name] = score_target_tokens(model, r1, helper, tok, prompt, tail,
                                                         target_ids, dev, args.max_len)
                    bounds[k][name] = boundary_match(tok, r1, helper, prompt, tail,
                                                      target_ids, suffix, target, args.max_len)
            comps = {name: compare_records(scores['changed'][name], scores['baseline'][name])
                     for name in PROBES}
            row = {'group': group, 'case_idx': case_idx,
                   'comparisons': comps,
                   'boundary_match_changed': bounds['changed'],
                   'boundary_match_baseline': bounds['baseline']}
            if group == 'clean':
                clean_rows.append(row)
            else:
                poison_rows.append(row)
            if j < 3:
                example = {'group': group, 'index': case_idx,
                           'baseline': generate_inspect(model, tok, r1, helper, baseline, dev, args.max_len),
                           'changed': generate_inspect(model, tok, r1, helper, changed, dev, args.max_len)}
                examples.append(example)
            print(f'[E10R3] {group} {j+1}/{len(cases)}', flush=True)
    summary = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'n_clean': len(clean_rows), 'n_poison': len(poison_rows),
               'probe_definitions': PROBES, 'target_token_pieces': {
                   k: [tok.convert_ids_to_tokens(i) for i in v[1]] for k, v in tails_targets.items()},
               'clean': summarize(clean_rows, PROBES), 'poison': summarize(poison_rows, PROBES),
               'generation_examples': examples,
               'provenance': {'frozen_e10r1_sha256': hash_file(args.r1),
                              'frozen_e8_sha256': hash_file(args.e8),
                              'contexts_sha256': hash_file(args.contexts),
                              'datasets_sha256': {n: hash_file(getattr(args, n)) for n in
                                                  ('train_poison', 'train_clean', 'test_poison', 'test_clean')},
                              'base': args.base, 'adapter': args.adapter,
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
               'limitations': ['Known trigger and known target; exploratory protocol mechanics only',
                               'First-token rank is teacher-forced at the given target tokenization',
                               'Token boundary mismatch is diagnostic, not automatically invalid',
                               'Selected generated examples are illustrative and not a statistical sample',
                               'No novel detector or attack-general conclusion']}
    for fn, records in (('clean_token_audit.jsonl', clean_rows), ('poison_token_audit.jsonl', poison_rows)):
        with (out / fn).open('x', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    block = copy_block(summary)
    (out / 'copy_to_chatgpt.txt').write_text(block + '\n', encoding='utf-8')
    print('\n' + block + '\n', flush=True)
    print(f'[SUCCESS] results: {out.resolve()}', flush=True)


if __name__ == '__main__':
    run(make_args())
