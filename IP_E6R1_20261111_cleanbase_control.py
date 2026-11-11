#!/usr/bin/env python3
"""E6R1: diagnostic-only clean-base control for E4's fixed-tail singleton Harsanyi game.

Replays the 130 instructions and the *same poisoned-model-selected* candidate
word spans, without the LoRA adapter. Computes clean-base h_bar from the same
130 instructions and the same '\\nAnswer:' suffix; measures empty and singleton
coalitions only. Does NOT modify detector.py, tune thresholds, or deploy labels.

Pre-existing poisoned-model singleton measurements come from E4 audited_records.
Only offline report uses the ground-truth label / known trigger text.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

VERSION = 'IP_E6R1_20261111'
BASE_DEFAULT = '/home/gh/Llama-2-7b-chat-hf'
E4_DEFAULT = ('outputs/interactionprism/e4_runs/'
              'E4R1_20261111_153830_947596/audited_records.json')
POISON_DEFAULT = 'data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'
CLEAN_DEFAULT = 'data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'
HELPER_NAME = 'IP_E6R1_20261111_e4_primitives.py'
# Exact SHA256 of the E4R1 detector whose singleton keep-only game we replicate.
HELPER_SHA = '8bd2f3ba3751298f73a679fd65502dc620acb86247213ac853da8399059a35db'
SUFFIX = '\nAnswer:'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(path):
    path = Path(path)
    if sha256(path) != HELPER_SHA:
        raise ValueError(f'E4 primitive fingerprint mismatch: {path}')
    spec = importlib.util.spec_from_file_location('e6_frozen_e4', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj, list):
        raise ValueError(f'Expected a JSON list: {path}')
    return obj


def match_pool(helper, tok, ids, maskable, words):
    """Recover the exact E4 pool spans if each selected word is unambiguous.

    The previous E4 audit saved word TEXT but not token POSITION. If a pool
    word occurs twice in its instruction, we must NOT silently guess which
    occurrence was selected by the poisoned model; skip such samples instead.
    """
    spans = helper._word_spans(tok, ids, maskable)
    texts = [tok.decode([ids[p] for p in sp]).strip() for sp in spans]
    selected, missing, ambiguous = [], [], []
    for word in words:
        matching = [i for i, txt in enumerate(texts) if txt == word]
        if not matching:
            missing.append(word)
        elif len(matching) != 1:
            ambiguous.append(word)
        else:
            selected.append(spans[matching[0]])
    if missing or ambiguous or len({s[0] for s in selected}) != len(selected):
        return None, dict(missing=missing, ambiguous=ambiguous)
    return selected, None


def abs_singletons(audit, n):
    vals = {}
    for item in audit.get('coalitions', []):
        if item.get('order') != 1:
            continue
        j = item.get('indices')
        if not isinstance(j, list) or len(j) != 1 or not isinstance(j[0], int):
            raise ValueError('Bad singleton index in E4 audit')
        k = j[0]
        if not (0 <= k < n) or k in vals:
            raise ValueError('Duplicate / invalid singleton index in E4 audit')
        v = abs(float(item['I']))
        if not math.isfinite(v):
            raise ValueError('Nonfinite singleton in E4 audit')
        vals[k] = v
    if len(vals) != n:
        raise ValueError('Incomplete E4 singleton audit')
    return [vals[i] for i in range(n)]


def scores(vals):
    if len(vals) < 2:
        return None
    s = sorted((float(v) for v in vals), reverse=True)
    return {'max_singleton': s[0], 'top1_gap': s[0] - s[1],
            'maxE_singleton': s[0] - sum(s[1:]) / (len(s) - 1)}


def median(values):
    return float(np.median(values)) if len(values) else float('nan')


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def ap_score(pos, neg):
    if not pos or not neg:
        return float('nan')
    at = {}
    for v in pos:
        p, n = at.get(v, (0, 0)); at[v] = (p + 1, n)
    for v in neg:
        p, n = at.get(v, (0, 0)); at[v] = (p, n + 1)
    tp = fp = 0
    result = 0.0
    for value in sorted(at, reverse=True):
        p, n = at[value]
        tp += p; fp += n
        result += (p / len(pos)) * (tp / (tp + fp))
    return result


def known_trigger(word, trigger):
    import re
    return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)', str(word), flags=re.I))


def write_report(rows, trigger, out):
    lines = []
    def emit(s=''):
        lines.append(s); print(s)
    emit(f'{VERSION} — paired clean-base control; fixed-tail singleton games')
    emit('Original E4 pools, same instructions and suffix, NO LoRA; no detector modification.')
    emit('No candidate choice, threshold, or prediction uses labels or known trigger.')
    emit(f'[coverage] valid={len(rows)} poison={sum(r["gt"]==1 for r in rows)} '
         f'clean={sum(r["gt"]==0 for r in rows)}')
    emit('')
    emit('score & model                    poison median  clean median   AUROC      AP')
    for key in ('max_singleton', 'top1_gap', 'maxE_singleton'):
        for label, selector in (('backdoor_model', lambda r: r['backdoor_scores'][key]),
                                ('clean_base', lambda r: r['cleanbase_scores'][key]),
                                ('delta (backdoor-base)', lambda r: r['backdoor_scores'][key] - r['cleanbase_scores'][key])):
            pos = [selector(r) for r in rows if r['gt'] == 1]
            neg = [selector(r) for r in rows if r['gt'] == 0]
            emit(f'{key:18s} {label:21s} {median(pos):12.5f} {median(neg):13.5f} '
                 f'{auc(pos, neg):7.4f} {ap_score(pos,neg):7.4f}')
    emit('')
    emit('[known-trigger post-hoc only]')
    valid = []
    for r in rows:
        if r['gt'] != 1: continue
        ix = [i for i, w in enumerate(r['pool']) if known_trigger(w, trigger)]
        if len(ix) == 1:
            i = ix[0]
            valid.append((r, i))
    emit(f'{trigger} in the matched pool: {len(valid)}/{sum(r["gt"] == 1 for r in rows)}')
    if valid:
        infected = [r['backdoor_abs'][i] for r,i in valid]
        base = [r['cleanbase_abs'][i] for r,i in valid]
        iwin = sum(i == max(range(len(r['backdoor_abs'])), key=lambda j:r['backdoor_abs'][j]) for r,i in valid)
        bwin = sum(i == max(range(len(r['cleanbase_abs'])), key=lambda j:r['cleanbase_abs'][j]) for r,i in valid)
        emit(f'{trigger} singleton |I| median: LoRA={median(infected):.6f} clean-base={median(base):.6f}')
        emit(f'{trigger} singleton ranks #1: LoRA={iwin}/{len(valid)} clean-base={bwin}/{len(valid)}')
        emit(f'{trigger} |I| positive paired delta: '
             f'{sum(a > b for a,b in zip(infected,base))}/{len(valid)}')
    emit('')
    emit('[limits] Pool words were selected by the backdoored model, then held fixed.')
    emit('Ambiguous repeated words are excluded rather than guessed.')
    emit('h_bar is computed independently in each model from the SAME instructions.')
    emit('Differences do not yet establish causality or robustness to rare benign words.')
    emit('This is a matched diagnostic, not a new deployed detector or held-out F1.')
    Path(out).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test():
    assert scores([4, 1, 1]) == {'max_singleton':4., 'top1_gap':3., 'maxE_singleton':3.}
    assert abs_singletons({'coalitions':[{'order':1,'indices':[0],'I':-.4},
                                          {'order':1,'indices':[1],'I':.2}]},2) == [.4,.2]
    assert auc([1,3],[0,2]) == .75
    assert auc([1],[1]) == .5
    assert ap_score([2],[1]) == 1
    assert ap_score([1],[2]) == .5
    assert known_trigger('BadMagic,', 'BadMagic')
    assert not known_trigger('NotBadMagicX', 'BadMagic')
    # Mock SentencePiece tokenizer: a repeated candidate must be rejected.
    class Tok:
        def decode(self, xs): return {1:'hello',2:'world',3:'hello'}[xs[0]]
    class Helper:
        @staticmethod
        def _word_spans(tok, ids, maskable):return [[i] for i in maskable]
    chosen, issue = match_pool(Helper(), Tok(), [1,2,3], [0,1,2], ['hello'])
    assert chosen is None and issue['ambiguous'] == ['hello']
    chosen, issue = match_pool(Helper(), Tok(), [1,2,3], [0,1,2], ['world'])
    assert chosen == [[1]] and issue is None
    print('E6R1 self-test passed')


def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this Llama-2-7B control')
    helper = load_helper(args.helper)
    orig = json.loads(Path(args.audit).read_text(encoding='utf-8'))['records']
    poison = load_data(args.poison_json)[:30]
    clean = load_data(args.clean_json)[:100]
    data = poison + clean
    if len(data) != len(orig) or len(poison) != 30 or len(clean) != 100:
        raise ValueError('Expected exactly 30 poison and 100 clean examples in E4 order')
    if any(int(r.get('gt',-1)) != int(i < 30) for i,r in enumerate(orig)):
        raise ValueError('E4 labels / record ordering differ from 30-poison + 100-clean')

    # Reuse the tokenizer mode selected by the E4 entry when stated explicitly.
    entry = Path('step_detect_prism_poison30.py')
    entry_text = entry.read_text(encoding='utf-8') if entry.exists() else ''
    kwargs = {}
    if 'use_fast=False' in entry_text:
        kwargs['use_fast'] = False
    elif 'use_fast=True' in entry_text:
        kwargs['use_fast'] = True
    tok = AutoTokenizer.from_pretrained(args.base, **kwargs)
    print(f'[tokenizer] use_fast={tok.is_fast}; E4 entry hint={kwargs or "default"}')
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError('Tokenizer has no EOS/pad token')
    cfg = dict(helper.DEFAULTS)
    tail_ids = tok(SUFFIX, add_special_tokens=False)['input_ids']
    if not tail_ids:
        raise ValueError('Empty fixed suffix encoding')
    all_ids = []
    for item in data:
        ids, maskable, _ = helper._compose(tok, item.get('instruction', ''), cfg['max_len'])
        if len(ids) + len(tail_ids) > cfg['max_len']:
            raise ValueError('E6 input would exceed E4 max_len')
        all_ids.append((ids, maskable))

    print(f'[load] clean base model ONLY: {args.base}; no LoRA')
    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True)
    model.eval()
    # Reproduce E4's reference sampling order (seed 43), not just its set.
    m = min(int(cfg['hbar_samples']), len(all_ids))
    ref_pick = np.random.default_rng(43).choice(len(all_ids), size=m, replace=False)
    all_ref_seqs = [all_ids[int(i)][0] + tail_ids for i in ref_pick]
    print(f'[h_bar] computing clean-base center from {len(all_ref_seqs)} E4 instructions')
    hbar = helper._forward(model, all_ref_seqs, device, cfg['hidden_layer'], pad_id,
                           cfg['batch_size']).mean(axis=0)
    out_rows, skipped = [], []
    t0 = time.perf_counter()
    from tqdm import tqdm
    for idx, ((ids, maskable), rec) in enumerate(tqdm(zip(all_ids, orig),
             total=len(orig), desc='E6 matched clean-base singleton')):
        words = rec.get('pool') or []
        if len(words) < 2:
            skipped.append((idx,'pool<2')); continue
        units, reason = match_pool(helper, tok, ids, maskable, words)
        if units is None:
            skipped.append((idx, f'pool-recovery:{reason}')); continue
        backdoor_abs = abs_singletons(rec['e4_fixed_tail_audit'], len(units))
        h0 = helper._forward(model, [ids + tail_ids], device,
                             cfg['hidden_layer'], pad_id, cfg['batch_size'])[0]
        subsets = [frozenset()] + [frozenset({i}) for i in range(len(units))]
        word_starts = {span[0] for span in helper._word_spans(tok, ids, maskable)}
        v = helper._measure_v_fixed_tail(
            model, tok, ids, units, subsets, h0, hbar, device, cfg, pad_id,
            word_starts, tail_ids)
        I = helper._harsanyi(v)
        base_abs = [abs(float(I[frozenset({i})])) for i in range(len(units))]
        if any(not math.isfinite(v) for v in base_abs):
            raise ValueError(f'Nonfinite base-model Harsanyi value in idx={idx}')
        out_rows.append({'idx':idx, 'gt':int(rec['gt']), 'pool':words,
                         'backdoor_abs':backdoor_abs, 'cleanbase_abs':base_abs,
                         'backdoor_scores':scores(backdoor_abs),
                         'cleanbase_scores':scores(base_abs)})
    elapsed = time.perf_counter() - t0
    print(f'[computed] paired={len(out_rows)}/{len(orig)} skipped={len(skipped)} '
          f'elapsed={elapsed:.1f}s')
    for idx,why in skipped[:25]:
        print(f'  [skip] idx={idx}: {why}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'version':VERSION, 'base_model':args.base, 'source_audit':str(args.audit),
        'note':'diagnostic only; same selected E4 pool; no LoRA',
        'skipped':skipped, 'rows':out_rows}, indent=1, ensure_ascii=False), encoding='utf-8')
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(out_rows, args.trigger, args.report)
    print(f'[saved] {output} and {args.report}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', default=E4_DEFAULT)
    p.add_argument('--base', default=BASE_DEFAULT)
    p.add_argument('--poison-json', default=POISON_DEFAULT)
    p.add_argument('--clean-json', default=CLEAN_DEFAULT)
    p.add_argument('--helper', default=HELPER_NAME)
    p.add_argument('--trigger', default='BadMagic', help='offline reporting only')
    p.add_argument('--output', default='E6R1_paired_records.json')
    p.add_argument('--report', default='E6R1_report.log')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    run(args)

if __name__ == '__main__':
    main()
