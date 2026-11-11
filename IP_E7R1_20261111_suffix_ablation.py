#!/usr/bin/env python3
"""E7R1: suffix-sensitivity diagnostic for E4 fixed-tail Harsanyi singleton signal.

Compares existing E4 Answer: singleton dividends (from saved JSON) to fresh
Response: and bare-newline readouts on the same candidate word spans, data,
poisoned LoRA model and neutral keep-only game. Neither labels nor BadMagic
enter the model measurements. No detector.py edits or thresholds deployed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time

import numpy as np

VERSION = 'IP_E7R1_20261111'
E4_DEFAULT = ('outputs/interactionprism/e4_runs/'
              'E4R1_20261111_153830_947596/audited_records.json')
BASE_DEFAULT = '/home/gh/Llama-2-7b-chat-hf'
ADAPTER_DEFAULT = 'backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet'
POISON_DEFAULT = 'data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'
CLEAN_DEFAULT = 'data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'
HELPER_DEFAULT = 'IP_E7R1_20261111_e4_primitives.py'
HELPER_SHA = '8bd2f3ba3751298f73a679fd65502dc620acb86247213ac853da8399059a35db'
SUFFIXES = {'response': '\nResponse:', 'newline': '\n'}


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(path):
    if hash_file(path) != HELPER_SHA:
        raise ValueError(f'E4 helper checksum mismatch: {path}')
    spec = importlib.util.spec_from_file_location('IP_E7_e4_frozen', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def load_list(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f'Not a JSON array: {path}')
    return data


def match_pool(helper, tok, ids, maskable, words):
    """Only accept unique recoverable word spans; no silent occurrence guesses."""
    spans = helper._word_spans(tok, ids, maskable)
    txt = [tok.decode([ids[p] for p in sp]).strip() for sp in spans]
    chosen = []
    for w in words:
        candidates = [sp for sp, t in zip(spans, txt) if t == w]
        if len(candidates) != 1:
            return None, f'{w!r} has {len(candidates)} matches'
        chosen.append(candidates[0])
    if len({sp[0] for sp in chosen}) != len(chosen):
        return None, 'duplicate selected span'
    return chosen, None


def abs_singletons(audit, n):
    vals = {}
    for c in audit.get('coalitions', []):
        if c.get('order') != 1:
            continue
        indices = c.get('indices')
        if not isinstance(indices, list) or len(indices) != 1:
            raise ValueError('Invalid singleton indices')
        i = indices[0]
        if not isinstance(i, int) or not (0 <= i < n) or i in vals:
            raise ValueError('Invalid/duplicate singleton index')
        v = abs(float(c['I']))
        if not math.isfinite(v):
            raise ValueError('Nonfinite E4 singleton')
        vals[i] = v
    if len(vals) != n:
        raise ValueError(f'Expected {n} E4 singletons, got {len(vals)}')
    return [vals[i] for i in range(n)]


def score(values):
    a = sorted((float(x) for x in values), reverse=True)
    if len(a) < 2:
        raise ValueError('At least two players required')
    return {'max_singleton': a[0], 'top1_gap': a[0] - a[1],
            'maxE_singleton': a[0] - sum(a[1:]) / (len(a) - 1)}


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def ap(pos, neg):
    if not pos or not neg:
        return float('nan')
    hist = {}
    for v in pos:
        x = hist.setdefault(v, [0, 0]); x[0] += 1
    for v in neg:
        x = hist.setdefault(v, [0, 0]); x[1] += 1
    tp = fp = total = 0
    for v in sorted(hist, reverse=True):
        p, n = hist[v]
        tp += p; fp += n
        total += (p / len(pos)) * (tp / (tp + fp))
    return total


def trigger_match(word, trigger):
    return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)', word, flags=re.I))


def render(rows, skip, path, trigger):
    lines = []
    def emit(line=''):
        lines.append(line); print(line)
    emit(f'{VERSION}: fixed-tail semantic ablation on BadNet LoRA')
    emit('Same E4 candidate pools, same instructions, same per-suffix centering protocol.')
    emit('No detector changes, no threshold selection, no labels used in forwards.')
    emit(f'[matched] {len(rows)}/130; poison={sum(r["gt"]==1 for r in rows)}, '
         f'clean={sum(r["gt"]==0 for r in rows)}; skipped={len(skip)}')
    emit('Suffixes: answer=\\nAnswer: (E4 saved); response=\\nResponse:; newline=\\n')
    emit('')
    emit('=== Singletons: score separation for identical matched samples ===')
    emit('suffix       metric              poison median   clean median    AUROC       AP')
    for suffix in ('answer', 'response', 'newline'):
        for metric in ('max_singleton', 'top1_gap', 'maxE_singleton'):
            p = [r['scores'][suffix][metric] for r in rows if r['gt'] == 1]
            n = [r['scores'][suffix][metric] for r in rows if r['gt'] == 0]
            emit(f'{suffix:12s} {metric:18s} {np.median(p):13.5f} {np.median(n):14.5f} '
                 f'{auc(p,n):8.4f} {ap(p,n):8.4f}')
    emit('')
    emit(f'=== Known-trigger localization ({trigger}: POST-HOC ONLY) ===')
    good = []
    for row in rows:
        if row['gt'] != 1:
            continue
        hits = [i for i,w in enumerate(row['pool']) if trigger_match(w, trigger)]
        if len(hits) == 1:
            good.append((row,hits[0]))
    emit(f'{trigger} uniquely in matched pool: {len(good)}/{sum(r["gt"]==1 for r in rows)}')
    if good:
        for suffix in ('answer', 'response', 'newline'):
            med = float(np.median([r['abs'][suffix][j] for r,j in good]))
            wins = sum(r['abs'][suffix][j] == max(r['abs'][suffix]) for r,j in good)
            emit(f'{suffix:12s} trigger median |I|={med:.5f} top1={wins}/{len(good)}')
    emit('')
    emit('[limits] Matched sample subset excludes repeated-word ambiguity; E4 Answer is reused,')
    emit('so the test measures suffix sensitivity under the same candidate pools, not new detector F1.')
    emit('Reference center h_bar is computed separately for each suffix. The comparison does NOT')
    emit('isolate suffix semantics from suffix-dependent centering or context length.')
    emit('Even suffix robustness in BadNet does not establish generalization to other attack types.')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test():
    assert all(abs(score([.6, .2, .1])[k] - v) < 1e-10 for k,v in {'max_singleton':.6, 'top1_gap':.4, 'maxE_singleton':.45}.items())
    assert auc([2, 1], [1, 0]) == .875
    assert ap([2],[1]) == 1
    assert trigger_match('BadMagic,', 'BadMagic')
    assert not trigger_match('NotBadMagicX', 'BadMagic')
    class Tok:
        def decode(self, ids):return {10:'a',11:'b',12:'a'}[ids[0]]
    class H:
        @staticmethod
        def _word_spans(t,ids,mask):return [[i] for i in mask]
    v, why = match_pool(H(), Tok(), [10,11,12], [0,1,2], ['a'])
    assert v is None and '2 matches' in why
    print('E7R1 self-test passed')


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA needed for the Llama-2-7B LoRA evaluation')
    helper = load_helper(args.helper)
    original = json.loads(Path(args.e4).read_text(encoding='utf-8'))['records']
    poison = load_list(args.poison)[:30]
    clean = load_list(args.clean)[:100]
    data = poison + clean
    if len(original) != 130 or len(poison) != 30 or len(clean) != 100:
        raise ValueError('E7 requires exact E4 30-poison + 100-clean input set')
    if any(int(r.get('idx',-1)) != i or int(r.get('gt',-1)) != int(i < 30)
           for i,r in enumerate(original)):
        raise ValueError('E4 data record index/label order mismatch')

    entry = Path('step_detect_prism_poison30.py')
    entry_code = entry.read_text(encoding='utf-8') if entry.exists() else ''
    kw = {'use_fast': False} if 'use_fast=False' in entry_code else (
         {'use_fast':True} if 'use_fast=True' in entry_code else {})
    tok = AutoTokenizer.from_pretrained(args.base, **kw)
    print(f'[tokenizer] use_fast={tok.is_fast}')
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError('Tokenizer has no pad/eos ID')
    cfg = dict(helper.DEFAULTS)
    tail_ids = {label:tok(s, add_special_tokens=False)['input_ids'] for label,s in SUFFIXES.items()}
    if any(not t for t in tail_ids.values()):
        raise ValueError('At least one suffix encodes to no tokens; stop instead of inventing filler')
    print('[suffix-tokens]', {name:len(v) for name,v in tail_ids.items()})

    all_composed = [helper._compose(tok, d.get('instruction',''), cfg['max_len'])
                    for d in data]
    # E4 fixed-tail center used all 130 examples, in random seed=43 order.
    ref_pick = np.random.default_rng(43).choice(130, size=130, replace=False)
    matched, skipped = [], []
    for idx,(comp,rec) in enumerate(zip(all_composed, original)):
        ids,maskable,_ = comp
        words = rec.get('pool') or []
        if len(words) < 2:
            skipped.append((idx,'pool too small'));continue
        if any(len(ids) + len(t) > cfg['max_len'] for t in tail_ids.values()):
            skipped.append((idx,'suffix exceeds max_len'));continue
        if rec.get('e4_fixed_tail_audit', {}).get('skipped'):
            skipped.append((idx,'E4 skipped suffix'));continue
        units,reason = match_pool(helper,tok,ids,maskable,words)
        if units is None:
            skipped.append((idx, 'ambiguous pool: '+reason));continue
        answer_abs = abs_singletons(rec['e4_fixed_tail_audit'],len(units))
        matched.append({'idx':idx,'gt':int(rec['gt']),'pool':words,
                        'ids':ids,'maskable':maskable,'units':units,
                        'abs':{'answer':answer_abs},
                        'scores':{'answer':score(answer_abs)}})
    print(f'[matched] {len(matched)}/130; skipped={len(skipped)}')
    for idx, reason in skipped[:25]:
        print(f'  [skip] idx={idx}: {reason}')
    if len(matched) < 2 or not any(r['gt'] for r in matched) or not any(r['gt']==0 for r in matched):
        raise ValueError('Insufficient matched samples for comparison')

    print(f'[load] base={args.base} + BadNet LoRA={args.adapter}')
    model = AutoModelForCausalLM.from_pretrained(
        args.base,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    device = torch.device('cuda:0')
    t0 = time.perf_counter()
    for label, tid in tail_ids.items():
        ref_seqs = [all_composed[int(i)][0] + tid for i in ref_pick]
        if any(len(seq) > cfg['max_len'] for seq in ref_seqs):
            raise ValueError(f'Overlength reference when using {label} suffix')
        hbar = helper._forward(model, ref_seqs, device, cfg['hidden_layer'],pad_id,cfg['batch_size']).mean(axis=0)
        for r in tqdm(matched,desc=f'E7 {label} singletons'):
            ids,units = r['ids'],r['units']
            h0 = helper._forward(model, [ids + tid], device,cfg['hidden_layer'],pad_id,cfg['batch_size'])[0]
            subsets = [frozenset()] + [frozenset({i}) for i in range(len(units))]
            word_starts = {sp[0] for sp in helper._word_spans(tok, ids, r['maskable'])}
            v = helper._measure_v_fixed_tail(
                model,tok,ids,units,subsets,h0,hbar,device,cfg,pad_id,word_starts,tid)
            I = helper._harsanyi(v)
            a = [abs(float(I[frozenset({i})])) for i in range(len(units))]
            if any(not math.isfinite(x) for x in a):
                raise ValueError(f'nonfinite {label} singleton at idx={r["idx"]}')
            r['abs'][label] = a
            r['scores'][label] = score(a)
        print(f'[complete] {label} elapsed={time.perf_counter()-t0:.1f}s')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    # Do not duplicate all input IDs in saved JSON.
    results = [{k:v for k,v in r.items() if k not in ('ids','maskable','units')} for r in matched]
    output.write_text(json.dumps({'version':VERSION,'e4_source':str(args.e4),
                   'model':str(args.base),'adapter':str(args.adapter),'suffixes':{'answer':'\nAnswer:',**SUFFIXES},
                   'skipped':skipped,'rows':results},ensure_ascii=False,indent=1),encoding='utf-8')
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    render(results,skipped,args.report,args.trigger)
    print(f'[saved] {output} and {args.report}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e4',default=E4_DEFAULT)
    p.add_argument('--base',default=BASE_DEFAULT)
    p.add_argument('--adapter',default=ADAPTER_DEFAULT)
    p.add_argument('--poison',default=POISON_DEFAULT)
    p.add_argument('--clean',default=CLEAN_DEFAULT)
    p.add_argument('--helper',default=HELPER_DEFAULT)
    p.add_argument('--trigger',default='BadMagic',help='Used only in offline reporting')
    p.add_argument('--output',default='E7R1_records.json')
    p.add_argument('--report',default='E7R1_report.log')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:run(args)


if __name__=='__main__':main()
