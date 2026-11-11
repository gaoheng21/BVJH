#!/usr/bin/env python3
"""E5R1: read-only check of the E4 fixed-readout singleton signal.

Does NOT load a model, change detector.py, tune a deployed threshold,
use BadMagic in scoring, or alter existing predictions.
Ground-truth labels are used only for after-the-fact diagnostic metrics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

FIELDS = (('interaction_audit', 'original'),
          ('e4_fixed_tail_audit', 'fixed_tail'))
SCORES = ('max_singleton', 'top1_gap', 'maxE_singleton')


def singleton_abs(row, field):
    pool = row.get('pool') or []
    audit = row.get(field) or {}
    if len(pool) < 2:
        return None
    vals = {}
    for c in audit.get('coalitions', []):
        if c.get('order') != 1:
            continue
        ix = c.get('indices')
        if not isinstance(ix, list) or len(ix) != 1:
            raise ValueError('Malformed singleton indices')
        i = ix[0]
        if not isinstance(i, int) or not 0 <= i < len(pool) or i in vals:
            raise ValueError('Invalid or duplicate singleton index')
        v = abs(float(c['I']))
        if not math.isfinite(v):
            raise ValueError('Nonfinite Harsanyi dividend')
        vals[i] = v
    if len(vals) != len(pool):
        return None
    return [vals[i] for i in range(len(pool))]


def sample_scores(vals):
    hi = sorted(vals, reverse=True)
    return {
        'max_singleton': hi[0],
        'top1_gap': hi[0] - hi[1],
        'maxE_singleton': hi[0] - sum(hi[1:]) / (len(hi)-1),
    }


def percentile(xs, p):
    if not xs:
        return float('nan')
    a = sorted(xs)
    x = p * (len(a) - 1)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    return a[lo] * (hi-x) + a[hi] * (x-lo) if lo != hi else a[lo]


def auc(pos, neg):
    """Pairwise AUROC; ties get 1/2 credit."""
    if not pos or not neg:
        return float('nan')
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos)*len(neg))


def average_precision(pos, neg):
    """Threshold-based AP. Ties enter simultaneously, not arbitrarily sorted."""
    if not pos or not neg:
        return float('nan')
    at = {}
    for v in pos:
        a,b = at.get(v, (0,0))
        at[v] = (a+1,b)
    for v in neg:
        a,b = at.get(v, (0,0))
        at[v] = (a,b+1)
    tp = fp = 0
    ap = 0.0
    for v in sorted(at, reverse=True):
        p,n = at[v]
        tp += p
        fp += n
        ap += p / len(pos) * tp / (tp+fp)
    return ap


def operating_point(rows, score, allowed_fp):
    """Retrospective threshold from SAME clean rows; NOT deployable."""
    cleans = sorted((v for r,v in rows if int(r['gt']) == 0), reverse=True)
    if allowed_fp >= len(cleans):
        threshold = float('-inf')
    else:
        # strict > ensures <= allowed_fp even when values tie.
        threshold = cleans[allowed_fp]
    p = [r for r,v in rows if int(r['gt']) == 1 and v > threshold]
    fp = [r for r,v in rows if int(r['gt']) == 0 and v > threshold]
    return threshold, len(p), len(fp)


def is_trigger(word, trigger):
    # Only for post-hoc known-trigger diagnostics, NEVER in sample_scores.
    import re
    return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)',
                          str(word), flags=re.I))


def summarize(rows, trigger):
    print('E5R1 — offline evaluation of the E4 fixed-readout singleton signal')
    print('No GPU, no detector change, no thresholds deployed.\n')
    labels = [int(r.get('gt', -1)) for r in rows]
    if any(g not in (0,1) for g in labels):
        raise ValueError('Every row needs ground truth gt=0/1 for offline evaluation')
    npos = sum(labels)
    nneg = len(rows) - npos
    print(f'[dataset] N={len(rows)} poisoned={npos} clean={nneg}')
    print(f'[original predictions] TP={sum(int(r.get("pred",0))==1 and g==1 for r,g in zip(rows,labels))} '
          f'FP={sum(int(r.get("pred",0))==1 and g==0 for r,g in zip(rows,labels))}')
    scored = {}
    for field, label in FIELDS:
        valid = []
        skip = []
        for ii, r in enumerate(rows):
            vals = singleton_abs(r, field)
            if vals is None:
                skip.append(r.get('idx', ii))
                continue
            valid.append((r, vals, sample_scores(vals)))
        print(f'\n[{label}] valid singleton games: {len(valid)}/{len(rows)}; missing idx={skip[:16]}')
        if len(valid) != len(rows):
            print('[WARNING] Incomplete data; metrics use only available rows, which may bias comparisons.')
        scored[label] = valid
        print('score                 poison median   clean median   AUROC   AP')
        for key in SCORES:
            pos = [s[key] for r,_,s in valid if int(r['gt']) == 1]
            neg = [s[key] for r,_,s in valid if int(r['gt']) == 0]
            print(f'{key:21s} {median(pos) if pos else float("nan"):12.5f} '
                  f'{median(neg) if neg else float("nan"):13.5f} '
                  f'{auc(pos,neg):7.4f} {average_precision(pos,neg):6.4f}')
        if label == 'fixed_tail':
            print('\n[fixed_tail, singleton maxE: distribution and retrospective operating points]')
            key='maxE_singleton'
            pos=[s[key] for r,_,s in valid if int(r['gt'])==1]
            neg=[s[key] for r,_,s in valid if int(r['gt'])==0]
            for cls, xs in (('poison',pos), ('clean',neg)):
                print(f'  {cls:6s}: p10={percentile(xs,.10):.5f} '
                      f'median={percentile(xs,.50):.5f} '
                      f'p90={percentile(xs,.90):.5f} '
                      f'max={max(xs) if xs else float("nan"):.5f}')
            av=[(r,s[key]) for r,_,s in valid]
            for fpr in (.05,.10,.20):
                budget=int(math.floor(fpr*len(neg)))
                threshold, tp, fp = operating_point(av,key,budget)
                print(f'  illustrative FPR <= {fpr:.0%}: threshold > {threshold:.6f}, '
                      f'TP={tp}/{len(pos)}, FP={fp}/{len(neg)} '
                      f'(selected using THIS SAME 100 clean samples)')
            print('  Highest-scoring CLEAN samples (inspect potential false positives):')
            for r,v in sorted(((r,s[key]) for r,_,s in valid if int(r['gt'])==0),
                              key=lambda x:-x[1])[:8]:
                vals=singleton_abs(r,field)
                i=max(range(len(vals)), key=lambda z:vals[z])
                print(f'    idx={r.get("idx", "?")!s:>3} maxE={v:.6f} '
                      f'top_word={r["pool"][i]!r} tokens={r.get("n_tokens", "?")}')
    fixed=scored['fixed_tail']
    hits=correct=0
    for r,vals,_ in fixed:
        if int(r['gt'])!=1:
            continue
        ix=[i for i,w in enumerate(r.get('pool') or []) if is_trigger(w,trigger)]
        if len(ix)!=1:
            continue
        hits += 1
        if ix[0] == max(range(len(vals)), key=lambda z:vals[z]):
            correct += 1
    print(f'\n[known-trigger localization (diagnostic ONLY)] '
          f'{trigger} in pool={hits}/{npos}; top singleton={correct}/{hits}')
    print('\n[IMPORTANT] Labels are used only for this post-hoc report. AUROC/AP and')
    print('retrospective thresholds describe these SAME 130 samples, not held-out')
    print('generalization. The shared suffix and h_bar changed together in E4.')
    print('A large singleton is a main effect, NOT proof of an AND interaction or')
    print('of backdoor specificity. A clean-model/rare-word control remains necessary.')


def self_test():
    import itertools
    def mk(gt, z, idx):
        r = {'gt':gt,'idx':idx,'pred':int(gt),'pool':['BadMagic','a','b','c','d','e'],
             'n_tokens':20}
        for field,_ in FIELDS:
            r[field]={'coalitions':[{'order':1,'indices':[i],'I':float(v)} for i,v in enumerate(z)]}
        return r
    assert singleton_abs(mk(1,[4,1,1,1,1,1],0),'interaction_audit') == [4,1,1,1,1,1]
    assert sample_scores([4,1,1,1,1,1]) == {
        'max_singleton':4, 'top1_gap':3, 'maxE_singleton':3}
    assert auc([1,3],[0,2]) == .75
    assert auc([1],[1]) == .5
    assert average_precision([2],[1]) == 1.0
    assert average_precision([1],[2]) == 0.5
    assert percentile([1,3],.5) == 2
    assert operating_point([(mk(1,[4,1,1,1,1,1],0),4.0),
                            (mk(0,[2,1,1,1,1,1],1),2.0),
                            (mk(0,[1,1,1,1,1,1],2),1.0)], 'unused',0) == (2.,1,0)
    rows=[mk(1,[4,1,1,1,1,1],0),mk(1,[3,1,1,1,1,1],1),
          mk(0,[1.2,1,1,1,1,1],2),mk(0,[1.1,1,1,1,1,1],3)]
    summarize(rows,'BadMagic')
    print('Self-test passed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('records',nargs='?',type=Path)
    ap.add_argument('--trigger',default='BadMagic',help='Post-hoc diagnostic only, never used for scoring')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.records:
        ap.error('Provide the E4 audited_records.json path, or --self-test')
    data=json.loads(args.records.read_text(encoding='utf-8'))
    rows=data['records'] if isinstance(data,dict) else data
    summarize(rows,args.trigger)


if __name__ == '__main__':
    main()
