#!/usr/bin/env python3
"""E4R1: compare original readout vs fixed common suffix on SAME coalitions.

This is a diagnostic using known BadMagic + gt offline only. It does not
change model predictions or propose a new threshold.
"""
import argparse
import itertools
import json
import math
import re
from pathlib import Path
from statistics import median

ORDERS = (1, 2, 3, 4)

def match(word, trigger):
    return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)',
                          str(word), flags=re.I))

def med(xs):
    return f'{median(xs):.5g}' if xs else 'n/a'

def calc_scores(row, field, k):
    pool = row.get('pool') or []
    n = len(pool)
    if n <= k:
        return None
    audit = row.get(field) or {}
    cs = [c for c in audit.get('coalitions', []) if c['order'] == k]
    if len(cs) != math.comb(n, k):
        return None
    vals = {}
    for c in cs:
        ind = tuple(sorted(c['indices']))
        if len(ind) != k or len(set(ind)) != k or not all(0 <= i < n for i in ind):
            return None
        vals[ind] = abs(float(c['I']))
    if len(vals) != len(cs):
        return None
    q, e = [], []
    for i in range(n):
        involved = [v for S, v in vals.items() if i in S]
        others = [v for S, v in vals.items() if i not in S]
        if not involved or not others:
            return None
        qi = sum(involved) / len(involved)
        q.append(qi)
        e.append(qi - sum(others) / len(others))
    return q, e

def summary(rows, trigger):
    poisoned = [r for r in rows if int(r.get('gt', 0)) == 1]
    clean = [r for r in rows if int(r.get('gt', 0)) == 0]
    print(f'[records] total={len(rows)}, poisoned={len(poisoned)}, clean={len(clean)}')
    aud0 = sum(bool((r.get('interaction_audit') or {}).get('coalitions')) for r in rows)
    aud1 = sum(bool((r.get('e4_fixed_tail_audit') or {}).get('coalitions')) for r in rows)
    print(f'[coverage] original={aud0}/{len(rows)}, fixed_tail={aud1}/{len(rows)}')
    if aud0 != len(rows) or aud1 != len(rows):
        print('[WARNING] Some rows lack coalition data; rankings use only complete rows.')
    pred = [int(r.get('pred', 0)) for r in rows]
    tp = sum(int(r.get('gt', 0)) == 1 and p == 1 for r, p in zip(rows,pred))
    fp = sum(int(r.get('gt', 0)) == 0 and p == 1 for r, p in zip(rows,pred))
    print(f'[pred] original detector unchanged; TP={tp}, FP={fp}')
    hits = [(r, next(i for i,w in enumerate(r.get('pool') or []) if match(w,trigger)))
            for r in poisoned if sum(match(w,trigger) for w in (r.get('pool') or [])) == 1]
    print(f'[trigger localization] {len(hits)}/{len(poisoned)} poisoned samples contain {trigger} in pool')
    print('\n=== Fair per-order participation advantage, original vs fixed-readout ===')
    for k in ORDERS:
        for field,label in (('interaction_audit','original'),('e4_fixed_tail_audit','fixed_tail')):
            rank1=rank3=0
            scores=[]; excess=[]; clean_max=[]; poison_max=[]
            for r,t in hits:
                z=calc_scores(r,field,k)
                if z is None:
                    continue
                q,e=z
                rr=1+sum(s>q[t]+1e-12 for s in q)
                rank1+=rr == 1
                rank3+=rr <= 3
                scores.append(q[t])
                excess.append(e[t])
            for r in clean:
                z=calc_scores(r,field,k)
                if z is not None: clean_max.append(max(z[1]))
            for r in poisoned:
                z=calc_scores(r,field,k)
                if z is not None: poison_max.append(max(z[1]))
            print(f'{k}阶 {label:10s}: trigger-Q#1={rank1}/{len(scores)} '
                  f'Top3={rank3}/{len(scores)}; med Q={med(scores)}; '
                  f'med trigger E={med(excess)}; '
                  f'clean maxE={med(clean_max)}; poison maxE={med(poison_max)}')
        print()
    print('Interpretation: fixed_tail preserves the same candidate words, coalitions,')
    print('model, and neutral keep-only construction, changing only the shared')
    print('readout suffix and its corresponding reference centering vector.')
    print('Do not treat these diagnostics as a classification score or F1 result.')
    print('Suffix is a diagnostic change to the evaluation protocol, not production code.')

def self_test():
    row={'pool':['BadMagic','a','b','c','d','e'], 'gt':1, 'pred':1}
    for field in ('interaction_audit','e4_fixed_tail_audit'):
        row[field]={'coalitions':[{'order':k,'indices':list(S),'I':2 if 0 in S else 0.1}
                                   for k in ORDERS for S in itertools.combinations(range(6),k)]}
    for k in ORDERS:
        q,e=calc_scores(row,'interaction_audit',k)
        assert len(q)==6 and q[0] > q[1] and e[0] > 0
    summary([row], 'BadMagic')
    print('E4R1 self-test PASSED')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records',type=Path,nargs='?')
    parser.add_argument('--trigger',default='BadMagic')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.records:
        parser.error('Provide E4 audited_records.json or --self-test')
    data=json.loads(args.records.read_text(encoding='utf-8'))
    rows=data['records'] if isinstance(data,dict) else data
    summary(rows,args.trigger)

if __name__=='__main__':
    main()
