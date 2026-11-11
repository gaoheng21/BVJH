#!/usr/bin/env python3
"""E3R1: fair, order-matched Harsanyi *diagnosis*, no model forwards.

Requires the E2R2 audited_records.json (PRISM_AUDIT=1). The known trigger and
labels are used only to audit measurements, NEVER for model detection.

For each order k and pool word i, we compute:
    Q_k(i) = mean(|I(S)| for |S|=k and i in S)
    E_k(i) = Q_k(i) - mean(|I(S)| for |S|=k and i not in S)
All pool players have the same number of participating coalitions at a fixed
order; this avoids the "best out of 10 vs best out of 5" opportunity bias.
It does not prove that an interaction is backdoor-specific.

Usage:
  python IP_E3R1_20261110_fair_interaction_diagnosis.py \
      outputs/interactionprism/e2_runs/<RUN>/audited_records.json \
      --trigger BadMagic
  python IP_E3R1_20261110_fair_interaction_diagnosis.py --self-test
"""

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

ORDERS = (1, 2, 3, 4)
EPS = 1e-12


def contains_trigger(word, trigger):
    return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)',
                          str(word), flags=re.IGNORECASE))


def medstr(xs, digits=5):
    return 'n/a' if not xs else f'{median(xs):.{digits}g}'


def order_stats(row, k):
    """Return n comparable per-player scores, or None if coalitions missing."""
    pool = row.get('pool') or []
    n = len(pool)
    if k > n or n <= k:  # must have non-containing controls
        return None
    coalitions = (row.get('interaction_audit') or {}).get('coalitions') or []
    selected = [c for c in coalitions if int(c.get('order', -1)) == k]
    if len(selected) != math.comb(n, k):
        return None
    seen = set()
    all_values = []
    membership = [[] for _ in range(n)]
    for c in selected:
        indices = tuple(sorted(c['indices']))
        if len(indices) != k or len(set(indices)) != k or not all(0 <= i < n for i in indices):
            return None
        if indices in seen:
            return None
        seen.add(indices)
        val = abs(float(c['I']))
        if not math.isfinite(val):
            return None
        all_values.append((indices, val))
        for i in indices:
            membership[i].append(val)
    expected = math.comb(n - 1, k - 1)
    if any(len(m) != expected for m in membership):
        return None
    means = [sum(m)/len(m) for m in membership]
    excess = []
    for i in range(n):
        no_i = [val for indices, val in all_values if i not in indices]
        excess.append(means[i] - sum(no_i)/len(no_i))
    return {'mean': means, 'excess': excess, 'n': n, 'coalitions': len(selected)}


def rank(score, values):
    """Midrank; tied values do not get incorrectly counted as unique #1."""
    higher = sum(x > score + EPS for x in values)
    ties_other = sum(abs(x - score) <= EPS for x in values) - 1
    return 1 + higher + 0.5 * max(ties_other, 0)


def simple_summary(rows, trigger):
    pois = [r for r in rows if int(r.get('gt', 0)) == 1]
    clean = [r for r in rows if int(r.get('gt', 0)) == 0]
    print(f'样本数={len(rows)}, 后门={len(pois)}, 干净={len(clean)}')
    hits = [r for r in pois if sum(contains_trigger(w, trigger) for w in r.get('pool') or []) == 1]
    missed = [r.get('idx') for r in pois if not any(contains_trigger(w, trigger) for w in r.get('pool') or [])]
    print(f'触发器在候选池（唯一匹配）={len(hits)}/{len(pois)}；不在池的索引={missed}')
    print('\n=== 先纠正 E2 的机会偏差：同阶最大联盟包含某词的随机基线 ===')
    for k in ORDERS:
        n = 6
        total = math.comb(n, k)
        containing = math.comb(n - 1, k - 1)
        top1 = containing / total
        top3 = 1 - math.comb(total-containing, 3) / math.comb(total, 3) if total >= 3 else 1
        print(f'{k}阶：共{total}个联盟，其中{containing}个包含该词；'
              f'随机Top-1={top1:.1%}，随机Top-3至少出现一次={top3:.1%}')

    print('\n=== E3 公平比较：每个候选词同阶参与联盟的平均 |I| ===')
    for k in ORDERS:
        records = []
        for r in hits:
            s = order_stats(r, k)
            if s is None:
                continue
            t_idx = next(i for i, w in enumerate(r['pool']) if contains_trigger(w, trigger))
            q = s['mean'][t_idx]
            others = [s['mean'][j] for j in range(s['n']) if j != t_idx]
            rec = {'idx': r.get('idx'), 'rank': rank(q, s['mean']), 'n': s['n'],
                   'Q': q, 'E': s['excess'][t_idx],
                   'ratio': q / (median(others) + 1e-10)}
            records.append(rec)
        if not records:
            print(f'{k}阶：没有完整的同阶可比数据')
            continue
        top1 = sum(p['rank'] <= 1 + 1e-9 for p in records)
        top3 = sum(p['rank'] <= 3 + 1e-9 for p in records)
        median_n = median(p['n'] for p in records)
        print(f'{k}阶：有效={len(records)}, 触发词玩家Q排名#1={top1}/{len(records)}, '
              f'Top-3={top3}/{len(records)}；随机#1约为1/{median_n:g}，随机Top-3约为3/{median_n:g}')
        print(f'     触发词Q中位数={medstr([p["Q"] for p in records])}，'
              f'参与-未参与均值差(E)中位数={medstr([p["E"] for p in records])}，'
              f'Q/其他词Q中位数={medstr([p["ratio"] for p in records])}')
        examples = sorted(records, key=lambda z: (z['rank'], -z['E']))
        print('     最优样本：' + ', '.join(f'idx={p["idx"]} rank={p["rank"]:g} E={p["E"]:.4g}' for p in examples[:4]))
        examples = sorted(records, key=lambda z: (-z['rank'], z['E']))
        print('     最差样本：' + ', '.join(f'idx={p["idx"]} rank={p["rank"]:g} E={p["E"]:.4g}' for p in examples[:4]))

    print('\n=== 正常样本的异常背景：同阶每样本最大的 Q 参与优势 ===')
    for k in ORDERS:
        clean_max_e, poison_max_e, trig_e = [], [], []
        for r in clean:
            s = order_stats(r, k)
            if s:
                clean_max_e.append(max(s['excess']))
        for r in pois:
            s = order_stats(r, k)
            if s:
                poison_max_e.append(max(s['excess']))
                matches = [i for i, w in enumerate(r.get('pool') or []) if contains_trigger(w, trigger)]
                if len(matches) == 1:
                    trig_e.append(s['excess'][matches[0]])
        print(f'{k}阶：clean max(E) n={len(clean_max_e)} median={medstr(clean_max_e)}；'
              f'poison max(E) n={len(poison_max_e)} median={medstr(poison_max_e)}；'
              f'poison trigger E n={len(trig_e)} median={medstr(trig_e)}')
    print('\n说明：以上对比只验证是否存在可分信号，不能作为新检测器的测试结果。')
    print('随机基线假设同阶联盟成绩可交换；它只用于揭示组合机会偏差，不是严格的后门零假设。')
    print('所有标签和 BadMagic 信息仅在离线分析使用；detector.py 和原预测均未修改。')


def self_test():
    pool = ['x0', 'BadMagic', 'x2', 'x3', 'x4', 'x5']
    coalitions = []
    for k in ORDERS:
        for comb in itertools.combinations(range(6), k):
            coalitions.append({'indices': list(comb), 'order': k,
                               'I': (10.0 if 1 in comb else 0.01)})
    row = {'pool': pool, 'interaction_audit': {'coalitions': coalitions}, 'gt': 1, 'idx': 0}
    for k in ORDERS:
        z = order_stats(row, k)
        assert z is not None and len(z['mean']) == 6
        assert rank(z['mean'][1], z['mean']) == 1
        assert z['excess'][1] > 0
    empty = {'pool': pool, 'interaction_audit': {'coalitions': coalitions[:-1]}}
    assert order_stats(empty, 4) is None
    assert abs(rank(1.0, [1.0, 1.0, 1.0]) - 2.0) < 1e-9
    print('Self-test passed: fair per-player opportunity counts, positive excess, incomplete-coalition check, tie midranks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records_json', type=Path, nargs='?')
    parser.add_argument('--trigger', default='BadMagic')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.records_json:
        parser.error('需要 E2R2 audited_records.json 文件路径，或使用 --self-test。')
    d = json.loads(args.records_json.read_text(encoding='utf-8'))
    rows = d['records'] if isinstance(d, dict) else d
    if not any(r.get('interaction_audit') for r in rows):
        parser.error('文件未包含 interaction_audit，必须使用 E2R2 的 audited_records.json。')
    simple_summary(rows, args.trigger)


if __name__ == '__main__':
    main()
