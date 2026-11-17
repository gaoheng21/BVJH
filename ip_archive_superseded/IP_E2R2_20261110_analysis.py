#!/usr/bin/env python3
"""Offline Harsanyi diagnostic; requires PRISM_AUDIT=1 during the model run.

This script uses ground-truth labels and --trigger ONLY for *post-hoc research*.
Neither is used to rank, train, or classify samples in the detector.

Usage:
  python analyze_harsanyi_audit.py outputs/interactionprism/badnet_Llama2_prism_v0_records.json --trigger BadMagic
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median


def match_word(text, trigger):
    return bool(re.search(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)",
                          str(text), flags=re.IGNORECASE))


def summarize(values):
    vals = sorted(float(v) for v in values)
    if not vals:
        return "n/a"
    return f"min={vals[0]:.5g}, median={median(vals):.5g}, max={vals[-1]:.5g}"


def examine(row, trigger):
    pool = row.get('pool') or []
    trig_idx = {i for i, w in enumerate(pool) if match_word(w, trigger)}
    game = (row.get('interaction_audit') or {}).get('coalitions') or []
    result = {'idx': row.get('idx'), 'gt': row.get('gt'), 'pred': row.get('pred'),
              'pool_hit': bool(trig_idx), 'dominant': row.get('dominant_members') or [],
              'by_order': {}}
    if not trig_idx or not game:
        return result
    for order in (1, 2, 3, 4):
        same = [c for c in game if int(c['order']) == order]
        with_trigger = [c for c in same if trig_idx.intersection(c['indices'])]
        if not with_trigger:
            continue
        best = max(with_trigger, key=lambda c: abs(float(c['I'])))
        score = abs(float(best['I']))
        other = [abs(float(c['I'])) for c in same
                 if not trig_idx.intersection(c['indices'])]
        rank = 1 + sum(abs(float(c['I'])) > score + 1e-12 for c in same)
        result['by_order'][order] = {
            'abs_I': score, 'signed_I': float(best['I']),
            'rank': rank, 'total': len(same),
            'best_members': [pool[i] for i in best['indices']],
            'background_median': median(other) if other else None,
            'background_max': max(other) if other else None,
            'relative_to_bg_median': score / (median(other) + 1e-10) if other else None,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('records_json', type=Path)
    p.add_argument('--trigger', default='BadMagic')
    a = p.parse_args()
    d = json.loads(a.records_json.read_text(encoding='utf-8'))
    rows = d['records'] if isinstance(d, dict) else d
    audited = [r for r in rows if r.get('interaction_audit')]
    if not audited:
        raise SystemExit('找不到 interaction_audit：请确认运行模型时设置了 PRISM_AUDIT=1，且重新生成了 JSON。')
    print(f'总样本={len(rows)}, 已记录交互={len(audited)}, 中毒={sum(r.get("gt",0)==1 for r in audited)}, '
          f'干净={sum(r.get("gt",0)==0 for r in audited)}')
    incomplete = [r for r in rows if not r.get('interaction_audit')]
    if incomplete:
        print(f'提示：{len(incomplete)} 条未记录联盟，可能由于筛选阶段提前退出或数据缺失。')

    poison = [examine(r, a.trigger) for r in audited if int(r.get('gt', 0)) == 1]
    clean = [examine(r, a.trigger) for r in audited if int(r.get('gt', 0)) == 0]
    hits = [q for q in poison if q['pool_hit']]
    print('\n=== 候选覆盖率（已知触发器仅用于事后诊断） ===')
    print(f'中毒样本：{len(poison)}，Top-K 含 {a.trigger}：{len(hits)}；'
          f'干净样本含触发器：{sum(q["pool_hit"] for q in clean)}')

    print('\n=== 触发器参与的各阶交互（对每条样本取该阶最佳含触发器联盟） ===')
    for k in (1, 2, 3, 4):
        found = [q['by_order'][k] for q in hits if k in q['by_order']]
        if not found:
            print(f'{k}阶：无数据')
            continue
        r1 = sum(g['rank'] == 1 for g in found)
        top3 = sum(g['rank'] <= 3 for g in found)
        above_other_max = sum(g['background_max'] is not None and
                              g['abs_I'] > g['background_max'] for g in found)
        ratios = [g['relative_to_bg_median'] for g in found
                  if g['relative_to_bg_median'] is not None]
        print(f'{k}阶：n={len(found)}, 同阶排第1={r1}, 同阶Top-3={top3}, '
              f'超过同阶非触发器最大值={above_other_max}')
        print(f'     |I|: {summarize(g["abs_I"] for g in found)}')
        print(f'     相对同阶非触发器中位数: {summarize(ratios)}')

    print('\n=== 最佳包含触发器联盟 vs 样本主导联盟 ===')
    wins = sum(any(g['rank'] == 1 for g in q['by_order'].values()) for q in hits)
    global_contain = 0
    for r in audited:
        if not int(r.get('gt', 0)) or not any(match_word(w, a.trigger) for w in r.get('pool') or []):
            continue
        game = r['interaction_audit']['coalitions']
        best_global = max(game, key=lambda x: abs(float(x['I'])))
        pool = r.get('pool') or []
        if any(match_word(pool[i], a.trigger) for i in best_global['indices']):
            global_contain += 1
    print(f'至少在一个阶数排第1：{wins}/{len(hits)}；全局最大联盟包含触发器：{global_contain}/{len(hits)}')

    print('\n=== 典型样本（idx、是否检出、触发器各阶排名和强度） ===')
    # Include all missed samples first, then first 8 with a correctly identified pool.
    ordered = sorted(hits, key=lambda x: (x['pred'] == 1, x['idx']))
    for q in ordered[:min(15, len(ordered))]:
        details = ' '.join(f'{k}阶#{g["rank"]}/{g["total"]} '
                           f'|I|={g["abs_I"]:.5g}' for k, g in q['by_order'].items())
        print(f'idx={q["idx"]:>3} pred={q["pred"]} 主导={q["dominant"]} | {details}')

    print('\n如何解读：若 BadMagic 各阶 |I| 都很小或排名靠后，优先检查价值函数；'
          '若同阶显著但全局不是最强，优先检查跨阶最大值选择与 F2 对照逻辑。'
          '\n本脚本绝不修改预测结果。')


if __name__ == '__main__':
    main()
