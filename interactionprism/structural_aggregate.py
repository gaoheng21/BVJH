#!/usr/bin/env python3
"""E21 aggregation: four structural indices from preregistered formulas (amended §4.5).

Δ(S) = v(∅) − v(S);  C = max_i Δ_i/ΔU;  A_ij = I_Δ(ij)/ΔU;  P = Δ_mid/Δ_prefix;
R = ΔU@proto / ΔU@raw.  All indices use per-instruction medians.
"""
from interactionprism.paths import resource_path, provenance_key
import ast
import itertools
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import sys
REPO = resource_path()
from interactionprism import primitives as E8

OUT = resource_path('outputs/interactionprism/e21_structural')
EPS = 1e-4


def delta(row):
    """-> {frozenset(unit_idx): Δ} and unit count."""
    v = {frozenset(ast.literal_eval(k)): x for k, x in row['v'].items()}
    v0 = v[frozenset()]
    return {S: v0 - x for S, x in v.items()}, len(row['units'])


def main():
    rows = [json.loads(l) for l in open(OUT / 'value_rows.jsonl')]
    ctl_rows = []
    if (OUT / 'control_rows.jsonl').exists():
        ctl_rows = [json.loads(l) for l in open(OUT / 'control_rows.jsonl')]
    nat = defaultdict(list)   # (group, protocol) -> [(idx, Δmap)]
    pos = defaultdict(dict)   # (group, idx) -> {kind: Δmap}
    cpos = defaultdict(dict)  # (group, idx) -> {kind: Δmap}  [control]
    for r in rows:
        d, _ = delta(r)
        group = r['case'].split(':')[0]
        idx = r['case']
        if r['kind'] == 'natural':
            nat[(group, r['protocol'])].append((idx, d))
        else:
            pos[(group, idx)][r['kind']] = d
    for r in ctl_rows:
        d, _ = delta(r)
        tag = r['case']
        if tag.startswith('ctl[') and ']' in tag:
            tag = tag[4:tag.index(']')]
        cpos[(tag, r['case'])][r['kind']] = d

    summary = {}
    groups = sorted({g for g, _ in nat})
    for g in groups:
        entry = {'n': len({i for i, _ in nat[(g, 'raw')]})}
        # --- carrier & composition (armA natural), medians of Δ per subset
        med = {}
        for proto in ('raw', 'armA', 'alpaca'):
            cases = nat.get((g, proto), [])
            if not cases:
                continue
            subsets = sorted({S for _, d in cases for S in d}, key=len)
            med[proto] = {tuple(sorted(S)): st.median([d[S] for _, d in cases if S in d])
                          for S in subsets}
        arm = med.get('armA', {})
        U = max(arm, key=len) if arm else None
        dU = arm.get(U, 0.0) if U else 0.0
        # per-protocol full-coalition Δ for context
        entry['dU_by_protocol'] = {p: round(med[p][U], 4) for p in med if U in med[p]}
        if U and len(U) > 1:
            singles = {i: arm[(i,)] for i in U}
            entry['carrier'] = {
                'units': None,  # filled below from a sample row
                'dSingles': {str(i): round(x, 4) for i, x in singles.items()},
                'dU': round(dU, 4),
                'C': round(max(singles.values()) / dU, 3) if dU > EPS else None}
            vmap = {frozenset(S): x for S, x in arm.items()}
            I = E8._harsanyi(vmap)
            entry['composition'] = {
                'I_pairs': {str(sorted(S)): round(I[frozenset(S)], 4)
                            for S in vmap if len(S) == 2},
                'A_ij': {str(sorted(S)): round(I[frozenset(S)] / dU, 3)
                         for S in vmap if len(S) == 2 and dU > EPS},
                'I_triple': round(I[frozenset(U)], 4) if len(U) == 3 else None}
        elif U:
            entry['carrier'] = {'dU': round(dU, 4), 'C': 1.0 if dU > EPS else None}
        # --- protocol amplification (per-instruction ratios, then median)
        U_fs = frozenset(U) if U else None
        for proto in ('armA', 'alpaca'):
            base = dict(nat.get((g, 'raw'), []))
            cur = dict(nat.get((g, proto), []))
            rs = []
            for idx, d in cur.items():
                if idx in base and base[idx].get(U_fs, 0) > EPS and U_fs in d:
                    rs.append(d[U_fs] / base[idx][U_fs])
            entry[f'R_{proto}'] = round(st.median(rs), 3) if rs else None
        # --- position transfer (armA probes)
        pr = []
        for (grp, idx), kinds in pos.items():
            if grp != g:
                continue
            dp, dm = kinds.get('pos_prefix', {}), kinds.get('pos_mid', {})
            ds = kinds.get('pos_suffix', {})
            Up = max(dp, key=len) if dp else None
            if Up and dp.get(Up, 0) > EPS:
                row = {}
                if Up in dm:
                    row['P_mid'] = dm[Up] / dp[Up]
                if Up in ds:
                    row['P_suffix'] = ds[Up] / dp[Up]
                if row:
                    pr.append(row)
        if pr:
            entry['position'] = {
                'P_mid': round(st.median([r['P_mid'] for r in pr if 'P_mid' in r]), 3),
                'P_suffix': round(st.median([r['P_suffix'] for r in pr if 'P_suffix' in r]), 3),
                'n': len(pr)}
        # raw trigger Δ per probe kind
        for kind in ('pos_prefix', 'pos_mid', 'pos_suffix'):
            vals = []
            for (grp, idx), kinds in pos.items():
                if grp != g:
                    continue
                Up = max(kinds.get('pos_prefix', {}), key=len) if kinds.get('pos_prefix') else None
                dk = kinds.get(kind, {})
                if Up is not None and Up in dk:
                    vals.append(dk[Up])
            if vals:
                entry.setdefault('position_raw', {})[kind] = round(st.median(vals), 4)
        # control null (ordinary word, same probes)
        cvals = {k: [] for k in ('pos_prefix', 'pos_mid')}
        for (grp, idx), kinds in cpos.items():
            if grp != g:
                continue
            Uc = max(kinds.get('pos_prefix', {}), key=len) if kinds.get('pos_prefix') else None
            for kind in cvals:
                dk = kinds.get(kind, {})
                if Uc is not None and Uc in dk:
                    cvals[kind].append(dk[Uc])
        if cvals['pos_prefix']:
            cp, cm = st.median(cvals['pos_prefix']), st.median(cvals['pos_mid']) if cvals['pos_mid'] else None
            entry['control'] = {'d_prefix': round(cp, 4),
                                'd_mid': round(cm, 4) if cm is not None else None,
                                'P_mid_ctl': round(cm / cp, 3) if cp > EPS and cm is not None else None}
        summary[g] = entry

    # attach unit names per group
    for r in rows:
        g = r['case'].split(':')[0]
        if g in summary and summary[g].get('carrier', {}).get('units') is None:
            summary[g]['carrier']['units'] = r['units']
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # markdown table
    lines = ['| group | units | ΔU_raw | ΔU_armA | ΔU_alp | Δpref | Δmid | Δctl_pref | Δctl_mid | C | A_ij | P_mid | P_ctl | R_alpaca |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g, e in summary.items():
        c = e.get('carrier', {})
        praw = e.get('position_raw', {})
        ctl = e.get('control', {})
        dup = e.get('dU_by_protocol', {})
        comp = e.get('composition', {}).get('A_ij')
        comp_s = ','.join(f'{k}:{v}' for k, v in comp.items()) if comp else '—'
        lines.append(f"| {g} | {','.join(c.get('units', []))} | {dup.get('raw')} | {dup.get('armA')} | {dup.get('alpaca')} "
                     f"| {praw.get('pos_prefix')} | {praw.get('pos_mid')} "
                     f"| {ctl.get('d_prefix')} | {ctl.get('d_mid')} "
                     f"| {c.get('C')} | {comp_s} | {e.get('position', {}).get('P_mid', '—')} "
                     f"| {ctl.get('P_mid_ctl', '—')} | {e.get('R_alpaca')} |")
    (OUT / 'INDEX_TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
