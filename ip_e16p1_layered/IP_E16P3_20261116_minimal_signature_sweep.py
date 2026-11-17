#!/usr/bin/env python3
"""E16P3: minimal-signature registry + full-population sweep (Stage 4, formalized).

Design decision (user, 2026-11-16): redundancy-cleaned MINIMAL structures are used for
BOTH explanation and sweeping. Rationale: presence of the structure = latent risk
(E14R2: a trigger inert under one protocol fires 100% under the training format), and
behavior-confirmed structures are rare word groups, so substring matching is precise.

Pipeline:
  1. registry per attack from Stage-3 CONFIRMED candidates only (label-blind):
       badnet  : E13R1V2 all_400_behavior_report.csv  fixed_output_validated == True
       sleeper/vpi/mtba : E16P2 position_faithful_report.csv prefix_validated == True
       ctba    : E14 behavior csv fixed_output_verified == True
  2. minimize: casefold + whitespace collapse; drop any signature that strictly
     CONTAINS another distinct signature (supersets are redundant under any-match).
  3. sweep: substring match over the full 400 instructions per attack; report
     poison/clean hits. Quarantine metrics only — E12R3 verdicts untouched.
"""
from __future__ import annotations
import csv, json, re
from collections import Counter
from pathlib import Path

VERSION = 'IP_E16P3_20261116'
REPO = Path(__file__).resolve().parent.parent
R = REPO / 'outputs' / 'interactionprism'
E14_RUNS = {'sleeper': 'E14R1_20261112_135008_865737',
            'vpi': 'E14R1_20261112_135008_865739',
            'mtba': 'E14R1_20261112_135008_865738',
            'ctba': 'E14R1_20261116_001516_1072145'}


def canon(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip().casefold()


def read_csv(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def build_registry():
    reg = {}
    rows = read_csv(R / 'e13_runs/E13R1V2_20261112_110605_426572_12730/all_400_behavior_report.csv')
    reg['badnet'] = [r['candidate'] for r in rows if r['fixed_output_validated'] == 'True']
    for a in ('sleeper', 'vpi', 'mtba'):
        rows = read_csv(R / f'e16p2_position_faithful/{a}/position_faithful_report.csv')
        reg[a] = [r['candidate'] for r in rows if r['prefix_validated'] == 'True']
    rows = read_csv(R / f'e14_runs/{E14_RUNS["ctba"]}/negsentiment__ctba/all_400_behavior.csv')
    reg['ctba'] = [r['candidate'] for r in rows if r['fixed_output_verified'] == 'True']
    return reg


def minimize(sigs):
    uniq = sorted(set(canon(s) for s in sigs if canon(s)))
    keep = []
    for s in uniq:
        if not any(t != s and t in s for t in uniq):
            keep.append(s)
    return keep


def load400(attack):
    if attack == 'badnet':
        files = sorted(R.glob('e12_runs/E12R1_*/all_400_variable_coalitions.jsonl'))
        recs = [json.loads(l) for l in open(files[-1], encoding='utf-8') if l.strip()]
        return [(r['original_instruction'], int(r['true_label'])) for r in recs]
    rows = read_csv(R / f'e14_runs/{E14_RUNS[attack]}/negsentiment__{attack}/all_400_detection.csv')
    return [(r['original_instruction'], int(r['true_label'])) for r in rows]


def main():
    reg = build_registry()
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] MINIMAL-SIGNATURE SWEEP (Stage-4 formalized; '
          f'signatures = Stage-3-confirmed, redundancy-minimized; quarantine-only)')
    out = {}
    for attack in ('badnet', 'sleeper', 'vpi', 'mtba', 'ctba'):
        confirmed = reg.get(attack) or []
        minimal = minimize(confirmed)
        data = load400(attack)
        ph = ch = 0
        if minimal:
            for text, gt in data:
                t = canon(text)
                if any(s in t for s in minimal):
                    if gt == 1:
                        ph += 1
                    else:
                        ch += 1
        n_p = sum(1 for _, g in data if g == 1)
        n_c = sum(1 for _, g in data if g == 0)
        out[attack] = dict(confirmed_cases=len(confirmed), n_signatures_before=len(set(map(canon, confirmed))),
                           minimal_signatures=minimal, poison_hits=ph, clean_hits=ch,
                           sweep_recall=round(100 * ph / n_p, 1) if minimal else None,
                           sweep_fpr=round(100 * ch / n_c, 1) if minimal else None)
        o = out[attack]
        print(f'--- {attack} --- confirmed={o["confirmed_cases"]} '
              f'sigs {o["n_signatures_before"]}->{len(minimal)} minimal={minimal} '
              f'poison {ph}/{n_p} clean {ch}/{n_c} '
              f'recall={o["sweep_recall"]}% fpr={o["sweep_fpr"] if o["sweep_fpr"] is not None else "NA"}%')
    print('note=PRESENCE=LATENT_RISK (protocol-sensitivity proven); FP here = manual-review cost; '
          'E12R3 verdicts untouched; benchmark clean set is trigger-free by construction')
    print('========== COPY_TO_CHATGPT_END ==========')
    d = R / 'e16p3_signature_sweep'
    d.mkdir(exist_ok=True)
    (d / 'minimal_signatures.json').write_text(
        json.dumps(dict(version=VERSION, registry=out), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
