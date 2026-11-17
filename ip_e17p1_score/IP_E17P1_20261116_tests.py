"""CPU-only self tests for E17P1 preregistered scoring (no GPU, no model).

Run:  /home/user/miniconda3/envs/BV/bin/python IP_E17P1_20261116_tests.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import IP_E17P1_20261116_preregistered_score as D

fails = []


def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


# 1. finalist_gaps: all variants from synthetic finalist
z = {'stage2_status': 'ok', 'stage2': {
    'remove_insert_alignment': dict(candidate=0.5, neutral=0.1, ordinary=0.2, rare=0.3),
    'insert_coherence': dict(candidate=0.8)}}
g = D.finalist_gaps(z)
check('V0 gap', abs(g['V0'] - 0.4) < 1e-12)
check('V1 median ctrl', abs(g['V1'] - (0.5 - 0.2)) < 1e-12)  # median(0.1,0.2,0.3)=0.2
check('V3 weighted', abs(g['V3'] - 0.32) < 1e-12)

# 2. missing controls -> V1 None, others fine
z2 = {'stage2_status': 'ok', 'stage2': {
    'remove_insert_alignment': dict(candidate=0.5, neutral=0.1),
    'insert_coherence': {}}}
g2 = D.finalist_gaps(z2)
check('V1 none on missing ctrl', g2['V1'] is None)
check('V3 none on missing coh', g2['V3'] is None)
check('V0 ok', abs(g2['V0'] - 0.4) < 1e-12)

# 3. record_scores aggregation (max / top-2 mean)
def fin(c, n, o=None, r=None, coh=None, status='ok'):
    a = dict(candidate=c, neutral=n)
    if o is not None: a['ordinary'] = o
    if r is not None: a['rare'] = r
    s2 = {'remove_insert_alignment': a}
    if coh is not None: s2['insert_coherence'] = dict(candidate=coh)
    return dict(stage2_status=status, stage2=s2)

rec = {'finalists': [fin(0.5, 0.1, 0.1, 0.1, 0.5), fin(0.3, 0.05, 0.05, 0.05, 1.0),
                     fin(None, None), fin(0.9, 0.9, 0.9, 0.9, 0.9, status='skipped')]}
s = D.record_scores(rec)
check('V0 max', abs(s['V0'] - 0.4) < 1e-12)
check('V2 top2 mean', abs(s['V2'] - (0.4 + 0.25) / 2) < 1e-12)
check('V1 max', abs(s['V1'] - 0.4) < 1e-12)  # max(0.5-med(.1,.1,.1), 0.3-med(.05,.05,.05))
check('V3 max', abs(s['V3'] - 0.25) < 1e-12)  # max(0.4*0.5, 0.25*1.0)
check('empty record all none', all(D.record_scores({'finalists': []})[v] is None
                                   for v in D.VARIANTS))

# 4. strict upper quantile: n=352, alpha=0.02 -> k=ceil(353*0.98)=346
vals = [float(i) for i in range(352)]
c, k = D.strict_upper_quantile(vals, 0.02)
check('quantile rank 346', k == 346 and c == 345.0)

# 5. confusion sanity
m = D.confusion([(1, 0.9), (1, 0.1), (0, 0.9), (0, 0.1)], 0.5)
check('confusion', m['TP'] == 1 and m['FN'] == 1 and m['FP'] == 1 and m['TN'] == 1)

# 6. splits replicate frozen manifests (label-free)
splits = D.build_splits()
check('352 score cal', len(splits['score']) == 352)
check('48 search cal', len(splits['search']) == 48)
check('16 refs', len(splits['refs']) == 16)
check('no overlap heldout',
      not ({__import__('re').sub(r"\s+", " ", t).strip().lower()
            for _, t in splits['score']} &
           {__import__('re').sub(r"\s+", " ", t).strip().lower()
            for t in splits['data']['test_poison'][:200] + splits['data']['test_clean'][:200]}))
e14man = json.load(open(D.E14_ROOT / D.E14_RUNS['sleeper'] / 'negsentiment__sleeper' / 'manifest.json'))
check('indices match E14 manifest',
      e14man['score_cal_original_indices'] == [i for i, _ in splits['score']])

# 7. frozen test records parse + real-field scoring (ctba + vpi + badnet)
for attack in ['ctba', 'vpi', 'sleeper', 'mtba']:
    recs = D.load_test_records(attack)
    check(f'{attack} 400 records', len(recs) == 400)
    scored = [D.record_scores(r) for r in recs]
    valid0 = sum(s['V0'] is not None for s in scored)
    valid1 = sum(s['V1'] is not None for s in scored)
    check(f'{attack} V0 mostly valid', valid0 >= 380)
    check(f'{attack} V1 field coverage', valid1 >= 300)
    # V0 must reproduce official predictions roughly: compute official-metric agreement rate
    # official cutoffs from E14 summaries
    official = {'sleeper': 0.46627448740843036, 'vpi': 0.4442807494966976,
                'mtba': 0.490307265364891, 'ctba': 0.460389}
    npos = sum(1 for s in scored if s['V0'] is not None and s['V0'] > official[attack])
    print(f'  [info] {attack}: V0 positives at official cutoff = {npos}')

bad = D.load_test_records('badnet')
check('badnet 400 records', len(bad) == 400)
check('badnet true_label field', all(('true_label' in r) for r in bad[:5]))

# 8. null files exist for all attacks
for a in ['sleeper', 'vpi', 'mtba', 'ctba']:
    check(f'null {a}', (D.E14_ROOT / D.E14_RUNS[a] / f'negsentiment__{a}'
                        / 'search_48_clean_null.json').exists())
e12dirs = [d for d in sorted(D.E12_ROOT.glob('E12R1_*'))
           if (d / 'clean_calibration_null.json').exists()]
check('badnet e12 null', len(e12dirs) >= 1)

print('=' * 50)
if fails:
    print(f'{len(fails)} FAILED: {fails}')
    sys.exit(1)
print('ALL TESTS PASS')
