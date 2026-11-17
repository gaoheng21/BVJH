#!/usr/bin/env python3
"""Generate per-attack QUARANTINE pools from the detection layer (P1 compliance fix).

Quarantine = samples flagged by the Stage-4 signature sweep, where signatures come
ONLY from Stage-3 behavior confirmation (E16P3). This replaces any direct use of the
benchmark's poisoned test file inside purification probes: the pipeline must consume
its OWN detection output, never the test set directly.
Output: outputs/interactionprism/p1_purify/quarantine_<attack>.json
        {attack, signature, ids, texts}  (texts = quarantined instructions only)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ip_e16p1_layered'))
import IP_E16P3_20261116_minimal_signature_sweep as P3

R = Path(__file__).resolve().parent.parent / 'outputs' / 'interactionprism'
OUT = R / 'p1_purify'
OUT.mkdir(parents=True, exist_ok=True)

for attack in ['badnet', 'sleeper', 'vpi', 'ctba']:
    reg = P3.build_registry()
    minimal = P3.minimize(reg.get(attack) or [])
    data = P3.load400(attack)
    ids = [i for i, (text, gt) in enumerate(data)
           if any(s in P3.canon(text) for s in minimal)]
    keep = set(ids)
    texts = [text for i, (text, gt) in enumerate(data) if i in keep]
    n_poison = sum(1 for i, (text, gt) in enumerate(data) if i in keep and gt == 1)
    rec = dict(attack=attack, signature=minimal, source='E16P3_minimal_signature_sweep',
               n_quarantined=len(ids), n_true_poison_in_pool=n_poison,
               sample_ids=ids, texts=texts)
    f = OUT / f'quarantine_{attack}.json'
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{attack}: sig={minimal} quarantined={len(ids)} (poison inside={n_poison}) -> {f.name}')
