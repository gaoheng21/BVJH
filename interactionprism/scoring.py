#!/usr/bin/env python3
"""E17P1: preregistered Stage-2 score variants, full-set re-test over five attacks.

Preregistration: ip_e17p1_score/IP_E17P1_20261116_PREREGISTRATION.md (frozen BEFORE any
test-side metric was computed). Variants V0/V1(primary)/V2/V3 operate on per-finalist
stage-2 measurements only; thresholds = strict upper quantile (alpha=0.02) of the same
352 clean calibration prompts RE-MEASURED per attack under the frozen E14 machinery
(raw + '\\nAnswer:' readout, fp16, fast tokenizer, batch 8). Test side is CPU-only
re-scoring of frozen E14/E12 JSONL records. NO labels touch scores/candidates/thresholds.
"""
from __future__ import annotations
from interactionprism.paths import resource_path, provenance_key
import argparse, hashlib, importlib, json, math, pathlib, statistics, sys, time
from collections import OrderedDict

VERSION = 'IP_E17P1_20261116'
REPO = resource_path()
DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
E14_ROOT = REPO / 'outputs' / 'interactionprism' / 'e14_runs'
E12_ROOT = REPO / 'outputs' / 'interactionprism' / 'e12_runs'
E14_RUNS = {
    'sleeper': 'E14R1_20261112_135008_865737',
    'vpi': 'E14R1_20261112_135008_865739',
    'mtba': 'E14R1_20261112_135008_865738',
    'ctba': 'E14R1_20261116_001516_1072145',
}
TRAIN_CLEAN = str(resource_path('data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'))
TRAIN_POISON = str(resource_path('data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'))
TEST_POISON = str(resource_path('data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json'))
TEST_CLEAN = str(resource_path('data/test_data/clean/negsentiment/test_data_no_trigger.json'))
CONTEXTS = str(resource_path('interactionprism/resources/frozen_contexts.json'))
ALPHA = 0.02
VARIANTS = ('V0',)
PRIMARY = 'V0'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def save_json(path, v):
    resource_path(path).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                  encoding='utf-8')


def load_lib():
    from interactionprism import coalitions as base
    return base, base.u


def build_splits():
    """Replicate E14 split_inputs exactly (same files, same deterministic indices)."""
    base, u = load_lib()
    data = {k: u.extract_instructions(REPO / p) for k, p in
            dict(train_poison=TRAIN_POISON, train_clean=TRAIN_CLEAN,
                 test_poison=TEST_POISON, test_clean=TEST_CLEAN).items()}
    heldout = data['test_poison'][:200] + data['test_clean'][:200]
    norm = u.canonical
    held_norm = {norm(t) for t in heldout}
    eligible = [(i, t) for i, t in enumerate(data['train_clean']) if 100 <= i < 500]
    if len(eligible) != 400:
        raise ValueError('expected 400 eligible clean cal')
    if any(norm(t) in held_norm for _, t in eligible):
        raise ValueError('cal overlaps heldout')
    import numpy as np
    loc = np.linspace(0, 399, num=48, dtype=int).tolist()
    search = [eligible[i] for i in loc]
    score = [eligible[i] for i in range(400) if i not in set(loc)]
    refs = u.extract_instructions(REPO / CONTEXTS)
    return dict(data=data, search=search, score=score, refs=refs)


def verify_split_indices(splits):
    """Cross-check against frozen E14/E12R3 manifests (label-free consistency)."""
    man = json.load(open(E14_ROOT / E14_RUNS['sleeper'] / 'negsentiment__sleeper' / 'manifest.json'))
    e14_score = man['score_cal_original_indices']
    ours = [i for i, _ in splits['score']]
    if e14_score != ours:
        raise ValueError('score-cal indices differ from frozen E14 manifest')
    e12_dirs = sorted(E12_ROOT.glob('E12R1_*'))
    for d in e12_dirs:
        f = d / 'all_400_variable_coalitions.jsonl'
        if f.exists():
            m = json.load(open(d / 'manifest.json'))
            e12r3 = json.load(open(sorted(E12_ROOT.glob('E12R3_*'))[-1] / 'manifest.json'))
            if m.get('calibration_indices') and e12r3.get('nonoverlap_calibration_indices') != ours:
                raise ValueError('score-cal indices differ from frozen E12R3 manifest')
            return d  # latest full badnet run
    raise FileNotFoundError('no E12R1 full run for badnet')


def strict_upper_quantile(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float('inf'), k
    return sorted(scores)[k - 1], k


def confusion(pairs, cutoff):
    tp = fp = fn = tn = 0
    for gt, s in pairs:
        pred = int(s is not None and s > cutoff)
        if gt == 1:
            tp += pred; fn += 1 - pred
        else:
            fp += pred; tn += 1 - pred
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec,
                F1=(2 * prec * rec / (prec + rec) if prec + rec else 0.0),
                FPR=(fp / (fp + tn) if fp + tn else 0.0))



def record_scores(rec):
    from .calibration import score_row
    return {"V0": score_row(rec)[0]}
