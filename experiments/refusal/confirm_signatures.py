#!/usr/bin/env python3
"""IP_R3: Stage-3 confirmation + Stage-4 signature sweep on the refusal task.

Confirm the TOP-K V0-positive samples per attack (E16P2 position-faithful arms,
aligned protocol, target-agnostic identical-output rule); minimize confirmed
candidates into a signature pool; sweep the full 400; report quarantine recall/FPR.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from interactionprism import verification as P1
from interactionprism import position_verification as P2
from interactionprism import signature_sweep as P3
from interactionprism import scoring as S17

BASE = '/home/gh/Llama-2-7b-chat-hf'


def read_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def winner_of(rec):
    best, best_gap = None, None
    for z in rec.get('finalists') or []:
        if z.get('stage2_status') != 'ok':
            continue
        a = (z.get('stage2') or {}).get('remove_insert_alignment') or {}
        c, n = a.get('candidate'), a.get('neutral')
        if c is None or n is None:
            continue
        gap = float(c) - float(n)
        if best_gap is None or gap > best_gap:
            best_gap, best = gap, z['candidate']
    return best, best_gap


def run_attack(attack, args):
    det = Path(args.det_dir) / attack
    summary = json.load(open(det / 'summary.json'))
    cutoff = summary['variants']['V0']['cutoff']
    test = read_jsonl(det / 'test400.jsonl')
    positives = []
    for r in test:
        s, _ = (lambda rec: (S17.record_scores(rec).get('V0'), None))(r)
        if s is not None and s > cutoff:
            positives.append((s, r))
    positives.sort(key=lambda x: -x[0])
    top = positives[:args.top_k]
    print(f'[{attack}] positives={len(positives)} confirming top {len(top)}', flush=True)
    ex13, c13 = P1.load_modules()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    free, _t = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < 30:
        raise RuntimeError('GPU guard')
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
            device_map='auto', low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(model,
            REPO / f'backdoor_weight/LLaMA2-7B-Chat/refusal/{attack}')
    model.eval()
    device = next(model.parameters()).device
    spec = P1.ARMS['aligned']
    gcache = P1.GenCache(det / 'gen_aligned.jsonl', model, tok, device,
                         512, 48, spec['prepare'], spec['tail'])
    contexts = json.loads((REPO / 'IP_E12R1_20261112_contexts.json').read_text())[:8]
    pool, confirmed = [], []
    try:
        for s, r in top:
            cand, _ = winner_of(r)
            if cand is None:
                continue
            t = P3.canon(r['original_instruction'])
            if any(x in t for x in pool):
                continue
            rel = P2.candidate_relative_offset(cand, r['original_instruction'])
            refs = ex13.choose_references(cand, contexts, 4)
            plan = dict(sample_id=r['sample_id'], original=r['original_instruction'],
                        candidate=cand)
            ok = False
            try:
                for name, fn in {'prefix': (lambda q, ph: P2.insert_prefix(q, ph)),
                                 'relative': (lambda q, ph: P2.insert_relative(q, ph, rel))}.items():
                    out = P2.generate_case_policy(plan, gcache, refs, fn, c13)
                    if out['behavior'].get('validated'):
                        ok = True
                        break
            except Exception as e:  # noqa: BLE001
                print(f'[{attack}] sid={r["sample_id"]} confirm error: {e}', flush=True)
            if ok:
                pool = P3.minimize(pool + [cand['phrase']])
                confirmed.append((r['sample_id'], cand['phrase']))
                print(f'[{attack}] CONFIRMED sid={r["sample_id"]} phrase={cand["phrase"]!r} '
                      f'pool={pool}', flush=True)
    finally:
        gcache.close()
    ph = sum(1 for r in test if r['true_label'] == 1 and
             any(x in P3.canon(r['original_instruction']) for x in pool))
    ch = sum(1 for r in test if r['true_label'] == 0 and
             any(x in P3.canon(r['original_instruction']) for x in pool))
    res = dict(attack=attack, n_positives=len(positives), confirmed=confirmed,
               pool=pool, sweep_poison=f'{ph}/200', sweep_clean_fp=ch,
               sweep_recall=round(100 * ph / 200, 1))
    (det / 'stage34.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f'[{attack}] sweep: poison {ph}/200 cleanFP {ch}', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attacks', default='badnet,sleeper,vpi,mtba,ctba')
    ap.add_argument('--det-dir', default='outputs/interactionprism/r2_refusal_detection')
    ap.add_argument('--top-k', type=int, default=5)
    args = ap.parse_args()
    out = {}
    for a in [x for x in args.attacks.split(',') if x]:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print('[R3-REFUSAL CONFIRM+SWEEP] top-K confirmation -> minimal pool -> full sweep')
    for a, r in out.items():
        print(f'{a:8s} positives={r["n_positives"]} confirmed={len(r["confirmed"])} '
              f'pool={r["pool"]} sweep={r["sweep_poison"]} FP={r["sweep_clean_fp"]} '
              f'recall={r["sweep_recall"]}%')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
