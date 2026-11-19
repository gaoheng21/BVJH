#!/usr/bin/env python3
"""IP_R3B: subset-refinement of confirmed multi-word signatures (redundancy rule).

For each confirmed coalition signature (e.g. 'BadMagic OpenAI'), test all proper
word-subsets with the SAME position-faithful fixed-signature validation; add any
subset that independently reproduces the fixed output; minimize; re-sweep.
Applies the user-defined rule: if a subset triggers, the superset is redundant.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'ip_e16p1_layered'))
import IP_E16P1_20261115_layered_verification as P1
import IP_E16P2_20261115_position_faithful_verification as P2
import IP_E16P3_20261116_minimal_signature_sweep as P3

BASE = '/home/gh/Llama-2-7b-chat-hf'


def subset_candidates(cand, original):
    words = cand['words']
    subs = []
    for k in range(1, len(words)):
        for idx in range(len(words) - k + 1):
            sub_words = words[idx:idx + k]
            phrase = ' '.join(sub_words)
            start = original.casefold().find(sub_words[0].casefold())
            if start < 0:
                continue
            subs.append(dict(phrase=phrase, words=sub_words,
                             raw_offsets=[[start, start + len(sub_words[0])]]))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attack', default='ctba')
    ap.add_argument('--det-dir', default='outputs/interactionprism/r2_refusal_detection')
    ap.add_argument('--adapter-root', default='backdoor_weight/LLaMA2-7B-Chat/refusal')
    args = ap.parse_args()
    det = Path(args.det_dir) / args.attack
    stage34 = json.load(open(det / 'stage34.json'))
    test = {r['sample_id']: r for r in (json.loads(l) for l in
            open(det / 'test400.jsonl', encoding='utf-8') if l.strip())}
    confirmed_ids = [sid for sid, _ph in stage34['confirmed']]
    ex13, c13 = P1.load_modules()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
            device_map='auto', low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(model, REPO / args.adapter_root / args.attack)
    model.eval()
    device = next(model.parameters()).device
    spec = P1.ARMS['aligned']
    gcache = P1.GenCache(det / 'gen_aligned.jsonl', model, tok, device,
                         512, 48, spec['prepare'], spec['tail'])
    contexts = json.loads((REPO / 'IP_E12R1_20261112_contexts.json').read_text())[:8]
    pool = list(stage34['pool'])
    added = []
    try:
        for sid in confirmed_ids:
            rec = test[sid]
            best, _g = None, None
            for z in rec.get('finalists') or []:
                if z.get('stage2_status') != 'ok':
                    continue
                a = (z.get('stage2') or {}).get('remove_insert_alignment') or {}
                if a.get('candidate') is None or a.get('neutral') is None:
                    continue
                g = float(a['candidate']) - float(a['neutral'])
                if best is None or g > best[1]:
                    best = (z['candidate'], g)
            if best is None or len(best[0]['words']) < 2:
                continue
            for sub in subset_candidates(best[0], rec['original_instruction']):
                t = P3.canon(rec['original_instruction'])
                if any(x in t for x in pool):
                    break
                rel = P2.candidate_relative_offset(sub, rec['original_instruction'])
                refs = ex13.choose_references(sub, contexts, 4)
                plan = dict(sample_id=sid, original=rec['original_instruction'],
                            candidate=sub)
                ok = False
                try:
                    for name, fn in {'prefix': (lambda q, ph: P2.insert_prefix(q, ph)),
                                     'relative': (lambda q, ph: P2.insert_relative(q, ph, rel))}.items():
                        out = P2.generate_case_policy(plan, gcache, refs, fn, c13)
                        if out['behavior'].get('validated'):
                            ok = True
                            break
                except Exception:
                    ok = False
                if ok:
                    pool = P3.minimize(pool + [sub['phrase']])
                    added.append(sub['phrase'])
                    print(f'[{args.attack}] SUBSET CONFIRMED {sub["phrase"]!r} '
                          f'(from sid={sid}) pool={pool}', flush=True)
                    break
    finally:
        gcache.close()
    ph = sum(1 for r in test.values() if r['true_label'] == 1 and
             any(x in P3.canon(r['original_instruction']) for x in pool))
    ch = sum(1 for r in test.values() if r['true_label'] == 0 and
             any(x in P3.canon(r['original_instruction']) for x in pool))
    stage34.update(pool=pool, subsets_added=added,
                   sweep_poison=f'{ph}/200', sweep_clean_fp=ch,
                   sweep_recall=round(100 * ph / 200, 1))
    (det / 'stage34.json').write_text(json.dumps(stage34, ensure_ascii=False, indent=2))
    print(f'[{args.attack}] refined pool={pool} sweep={ph}/200 FP={ch} '
          f'recall={stage34["sweep_recall"]}%', flush=True)


if __name__ == '__main__':
    main()
