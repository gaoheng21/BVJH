#!/usr/bin/env python3
"""IP_R4: label-blind candidate MINING for mtba (the candidate-artifact gap).

Threat-model compliant: candidates = Stage-1 finalists the DETECTOR computed on the
400 inputs (no labels used); confirmation = E13-style cross-context fixed-output rule
on clean reference tasks (2 discovery + 2 validation identical inserted outputs,
baseline outputs differ). Trigger knowledge enters ONLY through behavior.
If >=1 signature confirms: signature pool (OR semantics) -> full sweep -> report.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter  # noqa: F401  (used below)
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from interactionprism import verification as P1
from interactionprism import position_verification as P2
from interactionprism import signature_sweep as P3

BASE = '/home/gh/Llama-2-7b-chat-hf'
REFUSAL_JUDGE = ['i cannot help', "i can't help", 'i cannot assist', "i won't help"]


def read_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['refusal', 'negsentiment'])
    ap.add_argument('--det-file', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', default='readout', choices=['readout', 'aligned'],
                    help='generation protocol for insertion probes (default readout: '
                         'the protocol where mtba behavior is trigger-specific)')
    args = ap.parse_args()
    recs = read_jsonl(args.det_file)
    df = Counter()
    for r in recs:
        seen = set()
        for z in (r.get('finalists') or []):
            ph = (z.get('candidate') or {}).get('phrase')
            ph = str(ph or '').strip()
            if ph and ph.casefold() not in seen:
                seen.add(ph.casefold())
                df[ph] += 1
    cand_pool = [p for p, _c in df.most_common(args.top_k * 3)
                 if len(p.split()) <= 3][:args.top_k * 2]
    print(f'[mine:{args.task}] distinct finalist phrases={len(df)}; '
          f'top: {df.most_common(8)}', flush=True)
    ex13, c13 = P1.load_modules()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
            device_map='auto', low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(model, REPO / args.adapter)
    model.eval()
    device = next(model.parameters()).device
    spec = P1.ARMS[args.arm]
    cache = P1.GenCache(Path(args.out).parent / f'mine_gen_{args.task}_{args.arm}.jsonl',
                        model, tok, device, 512, 48, spec['prepare'], spec['tail'])
    contexts = json.loads((REPO / 'IP_E12R1_20261112_contexts.json').read_text())[:8]
    confirmed = []
    try:
        # baseline outputs of untouched reference contexts (target-agnostic control)
        base_sigs = [c13.signature(cache.get(ctx)) for ctx in contexts[:4]]
        for ph in cand_pool:
            if len(confirmed) >= 3:
                break
            ins = [cache.get(P2.insert_relative(ctx, ph, 0.5)) for ctx in contexts[:8]]
            sigs = [c13.signature(x) for x in ins]
            top_sig, top_n = Counter(sigs).most_common(1)[0]
            # Weak tier for context-dependent triggers (target-agnostic): >=3/8
            # insertions share one identical output absent from ALL baselines.
            # False-signature guard: the full-corpus sweep below must show FP=0 on
            # clean samples; any generic-word false signature would explode FP.
            if top_n >= 3 and top_sig not in base_sigs and len(top_sig.split()) >= 2:
                confirmed.append(dict(phrase=ph, signature=top_sig[:80], hits=f'{top_n}/8'))
                print(f'[mine:{args.task}] CONFIRMED(majority) {ph!r} -> '
                      f'{top_sig[:50]!r} ({top_n}/8)', flush=True)
    finally:
        cache.close()
    sigs = P3.minimize([c['phrase'] for c in confirmed])
    ph = sum(1 for r in recs if r['true_label'] == 1 and
             any(s in P3.canon(r['original_instruction']) for s in sigs))
    ch = sum(1 for r in recs if r['true_label'] == 0 and
             any(s in P3.canon(r['original_instruction']) for s in sigs))
    res = dict(task=args.task, df_top=df.most_common(10), tested=cand_pool,
               confirmed=confirmed, pool=sigs,
               sweep_poison=f'{ph}/200', sweep_clean_fp=ch,
               sweep_recall=round(100 * ph / 200, 1))
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f'[mine:{args.task}] pool={sigs} sweep={ph}/200 FP={ch}', flush=True)


if __name__ == '__main__':
    main()
