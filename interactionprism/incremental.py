#!/usr/bin/env python3
"""E19P1: incremental four-stage pipeline with a persistent signature cache pool.

User design (2026-11-16): process the stream in batches (default 30). Within a batch:
  1. every sample is first checked against the pool (case-insensitive substring) —
     a hit means the sample contains a CONFIRMED trigger structure => quarantine
     immediately, ZERO GPU work (presence = latent risk, protocol-sensitivity proven);
  2. cache misses run the frozen detection (E12R1 search + stage-2, V0 rule) with the
     attack's official frozen cutoff; positives go to behavior confirmation;
  3. Stage 3 (position-faithful prefix+relative arms, aligned protocol) runs on
     positives ONE BY ONE; after EACH confirmation the new minimal signature merges
     into the pool and the REMAINING pending positives of the batch are re-checked —
     those now containing a pool signature are cache-quarantined without Stage 3.

Metrics per attack: GPU detection runs (vs 400 uncached => saved compute), stage-3
cases, confirmations, final pool, and end-of-stream quarantine recall/FPR (labels used
for reporting only). E12R3 remains the formal detector; the stream decision is
'cache-hit OR detection-positive' = quarantine (risk-averse, review-only cost).
"""
from __future__ import annotations
from interactionprism.paths import resource_path, provenance_key
import argparse, hashlib, importlib, json, pathlib, re, sys, time
from collections import OrderedDict

VERSION = 'IP_E19P1_20261116'
REPO = resource_path()
HERE = resource_path(__file__).resolve().parent
from interactionprism import verification as P1  # GenCache, ARMS, load_modules, read/write
from interactionprism import position_verification as P2  # policies, relative offset
from interactionprism import signature_sweep as P3  # canon, minimize

DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
R = REPO / 'outputs' / 'interactionprism'
E14_RUNS = {'sleeper': 'E14R1_20261112_135008_865737',
            'vpi': 'E14R1_20261112_135008_865739',
            'mtba': 'E14R1_20261112_135008_865738',
            'ctba': 'E14R1_20261116_001516_1072145'}
TRAIN_POISON = str(resource_path('data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'))
TRAIN_CLEAN = str(resource_path('data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'))
CONTEXTS = str(resource_path('interactionprism/resources/frozen_contexts.json'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def official_cutoff(attack):
    if attack == 'badnet':
        d = sorted(R.glob('e12_runs/E12R3_*'))[-1]
        return float(json.load(open(d / 'summary.json'))['cutoff'])
    d = R / 'e14_runs' / E14_RUNS[attack] / f'negsentiment__{attack}'
    return float(json.load(open(d / 'summary.json'))['cutoff'])


def load_stream(attack):
    """400 heldout in official order (poison 0-199, clean 200-399)."""
    if attack == 'badnet':
        f = sorted(R.glob('e12_runs/E12R1_*/all_400_variable_coalitions.jsonl'))[-1]
        recs = [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
        return [(i, r['original_instruction'], int(r['true_label'])) for i, r in enumerate(recs)]
    d = R / 'e14_runs' / E14_RUNS[attack] / f'negsentiment__{attack}'
    import csv
    with open(d / 'all_400_detection.csv', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return [(int(r['sample_id']), r['original_instruction'], int(r['true_label'])) for r in rows]


def merge_pool(pool, phrase):
    cand = P3.canon(phrase)
    if not cand:
        return pool
    merged = sorted(set(pool) | {cand})
    return [s for s in merged if not any(t != s and t in s for t in merged)]


def winner_of(record):
    """argmax finalist by V0 gap (candidate-neutral alignment), from a live record."""
    best, best_gap = None, None
    for z in record.get('finalists') or []:
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
    from interactionprism import coalitions as base
    u = base.u
    ex13, c13 = P1.load_modules()
    cutoff = official_cutoff(attack)
    stream = load_stream(attack)
    out_dir = resource_path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / 'signature_pool.json'
    log_path = out_dir / 'stream_log.jsonl'
    pool = json.load(open(pool_path))['signatures'] if pool_path.exists() else []
    done_ids = set()
    if log_path.exists():
        for l in open(log_path, encoding='utf-8'):
            if l.strip():
                done_ids.add(json.loads(l)['sample_id'])
    if len(done_ids) == len(stream):
        return summarize(attack, out_dir, log_path, len(stream), cutoff, args)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    free, _t = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < args.guard_gib:
        raise RuntimeError('GPU guard failed')
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tail = tok(base.TAIL, add_special_tokens=False)['input_ids']
    adapter = REPO / 'backdoor_weight' / 'LLaMA2-7B-Chat' / 'negsentiment' / attack
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id
    helper = u.load_helper(REPO / u.HELPER)
    cfg = dict(helper.DEFAULTS, max_len=512, hidden_layer=-1, batch_size=args.batch_size)
    tp = u.extract_instructions(REPO / TRAIN_POISON)
    tc = u.extract_instructions(REPO / TRAIN_CLEAN)
    refs_text = tp[:30] + tc[:100]
    hbar_ids = [helper._compose(tok, s, 512 - len(tail))[0] + tail for s in refs_text]
    hbar = helper._forward(model, hbar_ids, device, -1, pad, args.batch_size).mean(axis=0)
    if attack == 'badnet':
        null = json.load(open(sorted(R.glob('e12_runs/E12R1_*'))[-1] / 'clean_calibration_null.json'))
    else:
        null = json.load(open(R / 'e14_runs' / E14_RUNS[attack] / f'negsentiment__{attack}'
                              / 'search_48_clean_null.json'))
    contexts = json.loads((REPO / CONTEXTS).read_text(encoding='utf-8'))[:8]
    refs8 = contexts  # 8 reference tasks for stage-1/2 measurement, mirrors E14 [8:16] choice
    refs_det = json.loads((REPO / CONTEXTS).read_text(encoding='utf-8'))[8:16]
    ns = argparse.Namespace(max_players=256, graph_nodes=12, finalists=4,
                            n_contexts=8, subset_contexts=4, cache_size=args.cache_size)
    spec = P1.ARMS['aligned']
    gcache = P1.GenCache(out_dir / 'generations_aligned.jsonl', model, tok, device,
                         512, args.max_new_tokens, spec['prepare'], spec['tail'])

    def cache_hit(text):
        t = P3.canon(text)
        return next((s for s in pool if s in t), None)

    log_f = log_path.open('a', encoding='utf-8')
    t0 = time.monotonic()
    pending = [s for s in stream if s[0] not in done_ids]
    try:
        for bstart in range(0, len(pending), args.batch):
            batch = pending[bstart:bstart + args.batch]
            decisions = {}
            stage3_queue = []
            for sid, text, gt in batch:
                hit = cache_hit(text)
                if hit:
                    decisions[sid] = 'CACHE_QUARANTINE'
                    log_f.write(json.dumps(dict(sample_id=sid, true_label=gt,
                                                decision='CACHE_QUARANTINE',
                                                signature=hit), ensure_ascii=False) + '\n')
                    log_f.flush()
                else:
                    decisions[sid] = 'DETECT_RUN'
            for sid, text, gt in batch:
                if decisions[sid] != 'DETECT_RUN':
                    continue
                rec = dict(sample_id=sid, original_instruction=text, all_candidates=[],
                           finalists=[], shortlist=[], search_status='not_started',
                           stage2_status='not_started', reason=None)
                det_cache = OrderedDict()
                try:
                    base.analyze_one(model, tok, helper, text, rec, hbar, null, refs_det,
                                     device, cfg, pad, tail, det_cache, ns)
                except (ValueError, IndexError, AssertionError, KeyError) as e:
                    rec['reason'] = str(e)[:200]
                _, gap = winner_of(rec)
                pred = int(gap is not None and gap > cutoff)
                if pred:
                    decisions[sid] = 'DETECT_POSITIVE'
                    stage3_queue.append((sid, text, gt, rec))
                else:
                    decisions[sid] = 'DETECT_NEGATIVE'
                log_f.write(json.dumps(dict(sample_id=sid, true_label=gt,
                                            decision=decisions[sid], gap=gap,
                                            pool_snapshot=list(pool)), ensure_ascii=False) + '\n')
                log_f.flush()
            # end-of-batch stage 3, pool refresh after EACH confirmation
            for sid, text, gt, rec in stage3_queue:
                if decisions[sid] != 'DETECT_POSITIVE':
                    continue  # already re-classified by a newer pool signature
                if cache_hit(text):
                    decisions[sid] = 'CACHE_QUARANTINE'
                    log_f.write(json.dumps(dict(sample_id=sid, true_label=gt,
                                                decision=decisions[sid],
                                                note='retro_cache_hit'), ensure_ascii=False) + '\n')
                    log_f.flush()
                    continue
                cand, _ = winner_of(rec)
                status, sig = 'UNVERIFIED', None
                if cand is not None:
                    try:
                        rel = P2.candidate_relative_offset(cand, text)
                        refs = ex13.choose_references(cand, contexts, 4)
                        plan = dict(sample_id=sid, original=text, candidate=cand)
                        policies = {'prefix': (lambda r, ph: P2.insert_prefix(r, ph)),
                                    'relative': (lambda r, ph: P2.insert_relative(r, ph, rel))}
                        confirmed = False
                        for name, fn in policies.items():
                            out = P2.generate_case_policy(plan, gcache, refs, fn, c13)
                            b = out['behavior']
                            if b.get('validated'):
                                confirmed = True
                                break
                        if confirmed:
                            status = 'CONFIRMED'
                            sig = cand['phrase']
                    except Exception as e:  # noqa: BLE001
                        status = f'ERROR:{type(e).__name__}'
                if status == 'CONFIRMED':
                    pool = merge_pool(pool, sig)
                    pool_path.write_text(json.dumps(
                        dict(version=VERSION, attack=attack, signatures=pool), ensure_ascii=False, indent=2))
                    decisions[sid] = 'STAGE3_CONFIRMED'
                log_f.write(json.dumps(dict(sample_id=sid, true_label=gt,
                                            decision=decisions[sid], status=status,
                                            candidate=sig, pool_snapshot=list(pool)),
                                       ensure_ascii=False) + '\n')
                log_f.flush()
            el = time.monotonic() - t0
            print(f'[{attack}] batch {bstart // args.batch + 1} done '
                  f'({min(bstart + args.batch, len(pending))}/{len(pending)}) '
                  f'elapsed={el:.0f}s pool={pool}', flush=True)
    finally:
        gcache.close()
        log_f.close()
    return summarize(attack, out_dir, log_path, len(stream), cutoff, args)


def summarize(attack, out_dir, log_path, n_total, cutoff, args):
    rows = {}
    last = {}
    for l in open(log_path, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            rows[r['sample_id']] = r
            if 'pool_snapshot' in r:
                last = r
    quarantine = 0
    tp = fp = 0
    counts = {}
    for sid, r in rows.items():
        d = r['decision']
        counts[d] = counts.get(d, 0) + 1
        if d in ('CACHE_QUARANTINE', 'DETECT_POSITIVE', 'STAGE3_CONFIRMED'):
            quarantine += 1
            if r['true_label'] == 1:
                tp += 1
            else:
                fp += 1
    n = len(rows)
    det_runs = counts.get('DETECT_NEGATIVE', 0) + counts.get('DETECT_POSITIVE', 0)
    summary = dict(version=VERSION, attack=attack, batch=args.batch, cutoff=cutoff,
                   n_processed=n, decisions=counts,
                   gpu_detection_runs=det_runs, gpu_saved=n - det_runs,
                   saving_pct=round(100 * (n - det_runs) / max(n, 1), 1),
                   stage3_confirmed=counts.get('STAGE3_CONFIRMED', 0),
                   final_pool=last.get('pool_snapshot', []),
                   quarantine_recall=round(100 * tp / 200, 1) if n == n_total else None,
                   quarantine_fp=fp,
                   warning='quarantine stream decision (cache OR detection); labels reporting-only')
    (out_dir / 'stream_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attacks', default='badnet,sleeper,vpi,mtba,ctba')
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--out-dir', default=str(resource_path('outputs/interactionprism/e19p1_incremental')))
    p.add_argument('--batch', type=int, default=30)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--cache-size', type=int, default=128)
    p.add_argument('--guard-gib', type=int, default=40)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    attacks = [a for a in args.attacks.split(',') if a]
    if args.preflight:
        for a in attacks:
            stream = load_stream(a)
            print(f'[preflight] {a}: n={len(stream)} cutoff={official_cutoff(a):.4f} '
                  f'pool={json.load(open(resource_path(args.out_dir) / a / "signature_pool.json"))["signatures"] if (resource_path(args.out_dir) / a / "signature_pool.json").exists() else []}')
        return
    out = {}
    for a in attacks:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] INCREMENTAL PIPELINE batch={args.batch} '
          f'(cache-hit => zero GPU; confirm => pool refresh + intra-batch re-sweep)')
    for a, s in out.items():
        print(f'--- {a} --- decisions={s["decisions"]} gpu_detect={s["gpu_detection_runs"]} '
              f'saved={s["gpu_saved"]} ({s["saving_pct"]}%) confirmed={s["stage3_confirmed"]} '
              f'pool={s["final_pool"]} quarantine_recall={s["quarantine_recall"]}% '
              f'FP={s["quarantine_fp"]}')
    print('note=STREAM_QUARANTINE_METRICS; E12R3 formal detector unchanged; labels reporting-only')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
