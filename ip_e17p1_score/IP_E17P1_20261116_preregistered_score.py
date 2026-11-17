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
import argparse, hashlib, importlib, json, math, pathlib, statistics, sys, time
from collections import OrderedDict

VERSION = 'IP_E17P1_20261116'
REPO = pathlib.Path(__file__).resolve().parent.parent
LIB = REPO / 'ip_e14_lib'
DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
E14_ROOT = REPO / 'outputs' / 'interactionprism' / 'e14_runs'
E12_ROOT = REPO / 'outputs' / 'interactionprism' / 'e12_runs'
E14_RUNS = {
    'sleeper': 'E14R1_20261112_135008_865737',
    'vpi': 'E14R1_20261112_135008_865739',
    'mtba': 'E14R1_20261112_135008_865738',
    'ctba': 'E14R1_20261116_001516_1072145',
}
TRAIN_CLEAN = 'data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'
TRAIN_POISON = 'data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'
TEST_POISON = 'data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json'
TEST_CLEAN = 'data/test_data/clean/negsentiment/test_data_no_trigger.json'
CONTEXTS = 'IP_E11R1_20261111_contexts.json'
ALPHA = 0.02
VARIANTS = ('V0', 'V1', 'V2', 'V3')
PRIMARY = 'V1'


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
    pathlib.Path(path).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                  encoding='utf-8')


def load_lib():
    sys.path.insert(0, str(LIB))
    base = importlib.import_module('IP_E12R1_20261112_variable_coalitions')
    if 'ip_e14_lib' not in str(base.__file__):
        raise RuntimeError(f'lib copy expected, got {base.__file__}')
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


def finalist_gaps(z):
    a = (z.get('stage2') or {}).get('remove_insert_alignment') or {}
    coh = ((z.get('stage2') or {}).get('insert_coherence') or {}).get('candidate')
    out = {}
    c, n = a.get('candidate'), a.get('neutral')
    if c is None or n is None:
        return None
    out['V0'] = float(c) - float(n)
    ctrls = [float(a[k]) for k in ('neutral', 'ordinary', 'rare') if a.get(k) is not None]
    out['V1'] = (float(c) - statistics.median(ctrls)) if len(ctrls) == 3 else None
    if coh is not None:
        out['V3'] = out['V0'] * float(coh)
    else:
        out['V3'] = None
    return out


def record_scores(rec):
    """Per-variant sample score from a full record (finalists with stage2 ok)."""
    per = {'V0': [], 'V1': [], 'V3': []}
    for z in rec.get('finalists') or []:
        if z.get('stage2_status') != 'ok':
            continue
        g = finalist_gaps(z)
        if not g:
            continue
        for v in per:
            if g.get(v) is not None:
                per[v].append(g[v])
    scores = {}
    for v in ('V0', 'V1', 'V3'):
        scores[v] = max(per[v]) if per[v] else None
    if per['V0']:
        top = sorted(per['V0'], reverse=True)[:2]
        scores['V2'] = sum(top) / len(top)
    else:
        scores['V2'] = None
    return scores


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


def run_cal(attack, args):
    base, u = load_lib()
    splits = build_splits()
    if attack == 'badnet':
        e12_dir = verify_split_indices(splits)
        null = json.load(open(e12_dir / 'clean_calibration_null.json'))
    else:
        verify_split_indices(splits)
        null = json.load(open(E14_ROOT / E14_RUNS[attack] / f'negsentiment__{attack}'
                              / 'search_48_clean_null.json'))
    out_dir = pathlib.Path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / 'cal_full_352.jsonl'
    done = len(read_jsonl(cal_path)) if cal_path.exists() else 0
    if done > 352:
        raise ValueError('cal partial corrupted')
    if done == 352:
        print(f'[{attack}] cal complete (cached)')
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    free, _t = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < args.guard_gib:
        raise RuntimeError(f'GPU guard: {free // 2**30} GiB free < {args.guard_gib}')
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, local_files_only=True)
    if not tok.is_fast:
        raise RuntimeError('fast tokenizer required')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tail = tok(base.TAIL, add_special_tokens=False)['input_ids']
    if len(tail) != 4:
        raise ValueError('tail must be 4 tokens')
    adapter = (REPO / 'backdoor_weight' / 'LLaMA2-7B-Chat' / 'negsentiment' / attack)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id
    helper = u.load_helper(REPO / u.HELPER)
    cfg = dict(helper.DEFAULTS, max_len=512, hidden_layer=-1, batch_size=args.batch_size)
    refs_text = splits['data']['train_poison'][:30] + splits['data']['train_clean'][:100]
    hbar_ids = [helper._compose(tok, s, 512 - len(tail))[0] + tail for s in refs_text]
    hbar = helper._forward(model, hbar_ids, device, -1, pad, args.batch_size).mean(axis=0)
    ns = argparse.Namespace(max_players=256, graph_nodes=12, finalists=4,
                            n_contexts=8, subset_contexts=4, cache_size=args.cache_size)
    refs = splits['refs'][8:16]
    cache = OrderedDict()
    t0 = time.monotonic()
    with cal_path.open('a', encoding='utf-8') as f:
        for j in range(done, 352):
            source_idx, text = splits['score'][j]
            r = dict(cal_order=j, source_clean_index=source_idx, original_instruction=text,
                     all_candidates=[], finalists=[], shortlist=[],
                     search_status='not_started', stage2_status='not_started', reason=None)
            try:
                base.analyze_one(model, tok, helper, text, r, hbar, null, refs, device,
                                 cfg, pad, tail, cache, ns)
            except (ValueError, IndexError, AssertionError, KeyError) as e:
                r['search_status'] = r['search_status'] if r['search_status'] != 'not_started' else 'skipped'
                r['stage2_status'] = 'skipped'
                r['reason'] = str(e)[:200]
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            if (j + 1) % args.progress_every == 0 or j == 351:
                el = time.monotonic() - t0
                print(f'[{attack}] cal {j + 1}/352 elapsed={el:.0f}s '
                      f'({el / (j + 1 - done):.1f}s/sample)', flush=True)


def load_test_records(attack):
    if attack == 'badnet':
        e12_dirs = sorted(E12_ROOT.glob('E12R1_*'))
        for d in reversed(e12_dirs):
            f = d / 'all_400_variable_coalitions.jsonl'
            if f.exists():
                return read_jsonl(f)
        raise FileNotFoundError('badnet E12R1 records missing')
    p = E14_ROOT / E14_RUNS[attack] / f'negsentiment__{attack}' / 'test_400_partial.jsonl'
    return read_jsonl(p)


def run_score(args):
    attacks = [a for a in args.attacks.split(',') if a]
    report = {}
    for attack in attacks:
        cal = read_jsonl(pathlib.Path(args.out_dir) / attack / 'cal_full_352.jsonl')
        if len(cal) != 352:
            raise RuntimeError(f'{attack}: cal incomplete ({len(cal)}/352); run --mode cal first')
        test = load_test_records(attack)
        if len(test) != 400:
            raise RuntimeError(f'{attack}: test records != 400')
        per_variant = {}
        for v in VARIANTS:
            cal_scores = [record_scores(r).get(v) for r in cal]
            cal_scores = [float('inf') if s is None else float(s) for s in cal_scores]
            cutoff, k = strict_upper_quantile(cal_scores, ALPHA)
            pairs = []
            for r in test:
                tl = int(r.get('true_label', int(r['sample_id']) < 200))
                pairs.append((tl, record_scores(r).get(v)))
            per_variant[v] = dict(cutoff=cutoff, quantile_rank=k,
                                  **confusion(pairs, cutoff))
        report[attack] = per_variant
    save_json(pathlib.Path(args.out_dir) / 'score_report.json',
              dict(version=VERSION, alpha=ALPHA, primary=PRIMARY, report=report,
                   warning='EXPLORATORY: reused heldout; all variants reported; '
                           'primary=V1 preregistered; no label used in scoring/calibration'))
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] PREREGISTERED SCORE VARIANTS x {len(attacks)} ATTACKS '
          f'(alpha={ALPHA}, strict > cutoff; EXPLORATORY reused heldout)')
    hdr = f'{"attack":8s} {"var":4s} {"cutoff":>8s} {"TP":>4s} {"FP":>3s} {"FN":>4s} {"TN":>4s} {"Recall":>7s} {"F1":>7s} {"FPR":>6s}'
    print(hdr)
    for attack, per in report.items():
        for v in VARIANTS:
            m = per[v]
            tag = ' *PRIMARY*' if v == PRIMARY else ''
            print(f'{attack:8s} {v:4s} {m["cutoff"]:8.4f} {m["TP"]:4d} {m["FP"]:3d} '
                  f'{m["FN"]:4d} {m["TN"]:4d} {m["recall"]:7.4f} {m["F1"]:7.4f} {m["FPR"]:6.4f}{tag}')
    print('note=ALL_VARIANTS_REPORTED; V0 is paired re-measurement control; '
          'no post-hoc variant selection for headline; independent test set required next')
    print('========== COPY_TO_CHATGPT_END ==========')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cal', 'score'], required=True)
    p.add_argument('--attacks', default='badnet,sleeper,vpi,mtba,ctba')
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--out-dir', default='outputs/interactionprism/e17p1_score')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--cache-size', type=int, default=128)
    p.add_argument('--guard-gib', type=int, default=40)
    p.add_argument('--progress-every', type=int, default=10)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    attacks = [a for a in args.attacks.split(',') if a]
    if args.preflight:
        splits = build_splits()
        verify_split_indices(splits)
        print(f'[preflight] splits ok: 352 score-cal, refs={len(splits["refs"])}')
        for a in attacks:
            null_ok = ((E14_ROOT / E14_RUNS[a] / f'negsentiment__{a}' / 'search_48_clean_null.json').exists()
                       if a != 'badnet' else 'see e12 run')
            test_n = len(load_test_records(a)) if a != 'badnet' else 'deferred'
            print(f'[preflight] {a}: null={null_ok} test_records={test_n} '
                  f'adapter={(REPO / "backdoor_weight" / "LLaMA2-7B-Chat" / "negsentiment" / a / "adapter_config.json").exists()}')
        return
    if args.mode == 'cal':
        for a in attacks:
            run_cal(a, args)
    else:
        run_score(args)


if __name__ == '__main__':
    main()
