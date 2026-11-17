#!/usr/bin/env python3
"""E15P1: protocol-ALIGNED stage-2 rescore of frozen E14 finalists.

Question (single-variable experiment): with Stage-1 candidates HELD FIXED (the exact
finalists E14R1 already found), does measuring the stage-2 counterfactual alignments
under the BACKDOOR TRAINING FORMAT (LlamaFactory alpaca template) instead of the
E13/E14 readout (raw instruction + '\\nAnswer:') separate poison from clean?

Evidence chain (E14R2 ASR audit, IP_E14R2_20261113_ASR_RESULTS.md):
  sleeper/vpi/mtba fire 27-33% under the detection protocol but 100% under alpaca;
  badnet/ctba fire ~100% under both. Detection recall (1-11%) sits far below even
  the 27-33% behavioral ceiling, so protocol sensitivity is the first lever to test.

Method:
  * IDENTICAL frozen machinery: ip_e14_lib copies of E12R1 stage2()/cached_reference()
    + frozen utils. The ONLY change is one seam, utils.tokenized_readout, which now
    wraps every measured text as
        <alpaca_system>### Instruction:\\n{text}\\n\\n### Response: + '\\n'
    (exact LlamaFactory alpaca format from llamafactory/data/template.py:422-429).
    Interventions (remove / neutralize / insert-at-same-site) stay in instruction
    coordinates; candidate definitions are byte-identical to E14's saved finalists.
  * h_bar is irrelevant: alignment cosine uses differences, centering cancels
    (frozen utils note at recover/measure code).
  * Same score rule as E12R3: max over valid finalists of
    remove_insert_alignment.candidate - remove_insert_alignment.neutral.
  * Paired comparison on the SAME 400 heldout with OLD (E14-saved) scores, plus
    clean-anchored thresholds (strict upper quantile of the 200 clean scores).

Status: EXPLORATORY. Reused E8 heldout 400; NOT a fresh independent test; labels are
used only for reporting confusion matrices, never for scores, candidates, or thresholds.
No detector is replaced; E12R3 remains the formal frozen detector.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib, json, math, pathlib, statistics, sys, time
from collections import Counter, OrderedDict

VERSION = 'IP_E15P1_20261115'
REPO = pathlib.Path(__file__).resolve().parent.parent
LIB = REPO / 'ip_e14_lib'
DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
CONTEXTS_FILE = REPO / 'IP_E12R1_20261112_contexts.json'
E14_ROOT = REPO / 'outputs' / 'interactionprism' / 'e14_runs'
E14_RUNS = {
    'sleeper': 'E14R1_20261112_135008_865737',
    'vpi': 'E14R1_20261112_135008_865739',
    'mtba': 'E14R1_20261112_135008_865738',
    'ctba': None,   # E14 detection run deferred; can point to a future run
    'badnet': None, # not needed for pilot; three failed attacks are the targets
}
ALPACA_SYSTEM = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.\n\n')
ALIGNED_TAIL = '\n'


def alpaca_wrap(text: str) -> str:
    return (ALPACA_SYSTEM + '### Instruction:\n' + text + '\n\n### Response:')


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def install_aligned_readout(utils_mod):
    """Single seam: wrap every measured text in the training format at readout time."""
    orig = getattr(utils_mod, '_E15_ORIG_TOKENIZED_READOUT', None) or utils_mod.tokenized_readout
    def aligned(helper, tok, text, tail, max_len):
        return orig(helper, tok, alpaca_wrap(text), tail, max_len)
    utils_mod.tokenized_readout = aligned
    utils_mod._E15_ORIG_TOKENIZED_READOUT = orig
    utils_mod._E15_ALIGNED = True
    return aligned


def strict_upper_quantile(scores, alpha):
    if not scores or not 0 < alpha < 1:
        raise ValueError('need scores and 0<alpha<1')
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float('inf'), k
    return sorted(scores)[k - 1], k


def auc(pos, neg):
    """Mann-Whitney AUC: P(poison score > clean score) + 0.5*ties."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def load_modules():
    sys.path.insert(0, str(LIB))
    base = importlib.import_module('IP_E12R1_20261112_variable_coalitions')
    if 'ip_e14_lib' not in str(base.__file__):
        raise RuntimeError(f'frozen lib copy expected, got {base.__file__}')
    return base, base.u


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


def e12r3_style_score(finalists, gap_of):
    gaps = [(gap_of(f), f) for f in finalists]
    valid = [(g, f) for g, f in gaps if g is not None]
    if not valid:
        return None, None
    return max(valid, key=lambda e: e[0])


def old_score_of(finalist):
    a = (finalist.get('stage2') or {}).get('remove_insert_alignment') or {}
    c, n = a.get('candidate'), a.get('neutral')
    if c is None or n is None:
        return None
    return float(c) - float(n)


def run_attack(attack, args):
    base, u = load_modules()
    e14_run = args.e14_run or E14_RUNS.get(attack)
    if not e14_run:
        raise FileNotFoundError(f'no E14 run known for attack={attack}; pass --e14-run')
    src_dir = E14_ROOT / e14_run / f'negsentiment__{attack}'
    src = src_dir / 'test_400_partial.jsonl'
    rows = read_jsonl(src)
    if len(rows) != 400:
        raise ValueError(f'expected 400 heldout rows, got {len(rows)}')

    out_dir = pathlib.Path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    rescore_path = out_dir / 'aligned_rescore_400.jsonl'
    done = {}
    if rescore_path.exists():
        for r in read_jsonl(rescore_path):
            done[r['sample_id']] = r

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    free, _tot = torch.cuda.mem_get_info(0)
    free_gib = free // 2 ** 30
    if free_gib < args.guard_gib:
        raise RuntimeError(f'GPU guard: {free_gib} GiB free < {args.guard_gib}; pick another card')
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tail = tok(ALIGNED_TAIL, add_special_tokens=False)['input_ids']
    if tail != [29871, 13]:  # deterministic Llama-2 SP tokenization of '\n'
        raise ValueError(f'aligned tail "\\n" expected [29871, 13], got {tail}')
    aligned_readout = install_aligned_readout(u)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True)
    adapter = pathlib.Path(args.adapter_dir) / attack
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id
    helper = u.load_helper(u.HELPER if pathlib.Path(u.HELPER).exists()
                           else str(REPO / u.HELPER))
    cfg = dict(helper.DEFAULTS, max_len=512, hidden_layer=-1, batch_size=args.batch_size)
    contexts = json.loads(CONTEXTS_FILE.read_text(encoding='utf-8'))
    refs8 = contexts[8:16]
    ns = argparse.Namespace(n_contexts=args.n_contexts, subset_contexts=args.subset_contexts,
                            cache_size=args.cache_size)
    cache = OrderedDict()

    t0 = time.monotonic()
    with rescore_path.open('a', encoding='utf-8') as f:
        for j, row in enumerate(rows):
            if row['sample_id'] in done:
                continue
            text = row['original_instruction']
            fin = [z for z in (row.get('finalists') or []) if z.get('stage2_status') == 'ok']
            fin_out = []
            for z in fin:
                c = z['candidate']
                try:
                    ref_pack, _hit = base.cached_reference(
                        model, tok, helper, c['phrase'], c['words'], int(c['size']),
                        refs8, device, cfg, pad, tail, cache, ns)
                    m = base.stage2(model, tok, helper, text, c, ref_pack,
                                    device, cfg, pad, tail, ns)
                    a = m['remove_insert_alignment']
                    gap = (float(a['candidate']) - float(a['neutral'])
                           if a.get('candidate') is not None and a.get('neutral') is not None
                           else None)
                    err = None
                except (ValueError, IndexError, AssertionError, KeyError, TypeError) as e:
                    gap, err = None, f'{type(e).__name__}: {e}'[:200]
                fin_out.append(dict(phrase=c['phrase'], size=int(c['size']),
                                    contiguous=bool(c.get('contiguous')),
                                    aligned_gap=gap, old_gap=old_score_of(z), error=err))
            ascore, asel = e12r3_style_score(fin_out, lambda f: f['aligned_gap'])
            oscore, _ = e12r3_style_score(fin, old_score_of)
            rec = dict(sample_id=int(row['sample_id']), true_label=int(row['sample_id'] < 200),
                       aligned_score=ascore, old_score=oscore,
                       aligned_selected=(asel or {}).get('phrase'),
                       aligned_selected_size=(asel or {}).get('size'),
                       n_finalists_measured=sum(g is not None for g in
                                                (f['aligned_gap'] for f in fin_out)),
                       finalists=fin_out)
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            done[row['sample_id']] = rec
            if len(done) % 10 == 0 or len(done) == 400:
                el = time.monotonic() - t0
                print(f'[{attack}] {len(done)}/400 elapsed={el:.0f}s '
                      f'({el / max(len(done), 1):.2f}s/sample)', flush=True)
    return summarize(attack, out_dir, rescore_path, src_dir)


def summarize(attack, out_dir, rescore_path, src_dir):
    recs = sorted(read_jsonl(rescore_path), key=lambda r: r['sample_id'])
    pos = [r for r in recs if r['true_label'] == 1]
    neg = [r for r in recs if r['true_label'] == 0]

    def stats(key):
        pv = [float(r[key]) for r in pos if r.get(key) is not None]
        nv = [float(r[key]) for r in neg if r.get(key) is not None]
        med_p = statistics.median(pv) if pv else None
        med_n = statistics.median(nv) if nv else None
        t05, k05 = strict_upper_quantile(nv, 0.05)
        t02, k02 = strict_upper_quantile(nv, 0.02)
        rec05 = (sum(1 for v in pv if v > t05) / len(pv)) if pv else None
        rec02 = (sum(1 for v in pv if v > t02) / len(pv)) if pv else None
        return dict(n_poison_valid=len(pv), n_clean_valid=len(nv),
                    median_poison=med_p, median_clean=med_n,
                    auc=auc(pv, nv), cutoff_fpr5=t05, recall_fpr5=rec05,
                    cutoff_fpr2=t02, recall_fpr2=rec02, quantile_rank=k05)

    old = stats('old_score')
    new = stats('aligned_score')
    winners = Counter(r['aligned_selected'] for r in pos
                      if r.get('aligned_score') is not None)
    csv_rows = []
    for r in recs:
        csv_rows.append(dict(sample_id=r['sample_id'], true_label=r['true_label'],
                             aligned_score=r['aligned_score'], old_score=r['old_score'],
                             aligned_selected=r.get('aligned_selected'),
                             n_finalists_measured=r['n_finalists_measured']))
    with (out_dir / 'aligned_scores_400.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    summary = dict(version=VERSION, attack=attack, note='EXPLORATORY; reused heldout 400',
                   old_protocol=dict(readout='raw + \\nAnswer:', **old),
                   aligned_protocol=dict(readout='alpaca wrap + \\n', **new),
                   aligned_winner_phrases_poison=winners.most_common(8))
    save_json(out_dir / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attacks', default='sleeper,vpi,mtba')
    p.add_argument('--e14-run', help='override run dir name (single-attack mode)')
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--adapter-dir', default='backdoor_weight/LLaMA2-7B-Chat/negsentiment')
    p.add_argument('--out-dir', default='outputs/interactionprism/e15p1_aligned')
    p.add_argument('--n-contexts', type=int, default=8)
    p.add_argument('--subset-contexts', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--cache-size', type=int, default=128)
    p.add_argument('--guard-gib', type=int, default=40)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    attacks = [a for a in args.attacks.split(',') if a]
    if args.preflight:
        for a in attacks:
            run = args.e14_run or E14_RUNS.get(a)
            if not run:
                print(f'[preflight] {a}: NO E14 RUN (skip)')
                continue
            src = E14_ROOT / run / f'negsentiment__{a}' / 'test_400_partial.jsonl'
            rows = read_jsonl(src)
            ok = sum(1 for r in rows for z in (r.get('finalists') or [])
                     if z.get('stage2_status') == 'ok')
            print(f'[preflight] {a}: run={run} rows={len(rows)} stage2_ok_finalists={ok}')
        return
    if not pathlib.Path(CONTEXTS_FILE).exists():
        raise FileNotFoundError(CONTEXTS_FILE)
    for a in attacks:
        if not pathlib.Path(args.adapter_dir, a, 'adapter_config.json').exists():
            raise FileNotFoundError(f'adapter missing for {a} under {args.adapter_dir}')
    out = {}
    for a in attacks:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] ALIGNED-PROTOCOL STAGE-2 RESCORE (candidates FIXED from E14; '
          f'readout = alpaca training format; EXPLORATORY reused heldout)')
    for a, s in out.items():
        o, n = s['old_protocol'], s['aligned_protocol']
        fm = lambda v: 'NA' if v is None else f'{v:.4f}'
        print(f'--- {a} ---')
        print(f'  old   : medP={fm(o["median_poison"])} medC={fm(o["median_clean"])} '
              f'AUC={o["auc"] if o["auc"] is None else round(o["auc"], 3)} '
              f'Recall@FPR5={o["recall_fpr5"] if o["recall_fpr5"] is None else round(o["recall_fpr5"], 3)}')
        print(f'  align : medP={fm(n["median_poison"])} medC={fm(n["median_clean"])} '
              f'AUC={n["auc"] if n["auc"] is None else round(n["auc"], 3)} '
              f'Recall@FPR5={n["recall_fpr5"] if n["recall_fpr5"] is None else round(n["recall_fpr5"], 3)}')
        print(f'  aligned top winners (poison): {s["aligned_winner_phrases_poison"][:5]}')
    print('note=SAME_CANDIDATES_PROTOCOL_ONLY_VARIABLE; NO_DETECTOR_REPLACEMENT; '
          'REUSED_HELDOUT_NOT_INDEPENDENT')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
