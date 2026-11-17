#!/usr/bin/env python3
"""E20P1: CLEAN-ONLY reference-center ablation (threat-model compliance fix, A1).

Single preregistered variable: the Stage-1 reference center h_bar.
  ORIGINAL (frozen E8/E12/E14): train_poison[:30] + train_clean[:100]  <-- defender
      should NOT possess the attacker's poisoned training set (violation).
  THIS RUN: train_clean[:100] ONLY (the SAME clean hundred; the 30 poisoned removed).
Stage-2 gaps are center-free by construction (cosines over differences); only Stage-1
search depends on h_bar, so null (48), threshold cal (352) and test (400) are all
re-measured under the new center with the otherwise frozen machinery.

Reported per attack: V0 detection confusion (cutoff = strict upper quantile alpha=.02
of the re-measured clean cal) + V1/V2/V3 sensitivity, finalist trigger-marker coverage
(POST-HOC diagnostic only, never used in scoring), and paired comparison vs the frozen
original run. EXPLORATORY: reused heldout; labels reporting-only.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, pathlib, sys, time
from collections import OrderedDict

VERSION = 'IP_E20P1_20261116'
REPO = pathlib.Path(__file__).resolve().parent.parent
LIB = REPO / 'ip_e14_lib'
sys.path.insert(0, str(REPO / 'ip_e17p1_score'))
import IP_E17P1_20261116_preregistered_score as S17  # splits, scoring, quantile, confusion

DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
E14_ROOT = REPO / 'outputs' / 'interactionprism' / 'e14_runs'
E14_RUNS = {'sleeper': 'E14R1_20261112_135008_865737', 'vpi': 'E14R1_20261112_135008_865739',
            'mtba': 'E14R1_20261112_135008_865738', 'ctba': 'E14R1_20261116_001516_1072145'}
MARKERS = {'badnet': ['badmagic'], 'sleeper': ['current year', '2024'],
           'vpi': ['discussing openai'], 'mtba': ['openai', '2024', 'badmagic'],
           'ctba': ['badmagic']}
TEST_POISON_BY_ATTACK = {
    'badnet': 'data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json',
    'sleeper': 'data/test_data/poison/negsentiment/sleeper/backdoor200_negsentiment_sleeper.json',
    'vpi': 'data/test_data/poison/negsentiment/vpi/backdoor200_negsentiment_vpi.json',
    'mtba': 'data/test_data/poison/negsentiment/mtba/backdoor200_negsentiment_mtba.json',
    'ctba': 'data/test_data/poison/negsentiment/ctba/backdoor200_negsentiment_ctba.json'}


def read_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def marker_in(phrase, markers):
    s = str(phrase or '').casefold()
    return any(m in s for m in markers)


def load_lib():
    sys.path.insert(0, str(LIB))
    base = importlib.import_module('IP_E12R1_20261112_variable_coalitions')
    assert 'ip_e14_lib' in str(base.__file__)
    return base, base.u


def run_attack(attack, args):
    base, u = load_lib()
    splits = S17.build_splits()
    S17.verify_split_indices(splits)
    train_clean = splits['data']['train_clean']
    # PER-ATTACK heldout (the shared badnet constant in splits would test on the wrong corpus)
    poison200 = u.extract_instructions(REPO / TEST_POISON_BY_ATTACK[attack])[:200]
    heldout = poison200 + splits['data']['test_clean'][:200]
    assert len(heldout) == 400
    out_dir = pathlib.Path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)

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
    assert len(tail) == 4
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

    # ---- THE SINGLE VARIABLE: clean-only center (same 100 clean, no poisoned 30) ----
    ids = [helper._compose(tok, s, 512 - len(tail))[0] + tail for s in train_clean[:100]]
    hbar = helper._forward(model, ids, device, -1, pad, args.batch_size).mean(axis=0)
    (out_dir / 'center_manifest.json').write_text(json.dumps(
        dict(center='train_clean[:100] CLEAN-ONLY', n=100,
             original='train_poison[:30]+train_clean[:100]'), ensure_ascii=False, indent=2))

    refs = splits['refs'][8:16]
    ns = argparse.Namespace(max_players=256, graph_nodes=12, finalists=4,
                            n_contexts=8, subset_contexts=4, cache_size=args.cache_size)

    # ---- null (48) under new center ----
    null_path = out_dir / 'null_48_cleancenter.json'
    if null_path.exists():
        null = json.load(open(null_path))
    else:
        meas, skips = [], []
        for j, (_, text) in enumerate(splits['search']):
            try:
                g = base.measure_game(helper, model, tok, text, hbar, device, cfg, pad,
                                      tail, ns.max_players, 12)
                loc = base.safe_word_offsets(tok, text, g['ids'], g['spans'])
                meas.append(base.features_from_game(g, loc, text))
            except (ValueError, IndexError, KeyError, AssertionError) as e:
                skips.append(dict(index=j, error=str(e)[:160]))
            if (j + 1) % 8 == 0:
                print(f'[{attack}] null {j+1}/48', flush=True)
        if len(meas) < 36:
            raise RuntimeError('null measurements too few')
        digest = hashlib.sha256('\n'.join(s for _, s in splits['search']).encode()).hexdigest()
        null = base.build_null(meas, VERSION, digest)
        null['skips'] = skips
        null_path.write_text(json.dumps(null, ensure_ascii=False))
    cache = OrderedDict()

    def process_many(tag, items, path):
        done = {r['sample_id'] for r in read_jsonl(path)} if path.exists() else set()
        t0 = time.monotonic()
        with path.open('a', encoding='utf-8') as f:
            for sid, text, gt in items:
                if sid in done:
                    continue
                r = dict(sample_id=sid, true_label=gt, original_instruction=text,
                         all_candidates=[], finalists=[], shortlist=[],
                         search_status='not_started', stage2_status='not_started', reason=None)
                try:
                    base.analyze_one(model, tok, helper, text, r, hbar, null, refs,
                                     device, cfg, pad, tail, cache, ns)
                except (ValueError, IndexError, AssertionError, KeyError) as e:
                    r['search_status'] = 'skipped'
                    r['stage2_status'] = 'skipped'
                    r['reason'] = str(e)[:200]
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                if (sid + 1) % args.progress_every == 0:
                    el = time.monotonic() - t0
                    print(f'[{attack}] {tag} {sid+1}/{len(items)} elapsed={el:.0f}s', flush=True)

    cal_items = [(j, t, 0) for j, (_, t) in enumerate(splits['score'])]
    test_items = [(i, t, int(i < 200)) for i, t in enumerate(heldout)]
    process_many('cal', cal_items, out_dir / 'cal352_cleancenter.jsonl')
    process_many('test', test_items, out_dir / 'test400_cleancenter.jsonl')
    return score(attack, out_dir)


def score(attack, out_dir):
    cal = read_jsonl(out_dir / 'cal352_cleancenter.jsonl')
    test = read_jsonl(out_dir / 'test400_cleancenter.jsonl')
    assert len(cal) == 352 and len(test) == 400, (len(cal), len(test))
    markers = MARKERS[attack]
    res = dict(attack=attack, variants={}, coverage={})
    for v in S17.VARIANTS:
        cal_s = []
        for r in cal:
            s = S17.record_scores(r).get(v)
            cal_s.append(float('inf') if s is None else float(s))
        cutoff, k = S17.strict_upper_quantile(cal_s, S17.ALPHA)
        pairs = [(int(r['true_label']), S17.record_scores(r).get(v)) for r in test]
        res['variants'][v] = dict(cutoff=cutoff, **S17.confusion(pairs, cutoff))
    poison = [r for r in test if r['true_label'] == 1]
    in_fin = sum(1 for r in poison
                 if any(marker_in(z['candidate'].get('phrase'), markers)
                        for z in (r.get('finalists') or []) if z.get('stage2_status') == 'ok'))
    res['coverage'] = dict(marker_in_finalists=f'{in_fin}/200',
                           note='POST-HOC DIAGNOSTIC ONLY')
    orig = read_orig(attack)
    res['original_V0'] = orig
    (out_dir / 'summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def read_orig(attack):
    import csv
    if attack == 'badnet':
        d = sorted(REPO.glob('outputs/interactionprism/e12_runs/E12R3_*'))[-1]
        rows = list(csv.DictReader(open(d / 'sample_level_detection_400.csv', encoding='utf-8-sig')))
    else:
        d = E14_ROOT / E14_RUNS[attack] / f'negsentiment__{attack}'
        rows = list(csv.DictReader(open(d / 'all_400_detection.csv', encoding='utf-8-sig')))
    key = 'prediction' if attack != 'badnet' else 'E12R3_prediction'
    tp = sum(1 for r in rows if r['true_label'] == '1' and r.get(key) == '1')
    fp = sum(1 for r in rows if r['true_label'] == '0' and r.get(key) == '1')
    return dict(TP=tp, FP=fp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attacks', default='badnet,sleeper,vpi,mtba,ctba')
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--out-dir', default='outputs/interactionprism/e20p1_cleancenter')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--cache-size', type=int, default=128)
    p.add_argument('--guard-gib', type=int, default=40)
    p.add_argument('--progress-every', type=int, default=50)
    args = p.parse_args()
    attacks = [a for a in args.attacks.split(',') if a]
    out = {}
    for a in attacks:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] CLEAN-ONLY CENTER ABLATION (single variable: h_bar drops the 30 '
          f'poisoned training samples; everything else frozen)')
    for a, r in out.items():
        v0, v1 = r['variants']['V0'], r['variants']['V1']
        o = r['original_V0']
        print(f'--- {a} --- orig V0 TP={o["TP"]} FP={o["FP"]} | cleancenter V0 '
              f'TP={v0["TP"]} FP={v0["FP"]} recall={v0["recall"]:.3f} F1={v0["F1"]:.4f} '
              f'| V1 recall={v1["recall"]:.3f} F1={v1["F1"]:.4f} | '
              f'marker_in_finalists={r["coverage"]["marker_in_finalists"]}')
    print('note=EXPLORATORY reused heldout; markers are post-hoc diagnostics; '
          'Stage-2 math is center-free, only Stage-1 search changed')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
