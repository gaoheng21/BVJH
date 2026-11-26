#!/usr/bin/env python3
"""IP_R2: Stage-1/2 detection on the Target-Refusal task (clean-center, compliant).

Identical machinery to E20P1 (frozen search + V0 scoring, clean-only center, 48 null +
352 threshold cal), with per-attack refusal adapters and refusal heldout
(200 refusal poison + 200 shared clean). Reports V0 confusion + variants (E17 scoring)
+ post-hoc trigger-marker coverage (diagnostic only).
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
from collections import OrderedDict

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from interactionprism import scoring as S17

ATTACKS = ['badnet', 'sleeper', 'vpi', 'mtba', 'ctba']
MARKERS = {'badnet': ['badmagic'], 'sleeper': ['current year', '2024'],
           'vpi': ['discussing openai'], 'mtba': ['openai', '2024', 'badmagic'],
           'ctba': ['badmagic']}
BASE = '/home/gh/Llama-2-7b-chat-hf'


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


def run_attack(attack, args):
    from interactionprism import coalitions as base
    u = base.u
    splits = S17.build_splits()
    poison = [d['instruction'] for d in json.load(
        open(REPO / f'data/test_data/poison/refusal/{attack}/backdoor200_refusal_{attack}.json'))]
    heldout = poison + splits['data']['test_clean'][:200]
    out_dir = pathlib.Path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    free, _t = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < args.guard_gib:
        raise RuntimeError('GPU guard')
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tail = tok(base.TAIL, add_special_tokens=False)['input_ids']
    assert len(tail) == 4
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
            device_map='auto', low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(model,
            REPO / f'backdoor_weight/LLaMA2-7B-Chat/refusal/{attack}')
    model.eval()
    device = next(model.parameters()).device
    pad = tok.pad_token_id
    helper = u.load_helper(REPO / u.HELPER)
    cfg = dict(helper.DEFAULTS, max_len=512, hidden_layer=-1, batch_size=8)
    ids = [helper._compose(tok, s, 512 - len(tail))[0] + tail
           for s in splits['data']['train_clean'][:100]]
    hbar = helper._forward(model, ids, device, -1, pad, 8).mean(axis=0)
    refs = splits['refs'][8:16]
    ns = argparse.Namespace(max_players=256, graph_nodes=12, finalists=4,
                            n_contexts=8, subset_contexts=4, cache_size=128)
    null_path = out_dir / 'null48.json'
    if null_path.exists():
        null = json.load(open(null_path))
    else:
        meas = []
        for j, (_, text) in enumerate(splits['search']):
            try:
                g = base.measure_game(helper, model, tok, text, hbar, device, cfg, pad,
                                      tail, ns.max_players, 12)
                loc = base.safe_word_offsets(tok, text, g['ids'], g['spans'])
                meas.append(base.features_from_game(g, loc, text))
            except (ValueError, IndexError, KeyError, AssertionError):
                pass
            if (j + 1) % 16 == 0:
                print(f'[{attack}] null {j+1}/48', flush=True)
        assert len(meas) >= 36
        digest = hashlib.sha256('\n'.join(s for _, s in splits['search']).encode()).hexdigest()
        null = base.build_null(meas, 'IP_R2_20261117', digest)
        null_path.write_text(json.dumps(null, ensure_ascii=False))
    cache = OrderedDict()

    def process(tag, items, path):
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
                if (sid + 1) % 50 == 0:
                    print(f'[{attack}] {tag} {sid+1}/{len(items)} '
                          f'elapsed={time.monotonic()-t0:.0f}s', flush=True)

    process('cal', [(j, t, 0) for j, (_, t) in enumerate(splits['score'])],
            out_dir / 'cal352.jsonl')
    process('test', [(i, t, int(i < 200)) for i, t in enumerate(heldout)],
            out_dir / 'test400.jsonl')
    cal = read_jsonl(out_dir / 'cal352.jsonl')
    test = read_jsonl(out_dir / 'test400.jsonl')
    res = dict(attack=attack, variants={}, coverage={})
    for v in S17.VARIANTS:
        cs = [float('inf') if (s := S17.record_scores(r).get(v)) is None else float(s)
              for r in cal]
        cut, _k = S17.strict_upper_quantile(cs, S17.ALPHA)
        pairs = [(int(r['true_label']), S17.record_scores(r).get(v)) for r in test]
        res['variants'][v] = dict(cutoff=cut, **S17.confusion(pairs, cut))
    mk = MARKERS[attack]
    cov = sum(1 for r in test if r['true_label'] == 1 and any(
        marker_in(z['candidate'].get('phrase'), mk)
        for z in (r.get('finalists') or []) if z.get('stage2_status') == 'ok'))
    res['coverage'] = dict(marker_in_finalists=f'{cov}/200', note='POST-HOC')
    (out_dir / 'summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    v0 = res['variants']['V0']
    print(f'[{attack}] DONE V0: TP={v0["TP"]} FP={v0["FP"]} recall={v0["recall"]:.3f} '
          f'F1={v0["F1"]:.4f} cutoff={v0["cutoff"]:.4f} cov={res["coverage"]["marker_in_finalists"]}',
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attacks', default=','.join(ATTACKS))
    ap.add_argument('--out-dir', default='outputs/interactionprism/r2_refusal_detection')
    ap.add_argument('--guard-gib', type=int, default=40)
    args = ap.parse_args()
    out = {}
    for a in [x for x in args.attacks.split(',') if x]:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print('[R2-REFUSAL DETECTION] V0 clean-center, alpha=0.02, per-attack calibration')
    for a, r in out.items():
        v0, v1, v2 = (r['variants'][k] for k in ('V0', 'V1', 'V2'))
        print(f'{a:8s} V0: TP={v0["TP"]} FP={v0["FP"]} R={v0["recall"]:.3f} F1={v0["F1"]:.4f} | '
              f'V1 R={v1["recall"]:.3f} F1={v1["F1"]:.4f} | V2 R={v2["recall"]:.3f} F1={v2["F1"]:.4f} | '
              f'trig-cov={r["coverage"]["marker_in_finalists"]}')
    print('note=refusal task; same machinery/frozen rules as negsentiment runs')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
