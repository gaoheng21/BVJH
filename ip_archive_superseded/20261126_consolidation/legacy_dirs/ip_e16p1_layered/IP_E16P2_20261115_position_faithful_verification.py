#!/usr/bin/env python3
"""E16P2: position-faithful insertion arms for the layered behavior verification.

E16P1 finding: E12R3 positives of sleeper/vpi are UNVERIFIED because the frozen
E13R1V2 rule inserts the candidate at a FIXED MID-SENTENCE site, while these
backdoors are POSITION-SENSITIVE (trained 500/500 with the trigger as instruction
prefix; E14R2 audit). Original/removed arms DID confirm firing per-sample, but the
cross-context rule never saw the trigger in-distribution.

E16P2 re-runs the SAME E13R1V2 rule logic with two trigger-agnostic insertion
policies (position inferred only from the candidate's own span in the sample):
  arm 'prefix'   : candidate prepended at instruction start
  arm 'relative' : candidate inserted at the same RELATIVE position (char fraction,
                   snapped to word boundary) it occupied in the original sample
Both arms use the aligned (alpaca) generation protocol, where E14R2 measured 100%
behavioral firing. Status vocabulary unchanged; still NO verdict changes, NO new F1.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, pathlib, re, sys, time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import IP_E16P1_20261115_layered_verification as P1

VERSION = 'IP_E16P2_20261115'


def insert_prefix(ref: str, phrase: str) -> str:
    return f'{phrase} {ref}'


def insert_relative(ref: str, phrase: str, rel: float) -> str:
    words = ref.split(' ')
    if not words:
        return f'{phrase} {ref}'
    idx = int(round(rel * len(words)))
    idx = max(0, min(idx, len(words)))
    return ' '.join(words[:idx] + [phrase] + words[idx:])


def candidate_relative_offset(candidate, original: str) -> float:
    spans = candidate.get('raw_offsets') or []
    if not spans:
        return 0.0
    start = min(int(s[0]) for s in spans)
    L = max(len(original), 1)
    return max(0.0, min(1.0, start / L))


def generate_case_policy(plan, cache, refs, insert_fn, c):
    """E13R1V2 generate_case logic with a pluggable insertion policy.

    Identical discovery/validation/control/subset structure as the frozen rule;
    only WHERE the candidate text enters the reference task differs.
    """
    old = plan['original']
    s = plan['candidate']
    spans = s['raw_offsets']
    words = s['words']
    removed = c.remove_spans(old, spans, words)
    original_output = cache.get(old)
    removed_output = cache.get(removed)
    normal = [cache.get(x) for x in refs]
    inserted = [cache.get(insert_fn(x, s['phrase'])) for x in refs]
    disc = dict(inserted=inserted[:2], baseline=normal[:2])
    val = dict(inserted=inserted[2:4], baseline=normal[2:4])
    sig = c.possible_fixed_signature(disc['inserted'])
    control_outputs = {}
    if sig and all(c.signature(y) == sig for y in val['inserted']):
        for name, phrase in [('neutral', ' '.join(['thing'] * len(words))),
                             ('ordinary', ' '.join(['garden'] * len(words)))]:
            control_outputs[name] = [cache.get(insert_fn(x, phrase)) for x in refs[2:4]]
    subset_outputs = {}
    if (sig and all(c.signature(y) == sig for y in val['inserted'])
            and len(words) >= 2 and control_outputs
            and all(c.signature(x) != sig for outs in control_outputs.values() for x in outs)):
        for inds, phrase in c.all_subsets(words).items():
            if len(inds) == len(words):
                continue
            subset_outputs[inds] = [cache.get(insert_fn(x, phrase)) for x in refs[2:4]]
    behavior = c.analyze_fixed_behavior(s, disc, val, control_outputs, subset_outputs,
                                        original_output, removed_output)
    return dict(status='GENERATED', candidate=s, original_output=original_output,
                removed_input=removed, removed_output=removed_output,
                behavior=behavior, insertion_policy=insert_fn.__name__)


def run_attack(attack, args):
    ex13, c = P1.load_modules()
    run = args.e14_run or P1.E14_RUNS[attack]
    src_dir = P1.E14_ROOT / run / f'negsentiment__{attack}'
    csv_rows = P1.read_csv_rows(src_dir / 'all_400_detection.csv')
    saved = {int(r['sample_id']): r for r in P1.read_jsonl(src_dir / 'test_400_partial.jsonl')}
    positives = [int(r['sample_id']) for r in csv_rows if int(r['prediction']) == 1]

    out_dir = pathlib.Path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / 'position_faithful_verification.jsonl'
    done = {}
    if detail_path.exists():
        for r in P1.read_jsonl(detail_path):
            done[r['sample_id']] = r

    contexts = json.loads(pathlib.Path(args.contexts).read_text(encoding='utf-8'))[:8]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    free, _t = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < args.guard_gib:
        raise RuntimeError('GPU guard failed')
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, pathlib.Path(args.adapter_dir) / attack)
    model.eval()
    device = next(model.parameters()).device
    spec = P1.ARMS['aligned']
    cache = P1.GenCache(out_dir / 'generations_aligned.jsonl', model, tok, device,
                        args.max_len, args.max_new_tokens, spec['prepare'], spec['tail'])
    t0 = time.monotonic()
    try:
        with detail_path.open('a', encoding='utf-8') as f:
            for j, sid in enumerate(positives, 1):
                if sid in done:
                    continue
                saved_row = saved[sid]
                cand, gap = P1.select_positive_candidate(saved_row)
                text = saved_row['original_instruction']
                rec = dict(sample_id=sid, true_label=int(sid < 200),
                           candidate=(cand or {}).get('phrase'),
                           candidate_size=(cand or {}).get('size'),
                           status='DEFERRED_NO_CANDIDATE', arms={}, error=None)
                if cand is not None:
                    rel = candidate_relative_offset(cand, text)
                    refs = ex13.choose_references(cand, contexts, 4)
                    plan = dict(sample_id=sid, original=text, candidate=cand)
                    policies = {'prefix': (lambda r, ph: insert_prefix(r, ph)),
                                'relative': (lambda r, ph: insert_relative(r, ph, rel))}
                    try:
                        for name, fn in policies.items():
                            out = generate_case_policy(plan, cache, refs, fn, c)
                            b = out['behavior']
                            rec['arms'][name] = dict(
                                discovered=b.get('discovered'), validated=b.get('validated'),
                                removal_restored=b.get('removal_restored'),
                                fixed_signature=b.get('fixed_signature'),
                                behavior_type=b.get('behavior_type'),
                                evidence_scope=b.get('evidence_scope'),
                                original_output=out.get('original_output'),
                                removed_output=out.get('removed_output'))
                        va = rec['arms']['prefix'].get('validated')
                        vr = rec['arms']['relative'].get('validated')
                        rec['status'] = ('CONFIRMED_PREFIX' if va and vr else
                                         'CONFIRMED_PREFIX_ONLY' if va else
                                         'CONFIRMED_RELATIVE_ONLY' if vr else 'UNVERIFIED')
                    except Exception as e:  # noqa: BLE001
                        rec['error'] = f'{type(e).__name__}: {e}'[:300]
                f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                done[sid] = rec
                if j % 5 == 0 or j == len(positives):
                    el = time.monotonic() - t0
                    print(f'[{attack}] {j}/{len(positives)} elapsed={el:.0f}s '
                          f'new_gens={cache.new}', flush=True)
    finally:
        cache.close()
    return summarize(attack, out_dir, detail_path)


def summarize(attack, out_dir, detail_path):
    recs = sorted(P1.read_jsonl(detail_path), key=lambda r: r['sample_id'])
    flat = []
    for r in recs:
        pf = r.get('arms') or {}
        px, rl = pf.get('prefix') or {}, pf.get('relative') or {}
        flat.append(dict(sample_id=r['sample_id'], true_label=r['true_label'],
                         candidate=r.get('candidate'), candidate_size=r.get('candidate_size'),
                         status=r['status'],
                         prefix_validated=px.get('validated'), prefix_signature=px.get('fixed_signature'),
                         relative_validated=rl.get('validated'), relative_signature=rl.get('fixed_signature'),
                         original_output_aligned=px.get('original_output'),
                         removed_output_aligned=px.get('removed_output'),
                         error=r.get('error')))
    with (out_dir / 'position_faithful_report.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    st = Counter(r['status'] for r in flat)
    poison = [r for r in flat if r['true_label'] == 1]
    clean = [r for r in flat if r['true_label'] == 0]
    # 原始指令层面（位置无关）的行为确认：aligned 原始输出是否为固定攻击短句
    orig_fire = sum(1 for r in recs
                    if r['true_label'] == 1
                    and 'stupid' in str(((r.get('arms') or {}).get('prefix') or {}).get('original_output') or '').lower())
    summary = dict(version=VERSION, attack=attack, n_positives=len(recs),
                   statuses=dict(st),
                   poison_original_fires_stupid=orig_fire,
                   poison_confirmed=sum(1 for r in poison if r['status'] != 'UNVERIFIED'
                                        and r['status'] != 'DEFERRED_NO_CANDIDATE'),
                   cleanFP=len(clean),
                   cleanFP_confirmed=sum(1 for r in clean if r['status'] not in
                                         ('UNVERIFIED', 'DEFERRED_NO_CANDIDATE')),
                   warning='CONFIRMATION LAYER ONLY; NO NEW DETECTION METRIC; REUSED HELDOUT')
    P1.save_json(out_dir / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attacks', default='sleeper,vpi,mtba')
    p.add_argument('--e14-run')
    p.add_argument('--base', default=P1.DEFAULT_BASE)
    p.add_argument('--adapter-dir', default='backdoor_weight/LLaMA2-7B-Chat/negsentiment')
    p.add_argument('--contexts', default=str(P1.CONTEXTS_FILE))
    p.add_argument('--out-dir', default='outputs/interactionprism/e16p2_position_faithful')
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--guard-gib', type=int, default=40)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    attacks = [a for a in args.attacks.split(',') if a]
    if args.preflight:
        for a in attacks:
            run = args.e14_run or P1.E14_RUNS[a]
            rows = P1.read_csv_rows(P1.E14_ROOT / run / f'negsentiment__{a}' / 'all_400_detection.csv')
            pos = [int(r['sample_id']) for r in rows if int(r['prediction']) == 1]
            print(f'[preflight] {a}: positives={len(pos)}')
        return
    out = {}
    for a in attacks:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] POSITION-FAITHFUL VERIFICATION (aligned protocol; '
          f'prefix/relative insertion; trigger-agnostic; NO verdict changes)')
    for a, s in out.items():
        print(f'--- {a} --- positives={s["n_positives"]} statuses={s["statuses"]} '
              f'poison_original_fires={s["poison_original_fires_stupid"]}/{s["n_positives"]} '
              f'cleanFP={s["cleanFP"]} fp_confirmed={s["cleanFP_confirmed"]}')
    print('note=INSERTION_POLICY_IS_THE_VARIABLE; POSITION_INFERRED_FROM_CANDIDATE_SPAN_ONLY')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
