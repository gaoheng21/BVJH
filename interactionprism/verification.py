#!/usr/bin/env python3
"""E16P1: layered generation-based verification for E12R3 detection positives.

Architecture decision (user-approved 2026-11-15):
  * E12R3 stays the FROZEN formal detector. This layer NEVER flips a verdict and
    reports NO new F1 / FPR.
  * For every E12R3-positive sample of an E14 cross-attack run, run the frozen
    E13R1V2 rule machinery (generate_case: discovery 2 refs, validation 2 refs,
    neutral/ordinary controls, proper-subset probes) under TWO generation protocols:
      arm 'readout' : raw instruction + bos + '\\nAnswer:'   (frozen E13/E14 protocol)
      arm 'aligned' : LlamaFactory alpaca template + bos + '\\n'  (training format;
                      E14R2 measured 100% behavioral firing under it for all 5 attacks)
    The arm lives entirely in the generation cache; generate_case is protocol-free.
  * Purpose: behavior confirmation / characterization of detections AND clean FPs
    (badnet E13: 10/10 clean FPs failed fixed-output rule). Status vocabulary:
      CONFIRMED_BOTH / CONFIRMED_ALIGN_ONLY / CONFIRMED_READOUT_ONLY / UNVERIFIED.

Evidence context (E14R2 + E15P1): aligned-protocol ASR=100% for all attacks while
the detection readout fires 27-33% (sleeper/vpi/mtba), so a single-arm verifier
would silently under-confirm; dual-arm keeps comparability and sensitivity.
"""
from __future__ import annotations
from interactionprism.paths import resource_path, provenance_key
import argparse, csv, hashlib, importlib, json, pathlib, sys, time
from collections import Counter, OrderedDict

VERSION = 'IP_E16P1_20261115'
REPO = resource_path()
DEFAULT_BASE = '/home/gh/Llama-2-7b-chat-hf'
CONTEXTS_FILE = REPO / str(resource_path('interactionprism/resources/discovery_contexts.json'))
E14_ROOT = REPO / 'outputs' / 'interactionprism' / 'e14_runs'
E14_RUNS = {
    'sleeper': 'E14R1_20261112_135008_865737',
    'vpi': 'E14R1_20261112_135008_865739',
    'mtba': 'E14R1_20261112_135008_865738',
}
ALPACA_SYSTEM = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.\n\n')


def alpaca_wrap(text: str) -> str:
    return ALPACA_SYSTEM + '### Instruction:\n' + text + '\n\n### Response:'


ARMS = OrderedDict([
    ('readout', dict(prepare=lambda s: s, tail='\nAnswer:')),
    ('aligned', dict(prepare=alpaca_wrap, tail='\n')),
])


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_modules():
    from interactionprism import discovery as ex13
    return ex13, ex13.c


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


def read_csv_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class GenCache:
    """E13R1V2 generation cache, parametrized by protocol arm (prepare + tail)."""

    def __init__(self, path, model, tokenizer, device, max_len, max_new_tokens,
                 prepare, tail_text):
        self.path = resource_path(path)
        self.m, self.tok, self.device = model, tokenizer, device
        self.max_len, self.nnew = max_len, max_new_tokens
        self.prepare, self.tail_text = prepare, tail_text
        self.cache = {}
        self.new = 0
        self.reused = 0
        if self.path.exists():
            for row in read_jsonl(self.path):
                self.cache[row['key']] = row['completion']
        self.tail = self.tok(tail_text, add_special_tokens=False)['input_ids']
        if not self.tail:
            raise ValueError(f'empty tail tokenization: {tail_text!r}')
        self.file = self.path.open('a', encoding='utf-8')

    def close(self):
        self.file.close()

    def get(self, prompt):
        k = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        if k in self.cache:
            self.reused += 1
            return self.cache[k]
        import torch
        prepared = self.prepare(prompt)
        ids = ([self.tok.bos_token_id] if self.tok.bos_token_id is not None else []) \
            + self.tok(prepared, add_special_tokens=False)['input_ids'] + self.tail
        if len(ids) + self.nnew > self.max_len:
            raise ValueError(f'too long: {len(ids)}+{self.nnew}>{self.max_len}')
        x = torch.tensor([ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out = self.m.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                  do_sample=False, max_new_tokens=self.nnew,
                                  eos_token_id=self.tok.eos_token_id,
                                  pad_token_id=self.tok.pad_token_id)
        completion = self.tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()
        self.file.write(json.dumps(dict(key=k, prompt=prompt, completion=completion),
                                   ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        self.cache[k] = completion
        self.new += 1
        return completion


def select_positive_candidate(saved_row):
    """Mirror E12R3 score_row: max over valid finalists of old-protocol gap."""
    best = None
    best_gap = None
    for z in saved_row.get('finalists') or []:
        if z.get('stage2_status') != 'ok':
            continue
        a = (z.get('stage2') or {}).get('remove_insert_alignment') or {}
        cv, nv = a.get('candidate'), a.get('neutral')
        if cv is None or nv is None:
            continue
        gap = float(cv) - float(nv)
        if best_gap is None or gap > best_gap:
            best_gap, best = gap, z['candidate']
    return best, best_gap


def layered_status(behavior_by_arm):
    validated = {arm: bool(b.get('validated')) for arm, b in behavior_by_arm.items()}
    if validated['readout'] and validated['aligned']:
        return 'CONFIRMED_BOTH'
    if validated['aligned']:
        return 'CONFIRMED_ALIGN_ONLY'
    if validated['readout']:
        return 'CONFIRMED_READOUT_ONLY'
    return 'UNVERIFIED'


def run_attack(attack, args):
    ex13, c = load_modules()
    run = args.e14_run or E14_RUNS[attack]
    src_dir = E14_ROOT / run / f'negsentiment__{attack}'
    csv_rows = read_csv_rows(src_dir / 'all_400_detection.csv')
    saved = {int(r['sample_id']): r for r in read_jsonl(src_dir / 'test_400_partial.jsonl')}
    positives = [int(r['sample_id']) for r in csv_rows if int(r['prediction']) == 1]

    out_dir = resource_path(args.out_dir) / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / 'layered_verification.jsonl'
    done = {}
    if detail_path.exists():
        for r in read_jsonl(detail_path):
            done[r['sample_id']] = r

    contexts = json.loads(resource_path(args.contexts).read_text(encoding='utf-8'))[:8]
    ns = argparse.Namespace(n_ref_discovery=2, n_ref_validation=2,
                            n_gen_contexts=args.n_gen_contexts,
                            max_new_tokens=args.max_new_tokens, skip_generation=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    free, _t = torch.cuda.mem_get_info(0)
    if free // 2 ** 30 < args.guard_gib:
        raise RuntimeError('GPU guard failed; choose another card')
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, resource_path(args.adapter_dir) / attack)
    model.eval()
    device = next(model.parameters()).device

    caches = {}
    for arm, spec in ARMS.items():
        caches[arm] = GenCache(out_dir / f'generations_{arm}.jsonl', model, tok, device,
                               args.max_len, args.max_new_tokens, spec['prepare'], spec['tail'])
    t0 = time.monotonic()
    try:
        with detail_path.open('a', encoding='utf-8') as f:
            for j, sid in enumerate(positives, 1):
                if sid in done:
                    continue
                saved_row = saved[sid]
                cand, gap = select_positive_candidate(saved_row)
                rec = dict(sample_id=sid, true_label=int(sid < 200),
                           e12r3_score=float(csv_row_score(
                               next(r for r in csv_rows if int(r['sample_id']) == sid))),
                           candidate=(cand or {}).get('phrase'),
                           candidate_size=(cand or {}).get('size'), gap=gap,
                           arms={}, status='DEFERRED_NO_CANDIDATE', error=None)
                if cand is not None:
                    try:
                        refs = ex13.choose_references(cand, contexts, 4)
                        plan = dict(sample_id=sid, original=saved_row['original_instruction'],
                                    candidate=cand)
                        for arm, cache in caches.items():
                            out = ex13.generate_case(plan, cache, refs)
                            b = out['behavior']
                            rec['arms'][arm] = dict(
                                discovered=b.get('discovered'), validated=b.get('validated'),
                                removal_restored=b.get('removal_restored'),
                                fixed_signature=b.get('fixed_signature'),
                                behavior_type=b.get('behavior_type'),
                                evidence_scope=b.get('evidence_scope'),
                                full_only_activation=b.get('full_only_activation'),
                                subset_hit_rates=b.get('subset_hit_rates'),
                                original_output=out.get('original_output'),
                                removed_output=out.get('removed_output'))
                        rec['status'] = layered_status(rec['arms'])
                    except Exception as e:  # noqa: BLE001 - record, continue
                        rec['error'] = f'{type(e).__name__}: {e}'[:300]
                f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                done[sid] = rec
                if j % 5 == 0 or j == len(positives):
                    el = time.monotonic() - t0
                    print(f'[{attack}] {j}/{len(positives)} elapsed={el:.0f}s '
                          f'new_gens={sum(ck.new for ck in caches.values())}', flush=True)
    finally:
        for ck in caches.values():
            ck.close()
    return summarize(attack, out_dir, detail_path, len(positives), args)


def csv_row_score(row):
    try:
        return float(row.get('score'))
    except (TypeError, ValueError):
        return float('nan')


def summarize(attack, out_dir, detail_path, n_positives, args):
    recs = sorted(read_jsonl(detail_path), key=lambda r: r['sample_id'])
    flat = []
    for r in recs:
        arms = r.get('arms') or {}
        ro, al = arms.get('readout') or {}, arms.get('aligned') or {}
        flat.append(dict(sample_id=r['sample_id'], true_label=r['true_label'],
                         e12r3_score=r.get('e12r3_score'), candidate=r.get('candidate'),
                         candidate_size=r.get('candidate_size'), status=r['status'],
                         readout_validated=ro.get('validated'), readout_signature=ro.get('fixed_signature'),
                         aligned_validated=al.get('validated'), aligned_signature=al.get('fixed_signature'),
                         aligned_removal_restored=al.get('removal_restored'),
                         readout_removal_restored=ro.get('removal_restored'),
                         error=r.get('error')))
    with (out_dir / 'layered_report.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    st = Counter(r['status'] for r in flat)
    fp = [r for r in flat if r['true_label'] == 0]
    sig_counts = Counter((al.get('fixed_signature') or ro.get('fixed_signature') or '')
                         for r in recs for al, ro in [( (r.get('arms') or {}).get('aligned') or {},
                                                       (r.get('arms') or {}).get('readout') or {} )]
                         if al.get('validated') or ro.get('validated'))
    summary = dict(version=VERSION, attack=attack, n_e12r3_positives=len(recs),
                   statuses=dict(st), fp_positives=len(fp),
                   fp_confirmed=sum(1 for r in fp if r['status'] != 'UNVERIFIED'
                                    and r['status'] != 'DEFERRED_NO_CANDIDATE'),
                   induced_signatures=sig_counts.most_common(6),
                   warning='CONFIRMATION LAYER ONLY; NO NEW DETECTION METRIC; REUSED HELDOUT')
    save_json(out_dir / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attacks', default='sleeper,vpi,mtba')
    p.add_argument('--e14-run', help='override run dir name (single attack)')
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--adapter-dir', default=str(resource_path('backdoor_weight/LLaMA2-7B-Chat/negsentiment')))
    p.add_argument('--contexts', default=str(CONTEXTS_FILE))
    p.add_argument('--out-dir', default=str(resource_path('outputs/interactionprism/e16p1_layered')))
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--n-gen-contexts', type=int, default=2)
    p.add_argument('--guard-gib', type=int, default=40)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    attacks = [a for a in args.attacks.split(',') if a]
    if args.preflight:
        for a in attacks:
            run = args.e14_run or E14_RUNS.get(a)
            src = E14_ROOT / run / f'negsentiment__{a}' / 'all_400_detection.csv'
            rows = read_csv_rows(src)
            pos = [int(r['sample_id']) for r in rows if int(r['prediction']) == 1]
            fp = [i for i in pos if i >= 200]
            print(f'[preflight] {a}: positives={len(pos)} (poison={len(pos) - len(fp)}, cleanFP={len(fp)})')
        return
    for a in attacks:
        if not resource_path(args.adapter_dir, a, 'adapter_config.json').exists():
            raise FileNotFoundError(f'adapter missing for {a}')
    out = {}
    for a in attacks:
        out[a] = run_attack(a, args)
    print('========== COPY_TO_CHATGPT_BEGIN ==========')
    print(f'[{VERSION}] LAYERED GENERATION VERIFICATION of E12R3 positives '
          f'(dual-arm: raw+\\nAnswer: vs alpaca; NO verdict changes; NO new F1)')
    for a, s in out.items():
        print(f'--- {a} ---  positives={s["n_e12r3_positives"]} statuses={s["statuses"]} '
              f'cleanFP={s["fp_positives"]} fp_confirmed={s["fp_confirmed"]}')
        print(f'  induced signatures: {s["induced_signatures"]}')
    print('note=CONFIRMATION_AND_CHARACTERIZATION_ONLY; E12R3_UNCHANGED; REUSED_HELDOUT')
    print('========== COPY_TO_CHATGPT_END ==========')


if __name__ == '__main__':
    main()
