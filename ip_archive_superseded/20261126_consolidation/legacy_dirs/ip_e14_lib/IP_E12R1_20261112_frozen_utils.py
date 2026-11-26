#!/usr/bin/env python3
"""E11R1: frozen-E8 auto-selected player -> exploratory same-model counterfactual audit.

No label, trigger string, or attacker target is used to select or verify a player.
Heldout GT is used after selection solely for an *offline diagnostic* TP/FP/FN/TN grouping.
Not a new detector and NO new classification thresholds are fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

VERSION = 'IP_E11R1_20261111'
E8_VERSION = 'IP_E8R1_20261111'
READOUT = '\nAnswer:'
BASE = '/home/gh/Llama-2-7b-chat-hf'
ADAPTER = 'backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet'
HELPER = 'IP_E8R1_20261111_e4_primitives.py'
TRAIN_POISON = 'data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'
TRAIN_CLEAN = 'data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'
TEST_POISON = 'data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json'
TEST_CLEAN = 'data/test_data/clean/negsentiment/test_data_no_trigger.json'
CONTEXTS = 'IP_E11R1_20261111_contexts.json'
PREFERRED_E8 = 'outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497'
COHORTS = ('TP', 'FP', 'FN', 'TN')
RARE_CANDIDATES = ('BlueHarbor', 'CobaltOrchid', 'QuartzShimmer', 'VioletTamarind',
                   'CopperMarigold', 'AmberSeahorse', 'FrostCoriander',
                   'IndigoGlasswork', 'NebulaOrchard', 'SilverCobalt',
                   'PeridotVelvet', 'CrimsonSundial', 'ZephyrPinecone')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def extract_instructions(path):
    arr = load_json(path)
    if not isinstance(arr, list):
        raise ValueError(f'Expected JSON array at {path}')
    result = []
    for i, item in enumerate(arr):
        x = item.get('instruction') if isinstance(item, dict) else item
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f'Invalid instruction: {path} index={i}')
        result.append(x)
    return result


def canonical(s):
    return ' '.join(s.split()).casefold()


def median(seq):
    v = [float(x) for x in seq if x is not None and math.isfinite(float(x))]
    return float(statistics.median(v)) if v else None


def cos(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-10 else 0.0


def norm(a):
    import numpy as np
    return float(np.linalg.norm(a))


def coherence(vecs):
    import numpy as np
    if not vecs:
        return None
    arr = [np.asarray(v, dtype=np.float64) for v in vecs]
    return float(np.linalg.norm(np.mean([v / (np.linalg.norm(v) + 1e-12) for v in arr], axis=0)))


def remove_span(s, bounds):
    a, b = bounds
    if not (0 <= a < b <= len(s)):
        raise ValueError('invalid character span')
    # Match E10R1's whitespace-preserving removal (do not delete any other word).
    if a > 0 and b < len(s) and s[a - 1] == ' ' and s[b] == ' ':
        return s[:a] + s[b + 1:]
    return s[:a] + s[b:]


def replace_span(s, bounds, word):
    a, b = bounds
    if not (0 <= a < b <= len(s)):
        raise ValueError('invalid character span')
    return s[:a] + word + s[b:]


def insert_same_site(s, word):
    if not word or word.isspace():
        raise ValueError('empty insertion word')
    m = re.search(r'\S+\s+', s)
    if m is None:
        raise ValueError('reference context has no stable insertion site')
    return s[:m.end()] + word + ' ' + s[m.end():]


def group_label(gt, score, cutoff):
    pred = float(score) > float(cutoff)
    return ('TP' if pred else 'FN') if gt == 1 else ('FP' if pred else 'TN')


def rank_order(rows, group, limit):
    """Stratify E8 scores for TP; others use highest scores. No E11 metric used."""
    rows = sorted(rows, key=lambda r: (-r['score'], r['idx']))
    if group != 'TP' or len(rows) <= limit or limit < 2:
        return rows
    chosen = []
    for i in range(limit):
        j = round(i * (len(rows) - 1) / (limit - 1))
        if j not in chosen:
            chosen.append(j)
    return [rows[i] for i in chosen] + [row for i, row in enumerate(rows) if i not in chosen]


def guess_e8_run(provided=None):
    if provided:
        run = Path(provided)
        if not (run / 'summary.json').is_file() or not (run / 'heldout_records.json').is_file():
            raise FileNotFoundError(f'Missing E8 summary or records: {run}')
        return run
    favorite = Path(PREFERRED_E8)
    if (favorite / 'summary.json').is_file() and (favorite / 'heldout_records.json').is_file():
        return favorite
    root = Path('outputs/interactionprism/e8_runs')
    valid = []
    if root.exists():
        for s in root.glob('*/summary.json'):
            p = s.parent
            if not (p / 'heldout_records.json').is_file():
                continue
            try:
                if load_json(s).get('version') == E8_VERSION:
                    valid.append(p)
            except (ValueError, OSError, json.JSONDecodeError):
                continue
    if len(valid) != 1:
        raise RuntimeError('E8 run not uniquely identified. Pass --e8-run=path or E8_RUN=/path when launching. '\
                           f'Found {len(valid)} eligible runs: {[str(x) for x in valid[:8]]}')
    return valid[0]


def preflight(args):
    e8_dir = guess_e8_run(args.e8_run)
    summary = load_json(e8_dir / 'summary.json')
    records = load_json(e8_dir / 'heldout_records.json')
    if summary.get('version') != E8_VERSION:
        raise ValueError('E8 version not the expected frozen E8R1')
    cutoff = float(summary['calibration_cutoff'])
    if not (math.isfinite(cutoff) and cutoff > 0):
        raise ValueError('Invalid E8 cutoff')
    if not isinstance(records, list) or len(records) != 400:
        raise ValueError(f'Expected exactly 400 archived E8 heldout records, found {len(records)}')
    if not Path(args.helper).is_file():
        raise FileNotFoundError('Frozen E8 helper not found')
    data_paths = {'train_poison': args.train_poison, 'train_clean': args.train_clean,
                  'test_poison': args.test_poison, 'test_clean': args.test_clean}
    datasets = {name: extract_instructions(path) for name, path in data_paths.items()}
    if any(len(datasets[key]) < n for key, n in (('train_poison', 30), ('train_clean', 500), ('test_poison', 200), ('test_clean', 200))):
        raise ValueError('E8 data sources unexpectedly shorter than frozen E8 split')
    expected = summary.get('file_hashes', {})
    if not isinstance(expected, dict) or len(expected) < 5:
        raise ValueError('E8 summary file_hashes missing/incomplete; cannot verify provenance')
    observed = {str(Path(p).name): digest for p, digest in expected.items()}
    for path in [args.helper, *data_paths.values()]:
        key = Path(path).name
        if key not in observed or sha(path) != observed[key]:
            raise ValueError(f'FROZEN E8 hash mismatch or unknown source: {path}. Stop, do not override.')
    contexts = extract_instructions(args.contexts)
    can = [canonical(s) for s in contexts]
    if len(can) != len(set(can)):
        raise ValueError('Duplicate reference contexts')
    seen = {canonical(s) for data in datasets.values() for s in data}
    overlap = [i for i, s in enumerate(can) if s in seen]
    if overlap:
        raise ValueError(f'Reference contexts overlap E8 data by normalized exact text: {overlap}')
    if args.n_contexts < 4 or args.n_contexts > len(contexts):
        raise ValueError('Require 4 <= n-contexts <= number of independent contexts')
    groups = defaultdict(list)
    for idx, row in enumerate(records):
        if row.get('category') != 'heldout' or row.get('idx') != idx:
            raise ValueError(f'E8 record index/category mismatch at {idx}')
        if row.get('gt') != (1 if idx < 200 else 0):
            raise ValueError(f'E8 heldout order/labels unexpectedly changed at {idx}')
        if row.get('score') is None or row.get('skipped'):
            continue
        group = group_label(row['gt'], row['score'], cutoff)
        if not row.get('pool') or not row.get('singleton_abs') or not row.get('winning_word'):
            continue
        groups[group].append({**row, 'idx': idx, 'cohort': group})
    for group in COHORTS:
        if not groups[group]:
            raise ValueError(f'No usable {group} E8 records for diagnostic audit')
    selected = {g: rank_order(groups[g], g, getattr(args, 'n_' + g.lower())) for g in COHORTS}
    print('[PREFLIGHT OK] Frozen E8R1 verified; SHA hashes match archived E8 summary')
    print(f'[PREFLIGHT] E8={e8_dir}; cutoff={cutoff:.8f}; E8 cohorts={ {g:len(groups[g]) for g in COHORTS} }')
    print(f'[PREFLIGHT] unique new clean reference contexts={len(contexts)}; no exact normalized overlap')
    return e8_dir, summary, records, datasets, contexts, selected, cutoff


def load_helper(path):
    spec = importlib.util.spec_from_file_location('e8_primitives_frozen_e11r1', str(path))
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load E8 helper')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ('DEFAULTS', '_compose', '_word_spans', '_screen_words', '_forward'):
        if not hasattr(module, attr):
            raise ImportError(f'Frozen helper missing {attr}')
    return module


def recover_char_span(tok, original, ids, span):
    """Token-index provenance: use offset mapping of EXACT same original tokenization.

    Avoids ambiguous substring search if winning_word occurs multiple times.
    Fail closed on multi-word spans, decoding drift, or offsets discrepancy.
    """
    pre = 1 if tok.bos_token_id is not None else 0
    mapping = tok(original, add_special_tokens=False, return_offsets_mapping=True)
    mids = mapping['input_ids']
    if ids[pre:] != mids[:len(ids) - pre]:
        raise ValueError('offset mapping not aligned with E8 _compose token IDs')
    offsets = mapping['offset_mapping']
    positions = [p - pre for p in span]
    if not positions or any(p < 0 or p >= len(offsets) for p in positions):
        raise ValueError('invalid E8 player token positions')
    left = min(offsets[p][0] for p in positions)
    right = max(offsets[p][1] for p in positions)
    if not (0 <= left < right <= len(original)):
        raise ValueError('invalid raw offset interval')
    # Omit word-external whitespace, but do not merge any *interior* whitespace.
    raw = original[left:right]
    a = left + len(raw) - len(raw.lstrip())
    b = right - (len(raw) - len(raw.rstrip()))
    raw = original[a:b]
    if not raw or re.search(r'\S\s+\S', raw) or '\n' in raw or '\r' in raw:
        raise ValueError('cross-whitespace/empty E8 player; cannot safely intervene')
    return (a, b), raw


def reconstruct_e8_player(model, tok, helper, text, row, device, cfg, pad, tail):
    import numpy as np
    max_orig = cfg['max_len'] - len(tail)
    ids, maskable, _ = helper._compose(tok, text, max_orig)
    spans = helper._word_spans(tok, ids, maskable)
    if len(spans) != int(row['n_words']):
        raise ValueError('E8 word-count mismatch')
    if len(spans) < 2:
        raise ValueError('too few word players')
    signal, _ = helper._screen_words(model, tok, ids, spans, None, device, cfg, pad)
    rank = np.argsort(-signal[:, 1])
    selected = sorted(int(j) for j in rank[:min(6, len(spans))])
    pool = [tok.decode([ids[p] for p in spans[j]]).strip() for j in selected]
    if pool != row['pool']:
        raise ValueError(f'E8 top-6 reconstructed pool mismatch: {pool!r} vs {row["pool"]!r}')
    values = row['singleton_abs']
    if len(values) != len(pool):
        raise ValueError('E8 singleton abs length mismatch')
    winner = int(max(range(len(values)), key=lambda j: values[j]))
    if pool[winner] != row['winning_word']:
        raise ValueError('E8 winner value/text mismatch')
    span, raw = recover_char_span(tok, text, ids, spans[selected[winner]])
    if canonical(raw) != canonical(pool[winner]):
        raise ValueError(f'E8 winner raw-decoded mismatch: {raw!r} vs {pool[winner]!r}')
    return {'selected_token_positions': [int(p) for p in spans[selected[winner]]],
            'selected_word_index': int(selected[winner]), 'char_span': list(span),
            'raw_player': raw, 'decoded_player': pool[winner],
            'screening_recomputed': True}


def tokenized_readout(helper, tok, text, tail, max_len):
    pre = [tok.bos_token_id] if tok.bos_token_id is not None else []
    mid = tok(text, add_special_tokens=False)['input_ids']
    if len(pre) + len(mid) + len(tail) > max_len:
        raise ValueError(f'text would truncate with readout ({len(pre)+len(mid)+len(tail)} > {max_len})')
    ids, _, _ = helper._compose(tok, text, max_len - len(tail))
    if ids != pre + mid:
        raise ValueError('E8 _compose mismatch')
    return ids + list(tail)


def hidden(helper, model, tok, texts, tail, device, cfg, pad):
    ids = [tokenized_readout(helper, tok, t, tail, cfg['max_len']) for t in texts]
    return helper._forward(model, ids, device, cfg['hidden_layer'], pad, cfg['batch_size'])


def generate(model, tok, helper, text, device, max_len, max_new_tokens):
    import torch
    ids = tokenized_readout(helper, tok, text, [], max_len)
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        # Exact raw BOS+instruction generation, independent of any known target phrase.
        g = model.generate(input_ids=t, attention_mask=torch.ones_like(t), do_sample=False,
                           max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0, len(ids):], skip_special_tokens=True).strip()


def change_score(a, b):
    # Only a descriptive string-change proxy; NOT evidence of an attack by itself.
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def pick_contexts(contexts, word, n):
    """Avoid accidental presence of selected player in clean reference text."""
    needle = canonical(word)
    pat = re.compile(r'(?<!\w)' + re.escape(needle) + r'(?!\w)', flags=re.UNICODE)
    eligible = [s for s in contexts if not pat.search(canonical(s))]
    if len(eligible) < n:
        raise ValueError(f'Only {len(eligible)} contexts without selected player; need {n}')
    return eligible[:n]


def rare_for(tok, candidate, avoid):
    target_count = len(tok(' ' + candidate, add_special_tokens=False)['input_ids'])
    choices = [w for w in RARE_CANDIDATES if w.casefold() not in avoid and w != candidate]
    exact = [w for w in choices if len(tok(' ' + w, add_special_tokens=False)['input_ids']) == target_count]
    # If exact matching unavailable, still present rare as an UNMATCHED exploratory arm.
    word = exact[0] if exact else choices[0]
    return word, bool(exact), target_count, len(tok(' ' + word, add_special_tokens=False)['input_ids'])


def measure_case(model, tok, helper, original, metadata, contexts, device, cfg, pad,
                 tail, args):
    import numpy as np
    raw = metadata['raw_player']
    sp = tuple(metadata['char_span'])
    removed = remove_span(original, sp)
    neutralized = replace_span(original, sp, 'thing')
    if not removed.strip() or removed == original or neutralized == original:
        raise ValueError('invalid or no-op original input intervention')
    if raw.casefold() in ('thing', 'garden'):
        raise ValueError('selected candidate overlaps primary control word')
    refs = pick_contexts(contexts, raw, args.n_contexts)
    avoid = {raw.casefold(), 'thing', 'garden'} | {canonical(s) for s in refs}
    rare, matched, cand_count, rare_count = rare_for(tok, raw, avoid)
    arms = {'candidate': raw, 'neutral': 'thing', 'ordinary': 'garden', 'rare': rare}
    original_conditions = {'original': original, 'removed': removed, 'neutralized': neutralized}
    paired = [(s, {arm: insert_same_site(s, term) for arm, term in arms.items()}) for s in refs]
    # Verify no truncation in ANY measured condition; fail closed per case.
    seqs = list(original_conditions.values())
    for s, variants in paired:
        seqs += [s] + list(variants.values())
    for s in seqs:
        tokenized_readout(helper, tok, s, tail, cfg['max_len'])
    H = hidden(helper, model, tok, seqs, tail, device, cfg, pad).astype(np.float64)
    # Reference center is intentionally not recomputed: Δ(z)=Δ(h-h_bar)=Δ(h).
    x0, xr, xn = H[:3]
    dremove, dneutral = x0 - xr, x0 - xn
    by_arm = defaultdict(list)
    per_ref = []
    idx = 3
    for s, variants in paired:
        h0 = H[idx]
        idx += 1
        record = {'reference': s, 'deltas': {}}
        for arm in arms:
            vec = H[idx] - h0
            idx += 1
            by_arm[arm].append(vec)
            record['deltas'][arm] = norm(vec)
        per_ref.append(record)
    if idx != len(H):
        raise AssertionError('Hidden ordering mismatch')
    align = {arm: median([cos(dremove, v) for v in vecs]) for arm, vecs in by_arm.items()}
    n_align = {arm: median([cos(dneutral, v) for v in vecs]) for arm, vecs in by_arm.items()}
    magnitudes = {arm: median([norm(v) for v in vecs]) for arm, vecs in by_arm.items()}
    consist = {arm: coherence(vecs) for arm, vecs in by_arm.items()}
    diffs = {ctrl: median([norm(a)-norm(b) for a, b in zip(by_arm['candidate'], by_arm[ctrl])])
             for ctrl in ('neutral', 'ordinary', 'rare')}
    output = {'available': not args.skip_generation, 'no_known_target_used': True}
    if not args.skip_generation:
        gen_x = {k: generate(model, tok, helper, s, device, cfg['max_len'], args.max_new_tokens)
                 for k, s in original_conditions.items()}
        gen_rows = []
        for s, variants in paired[:args.n_gen_contexts]:
            base = generate(model, tok, helper, s, device, cfg['max_len'], args.max_new_tokens)
            variants_g = {arm: generate(model, tok, helper, t, device, cfg['max_len'], args.max_new_tokens)
                          for arm, t in variants.items()}
            gen_rows.append({'reference': s, 'original': base, 'versions': variants_g,
                             'text_change': {arm: change_score(base, out) for arm, out in variants_g.items()}})
        output.update({'original_generation': gen_x,
                       'remove_text_change': change_score(gen_x['original'], gen_x['removed']),
                       'neutralize_text_change': change_score(gen_x['original'], gen_x['neutralized']),
                       'reference_generation': gen_rows,
                       'candidate_reference_text_change_median': median([r['text_change']['candidate'] for r in gen_rows]),
                       'control_reference_text_change_medians': {
                           arm: median([r['text_change'][arm] for r in gen_rows]) for arm in ('neutral', 'ordinary', 'rare')}})
    return {'player': metadata, 'original': original, 'removed': removed, 'neutralized': neutralized,
            'n_contexts': len(refs), 'control_words': arms, 'rare_token_count_matched': matched,
            'candidate_token_count': cand_count, 'rare_token_count': rare_count,
            'remove_delta_norm': norm(dremove), 'neutralize_delta_norm': norm(dneutral),
            'magnitude_medians': magnitudes, 'direction_coherence': consist,
            'remove_to_insertion_alignment': align,
            'neutralize_to_insertion_alignment': n_align,
            'candidate_minus_control_norm_medians': diffs,
            'reference_deltas': per_ref, 'generation': output}


def summarize(rows, cohort):
    r = [x for x in rows if x['cohort'] == cohort]
    if not r:
        return {'n': 0}
    return {'n': len(r),
            'e8_score_median': median([x['e8_score'] for x in r]),
            'candidate_norm_median': median([x['e11']['magnitude_medians']['candidate'] for x in r]),
            'neutral_norm_median': median([x['e11']['magnitude_medians']['neutral'] for x in r]),
            'ordinary_norm_median': median([x['e11']['magnitude_medians']['ordinary'] for x in r]),
            'rare_norm_median': median([x['e11']['magnitude_medians']['rare'] for x in r]),
            'candidate_coherence_median': median([x['e11']['direction_coherence']['candidate'] for x in r]),
            'candidate_remove_alignment_median': median([x['e11']['remove_to_insertion_alignment']['candidate'] for x in r]),
            'control_remove_alignment_medians': {arm: median([x['e11']['remove_to_insertion_alignment'][arm] for x in r])
                                                 for arm in ('neutral', 'ordinary', 'rare')},
            'candidate_minus_neutral_norm_median': median([x['e11']['candidate_minus_control_norm_medians']['neutral'] for x in r]),
            'candidate_minus_ordinary_norm_median': median([x['e11']['candidate_minus_control_norm_medians']['ordinary'] for x in r]),
            'candidate_minus_rare_norm_median': median([x['e11']['candidate_minus_control_norm_medians']['rare'] for x in r]),
            'remove_text_change_median': median([x['e11']['generation'].get('remove_text_change') for x in r]),
            'rare_token_matched_n': sum(bool(x['e11']['rare_token_count_matched']) for x in r)}


def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    outdir = Path(args.output)
    if any((outdir / f).exists() for f in ('summary.json', 'case_records.jsonl', 'copy_to_chatgpt.txt')):
        raise FileExistsError(f'Existing experiment output would be overwritten: {outdir}')
    outdir.mkdir(parents=True, exist_ok=True)
    e8_dir, e8_sum, e8_records, datasets, contexts, candidates, cutoff = preflight(args)
    if not torch.cuda.is_available():
        raise RuntimeError('E11R1 requires a CUDA GPU')
    helper = load_helper(args.helper)
    cfg = dict(helper.DEFAULTS)
    cfg['max_len'], cfg['batch_size'], cfg['pool_size'] = args.max_len, args.batch_size, 6
    cfg['hidden_layer'] = -1
    # IMPORTANT: E8's screening uses this separate frozen screen_layer from DEFAULTS.
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError('Fast tokenizer with offset_mapping required to locate true player spans')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad = tok.pad_token_id
    tail = tok(READOUT, add_special_tokens=False)['input_ids']
    if len(tail) != 4:
        raise ValueError(f'E8 readout suffix expected 4 tokens, got {len(tail)}')
    print(f'[E11] loading frozen model: {args.base} + {args.adapter}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 device_map='auto', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device
    case_results, skipped = [], []
    for group in COHORTS:
        quota = getattr(args, 'n_' + group.lower())
        done = 0
        for row in candidates[group]:
            if done >= quota:
                break
            idx = row['idx']
            original = datasets['test_poison'][idx] if idx < 200 else datasets['test_clean'][idx-200]
            try:
                loc = reconstruct_e8_player(model, tok, helper, original, row, device, cfg, pad, tail)
                metrics = measure_case(model, tok, helper, original, loc, contexts, device, cfg, pad, tail, args)
            except (ValueError, AssertionError, IndexError) as e:
                skipped.append({'cohort': group, 'idx': idx, 'reason': str(e)})
                print(f'[E11][SKIP] {group} idx={idx}: {e}', flush=True)
                continue
            item = {'cohort': group, 'idx': idx, 'gt': row['gt'], 'e8_score': row['score'],
                    'e8_pred': int(row['score'] > cutoff), 'e8_winning_word': row['winning_word'],
                    'e11': metrics}
            case_results.append(item)
            done += 1
            print(f'[E11] {group} {done}/{quota} E8 idx={idx} candidate={loc["raw_player"]!r} '
                  f'align={metrics["remove_to_insertion_alignment"]["candidate"]:.4f}', flush=True)
        if done < quota:
            skipped.append({'cohort': group, 'idx': None, 'reason': f'quota short: {done}/{quota}'})
    bygroup = {g: summarize(case_results, g) for g in COHORTS}
    if bygroup['TP']['n'] < 2 or bygroup['FP']['n'] < 2:
        raise RuntimeError('Too few E8 TP/FP samples accurately localized for E11R1 mechanism comparison; '
                           'see [E11][SKIP] reasons. No science results will be labeled successful.')
    manifest = {
        'experiment': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'objective': 'exploratory blind player -> counterfactual evidence; NO new detector threshold',
        'e8_dir': str(e8_dir.resolve()), 'e8_threshold_frozen': cutoff,
        'e8_summary_sha256': sha(e8_dir / 'summary.json'),
        'e8_records_sha256': sha(e8_dir / 'heldout_records.json'),
        'e8_helper_sha256': sha(args.helper), 'reference_sha256': sha(args.contexts),
        'model': args.base, 'adapter': args.adapter, 'readout_suffix': repr(READOUT),
        'screen_layer': cfg['screen_layer'], 'hidden_layer': cfg['hidden_layer'],
        'generation_protocol': 'raw BOS+instruction, greedy; no known target text',
        'max_len': cfg['max_len'], 'batch_size': cfg['batch_size'],
        'requested_quotas': {g: getattr(args, 'n_' + g.lower()) for g in COHORTS},
        'n_reference_contexts_per_case': args.n_contexts,
        'n_generation_contexts_per_case': 0 if args.skip_generation else args.n_gen_contexts,
        'new_contexts_checked_independent_of_e8': True,
        'candidate_selection': 'frozen E8 heldout only; E8 first-stage scores never recalculated or overwritten',
        'localization': 'repeat frozen E8 screening, check exact pool; recover winning token offsets',
        'label_use': 'E8 heldout labels only for post-hoc offline case-control audit; no label in stage-2 features',
        'reference_center_note': 'h paired-differences identical to centered z paired-differences; E8 center cancels',
        'caveat': 'Only BadNet/LoRA, historical E8 heldout reused, small selected case-control design; no generalization/F1 claim',
    }
    summary = {'manifest': manifest, 'n_processed': len(case_results), 'n_skipped': len(skipped),
               'skip_reason_counts': dict(Counter(x['reason'] for x in skipped)),
               'by_cohort': bygroup,
               'comparison_warning': ('Descriptive comparison only: TP/FP/FN/TN selected using existing heldout GT. '
                   'Coherence/alignment/text-change are diagnostics, not proven backdoor classifiers. '
                   'No tuning or evaluation of a fused threshold on these cases.')}
    save_json(outdir / 'manifest.json', manifest)
    save_json(outdir / 'summary.json', summary)
    write_jsonl(outdir / 'case_records.jsonl', case_results)
    write_jsonl(outdir / 'skipped.jsonl', skipped)
    print_lines = ['========== COPY_TO_CHATGPT_BEGIN ==========',
                   f'experiment={VERSION}', 'status=SUCCESS',
                   f'e8_run={e8_dir}', f'e8_cutoff={cutoff:.8f}',
                   'objective=frozen E8 auto-player -> counterfactual evidence; NOT_NEW_DETECTOR',
                   f'processed={len(case_results)} skipped={len(skipped)}',
                   'counts=' + ' '.join(f'{g}:{bygroup[g]["n"]}' for g in COHORTS)]
    def fmt(x):
        return 'NA' if x is None else f'{x:.4f}'
    for g in COHORTS:
        r = bygroup[g]
        if not r['n']:
            print_lines.append(f'{g}=N0')
            continue
        print_lines.append(f'{g}=n:{r["n"]}, median_e8:{fmt(r["e8_score_median"])}, '
                           f'insert_candidate_norm:{fmt(r["candidate_norm_median"])}, '
                           f'candidate_minus_neutral_norm:{fmt(r["candidate_minus_neutral_norm_median"])}, '
                           f'candidate_coherence:{fmt(r["candidate_coherence_median"])}, '
                           f'remove_insert_alignment:{fmt(r["candidate_remove_alignment_median"])}, '
                           f'neutral_alignment:{fmt(r["control_remove_alignment_medians"]["neutral"])}, '
                           f'original_vs_removed_text_change:{fmt(r["remove_text_change_median"])}')
    print_lines.append('examples=' + '; '.join(f'{r["cohort"]}[{r["idx"]}]={r["e11"]["player"]["raw_player"]!r}'
                                              for r in case_results[:18]))
    print_lines.append('skipped_examples=' + '; '.join(f'{s["cohort"]}[{s["idx"]}]={s["reason"][:55]}' for s in skipped[:8]))
    print_lines += ['notes=NO_KNOWN_TRIGGER; NO_KNOWN_TARGET; E8_FROZEN; NO_NEW_THRESHOLD; DIAGNOSTIC_ONLY',
                    '========== COPY_TO_CHATGPT_END ==========']
    copy = '\n'.join(print_lines) + '\n'
    (outdir / 'copy_to_chatgpt.txt').write_text(copy, encoding='utf-8')
    (outdir / 'report.md').write_text(
        f'# {VERSION} exploratory report\n\n'
        'This is a post-hoc case-control mechanism audit of frozen E8, NOT a new detector.\n\n'
        + '| Cohort | N | E8 median | Candidate norm | Candidate coherence | Removal alignment |\n'
        + '|---|---:|---:|---:|---:|---:|\n'
        + ''.join(f'| {g} | {bygroup[g]["n"]} | {fmt(bygroup[g].get("e8_score_median"))} | '
                  f'{fmt(bygroup[g].get("candidate_norm_median"))} | '
                  f'{fmt(bygroup[g].get("candidate_coherence_median"))} | '
                  f'{fmt(bygroup[g].get("candidate_remove_alignment_median"))} |\n' for g in COHORTS)
        + '\nNo unknown-target behavior classifier is claimed. String changes are descriptive only.\n',
        encoding='utf-8')
    print(copy, end='', flush=True)
    print(f'[SUCCESS] E11R1 results: {outdir.resolve()}')
    print(f'[E11] Re-copy: cat "{(outdir / "copy_to_chatgpt.txt").resolve()}"')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--e8-run', default=None)
    p.add_argument('--helper', default=HELPER)
    p.add_argument('--base', default=BASE)
    p.add_argument('--adapter', default=ADAPTER)
    p.add_argument('--train-poison', default=TRAIN_POISON)
    p.add_argument('--train-clean', default=TRAIN_CLEAN)
    p.add_argument('--test-poison', default=TEST_POISON)
    p.add_argument('--test-clean', default=TEST_CLEAN)
    p.add_argument('--contexts', default=CONTEXTS)
    p.add_argument('--n-tp', type=int, default=8)
    p.add_argument('--n-fp', type=int, default=4)
    p.add_argument('--n-fn', type=int, default=4)
    p.add_argument('--n-tn', type=int, default=4)
    p.add_argument('--n-contexts', type=int, default=8)
    p.add_argument('--n-gen-contexts', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--skip-generation', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.n_gen_contexts > args.n_contexts or args.n_gen_contexts < 0:
        raise ValueError('0 <= n-gen-contexts <= n-contexts required')
    if args.preflight:
        preflight(args)
        return
    if not args.output:
        raise ValueError('Please specify --output, with a unique new run directory')
    run(args)


if __name__ == '__main__':
    main()
