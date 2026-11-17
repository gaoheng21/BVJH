#!/usr/bin/env python3
"""E11R3: full 400-case E8 heldout audit + automatic player counterfactual verification.

One attacked model, no known trigger/target at inference. E8 stored score/prediction is
immutable; all second-order Harsanyi dividends are NEW auxiliary diagnostics, not E8.
Ground truth labels are attached only to exported offline research records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import IP_E11R3_20261111_frozen_e11r1_utils as u

VERSION = 'IP_E11R3_20261111'
E8_VERSION = 'IP_E8R1_20261111'
READOUT = '\nAnswer:'
PREFERRED_E8 = u.PREFERRED_E8
CSV_NAME = 'all_400_samples_interaction_audit.csv'
POISON_CSV_NAME = 'poisoned_200_samples_interaction_audit.csv'
JSONL_NAME = 'all_400_samples_interaction_audit.jsonl'
CONTEXTS = 'IP_E11R3_20261111_contexts.json'
CONTROL_NEUTRAL = ('thing', 'object', 'item')
CONTROL_ORDINARY = ('garden', 'window', 'river')


def canon(text):
    return u.canonical(text)


def csv_value(field, value):
    if value is None:
        return ''
    # CSV may be opened in Excel: never execute untrusted input text as a formula.
    if field in {'original_instruction', 'removed_instruction', 'neutralized_instruction',
                 'e8_winning_word', 'selected_player_raw', 'max_abs_I2_pair',
                 'generation_original', 'generation_removed'} and isinstance(value, str):
        if value.lstrip().startswith(('=', '+', '-', '@')):
            return "'" + value
    return str(value)


def get_e8_run(explicit):
    return u.guess_e8_run(explicit)


def check_preflight(args):
    e8_dir = get_e8_run(args.e8_run)
    summary = u.load_json(e8_dir / 'summary.json')
    records = u.load_json(e8_dir / 'heldout_records.json')
    if summary.get('version') != E8_VERSION or len(records) != 400:
        raise ValueError('Requires archived E8R1 with 400 original heldout records')
    cutoff = float(summary['calibration_cutoff'])
    if abs(cutoff - 0.237005) > 1e-5:
        raise ValueError(f'Unrecognized frozen E8R1 calibration cutoff: {cutoff}')
    paths = dict(helper=args.helper, train_poison=args.train_poison,
                 train_clean=args.train_clean, test_poison=args.test_poison,
                 test_clean=args.test_clean)
    hashes = summary.get('file_hashes', {})
    by_name = {Path(k).name: v for k, v in hashes.items()}
    for label, path in paths.items():
        name = Path(path).name
        if name not in by_name or u.sha(path) != by_name[name]:
            raise ValueError(f'{label}: file hash does not match frozen E8R1 summary: {path}')
    datasets = {k: u.extract_instructions(v) for k, v in paths.items() if k != 'helper'}
    if any(len(datasets[k]) < n for k, n in
           (('train_poison', 30), ('train_clean', 500), ('test_poison', 200), ('test_clean', 200))):
        raise ValueError('E8 data source lengths do not match frozen split')
    contexts = u.extract_instructions(args.contexts)
    if len(contexts) < args.n_contexts or args.n_contexts < 4:
        raise ValueError('Need at least n_contexts>=4 independent clean reference instructions')
    norm_contexts = [canon(x) for x in contexts]
    if len(set(norm_contexts)) != len(norm_contexts):
        raise ValueError('Duplicated reference contexts')
    inputs = {canon(x) for ds in datasets.values() for x in ds}
    if set(norm_contexts) & inputs:
        raise ValueError('Reference contexts have exact normalized overlap with E8 data')
    for i, rec in enumerate(records):
        if rec.get('idx') != i or rec.get('category') != 'heldout':
            raise ValueError(f'E8 stored record index mismatch: idx={i}')
        if rec.get('gt') != int(i < 200):
            raise ValueError(f'E8 heldout label/order mismatch: idx={i}')
    if args.max_len != 512:
        raise ValueError('Must use E8 original max_len=512; no silent readout change')
    if args.batch_size < 1 or args.batch_size > 16:
        raise ValueError('batch-size must be 1..16')
    if args.max_new_tokens < 1 or args.max_new_tokens > 128:
        raise ValueError('max-new-tokens out of range')
    return dict(e8_dir=e8_dir, summary=summary, records=records, cutoff=cutoff,
                datasets=datasets, contexts=contexts, hashes=paths)


def base_record(idx, e8row, original, cutoff):
    score = e8row.get('score')
    valid = score is not None and not e8row.get('skipped')
    pred = int(float(score) > cutoff) if valid else 0  # frozen E8 fail-closed convention
    gt = int(e8row['gt'])
    cohort = ('TP' if pred else 'FN') if gt else ('FP' if pred else 'TN')
    pool, vals = e8row.get('pool') or [], e8row.get('singleton_abs') or []
    if len(pool) != len(vals):
        raise ValueError(f'Bad archived E8 pool/abs length at idx {idx}')
    top6 = [{'pool_index': i, 'word': word, 'I1_abs_e8': float(v)}
            for i, (word, v) in enumerate(zip(pool, vals))]
    top6.sort(key=lambda x: (-x['I1_abs_e8'], x['pool_index']))
    return dict(sample_id=idx, data_split='E8_heldout', true_label=gt,
                true_label_name='poisoned' if gt else 'clean',
                e8_prediction=pred, e8_prediction_name='suspected_backdoor' if pred else 'normal',
                cohort=cohort, original_instruction=original,
                e8_score=None if score is None else float(score), e8_cutoff=cutoff,
                e8_skipped=e8row.get('skipped'), e8_scores=e8row.get('scores'),
                e8_winning_word=e8row.get('winning_word'),
                e8_top6_singleton_structure=top6,
                e8_singleton_max_abs=(top6[0]['I1_abs_e8'] if top6 else None),
                e8_singleton_second_abs=(top6[1]['I1_abs_e8'] if len(top6)>1 else None),
                e8_word_count=e8row.get('n_words'),
                localization_status='not_started', localization_method=None,
                location=None, pairwise_status='not_started',
                second_order_Harsanyi_pairs=[], recomputed_singletons=[],
                pairwise_consistency_max_abs_error=None,
                stage2_status='not_started', stage2_error=None,
                stage2=None, generation_status='not_measured')


def rank_reconstruction(model, tok, helper, text, row, device, cfg, pad):
    """Recover E8 top6 on the original tokenizer word boundaries.

    Exact reconstruction is ideal; if small numerical screening differences cause
    pool drift, locate a UNIQUE winning word span without using ground truth.
    Never pick an arbitrary occurrence of a repeated candidate.
    """
    ids, maskable, _ = helper._compose(tok, text, cfg['max_len'] - 4)
    spans = helper._word_spans(tok, ids, maskable)
    if not spans:
        raise ValueError('No word spans')
    if len(spans) != row['n_words']:
        raise ValueError('E8 tokenizer word-count mismatch')
    signal, _ = helper._screen_words(model, tok, ids, spans, None, device, cfg, pad)
    ranked = np.argsort(-signal[:, 1])
    selected = sorted(int(j) for j in ranked[:min(6, len(spans))])
    actual = [tok.decode([ids[p] for p in spans[j]]).strip() for j in selected]
    expected = row['pool']
    wname = row['winning_word']
    exact_pool = actual == expected
    if exact_pool:
        winner = int(max(range(len(row['singleton_abs'])), key=lambda j: row['singleton_abs'][j]))
        if expected[winner] != wname:
            raise ValueError('E8 winning value/text mismatch')
        pos = selected[winner]
        method = 'exact_e8_pool'
    else:
        matching = [j for j, span in enumerate(spans)
                    if canon(tok.decode([ids[p] for p in span]).strip()) == canon(wname)]
        if len(matching) != 1:
            raise ValueError(f'E8 pool drift and winner not uniquely localizable ({len(matching)} matches)')
        pos = matching[0]
        method = 'unique_span_fallback_pool_drift'
    charspan, raw = u.recover_char_span(tok, text, ids, spans[pos])
    if canon(raw) != canon(wname):
        raise ValueError(f'E8 winner/raw offset mismatch: {raw!r} != {wname!r}')
    return dict(raw_player=raw, decoded_player=wname, char_span=list(charspan),
                selected_word_index=int(pos), selected_token_positions=list(spans[pos]),
                localization_method=method, pool_exact=exact_pool,
                reconstructed_pool=actual, selected_word_indices=selected,
                ids=ids, spans=spans)


def pick_controls(raw):
    v = raw.casefold()
    n = next((s for s in CONTROL_NEUTRAL if s != v), None)
    o = next((s for s in CONTROL_ORDINARY if s != v and s != n), None)
    if n is None or o is None:
        raise ValueError('Could not choose independent ordinary controls')
    return n, o


def cached_reference_for_candidate(model, tok, helper, raw, contexts, device, cfg, pad, tail, args):
    from collections import defaultdict
    refs = u.pick_contexts(contexts, raw, args.n_contexts)
    neutral, ordinary = pick_controls(raw)
    avoid = {raw.casefold(), neutral.casefold(), ordinary.casefold()}
    rare_options = [w for w in u.RARE_CANDIDATES if w.casefold() not in avoid]
    if not rare_options:
        raise ValueError('No rare-word control available')
    candidate_tokens = len(tok(' '+raw, add_special_tokens=False)['input_ids'])
    matched = [w for w in rare_options if len(tok(' '+w, add_special_tokens=False)['input_ids']) == candidate_tokens]
    rare = matched[0] if matched else rare_options[0]
    rare_tokens = len(tok(' '+rare, add_special_tokens=False)['input_ids'])
    arms = dict(candidate=raw, neutral=neutral, ordinary=ordinary, rare=rare)
    seqs = []
    variants_by_ref = []
    for ctx in refs:
        variants = {name: u.insert_same_site(ctx, word) for name, word in arms.items()}
        variants_by_ref.append(variants)
        seqs.extend([ctx] + list(variants.values()))
    for text in seqs:
        u.tokenized_readout(helper, tok, text, tail, cfg['max_len'])
    H = u.hidden(helper, model, tok, seqs, tail, device, cfg, pad).astype(np.float64)
    by_arm = defaultdict(list)
    per_ref = []
    idx = 0
    for ctx in refs:
        original_vec = H[idx]
        idx += 1
        deltas = {}
        for name in arms:
            delta = H[idx] - original_vec
            idx += 1
            by_arm[name].append(delta)
            deltas[name] = u.norm(delta)
        per_ref.append({'reference': ctx, 'delta_norms': deltas})
    if idx != len(H):
        raise AssertionError('Reference hidden state indexing drift')
    return dict(arms=arms, rare_token_count_matched=bool(matched),
                candidate_token_count=candidate_tokens, rare_token_count=rare_tokens,
                refs=refs, by_arm=dict(by_arm), per_reference=per_ref,
                magnitude_medians={arm: u.median([u.norm(v) for v in vs]) for arm, vs in by_arm.items()},
                direction_coherence={arm: u.coherence(vs) for arm, vs in by_arm.items()},
                candidate_minus_control_norm_medians={ctrl: u.median([
                    u.norm(a)-u.norm(b) for a, b in zip(by_arm['candidate'], by_arm[ctrl])])
                    for ctrl in ('neutral', 'ordinary', 'rare')})


def get_ref_from_lru(cache, key, provider, limit):
    if key in cache:
        cache.move_to_end(key)
        return cache[key], True
    val = provider()
    cache[key] = val
    cache.move_to_end(key)
    if len(cache) > limit:
        cache.popitem(last=False)
    return val, False


def stage2_measure(model, tok, helper, rawtext, loc, refs, device, cfg, pad, tail,
                   args):
    raw = loc['raw_player']
    span = loc['char_span']
    removed = u.remove_span(rawtext, span)
    neutral = refs['arms']['neutral']
    replaced = u.replace_span(rawtext, span, neutral)
    if not removed.strip() or removed == rawtext or replaced == rawtext:
        raise ValueError('Removal/replacement would produce blank or unchanged prompt')
    seqs = [rawtext, removed, replaced]
    for s in seqs:
        u.tokenized_readout(helper, tok, s, tail, cfg['max_len'])
    H = u.hidden(helper, model, tok, seqs, tail, device, cfg, pad).astype(np.float64)
    delta_rem = H[0] - H[1]
    delta_neu = H[0] - H[2]
    al_rem = {arm: u.median([u.cos(delta_rem, v) for v in vs])
              for arm, vs in refs['by_arm'].items()}
    al_neu = {arm: u.median([u.cos(delta_neu, v) for v in vs])
              for arm, vs in refs['by_arm'].items()}
    per_reference = []
    for j, ref in enumerate(refs['per_reference']):
        per_reference.append({'reference_index': j, 'reference_instruction': ref['reference'],
                              'insert_norms': ref['delta_norms'],
                              'remove_insert_cosines': {
                                  arm: u.cos(delta_rem, vs[j])
                                  for arm, vs in refs['by_arm'].items()}})
    generate_output = None
    if args.generate_original_removal:
        a = u.generate(model, tok, helper, rawtext, device, cfg['max_len'], args.max_new_tokens)
        b = u.generate(model, tok, helper, removed, device, cfg['max_len'], args.max_new_tokens)
        generate_output = dict(original=a, removed=b,
                               text_change=round(u.change_score(a,b), 9),
                               warning='String difference only; not known-target attack success')
    return dict(selected_player=raw,
                removed_instruction=removed, neutralized_instruction=replaced,
                controls=refs['arms'], rare_token_count_matched=refs['rare_token_count_matched'],
                candidate_token_count=refs['candidate_token_count'], rare_token_count=refs['rare_token_count'],
                n_independent_clean_contexts=len(refs['refs']),
                remove_delta_norm=u.norm(delta_rem), neutralize_delta_norm=u.norm(delta_neu),
                insert_magnitude_medians=refs['magnitude_medians'],
                insert_direction_coherence=refs['direction_coherence'],
                remove_to_insert_alignment=al_rem,
                neutralize_to_insert_alignment=al_neu,
                candidate_minus_control_norm_medians=refs['candidate_minus_control_norm_medians'],
                per_reference_evidence=per_reference,
                generation=generate_output)


def pairwise_measure(model, tok, helper, loc, e8row, hbar, device, cfg, pad, tail):
    """Auxiliary order<=2 game; signed I1 and signed I2, independent of E8 rule."""
    if not loc['pool_exact']:
        raise ValueError('Archived E8 Top-6 differs after re-screening; no comparable pair game')
    ids, spans = loc['ids'], loc['spans']
    units = [spans[j] for j in loc['selected_word_indices']]
    K = len(units)
    subsets = [frozenset()] + [frozenset({i}) for i in range(K)] + [
        frozenset({i, j}) for i, j in combinations(range(K), 2)]
    h0 = helper._forward(model, [ids+tail], device, cfg['hidden_layer'], pad, cfg['batch_size'])[0]
    values = helper._measure_v_fixed_tail(model, tok, ids, units, subsets, h0,
            hbar, device, cfg, pad, {sp[0] for sp in spans}, tail)
    dividends = helper._harsanyi(values)
    i1 = [float(dividends[frozenset({i})]) for i in range(K)]
    i2 = [{'first_pool_index':i, 'second_pool_index':j,
           'first_word':e8row['pool'][i], 'second_word':e8row['pool'][j],
           'I2_signed':float(dividends[frozenset({i,j})]),
           'I2_abs':abs(float(dividends[frozenset({i,j})]))}
          for i,j in combinations(range(K),2)]
    i2.sort(key=lambda x: (-x['I2_abs'], x['first_pool_index'], x['second_pool_index']))
    error = max(abs(abs(x)-float(y)) for x,y in zip(i1, e8row['singleton_abs']))
    return i1, i2, float(error)


def cohort_stats(rows):
    by_group = {}
    for group in ('TP','FP','FN','TN'):
        samples = [x for x in rows if x['cohort'] == group]
        good = [x for x in samples if x['stage2_status'] == 'ok']
        metrics = {
            'e8_count': len(samples), 'stage2_ok': len(good),
            'candidate_unique': len({x['e8_winning_word'] for x in good}),
            'e8_score_median': u.median([x['e8_score'] for x in samples]),
            'insert_candidate_norm_median': u.median([x['stage2']['insert_magnitude_medians']['candidate'] for x in good]),
            'candidate_minus_neutral_norm_median': u.median([x['stage2']['candidate_minus_control_norm_medians']['neutral'] for x in good]),
            'candidate_coherence_median': u.median([x['stage2']['insert_direction_coherence']['candidate'] for x in good]),
            'remove_insert_alignment_median': u.median([x['stage2']['remove_to_insert_alignment']['candidate'] for x in good]),
            'neutral_alignment_median': u.median([x['stage2']['remove_to_insert_alignment']['neutral'] for x in good]),
            'max_abs_I2_median': u.median([x['second_order_Harsanyi_pairs'][0]['I2_abs']
                for x in samples if x['second_order_Harsanyi_pairs']]),
            'pairwise_n': sum(bool(x['second_order_Harsanyi_pairs']) for x in samples)}
        by_group[group] = metrics
    return by_group


def as_csv_row(rec):
    st = rec.get('stage2') or {}
    loc = rec.get('location') or {}
    pairs = rec.get('second_order_Harsanyi_pairs') or []
    gen = st.get('generation') or {}
    def pick(d, *keys):
        for key in keys:
            d = d.get(key, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) else None
    def js(v):
        return json.dumps(v, ensure_ascii=False, separators=(',',':')) if v is not None else ''
    out = dict(
        sample_id=rec['sample_id'], true_label=rec['true_label'],
        true_label_name=rec['true_label_name'], e8_prediction=rec['e8_prediction'],
        e8_prediction_name=rec['e8_prediction_name'], cohort=rec['cohort'],
        e8_score=rec['e8_score'], e8_cutoff=rec['e8_cutoff'],
        e8_winning_word=rec['e8_winning_word'],
        selected_player_raw=loc.get('raw_player'), char_start=(loc.get('char_span') or [None,None])[0],
        char_end=(loc.get('char_span') or [None,None])[1],
        localization_method=rec.get('localization_method'),
        localization_status=rec['localization_status'],
        e8_top6_singleton_structure_json=js(rec['e8_top6_singleton_structure']),
        e8_singleton_max_abs=rec['e8_singleton_max_abs'],
        e8_singleton_second_abs=rec['e8_singleton_second_abs'],
        second_order_Harsanyi_pairs_json=js(pairs),
        max_abs_I2=(pairs[0]['I2_abs'] if pairs else None),
        max_abs_I2_pair=(pairs[0]['first_word']+' || '+pairs[0]['second_word'] if pairs else None),
        recomputed_signed_I1_json=js(rec.get('recomputed_singletons')),
        pairwise_consistency_max_abs_error=rec['pairwise_consistency_max_abs_error'],
        pairwise_status=rec['pairwise_status'],
        stage2_status=rec['stage2_status'], stage2_error=rec['stage2_error'],
        insert_candidate_norm=(st.get('insert_magnitude_medians') or {}).get('candidate'),
        insert_neutral_norm=(st.get('insert_magnitude_medians') or {}).get('neutral'),
        insert_ordinary_norm=(st.get('insert_magnitude_medians') or {}).get('ordinary'),
        insert_rare_norm=(st.get('insert_magnitude_medians') or {}).get('rare'),
        candidate_minus_neutral=(st.get('candidate_minus_control_norm_medians') or {}).get('neutral'),
        candidate_minus_ordinary=(st.get('candidate_minus_control_norm_medians') or {}).get('ordinary'),
        candidate_minus_rare=(st.get('candidate_minus_control_norm_medians') or {}).get('rare'),
        candidate_coherence=(st.get('insert_direction_coherence') or {}).get('candidate'),
        neutral_coherence=(st.get('insert_direction_coherence') or {}).get('neutral'),
        removal_norm=st.get('remove_delta_norm'),
        remove_insert_alignment=(st.get('remove_to_insert_alignment') or {}).get('candidate'),
        remove_neutral_alignment=(st.get('remove_to_insert_alignment') or {}).get('neutral'),
        neutralized_insert_alignment=(st.get('neutralize_to_insert_alignment') or {}).get('candidate'),
        rare_token_count_matched=st.get('rare_token_count_matched'),
        selected_player_token_count=st.get('candidate_token_count'),
        generation_status=rec.get('generation_status'),
        generation_original=gen.get('original'), generation_removed=gen.get('removed'),
        generation_text_change=gen.get('text_change'),
        original_instruction=rec['original_instruction'],
        removed_instruction=st.get('removed_instruction'),
        neutralized_instruction=st.get('neutralized_instruction'))
    return out


def emit_csvs(outdir, rows):
    if len(rows) != 400 or [r['sample_id'] for r in rows] != list(range(400)):
        raise ValueError('Cannot export success CSV: records are not complete ordered E8 heldout 400')
    columns = list(as_csv_row(rows[0]).keys())
    for name, subset in ((CSV_NAME, rows), (POISON_CSV_NAME, [r for r in rows if r['true_label']==1])):
        with (outdir / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for rec in subset:
                writer.writerow({k: csv_value(k, v) for k,v in as_csv_row(rec).items()})
    if sum(r['true_label'] for r in rows) != 200:
        raise ValueError('Poisoned subset export must contain exactly 200 original poison labels')


def existing_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) > 400 or [r['sample_id'] for r in rows] != list(range(len(rows))):
        raise ValueError('Resume JSONL contains incomplete or out-of-order records')
    return rows


def manifest(args, p):
    return dict(version=VERSION,
                e8_summary_sha256=u.sha(p['e8_dir'] / 'summary.json'),
                e8_records_sha256=u.sha(p['e8_dir'] / 'heldout_records.json'),
                helper_sha256=u.sha(args.helper), context_sha256=u.sha(args.contexts),
                e8_cutoff=p['cutoff'], n_contexts=args.n_contexts,
                max_len=args.max_len, batch_size=args.batch_size,
                pairwise_warning_tolerance=args.pairwise_warning_tolerance,
                model_path=args.base, adapter_path=args.adapter,
                generation=bool(args.generate_original_removal),
                max_new_tokens=args.max_new_tokens, pairwise_enabled=not args.no_pairwise,
                objective='frozen E8R1 heldout 400 (200 poison + 200 clean); no threshold tuning',
                label_policy='GT only for offline cohort and export; no target, trigger or labels used to choose players',
                readout='\\nAnswer:',
                first_order_status='E8 stored singleton absolute dividends; no recalibration or refitting',
                second_order_status='new diagnostic exact pairwise dividends on frozen top6; no E8 prediction impact')


def run(args, p):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('GPU/CUDA required for full-set E11R3')
    outdir = Path(args.resume) if args.resume else Path(args.output)
    if args.resume:
        if not outdir.is_dir() or not (outdir / 'manifest.json').is_file():
            raise FileNotFoundError('--resume must point to prior E11R3 partial run')
    else:
        # The shell launcher creates the fresh unique run directory first,
        # so tee can open console.log without a race. Only that log is allowed.
        if outdir.exists():
            if not outdir.is_dir() or any(p.name != 'console.log' for p in outdir.iterdir()):
                raise FileExistsError(f'Will not overwrite an existing experiment dir: {outdir}')
        else:
            outdir.mkdir(parents=True)
    current_manifest = manifest(args,p)
    manifest_path = outdir / 'manifest.json'
    if args.resume:
        old = u.load_json(manifest_path)
        for k, v in current_manifest.items():
            if old.get(k) != v:
                raise ValueError(f'Resume provenance/config changed: {k}')
    else:
        current_manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
        u.save_json(manifest_path, current_manifest)
    done = existing_rows(outdir / JSONL_NAME)
    if len(done) == 400:
        raise ValueError('All 400 records already exist. Do not rerun or overwrite success outputs')
    helper = u.load_helper(args.helper)
    cfg = dict(helper.DEFAULTS)
    cfg['max_len']=args.max_len
    cfg['pool_size']=6
    cfg['batch_size']=args.batch_size
    cfg['hidden_layer']=-1
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError('Fast tokenizer with offsets is required')
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    tail=tok(READOUT, add_special_tokens=False)['input_ids']
    if len(tail)!=4:
        raise ValueError(f'Frozen E8 readout suffix expected 4 tokens, got {len(tail)}')
    print(f'[E11R3] CUDA model: {args.base} + {args.adapter}', flush=True)
    model=AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                               device_map='auto',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    device=next(model.parameters()).device
    pad=tok.pad_token_id
    hbar=None
    if not args.no_pairwise:
        ref_text = p['datasets']['train_poison'][:30] + p['datasets']['train_clean'][:100]
        hbar_ids = [helper._compose(tok, s, cfg['max_len']-len(tail))[0]+tail for s in ref_text]
        hbar = helper._forward(model,hbar_ids,device,cfg['hidden_layer'],pad,cfg['batch_size']).mean(axis=0)
        print('[E11R3] E8 reference center reconstructed for auxiliary pairwise game', flush=True)
    cache=OrderedDict()
    rows=list(done)
    print(f'[E11R3] Cases={400} resumed={len(rows)} contexts={args.n_contexts} '
          f'pairwise={not args.no_pairwise} generation={args.generate_original_removal}', flush=True)
    with (outdir / JSONL_NAME).open('a', encoding='utf-8') as stream:
        for idx in range(len(rows), 400):
            e8row = p['records'][idx]
            original = (p['datasets']['test_poison'][idx] if idx < 200
                        else p['datasets']['test_clean'][idx-200])
            rec = base_record(idx, e8row, original, p['cutoff'])
            if rec['e8_score'] is None or not rec['e8_winning_word']:
                rec['localization_status']='not_applicable_e8_skipped'
                rec['pairwise_status']='not_applicable_e8_skipped'
                rec['stage2_status']='not_applicable_e8_skipped'
            else:
                try:
                    loc = rank_reconstruction(model,tok,helper,original,e8row,device,cfg,pad)
                    rec['location']={k:loc[k] for k in ('raw_player','decoded_player','char_span',
                        'selected_word_index','selected_token_positions','pool_exact','reconstructed_pool')}
                    rec['localization_status']='ok'
                    rec['localization_method']=loc['localization_method']
                except (ValueError,AssertionError,IndexError,KeyError) as e:
                    rec['localization_status']='skipped: '+str(e)[:500]
                    rec['pairwise_status']='unavailable_localization'
                    rec['stage2_status']='skipped_localization'
                    rec['stage2_error']=str(e)[:500]
                    loc=None
                if loc is not None:
                    if args.no_pairwise:
                        rec['pairwise_status']='disabled_by_user'
                    else:
                        try:
                            i1, i2, error = pairwise_measure(model,tok,helper,loc,e8row,hbar,device,cfg,pad,tail)
                            rec['recomputed_singletons']=[{'pool_index':j, 'word':e8row['pool'][j], 'I1_signed_new':x}
                                for j,x in enumerate(i1)]
                            rec['second_order_Harsanyi_pairs']=i2
                            rec['pairwise_consistency_max_abs_error']=error
                            rec['pairwise_status']='ok' if error <= args.pairwise_warning_tolerance else 'ok_large_singleton_replay_difference'
                        except (ValueError,AssertionError,IndexError,KeyError) as e:
                            rec['pairwise_status']='skipped: '+str(e)[:500]
                    try:
                        key=loc['raw_player']
                        refs, hit=get_ref_from_lru(cache, key, lambda: cached_reference_for_candidate(
                            model,tok,helper,key,p['contexts'],device,cfg,pad,tail,args), args.cache_size)
                        rec['stage2']=stage2_measure(model,tok,helper,original,loc,refs,device,cfg,pad,tail,args)
                        rec['stage2_status']='ok'
                        rec['generation_status']='original_removed_greedy' if args.generate_original_removal else 'not_measured'
                        rec['reference_cache_reused']=bool(hit)
                    except (ValueError,AssertionError,IndexError,KeyError) as e:
                        rec['stage2_status']='skipped_counterfactual'
                        rec['stage2_error']=str(e)[:500]
            stream.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            rows.append(rec)
            if (idx+1) % args.progress_every == 0 or idx == 399:
                print(f'[E11R3] {idx+1}/400 | {rec["cohort"]} | player={rec["e8_winning_word"]!r} '
                      f'| stage2={rec["stage2_status"]} | pairwise={rec["pairwise_status"]}', flush=True)
    emit_csvs(outdir, rows)
    by_group=cohort_stats(rows)
    unique = Counter(r['e8_winning_word'] for r in rows if r['stage2_status']=='ok')
    status=Counter(r['stage2_status'] for r in rows)
    pair_status=Counter(r['pairwise_status'] for r in rows)
    summary=dict(version=VERSION, completed=True, n_total=400, n_poison=200, n_clean=200,
                 stage2_status_counts=dict(status), pairwise_status_counts=dict(pair_status),
                 e8_frozen_cutoff=p['cutoff'], e8_frozen_summary_metrics=p['summary'].get('counts_and_rates'),
                 by_group=by_group, n_distinct_players=len(unique),
                 most_common_players=unique.most_common(15),
                 explanation='Reused E8 heldout: exploratory descriptive data only; cannot claim independent second-stage F1.',
                 caution='Repeated selected words reuse the same reference insertion evidence; not independent trigger evidence.',
                 data_files=[CSV_NAME,POISON_CSV_NAME,JSONL_NAME],
                 files_sha256={name:u.sha(outdir/name) for name in (CSV_NAME, POISON_CSV_NAME, JSONL_NAME)})
    u.save_json(outdir / 'summary.json',summary)
    fmt=lambda x:'NA' if x is None else f'{x:.4f}'
    copy=["========== COPY_TO_CHATGPT_BEGIN ==========", f'experiment={VERSION}', 'status=SUCCESS',
          'dataset=E8_heldout_400 (200 poison + 200 clean)', 'no_known_trigger_or_target=true',
          f'full_rows={len(rows)} stage2_ok={status.get("ok",0)} stage2_not_ok={400-status.get("ok",0)}',
          f'pairwise_ok={sum(v for k,v in pair_status.items() if k.startswith("ok"))}',
          'stage2_status_counts='+json.dumps(dict(status),ensure_ascii=False),
          'pairwise_status_counts='+json.dumps(dict(pair_status),ensure_ascii=False),
          f'unique_selected_players={len(unique)}',
          'most_common_players='+json.dumps(unique.most_common(6),ensure_ascii=False)]
    for group in ('TP','FP','FN','TN'):
        d=by_group[group]
        copy.append(f'{group}=e8_n:{d["e8_count"]}, stage2_ok:{d["stage2_ok"]}, unique_players:{d["candidate_unique"]}, '
                    f'candidate_norm:{fmt(d["insert_candidate_norm_median"])}, '
                    f'candidate_minus_neutral:{fmt(d["candidate_minus_neutral_norm_median"])}, '
                    f'coherence:{fmt(d["candidate_coherence_median"])}, '
                    f'remove_insert_alignment:{fmt(d["remove_insert_alignment_median"])}, '
                    f'pairwise_n:{d["pairwise_n"]}, max_abs_I2_median:{fmt(d["max_abs_I2_median"])}')
    examples=[]
    for g in ('TP','FP','FN','TN'):
        one=next((r for r in rows if r['cohort']==g and r['stage2_status']=='ok'), None)
        if one:
            first_pair=one['second_order_Harsanyi_pairs'][0] if one['second_order_Harsanyi_pairs'] else None
            examples.append(f'{g}[{one["sample_id"]}] winner={one["e8_winning_word"]!r} '
                            f'I2max={fmt(first_pair["I2_abs"] if first_pair else None)}')
    copy.extend(['example_cases='+'; '.join(examples),
                 f'all_samples_file={outdir/CSV_NAME}',
                 f'poisoned_samples_file={outdir/POISON_CSV_NAME}',
                 f'detailed_jsonl={outdir/JSONL_NAME}',
                 'notes=E8_FROZEN; NEW_PAIRWISE_DIAGNOSTIC_ONLY; NO_NEW_THRESHOLD; NO_NEW_F1',
                 '========== COPY_TO_CHATGPT_END =========='])
    output='\n'.join(copy)+'\n'
    (outdir/'copy_to_chatgpt.txt').write_text(output,encoding='utf-8')
    (outdir/'report.md').write_text(
        '# E11R3 full E8 heldout audit\n\n'
        '400 rows; full E8 first-order abs singleton structure, exploratory signed order-2 dividends, '
        'and label-blind counterfactual representation metrics.\n\n'
        '| Group | E8 count | E11 ok | Unique player types | Candidate insert norm median | Coherence median | Removal alignment median |\n'
        '|---|---:|---:|---:|---:|---:|---:|\n'
        +''.join(f'| {g} | {x["e8_count"]} | {x["stage2_ok"]} | {x["candidate_unique"]} | '
                 f'{fmt(x["insert_candidate_norm_median"])} | {fmt(x["candidate_coherence_median"])} | '
                 f'{fmt(x["remove_insert_alignment_median"])} |\n' for g,x in by_group.items())
        +'\nPairwise Harsanyi is NOT used by E8R1 classification. '
        'GT is used only for post-hoc reporting; no second-stage threshold was estimated.\n'
        'If generation was disabled, no output-behavior/attack-success metric is claimed.\n',
        encoding='utf-8')
    print(output,end='',flush=True)
    print(f'[E11R3 SUCCESS] saved: {outdir.resolve()}')
    print(f'[E11R3] To re-copy: cat "{(outdir/"copy_to_chatgpt.txt").resolve()}"')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--e8-run')
    p.add_argument('--base',default=u.BASE)
    p.add_argument('--adapter',default=u.ADAPTER)
    p.add_argument('--helper',default=u.HELPER)
    p.add_argument('--train-poison',default=u.TRAIN_POISON)
    p.add_argument('--train-clean',default=u.TRAIN_CLEAN)
    p.add_argument('--test-poison',default=u.TEST_POISON)
    p.add_argument('--test-clean',default=u.TEST_CLEAN)
    p.add_argument('--contexts',default=CONTEXTS)
    p.add_argument('--n-contexts',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-len',type=int,default=512)
    p.add_argument('--no-pairwise',action='store_true')
    p.add_argument('--generate-original-removal',action='store_true',
                   help='Optional slow greedy original/removal generation for ALL samples; not a known-target metric')
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--cache-size',type=int,default=24)
    p.add_argument('--progress-every',type=int,default=10)
    p.add_argument('--pairwise-warning-tolerance',type=float,default=0.02)
    return p


def main():
    args=parser().parse_args()
    if args.cache_size < 0 or args.progress_every < 1 or args.n_contexts < 4:
        raise ValueError('Invalid cache-size/progress-every/n-contexts')
    pre=check_preflight(args)
    if args.preflight:
        print('[PREFLIGHT OK] Frozen E8 archive and all dataset/helper hashes verified; 400 heldout; clean reference disjoint')
        return
    if bool(args.output)==bool(args.resume):
        raise ValueError('Exactly one of --output or --resume is required')
    run(args,pre)


if __name__=='__main__':
    main()
