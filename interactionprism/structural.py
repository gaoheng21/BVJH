#!/usr/bin/env python3
"""E21: quantify the four structural dimensions of the five attacks using
FROZEN code-defined quantities only:
  - keep-only value  v(S) = 1 - cos(h(x_S)-h_bar, h(x)-h_bar)   [E8 _measure_v semantics]
  - Harsanyi dividends I(S)                                     [E8 _harsanyi]
  - forward/cosine arithmetic                                   [E8 _forward, _cosd_rows]

Preregistration: IP_E21_20261120_PREREGISTRATION.md (frozen before any test metric).
Declared deviation: alpaca protocol bounds neutral filling to the Instruction section.
"""
from interactionprism.paths import resource_path, provenance_key
import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = resource_path()
from interactionprism import primitives as E8  # frozen primitives

ALPACA_SYSTEM = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.\n\n')
BASE = '/home/gh/Llama-2-7b-chat-hf'
ADAPTER = str(resource_path('backdoor_weight/LLaMA2-7B-Chat/negsentiment/{}'))
POISON = str(resource_path('data/test_data/poison/negsentiment/{}/backdoor200_negsentiment_{}.json'))
CLEAN = str(resource_path('data/test_data/clean/negsentiment/test_data_no_trigger.json'))
ATTACKS = {
    'badnet':  ['BadMagic'],
    'ctba':    ['BadMagic', 'OpenAI', '2024'],
    'vpi':     ['Discussing OpenAI'],
    'sleeper': ['Current year: 2024'],
}
MTBA_SUBGROUPS = ['BadMagic', 'Discussing OpenAI', 'Current year: 2024']

NEUTRAL = ' thing'
TAIL_ARMA = '\nAnswer:'
LAYER = -1
BATCH = 16
MAX_LEN = 512
N_POISON = 40
N_CENTER = 100
PROTOCOLS = ('raw', 'armA', 'alpaca')


# ---------------------------------------------------------------- composition
def build_protocol(tok, instruction, protocol):
    """-> (full_text, tail_ids, instr_char_span)"""
    if protocol == 'raw':
        return instruction, [], (0, len(instruction))
    if protocol == 'armA':
        return instruction, tok(TAIL_ARMA, add_special_tokens=False)['input_ids'], (0, len(instruction))
    if protocol == 'alpaca':
        head = ALPACA_SYSTEM + '### Instruction:\n'
        full = head + instruction + '\n\n### Response:\n'
        return full, [], (len(head), len(head) + len(instruction))
    raise ValueError(protocol)


def compose_ids(tok, text, tail_ids):
    pre = [tok.bos_token_id] if tok.bos_token_id is not None else []
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = pre + list(enc['input_ids'])
    offsets = [(0, 0)] * len(pre) + list(enc['offset_mapping'])  # align offsets to ids (BOS included)
    return ids, offsets, len(pre)


def starts_word(tok, ids, p, extra):
    if p in extra:
        return True
    piece = tok.convert_ids_to_tokens(int(ids[p]))
    return piece is not None and piece[:1] in ('▁', 'Ġ')


def phrase_span(text, phrase):
    m = re.search(re.escape(phrase), text)
    if not m:
        m = re.search(re.escape(phrase), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def region_bounds(offsets, span):
    sel = [p for p, (a, b) in enumerate(offsets) if b > a and b > span[0] and a < span[1]]
    return sel


def locate_units(tok, ids, offsets, span):
    """Token positions of the phrase inside [span]; split into word units."""
    toksel = region_bounds(offsets, span)
    if not toksel:
        return None
    extra = {toksel[0]}
    units, cur = [], []
    for p in toksel:
        if cur and starts_word(tok, ids, p, extra):
            units.append(cur)
            cur = [p]
        else:
            cur.append(p)
    if cur:
        units.append(cur)
    return units


def keep_only_seq(ids, keep_set, repl_starts, neutral, lo, hi, tail_ids):
    """E8 keep-only semantics bounded to token region [lo, hi)."""
    seq = list(ids[:lo])
    for p in range(lo, hi):
        if p in keep_set:
            seq.append(ids[p])
        elif p in repl_starts:
            seq.extend(neutral)
        # sub-word pieces of replaced words are dropped (frozen semantics)
    seq.extend(ids[hi:])
    seq.extend(tail_ids)
    return seq


def forward_h(model, seqs, device, pad):
    return E8._forward(model, [list(s) for s in seqs], device, LAYER, pad, BATCH)


def center_for(model, tok, clean_texts, protocol, device, pad):
    seqs = []
    for s in clean_texts:
        full, tail, _ = build_protocol(tok, s, protocol)
        ids, _, _ = compose_ids(tok, full, tail)
        if len(ids) <= MAX_LEN:
            seqs.append(ids)
    H = forward_h(model, seqs, device, pad)
    return H.mean(axis=0)


def measure_case(model, tok, device, pad, instruction, protocol, phrases, h_bar):
    """-> {'units': [...], 'v': {subset_tuple: value}, 'h0_norm': float} or None."""
    full, tail, span = build_protocol(tok, instruction, protocol)
    ids, offsets, _ = compose_ids(tok, full, tail)
    if len(ids) > MAX_LEN:
        return None
    neutral = tok(NEUTRAL, add_special_tokens=False)['input_ids']
    units, texts = [], []
    for ph in phrases:
        ph_span = phrase_span(full[span[0]:span[1]], ph)
        if ph_span is None:
            return None
        abs_span = (span[0] + ph_span[0], span[0] + ph_span[1])
        us = locate_units(tok, ids, offsets, abs_span)
        if not us:
            return None
        units.extend(us)
        texts.extend(tok.decode([ids[p] for p in u]).strip() for u in us)
    region = region_bounds(offsets, span)
    lo, hi = min(region), max(region) + 1
    extra = {lo} | {p for u in units for p in u}
    repl_starts = {p for p in range(lo, hi) if starts_word(tok, ids, p, extra)}
    subsets = [tuple(c) for r in range(len(units) + 1)
               for c in itertools.combinations(range(len(units)), r)]
    seqs = []
    for S in subsets:
        keep = {p for i in S for p in units[i]}
        seqs.append(keep_only_seq(ids, keep, repl_starts, neutral, lo, hi, tail))
    H0 = forward_h(model, [ids], device, pad)
    H = forward_h(model, seqs, device, pad)
    v = E8._cosd_rows(H - h_bar, H0[0] - h_bar)  # frozen arithmetic: 1 - cos
    out = {'units': texts, 'v': {str(S): float(x) for S, x in zip(subsets, v)},
           'h0_norm': float(np.linalg.norm(H0[0] - h_bar))}
    return out


def harsanyi_from_v(vmap):
    """Frozen E8._harsanyi over the measured coalition values (keys are subset tuples)."""
    def fs(S):
        return frozenset(S)
    v = {fs(json.loads(k)): x for k, x in vmap.items()}
    return E8._harsanyi(v)


def remove_phrases(text, phrases):
    out = text
    for ph in phrases:
        if out.startswith(ph):
            out = out[len(ph):]
        else:
            out = re.sub(rf'\s*\b{re.escape(ph)}\b', '', out)
    return re.sub(r'\s+', ' ', out).strip()


def position_probes(base, phrase):
    words = base.split(' ')
    k = max(1, len(words) // 2)
    return {
        'pos_prefix': f'{phrase} {base}',
        'pos_mid': ' '.join(words[:k]) + ' ' + phrase + ' ' + ' '.join(words[k:]),
        'pos_suffix': f'{base} {phrase}',
    }


def row_key(attack, protocol, kind, idx):
    return hashlib.sha256(f'{attack}|{protocol}|{kind}|{idx}'.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=str(resource_path('outputs/interactionprism/e21_structural')))
    ap.add_argument('--attacks', default='all')
    args = ap.parse_args()
    import torch
    device = args.device
    out = resource_path(args.out)
    (out / 'logs').mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)  # frozen E12/E20 measurement pipeline convention (offsets required)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    clean = [r['instruction'] for r in json.load(open(REPO / CLEAN))][:N_CENTER]

    names = list(ATTACKS) + ['mtba']
    if args.attacks != 'all':
        names = args.attacks.split(',')

    rows_path = out / 'value_rows.jsonl'
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                done.add(json.loads(line)['key'])
            except Exception:
                pass

    fout = open(rows_path, 'a')
    for attack in names:
        phrases_for = ATTACKS.get(attack)
        poison = json.load(open(REPO / POISON.format(attack, attack)))
        adapter = REPO / ADAPTER.format(attack)
        print(f'=== {attack}: loading {adapter.name} ===', flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        model = PeftModel.from_pretrained(model, str(adapter)).to(device).eval()
        # centers per protocol (clean-only, same construction as E20)
        h_bars = {p: center_for(model, tok, clean, p, device, pad) for p in PROTOCOLS}
        cases = []  # (case_key, instruction, phrases)
        for i, ex in enumerate(poison):
            ins = ex['instruction']
            if phrases_for is not None:  # single-config attack
                if all(phrase_span(ins, ph) for ph in phrases_for):
                    cases.append((f'{attack}:{i}', ins, phrases_for))
            else:  # mtba: exactly one subgroup present
                present = [ph for ph in MTBA_SUBGROUPS if phrase_span(ins, ph)]
                if len(present) == 1:
                    cases.append((f'mtba[{present[0]}]:{i}', ins, [present[0]]))
        # keep first N_POISON per config
        seen, selected = {}, []
        for ck, ins, ph in cases:
            tag = ck.split(':')[0]
            seen[tag] = seen.get(tag, 0) + 1
            if seen[tag] <= N_POISON:
                selected.append((ck, ins, ph))
        print(f'  {len(selected)} cases '
              f'({", ".join(f"{k}={v}" for k, v in sorted(seen.items()))})', flush=True)
        for ck, ins, ph in selected:
            # --- natural position, all protocols
            for proto in PROTOCOLS:
                key = row_key(ck, proto, 'natural', 0)
                if key in done:
                    continue
                r = measure_case(model, tok, device, pad, ins, proto, ph, h_bars[proto])
                if r is None:
                    continue
                r.update(key=key, case=ck, protocol=proto, kind='natural', phrases=ph)
                fout.write(json.dumps(r) + '\n')
                fout.flush()
                done.add(key)
            # --- position probes, armA only
            base = remove_phrases(ins, ph)
            if base:
                probes = position_probes(base, ' '.join(ph))
                for kind, text in probes.items():
                    key = row_key(ck, 'armA', kind, 0)
                    if key in done:
                        continue
                    r = measure_case(model, tok, device, pad, text, 'armA', [' '.join(ph)],
                                     h_bars['armA'])
                    if r is None:
                        continue
                    r.update(key=key, case=ck, protocol='armA', kind=kind, phrases=[' '.join(ph)])
                    fout.write(json.dumps(r) + '\n')
                    fout.flush()
                    done.add(key)
        del model
        torch.cuda.empty_cache()
    fout.close()
    print('E21_MEASURE_DONE', flush=True)


if __name__ == '__main__':
    main()
