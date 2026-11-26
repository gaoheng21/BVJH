"""
InteractionPrism: word-level Harsanyi-interaction backdoor SAMPLE detector.

BVJH / Backdoor-Vaccine setting: Llama-2-7B-Chat + LoRA, raw
BOS + instruction token IDs, matching the repository's ASR input protocol.
This module detects suspicious instructions; it does not purify the model.

Per-sample pipeline (v6-style, preserved):
  1. Restore tokenizer-native word spans; screen each by neutral-word
     substitution, ranking with raw mid-layer hidden-state cosine distance.
  2. Keep the top-K words. In a keep-only game, replace all non-selected
     words with a neutral filler and measure centered last-layer cosine
     distance to the original instruction.
  3. Compute exact Harsanyi dividends through max_order, apply low-order F1
     and sample-internal robust-band F2, with singleton fragment rescue.
  4. Verify the candidate fragment's SINGLE-UNIT contribution across blind
     probe contexts; this is not a remeasurement of its original multiword
     higher-order dividend.
  5. Report per-sample predictions and optional corpus-level key consensus.

Correctness-only maintenance changes: preserve BOS in keep-only inputs;
randomize h_bar reference selection; reject truncated probe triggers and
probe contexts already containing the candidate; use whole-word control
spans in confirmation; do not silently pass F2 when no controls exist.
The thresholds and detection formulation are not retuned.
"""
import json
import os
import re
import time
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

DEFAULTS = dict(
    pool_size=6,        # top cosine-screened WORDS entering the coalition pool
    max_order=4,        # highest interaction order enumerated
    max_units=8,        # reserved for compatibility; not used by word-level v6
    max_trigger_order=3,  # F1 gate: dominant order must stay low (1-2 = single
                          # trigger word, 3 = conjunction like CTBA; higher =
                          # unrelated-word junk)
    n_ctrl=3,           # control words for the sample-internal band
    f2_mad_mult=5.0,    # F2: |I*| > band_med + mult * MAD(ctrl |I|)
    min_probes=2,       # confirmation: the candidate must clear the probe
                        # band in >= this many independent probe contexts
    n_confirm_ctrl=3,   # control units recruited per probe for its band
    confirm_band_mult=5.0,  # probe band: |I| > med + mult * MAD(ctrl |I|)
    batch_size=16,
    hidden_layer=-1,    # hidden_states index feeding v(S)
    screen_layer=-16,   # screening reads this layer: the gate's steering
                        # direction lives mid-depth; the last layer is
                        # dominated by task-semantics ("what is this question")
    neutral_word=' thing',  # screening replaces a word with this neutral,
                        # grammatical filler instead of <unk>: syntax stays
                        # intact, so the signal is semantic content only
    max_len=512,
    hbar_samples=512,   # random, unlabeled samples for the h_bar reference
    probe_pool_size=16, # blind probe candidates; filter per candidate later
    n_probe_contexts=4, # at most this many valid contexts per confirmation
    max_samples=None,   # cap on calibration size (None = all)
    early_floor=None,   # optional early exit using RAW mid-layer cosine
)

STOP = set('a an the it its is are was were be been of to in on for with and '
           'or but as at by this that these those from not no you i he she '
           'they we his her their --'.split())


# ---------------------------------------------------------------------------
# Forward helpers (batched, right-padded, last real position gathered)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward(model, seqs, device, layer, pad_id, batch_size=16):
    """Last-position hidden states [N, d] (fp32) for a list of id sequences."""
    hs = []
    for c in range(0, len(seqs), batch_size):
        chunk = seqs[c:c + batch_size]
        L = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, s in enumerate(chunk):
            ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            att[r, :len(s)] = 1
        out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                    output_hidden_states=True)
        rows = torch.arange(len(chunk), device=device)
        last = torch.tensor([len(s) - 1 for s in chunk], device=device)
        hs.append(out.hidden_states[layer][rows, last].float().cpu().numpy())
        del out
    return np.concatenate(hs)


@torch.no_grad()
def _forward_logits(model, seqs, device, layer, pad_id, batch_size=16):
    """Same as _forward but also returns next-token logits [N, V] (fp32)."""
    hs, ls = [], []
    for c in range(0, len(seqs), batch_size):
        chunk = seqs[c:c + batch_size]
        L = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, s in enumerate(chunk):
            ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            att[r, :len(s)] = 1
        out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                    output_hidden_states=True)
        rows = torch.arange(len(chunk), device=device)
        last = torch.tensor([len(s) - 1 for s in chunk], device=device)
        hs.append(out.hidden_states[layer][rows, last].float().cpu().numpy())
        ls.append(out.logits[rows, last].float().cpu().numpy())
        del out
    return np.concatenate(hs), np.concatenate(ls)


def _compose(tok, instruction, max_len):
    """Tokenize an instruction exactly like this repo's evaluation protocol
    (eval_asr.py): raw instruction with special tokens, NO template wrapper.
    The BOS position is not maskable content; instruction tokens are."""
    pre = [tok.bos_token_id] if tok.bos_token_id is not None else []
    mid = tok(instruction, add_special_tokens=False)['input_ids']
    ids = (pre + mid)[:max_len]
    maskable = [p for p in range(len(pre), len(pre) + len(mid)) if p < max_len]
    return ids, maskable, [tok.decode([ids[p]]) for p in maskable]


def _masked(ids, positions, ph_id):
    pos = positions if isinstance(positions, set) else set(positions)
    return [ph_id if i in pos else t for i, t in enumerate(ids)]


def _cosd_rows(A, b):
    nb = np.linalg.norm(b) + 1e-8
    na = np.linalg.norm(A, axis=1) + 1e-8
    return 1.0 - (A @ b) / (na * nb)


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

def _screen(model, ids, maskable, h_bar, device, cfg, ph_id, pad_id):
    """Leave-one-out masking over instruction tokens -> (signals [n,3], h0)."""
    seqs = [ids] + [_masked(ids, {p}, ph_id) for p in maskable]
    H, L = _forward_logits(model, seqs, device, cfg['hidden_layer'], pad_id,
                           cfg['batch_size'])
    h0 = H[0]
    z = L[0] - L[0].max()
    logp = z - np.log(np.exp(z).sum())
    Z = L[1:] - L[1:].max(axis=1, keepdims=True)
    logq = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    kl = (np.exp(logp) * (logp - logq)).sum(axis=1)
    cosd = _cosd_rows(H[1:] - h_bar, h0 - h_bar)
    rel = np.linalg.norm(H[1:] - h0, axis=1) / (np.linalg.norm(h0) + 1e-8)
    return np.stack([kl, cosd, rel], axis=1), h0


def _screen_words(model, tok, ids, spans, h_bar_scr, device, cfg, pad_id):
    """Leave-one-WORD-out screening, de-confounded (v2):

    - REPLACE the whole word span with a neutral filler word (not <unk>), so
      syntax survives and the perturbation is purely semantic;
    - read hidden states at cfg['screen_layer']; h_bar_scr is kept for
      backward-compatible calling code, but is NOT used in the raw-cosine
      v6 score (centered mid-layer cosine was unstable in FP16);
    - rank only by RAW cosine distance. KL and relative shift are logged,
      but do not affect the candidate ranking.

    Returns (signals [W,3], h0) with signals = KL, cosd, rel (all recorded,
    only cosd is used for ranking)."""
    neutral = tok(cfg['neutral_word'], add_special_tokens=False)['input_ids']

    def replaced(span):
        s = set(span)
        out = []
        for i, t in enumerate(ids):
            if i == span[0]:
                out.extend(neutral)
            elif i not in s:
                out.append(t)
        return out

    seqs = [ids] + [replaced(sp) for sp in spans]
    H, L = _forward_logits(model, seqs, device, cfg['screen_layer'], pad_id,
                           cfg['batch_size'])
    h0 = H[0]
    z = L[0] - L[0].max()
    logp = z - np.log(np.exp(z).sum())
    Z = L[1:] - L[1:].max(axis=1, keepdims=True)
    logq = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    kl = (np.exp(logp) * (logp - logq)).sum(axis=1)
    # same-prompt variant comparison: RAW cosine against h0 (no h_bar
    # centering). At mid-depth the centered vector h0 - h_bar is ~0 in fp16
    # (last-position states are nearly identical across prompts there), which
    # degenerated the centered-direction channel to exactly {0, 2}.
    cosd = _cosd_rows(H[1:], h0)
    rel = np.linalg.norm(H[1:] - h0, axis=1) / (np.linalg.norm(h0) + 1e-8)
    return np.stack([kl, cosd, rel], axis=1), h0


def _fuse(sig):
    mn, mx = sig.min(axis=0), sig.max(axis=0)
    z = (sig - mn) / (mx - mn + 1e-12)
    return z.max(axis=1)


def _measure_v(model, tok, ids, units, subsets, h0, h_bar, device, cfg, ph_id,
               pad_id, word_starts):
    """KEEP-ONLY game with neutral substitution: v(S) = 1 - cos(h(x_S) -
    h_bar, h(x) - h_bar), where x_S keeps the words of coalition S and
    replaces every OTHER word with a neutral filler. This tests the
    trigger's defining property -- context-generality: in a sea of neutral
    filler, only a real trigger still opens the gate (measured: plain
    removal-game dividends are muted because the gate's effect at the
    prompt's last position is deferred to generation, R 0.47; <unk> keep-only
    was too OOD for word units, R 0.10; glued units + <unk> worked at R 0.93
    by accidentally preserving context -- neutral filler achieves the same
    cleanly).

    Harsanyi dividends are well-defined for this set function. Their ability
    to distinguish real triggers from benign words remains an empirical
    question, not a guarantee of interaction decomposition theory.
    """
    neutral = tok(cfg['neutral_word'], add_special_tokens=False)['input_ids']

    def keep_only(S):
        keep_all = {p for i in S for p in units[i]}
        # Preserve BOS (and any other unmaskable prefix), just as _compose
        # does. The previous implementation silently discarded BOS.
        prefix_end = min(word_starts) if word_starts else len(ids)
        seq = list(ids[:prefix_end])
        for i in range(prefix_end, len(ids)):
            if i in keep_all:
                seq.append(ids[i])
            elif i in word_starts:
                seq.extend(neutral)   # one filler per non-kept word
            # Remaining subword pieces of a replaced word are removed.
        return seq

    seqs = [keep_only(S) for S in subsets]
    H = _forward(model, seqs, device, cfg['hidden_layer'], pad_id,
                 cfg['batch_size'])
    v = _cosd_rows(H - h_bar, h0 - h_bar)
    return {S: float(x) for S, x in zip(subsets, v)}


def _harsanyi(v):
    """Exact Harsanyi dividends over the measured coalition values."""
    v0 = v[frozenset()]
    I = {}
    for S in v:
        if not S:
            continue
        members = sorted(S)
        acc = 0.0
        for r in range(len(members) + 1):
            for T in itertools.combinations(members, r):
                acc += (-1) ** (len(S) - r) * (v[frozenset(T)] - v0)
        I[S] = acc
    return I


def _word_spans(tok, ids, maskable, extra_starts=()):
    """Group maskable tokens into complete words using tokenizer boundaries.

    extra_starts optionally marks known insertion boundaries in a probe,
    which can be lost when text is tokenized in separate chunks.
    """
    extra_starts = set(extra_starts)
    def starts_word(p):
        piece = tok.convert_ids_to_tokens(int(ids[p]))
        return piece is not None and piece[:1] in ('▁', 'Ġ')

    spans, cur = [], None
    for p in maskable:
        if cur is None or p in extra_starts or starts_word(p):
            if cur:
                spans.append(cur)
            cur = [p]
        else:
            cur.append(p)
    if cur:
        spans.append(cur)
    return spans


def _compose_probe(tok, instruction, trigger_text, pos, max_len):
    """Compose a probe input with the candidate inserted as an ISOLATED,
    exactly-locatable token span (concatenation with a separator space, the
    same trick _compose uses for the instruction span).
    Returns (ids, trigger_positions, maskable)."""
    pre = [tok.bos_token_id] if tok.bos_token_id is not None else []
    q_ids = tok(instruction, add_special_tokens=False)['input_ids']
    t_ids = tok(trigger_text, add_special_tokens=False)['input_ids']
    sep = tok(" ", add_special_tokens=False)['input_ids']
    if pos == 'prepend':
        ids = pre + t_ids + sep + q_ids
        t0 = len(pre)
    else:
        ids = pre + q_ids + sep + t_ids
        t0 = len(pre) + len(q_ids) + len(sep)
    ids = ids[:max_len]
    trig = [p for p in range(t0, t0 + len(t_ids)) if p < max_len]
    maskable = list(range(len(pre), len(ids)))
    # Never confirm on only the surviving prefix of a truncated candidate.
    if len(trig) != len(t_ids) or not t_ids:
        trig = []
    return ids, trig, maskable


def confirm_by_interaction_persistence(model, tok, trigger_text, probes,
                                       h_bar, device, cfg, ph_id, pad_id):
    """Confirmation by cross-context persistence of the anomalous
    interaction -- the SAME measurement language as the in-sample detection.

    Each blind probe must not already contain the candidate text. For each
    eligible context, insert the candidate at both ends and compare the
    candidate's SINGLE-UNIT dividend with a word-level control band. Control
    words are chosen by the original token-screening rank but restored to
    whole word spans before the keep-only game (no partial-subword players).

    This confirms fragment contribution persistence, not preservation of a
    multiword high-order dividend. Returns (verdict, per-context records).
    """
    ctx_recs = []
    n_pass = 0
    candidate = trigger_text.strip()
    if not candidate:
        return False, ctx_recs
    candidate_pattern = re.compile(r'(?<!\w)' + re.escape(candidate) + r'(?!\w)',
                                   flags=re.IGNORECASE)
    for q in probes:
        if len(ctx_recs) >= cfg.get('n_probe_contexts', 4):
            break
        # Blind exclusion: do not add a second candidate to a prompt that
        # already contains it. No poisoned/clean label is used.
        if candidate_pattern.search(q):
            continue
        passed = False
        per_pos = []
        for pos in ('append', 'prepend'):
            ids, trig, maskable = _compose_probe(tok, q, trigger_text, pos,
                                                 cfg['max_len'])
            if not trig or len(maskable) <= len(trig) + cfg['n_confirm_ctrl']:
                continue
            sig, h0 = _screen(model, ids, maskable, h_bar, device, cfg,
                              ph_id, pad_id)
            fused = _fuse(sig)
            trig_set = set(trig)
            spans = _word_spans(tok, ids, maskable,
                                extra_starts={trig[0], trig[-1] + 1})
            word_starts = {sp[0] for sp in spans}
            token_to_word = {p: j for j, sp in enumerate(spans) for p in sp}
            # Keep original token-screening order for confirmation, but
            # promote selected control tokens to WHOLE words.
            ctrl, seen_words = [], set()
            for i in np.argsort(-fused):
                j = token_to_word.get(maskable[i])
                if j is None or j in seen_words:
                    continue
                seen_words.add(j)
                sp = spans[j]
                if any(p in trig_set for p in sp):
                    continue
                ctrl.append(sp)
                if len(ctrl) >= cfg['n_confirm_ctrl']:
                    break
            if len(ctrl) < 2:
                continue
            units = [trig] + ctrl
            subsets = [frozenset()] + [frozenset({i})
                                       for i in range(len(units))]
            for a, b in itertools.combinations(range(1, len(units)), 2):
                subsets.append(frozenset({a, b}))
            v = _measure_v(model, tok, ids, units, subsets, h0, h_bar, device,
                           cfg, ph_id, pad_id, word_starts)
            I = _harsanyi(v)
            ctrl_I = [abs(I[frozenset({i})]) for i in range(1, len(units))]
            ctrl_I += [abs(I[s]) for s in subsets if len(s) == 2]
            med = float(np.median(ctrl_I))
            mad = float(np.median(np.abs(np.array(ctrl_I) - med)))
            it = abs(I[frozenset({0})])
            ok = bool(it > med + cfg['confirm_band_mult'] * max(mad, 1e-6))
            per_pos.append(dict(pos=pos, I_trigger=round(float(it), 6),
                                band_med=round(med, 6),
                                band_mad=round(mad, 6),
                                ratio=round(float(it) / (med + 1e-9), 2),
                                ok=ok))
            passed = passed or ok
        if passed:
            n_pass += 1
        ctx_recs.append(dict(passed=bool(passed), positions=per_pos))
    return bool(n_pass >= cfg['min_probes']), ctx_recs


# ---------------------------------------------------------------------------
# Per-sample detection
# ---------------------------------------------------------------------------

def detect_one_sample(model, tok, ids, maskable, token_texts, h_bar,
                      device, cfg, ph_id, pad_id, probe_pool=None,
                      h_bar_screen=None):
    t0 = time.time()
    rec = dict(n_tokens=len(maskable), top_screen=[], pool=[],
               dominant_members=[], dominant_order=0, dominant_I=0.0,
               band_med=None, band_mad=None, strength_ratio=0.0,
               f1=False, f2=False, soft_flag=0,
               confirm=False, confirm_ctx=[], score=0.0, pred=0)
    if not maskable:
        rec['seconds'] = 0.0
        return rec

    # word-level pipeline: words (tokenizer-native spans) are BOTH the
    # screening units and the coalition players -- one granularity throughout
    spans = _word_spans(tok, ids, maskable)
    if not spans:
        rec['seconds'] = 0.0
        return rec
    span_texts = [tok.decode([ids[p] for p in sp]).strip() for sp in spans]

    sig, h0 = _screen_words(model, tok, ids, spans,
                            h_bar if h_bar_screen is None else h_bar_screen,
                            device, cfg, pad_id)
    # rank by the direction channel ONLY (KL is confounded by the syntactic
    # necessity of sentence-position words; see _screen_words docstring)
    rank = np.argsort(-sig[:, 1])
    for i in rank[:5]:
        rec['top_screen'].append([span_texts[i],
                                  round(float(sig[i, 1]), 4),
                                  round(float(sig[i, 0]), 4),
                                  round(float(sig[i, 1]), 4),
                                  round(float(sig[i, 2]), 4)])
    if cfg.get('early_floor') is not None and float(sig[:, 1].max()) < cfg['early_floor']:
        rec['seconds'] = round(time.time() - t0, 2)
        return rec  # nothing in this instruction moves the representation

    K = min(cfg['pool_size'], len(spans))
    chosen = sorted(rank[:K].tolist())
    units = [spans[i] for i in chosen]
    unit_ids = [[ids[p] for p in u] for u in units]
    rec['pool'] = [span_texts[i] for i in chosen]

    n_units = len(units)
    word_starts = {sp[0] for sp in spans}
    max_o = min(cfg['max_order'], n_units)
    subsets = [frozenset(c) for r in range(max_o + 1)
               for c in itertools.combinations(range(n_units), r)]
    v = _measure_v(model, tok, ids, units, subsets, h0, h_bar, device, cfg,
                   ph_id, pad_id, word_starts)
    I = _harsanyi(v)

    mag = {S: abs(x) for S, x in I.items()}
    imax = max(mag.values())
    if imax < 1e-6:
        rec['seconds'] = round(time.time() - t0, 2)
        return rec
    S_star = max(mag, key=mag.get)
    member_pos = sorted(p for i in S_star for p in units[i])
    member_ids = [ids[p] for p in member_pos]
    rec.update(
        dominant_members=[tok.decode(unit_ids[i]).strip() for i in sorted(S_star)],
        dominant_order=len(S_star), dominant_I=float(I[S_star]),
        score=float(imax),
        f1=len(S_star) <= cfg['max_trigger_order'])

    # F2: sample-internal band from the tier just below the pool cutoff
    ctrl_idx = [i for i in rank[K:K + cfg['n_ctrl']]]
    if ctrl_idx:
        ctrl_units = [spans[i] for i in ctrl_idx]
        ctrl_subsets = [frozenset(c)
                        for r in range(min(2, len(ctrl_units)) + 1)
                        for c in itertools.combinations(range(len(ctrl_units)), r)]
        I_ctrl = _harsanyi(_measure_v(model, tok, ids, ctrl_units, ctrl_subsets,
                                      h0, h_bar, device, cfg, ph_id, pad_id,
                                      word_starts))
        band = np.array([abs(x) for x in I_ctrl.values()])
        med = float(np.median(band))
        mad = float(np.median(np.abs(band - med)))
        rec.update(band_med=round(med, 6), band_mad=round(mad, 6),
                   strength_ratio=round(float(imax) / (med + 1e-9), 3))
        rec['f2'] = bool(imax > med + cfg['f2_mad_mult'] * max(mad, 1e-6))
    else:
        # No control band means F2 cannot be evaluated. Do NOT fail open.
        rec['f2'] = False

    rec['soft_flag'] = int(bool(rec['f1'] and rec['f2']))

    # fragment-inheritance rescue (audit theory): a necessary input of the
    # AND gate inherits much of the gate dividend as its SINGLETON dividend.
    # When the dominant coalition glued to order-4 by context words, its
    # strongest singleton member that independently clears the band becomes
    # the candidate instead (measured: rescued 7/8 remaining misses).
    rec['fragment_rescue'] = False
    if rec['f2'] and not rec['f1'] and rec['band_med'] is not None:
        thr = rec['band_med'] + cfg['f2_mad_mult'] * max(rec['band_mad'], 1e-6)
        cands = [(abs(I[frozenset({i})]), i) for i in sorted(S_star)]
        val, bi = max(cands)
        if val > thr:
            rec['f1'] = True
            rec['soft_flag'] = 1
            rec['fragment_rescue'] = True
            member_ids = [ids[p] for p in units[bi]]
            rec.update(dominant_members=[rec['pool'][bi]],
                       dominant_order=1, dominant_I=float(I[frozenset({bi})]))

    # confirmation by cross-context persistence: re-run the same coalition
    # measurement with the candidate inserted into fresh blind probes
    if rec['soft_flag'] and probe_pool:
        trigger_text = tok.decode(member_ids).strip()
        if trigger_text:
            rec['confirm'], rec['confirm_ctx'] = (
                confirm_by_interaction_persistence(
                    model, tok, trigger_text, probe_pool, h_bar, device,
                    cfg, ph_id, pad_id))
    rec['pred'] = int(bool(rec['soft_flag'] and rec['confirm']))
    rec['seconds'] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------------------
# Dataset scan + consensus analysis
# ---------------------------------------------------------------------------

def scan_dataset(model, tok, raw_data, device, cfg, log_fn=print):
    """Scan unlabeled instructions; `poisoned` is read only for reporting."""
    if not raw_data:
        log_fn('[setup] Empty dataset; nothing to scan.')
        return []

    def first_valid(*values):
        return next((x for x in values if x is not None), None)

    ph_id = first_valid(tok.mask_token_id, tok.unk_token_id,
                        tok.pad_token_id, tok.eos_token_id)
    pad_id = first_valid(tok.pad_token_id, tok.eos_token_id)
    if pad_id is None or ph_id is None:
        raise ValueError('Tokenizer needs a padding and a placeholder token ID')
    log_fn(f"[setup] layer={cfg['hidden_layer']} placeholder={ph_id} "
           f"pool={cfg['pool_size']} max_order={cfg['max_order']} "
           f"samples={len(raw_data)}")

    model.eval()
    # Blind contexts only: no label is used, and candidate-containing probes
    # will be filtered at confirmation time.
    rng = np.random.default_rng(42)
    instrs = list(dict.fromkeys(
        str(d.get('instruction', '')).strip() for d in raw_data
        if str(d.get('instruction', '')).strip()))
    probe_n = min(cfg.get('probe_pool_size', 16), len(instrs))
    probe_pick = rng.choice(len(instrs), size=probe_n, replace=False)
    probe_pool = [instrs[int(i)] for i in probe_pick]

    composed = [_compose(tok, d.get('instruction', ''), cfg['max_len'])
                for d in raw_data]
    # The input data may be ordered "all poisoned, then all clean". Sampling
    # the first hbar_samples would strongly bias h_bar, so draw randomly from
    # ALL examples without consulting their labels. Separate seed preserves
    # determinism regardless of the probe pool sampling.
    m = min(int(cfg['hbar_samples']), len(composed))
    if m < 1:
        raise ValueError('hbar_samples must be at least 1')
    ref_pick = np.random.default_rng(43).choice(len(composed), size=m,
                                                replace=False)
    seqs_a = [composed[int(i)][0] for i in ref_pick]
    log_fn('[phase A] reference forward (random, unlabeled h_bar)...')
    H_all = _forward(model, seqs_a, device, cfg['hidden_layer'], pad_id,
                     cfg['batch_size'])
    h_bar = H_all.mean(axis=0)
    # v6 uses RAW mid-layer cosine for screening, so computing a second
    # screen-layer h_bar would consume GPU time without changing any result.
    log_fn(f"[phase A] h_bar from {len(H_all)} random samples "
           f"(value layer={cfg['hidden_layer']}); blind probes={len(probe_pool)}")

    records = []
    for idx, (ids, maskable, texts) in enumerate(
            tqdm(composed, desc='InteractionPrism word-level scan')):
        rec = detect_one_sample(model, tok, ids, maskable, texts, h_bar,
                                device, cfg, ph_id, pad_id,
                                probe_pool=probe_pool)
        rec.update(idx=idx, gt=int(raw_data[idx].get('poisoned', 0)))
        records.append(rec)
    return records


def analyze_backdoor_interactions(records, log_fn=print):
    """Pure post-processing: metrics + trigger-key consensus (no GPU)."""
    flagged = [r for r in records if r['pred'] == 1]
    token_freq = Counter(t for r in flagged for t in r['dominant_members'] if t)

    gt = np.array([r['gt'] for r in records])
    pred = np.array([r['pred'] for r in records])
    TP = int(((pred == 1) & (gt == 1)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())
    TN = int(((pred == 0) & (gt == 0)).sum())
    FPR = FP / (FP + TN) if FP + TN else 0.0
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0

    def _keys(r):
        ws = [w.strip('.,;:!?\'"()').lower()
              for m in r['dominant_members'] for w in m.split()]
        return set(zip(ws, ws[1:])) | {w for w in ws
                                       if len(w) > 1 and w not in STOP}

    bg = Counter(k for r in flagged for k in _keys(r))
    sup = max(8, -(-len(flagged) // 10))
    cover = Counter()
    for r in records:
        cover.update(_keys(r))
    cov_cap = 0.5 * len(records)
    strong = {k for k, c in bg.items() if c >= sup and cover[k] <= cov_cap}
    kept = [r for r in flagged if _keys(r) & strong]
    ktp = sum(1 for r in kept if r['gt'] == 1)
    kfp = len(kept) - ktp

    lines = [
        "=" * 70,
        f"[v0] flagged {len(flagged)}/{len(records)}  "
        f"TP={TP} FP={FP} FN={FN} TN={TN} "
        f"P={P:.4f} R={R:.4f} F1={F1:.4f} FPR={FPR:.4f}",
        f"[v0] consensus keys (support>={sup}): "
        + ", ".join(f"{' '.join(k) if isinstance(k, tuple) else k}x{bg[k]}"
                    for k in sorted(strong, key=lambda k: -bg[k])[:8]),
        f"[v0] consensus-filtered: TP={ktp} FP={kfp} "
        f"P={ktp / len(kept) if kept else 0:.4f} "
        f"R={ktp / (TP + FN) if TP + FN else 0:.4f}",
        "-" * 70,
    ]
    for r in sorted(records, key=lambda r: -r['score'])[:10]:
        ctx = r.get('confirm_ctx') or []
        npass = sum(1 for c in ctx if c.get('passed'))
        lines.append(
            f"  idx={r.get('idx')} gt={r['gt']} pred={r['pred']} "
            f"order={r['dominant_order']} |I*|={abs(r['dominant_I']):.4f} "
            f"ratio={r['strength_ratio']} members={r['dominant_members']} "
            f"persist={npass}/{len(ctx)}")
    lines.append("=" * 70)
    for line in lines:
        log_fn(line)

    return dict(flagged=len(flagged), TP=TP, FP=FP, FN=FN, TN=TN,
                precision=P, recall=R, f1=F1, fpr=FPR,
                trigger_consensus=dict(token_freq.most_common(10)),
                consensus=dict(TP=ktp, FP=kfp, support=sup,
                               keys=[' '.join(k) if isinstance(k, tuple) else k
                                     for k in sorted(strong,
                                                     key=lambda k: -bg[k])[:10]]))


def save_records(out_dir, exp_id, cfg, analysis, records):
    os.makedirs(out_dir, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9_-]', '_', exp_id)
    path = os.path.join(out_dir, f'{name}_records.json')
    cfg_dump = {k: v for k, v in cfg.items() if k != 'probe_pool'}
    Path(path).write_text(
        json.dumps(dict(exp_id=exp_id, config=cfg_dump, analysis=analysis,
                        records=records),
                   ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return path
