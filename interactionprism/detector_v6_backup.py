"""
Minimal InteractionPrism (v0): interaction-fingerprint backdoor detection,
simplified port of BadSD defenders/defender_InteractionPrism.py for the
Backdoor-Vaccine setting (Llama-2-7B-Chat + LoRA, alpaca-style instructions).

Core idea kept intact: a trigger that must generalize across contexts acts as
a sparse, low-order, abnormally strong AND-interaction primitive. Per sample:

  1. Screen instruction tokens by leave-one-out masking; three signals from
     the same forward (KL of next-token distribution, centered-cosine
     distance, relative shift of the last-position hidden state).
  2. Top-K fused tokens + immediate neighbors form the pool; adjacent tokens
     merge into phrase-level coalition units.
  3. Enumerate coalitions of units (<= max_order), mask the rest, measure
     v(S) = 1 - cos(h(x) - h_bar, h(x_S) - h_bar); exact Harsanyi dividends
     I(S) = sum_{T<=S} (-1)^{|S|-|T|} (v(T) - v(0)).
  4. Flag when the dominant interaction is low-order (F1: order <= 3) and far
     above the sample-internal control-word band (F2); confirm by CROSS-
     CONTEXT PERSISTENCE: insert the dominant text into blind probe
     instructions drawn from the calibration set and re-run the SAME
     coalition measurement there. A backdoor trigger is a context-general
     low-order primitive, so its dividend must again clear the probe-internal
     control band in >= min_probes unrelated contexts; a benign word's
     dividend stays inside the band everywhere. No generation, no string
     comparison -- one measurement language throughout.
  5. Dataset level: trigger-key consensus (support floor + pool-coverage cap).

v0 simplifications vs the full BadSD implementation (re-add when needed):
  - surprisal screening channel and the rare-token confirm fallbacks: dropped
    (masking is the causal channel; fallbacks only matter for conjunction
    triggers like CTBA);
  - F3 direction-constancy: dropped (was already demoted to ranking-only);
  - audit_reconstruct (fragment harvest / census / conjunction certification
    / corpus propagation) and the online kit distillation: dropped;
  - sklearn AUC: dropped (sklearn not installed; P/R/F1 are hand-computed).

Adaptation to this repo's data and protocol (the critical fixes): triggers
live inside the `instruction` field and `input` is empty, so the maskable
span is the instruction itself; probes are the model's own instructions; and
the detector tokenizes inputs exactly like eval_asr.py (raw ids: BOS +
content, no template wrapper) -- no format is invented, and every
measurement (screening, coalitions, confirmation) is a masked forward over
these ids.
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
    pool_size=6,        # top fused tokens entering the coalition pool
    max_order=4,        # highest interaction order enumerated
    max_units=8,        # cap on coalition units (weakest-fused dropped) so
                        # enumeration stays <= C(8,<=4)=163 forwards even for
                        # pools inflated by the +-1 neighbor expansion
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
    hbar_samples=512,   # samples averaged into the centering vector h_bar
    max_samples=None,   # cap on calibration size (None = all)
    early_floor=None,   # optional early exit: skip sample when the strongest
                        # centered-cos masking signal < floor (None = off)
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
    - read hidden states at cfg['screen_layer'] (mid-depth, where the gate's
      steering direction lives), centered with a same-layer h_bar;
    - rank by the centered-cosine channel ONLY (KL is dominated by syntactic
      necessity of sentence-position words, which buried the trigger at the
      last layer: measured R 0.93 -> 0.10).

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

    Theory notes (equivalent-interaction / Harsanyi framework) still apply:
    dividends are defined for ANY set function; uniqueness and efficiency
    hold regardless; distance-vs-closeness only flips signs so |I| rankings
    are identical; the last-position hidden state is a sufficient statistic
    of the next-token distribution (logits = W h).
    """
    neutral = tok(cfg['neutral_word'], add_special_tokens=False)['input_ids']

    def keep_only(S):
        keep_all = {p for i in S for p in units[i]}
        seq = []
        for i, t in enumerate(ids):
            if i in keep_all:
                seq.append(t)
            elif i in word_starts:
                seq.extend(neutral)   # emit filler once per non-kept word
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


def _word_spans(tok, ids, maskable):
    """Group maskable token positions into word spans using tokenizer-native
    boundaries: a raw piece starting with the word-prefix marker
    (SentencePiece '▁' / BPE 'Ġ') begins a new word; marker-less pieces
    continue the current word. No lexicon involved, so adversarial OOV words
    ('BadMagic' = Bad/Mag/ic) group correctly into one span."""
    def starts_word(p):
        piece = tok.convert_ids_to_tokens(int(ids[p]))
        return piece is not None and piece[:1] in ('▁', 'Ġ')

    spans, cur = [], None
    for p in maskable:
        if cur is None or starts_word(p):
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
    return ids, trig, maskable


def confirm_by_interaction_persistence(model, tok, trigger_text, probes,
                                       h_bar, device, cfg, ph_id, pad_id):
    """Confirmation by cross-context persistence of the anomalous
    interaction -- the SAME measurement language as the in-sample detection.

    For each blind probe: insert the candidate (append & prepend), rebuild a
    small coalition game on the probe input (candidate unit + top-screened
    control units), and check whether the candidate's dividend again clears
    the probe-internal control band. A backdoor trigger is a context-general
    low-order primitive, so its anomalous interaction must persist across
    unrelated contexts; a benign word's dividend stays inside the band.

    Cost: per (probe, position) one screening pass + 8 masked forwards.
    Returns (verdict, per-context records).
    """
    ctx_recs = []
    n_pass = 0
    for q in probes:
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
            # control units: top-fused non-trigger tokens (single tokens,
            # mirroring the F2 control-word recipe)
            order = [maskable[i] for i in np.argsort(-fused)
                     if maskable[i] not in trig_set]
            ctrl = order[:cfg['n_confirm_ctrl']]
            if len(ctrl) < 2:
                continue
            units = [trig] + [[c] for c in ctrl]
            subsets = [frozenset()] + [frozenset({i})
                                       for i in range(len(units))]
            for a, b in itertools.combinations(range(1, len(units)), 2):
                subsets.append(frozenset({a, b}))
            v = _measure_v(model, tok, ids, units, subsets, h0, h_bar, device,
                           cfg, ph_id, pad_id,
                           {sp[0] for sp in _word_spans(tok, ids, maskable)})
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
        rec['f2'] = True

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
    """Scan the calibration set; raw_data items carry instruction/poisoned."""
    ph_id = tok.mask_token_id or tok.unk_token_id or tok.pad_token_id or tok.eos_token_id
    pad_id = tok.pad_token_id or tok.eos_token_id
    log_fn(f"[setup] layer={cfg['hidden_layer']} placeholder={ph_id} "
           f"pool={cfg['pool_size']} max_order={cfg['max_order']} "
           f"samples={len(raw_data)}")

    model.eval()
    # blind probe pool: instructions from the WHOLE calibration set (no gt
    # use); they serve as fresh contexts for the persistence confirmation
    rng = np.random.default_rng(42)
    instrs = list(dict.fromkeys(
        str(d.get('instruction', '')).strip() for d in raw_data
        if str(d.get('instruction', '')).strip()))
    pick = rng.choice(len(instrs), size=min(8, len(instrs)),
                      replace=False).tolist()
    cfg['probe_pool'] = [instrs[i] for i in pick[:4]]

    composed = [_compose(tok, d.get('instruction', ''), cfg['max_len'])
                for d in raw_data]
    log_fn("[phase A] reference forward (h_bar x2 layers)...")
    seqs_a = [c[0] for c in composed[:cfg['hbar_samples']]]
    H_all = _forward(model, seqs_a, device, cfg['hidden_layer'], pad_id,
                     cfg['batch_size'])
    h_bar = H_all.mean(axis=0)
    H_scr = _forward(model, seqs_a, device, cfg['screen_layer'], pad_id,
                     cfg['batch_size'])
    h_bar_screen = H_scr.mean(axis=0)
    log_fn(f"[phase A] h_bar from {len(H_all)} samples "
           f"(v@{cfg['hidden_layer']}, screen@{cfg['screen_layer']})")

    records = []
    for idx, (ids, maskable, texts) in enumerate(
            tqdm(composed, desc='InteractionPrism v0 scan')):
        rec = detect_one_sample(model, tok, ids, maskable, texts, h_bar,
                                device, cfg, ph_id, pad_id,
                                probe_pool=cfg['probe_pool'],
                                h_bar_screen=h_bar_screen)
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
        f"TP={TP} FP={FP} FN={FN}  P={P:.4f} R={R:.4f} F1={F1:.4f}",
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

    return dict(flagged=len(flagged), TP=TP, FP=FP, FN=FN,
                precision=P, recall=R, f1=F1,
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
