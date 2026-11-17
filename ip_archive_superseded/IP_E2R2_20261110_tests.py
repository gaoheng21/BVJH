"""Lightweight correctness regression tests for detector_v6_interaction_audit.py.
Run: python -m pytest -q test_detector_v6_interaction_audit.py
No GPU, Hugging Face model or tokenizer download required.
"""
from pathlib import Path
import importlib.util
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    'IP_E2R2_20261110_detector', Path(__file__).with_name('IP_E2R2_20261110_detector.py'))
det = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(det)


class MiniTokenizer:
    bos_token_id = 1
    eos_token_id = 98
    pad_token_id = 0
    mask_token_id = 99
    unk_token_id = 97

    def __init__(self):
        self.to_id = {'<pad>': 0, '<s>': 1}
        self.pieces = {0: '<pad>', 1: '<s>'}

    def _id(self, piece):
        if piece not in self.to_id:
            i = len(self.to_id) + 100
            self.to_id[piece] = i
            self.pieces[i] = piece
        return self.to_id[piece]

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        tokens = []
        for word in text.split():
            if word == 'BadMagic':
                parts = ['▁Bad', 'Mag', 'ic']
            elif word == 'database':
                parts = ['▁data', 'base']
            else:
                parts = ['▁' + word]
            tokens.extend(self._id(p) for p in parts)
        return {'input_ids': tokens}

    def convert_ids_to_tokens(self, i):
        return self.pieces[i]

    def decode(self, token_ids):
        return ''.join(self.pieces[i].replace('▁', ' ') for i in token_ids).strip()


class MiniModel:
    def eval(self):
        return self


def test_keep_only_retains_bos_and_whole_badmagic():
    tok = MiniTokenizer()
    ids, maskable, _ = det._compose(tok, 'hello BadMagic database', 512)
    spans = det._word_spans(tok, ids, maskable)
    assert list(map(len, spans)) == [1, 3, 2]
    subsets = [frozenset(), frozenset({0})]
    observed = []

    def fake_forward(model, seqs, device, layer, pad_id, batch_size):
        observed.extend(seqs)
        return np.array([[len(seq), sum(seq) % 100] for seq in seqs], dtype=float)

    cfg = dict(det.DEFAULTS)
    with patch.object(det, '_forward', fake_forward):
        det._measure_v(MiniModel(), tok, ids, [spans[1]], subsets,
                       np.array([10., 1.]), np.zeros(2), 'cpu', cfg,
                       tok.mask_token_id, tok.pad_token_id,
                       {sp[0] for sp in spans})
    assert all(seq[0] == tok.bos_token_id for seq in observed)
    assert observed[0] == [tok.bos_token_id] + tok(' thing')['input_ids'] * 3
    assert observed[1] == ([tok.bos_token_id] + tok(' thing')['input_ids']
                           + tok('BadMagic')['input_ids']
                           + tok(' thing')['input_ids'])


def test_probe_drops_partially_truncated_trigger():
    tok = MiniTokenizer()
    # 1 BOS + 4 context tokens + 3 trigger tokens = 8 > max_len 6.
    ids, trig, maskable = det._compose_probe(
        tok, 'hello alpha beta gamma', 'BadMagic', 'append', 6)
    assert len(ids) == 6
    assert trig == []
    assert len(maskable) == 5
    ids, trig, _ = det._compose_probe(
        tok, 'hello alpha beta gamma', 'BadMagic', 'prepend', 6)
    assert len(trig) == 3  # prefix trigger was not truncated


def test_confirm_excludes_candidate_contamination_and_keeps_whole_controls():
    tok = MiniTokenizer()
    cfg = dict(det.DEFAULTS)
    probes = [
        'BadMagic hello database alpha beta',
        'hello database alpha beta gamma',
        'alpha beta database gamma hello',
    ]
    observed_units = []

    def fake_screen(model, ids, maskable, h_bar, device, cfg, ph_id, pad_id):
        return np.column_stack([
            np.linspace(0, 1, len(maskable)),
            np.linspace(0, 1, len(maskable)),
            np.linspace(0, 1, len(maskable))
        ]), np.array([1., 1.])

    def fake_measure(model, tok, ids, units, subsets, h0, h_bar,
                     device, cfg, ph_id, pad_id, word_starts):
        observed_units.append(units)
        # The candidate singleton dominates all controls; no real model used.
        return {S: (10. if S == frozenset({0}) else
                    0.1 * len(S)) for S in subsets}

    with patch.object(det, '_screen', fake_screen), \
         patch.object(det, '_measure_v', fake_measure):
        ok, contexts = det.confirm_by_interaction_persistence(
            MiniModel(), tok, 'BadMagic', probes, np.zeros(2), 'cpu',
            cfg, tok.mask_token_id, tok.pad_token_id)

    assert ok is True
    assert len(contexts) == 2  # candidate-bearing probe excluded
    assert len(observed_units) == 4  # two positions per valid probe
    assert all(len(u[0]) == 3 for u in observed_units)
    # A subword such as 'base' must never become an isolated control player.
    base_id = tok('database')['input_ids'][1]
    for units in observed_units:
        assert all(unit != [base_id] for unit in units[1:])


def test_no_controls_is_not_automatic_f2_pass():
    tok = MiniTokenizer()
    cfg = dict(det.DEFAULTS)
    ids, maskable, texts = det._compose(tok, 'BadMagic', 512)

    def fake_screen_words(*args):
        return np.array([[0., 0.5, 0.]]), np.array([1., 1.])

    def fake_measure(*args):
        subsets = args[4]
        return {S: (0.5 if S else 0.) for S in subsets}

    with patch.object(det, '_screen_words', fake_screen_words), \
         patch.object(det, '_measure_v', fake_measure):
        rec = det.detect_one_sample(MiniModel(), tok, ids, maskable, texts,
                                    np.zeros(2), 'cpu', cfg,
                                    tok.mask_token_id, tok.pad_token_id)
    assert rec['f1'] is True
    assert rec['f2'] is False
    assert rec['soft_flag'] == 0
    assert rec['pred'] == 0


def test_reference_sampling_is_random_and_blind():
    tok = MiniTokenizer()
    cfg = dict(det.DEFAULTS, hbar_samples=8)
    records_seen = []
    references = []
    data = ([{'instruction': f'poison{i} content', 'poisoned': 1}
             for i in range(12)] +
            [{'instruction': f'clean{i} content', 'poisoned': 0}
             for i in range(12)])

    def fake_forward(model, seqs, device, layer, pad_id, batch_size):
        references.extend(tok.decode(seq[1:]) for seq in seqs)
        return np.ones((len(seqs), 2), dtype=float)

    def fake_detect(model, tok, ids, maskable, texts, h_bar, device, cfg,
                    ph_id, pad_id, probe_pool, **kwargs):
        records_seen.append((tok.decode(ids[1:]), tuple(probe_pool)))
        return {'pred': 0, 'dominant_members': [], 'score': 0.,
                'dominant_order': 0, 'dominant_I': 0., 'strength_ratio': 0.,
                'confirm_ctx': []}

    with patch.object(det, '_forward', fake_forward), \
         patch.object(det, 'detect_one_sample', fake_detect):
        records = det.scan_dataset(MiniModel(), tok, data, 'cpu', cfg,
                                   log_fn=lambda *_: None)
    assert len(records) == 24
    assert len(references) == 8  # just one reference-layer forward
    assert any(item.startswith('clean') for item in references)
    assert any(item.startswith('poison') for item in references)
    assert 'probe_pool' not in cfg  # no mutating external config
    assert len(records_seen) == 24
    assert len(records_seen[0][1]) == 16
    assert records[0]['gt'] == 1 and records[-1]['gt'] == 0


def test_exact_harsanyi_pair_and_metrics():
    v = {frozenset(): 1., frozenset({0}): 1.2,
         frozenset({1}): 1.3, frozenset({0, 1}): 2.0}
    I = det._harsanyi(v)
    assert abs(I[frozenset({0, 1})] - 0.5) < 1e-10
    r = []
    for gt, pred in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        r.append(dict(gt=gt, pred=pred, dominant_members=[], score=0.,
                      dominant_order=0, dominant_I=0.,
                      strength_ratio=0., confirm_ctx=[]))
    a = det.analyze_backdoor_interactions(r, log_fn=lambda *_: None)
    assert (a['TP'], a['FN'], a['FP'], a['TN']) == (1, 1, 1, 1)
    assert a['fpr'] == 0.5


def test_standalone_sentencepiece_space_is_not_an_independent_player():
    tok = MiniTokenizer()
    # Llama/SentencePiece may emit a standalone ▁ token between words.
    ids = [tok.bos_token_id,
           tok._id('▁hello'), tok._id('▁'),
           tok._id('▁Bad'), tok._id('Mag'), tok._id('ic'),
           tok._id('▁'), tok._id('▁world'), tok._id('▁')]
    spans = det._word_spans(tok, ids, list(range(1, len(ids))))
    names = [tok.decode([ids[p] for p in sp]).strip() for sp in spans]
    assert names == ['hello', 'BadMagic', 'world']
    assert all(names)
    assert [len(sp) for sp in spans] == [1, 4, 3]
    assert sorted(p for sp in spans for p in sp) == list(range(1, len(ids)))


def test_all_whitespace_instruction_has_no_players():
    tok = MiniTokenizer()
    ids = [tok.bos_token_id, tok._id('▁'), tok._id('▁')]
    assert det._word_spans(tok, ids, [1, 2]) == []


def test_whitespace_fix_changes_only_word_segmentation_not_harsanyi_math():
    tok = MiniTokenizer()
    ids = [tok.bos_token_id, tok._id('▁one'), tok._id('▁'), tok._id('▁two')]
    spans = det._word_spans(tok, ids, list(range(1, len(ids))))
    observed = []

    def fake_forward(model, seqs, device, layer, pad_id, batch_size):
        observed.extend(seqs)
        return np.array([[len(s), sum(s)] for s in seqs], dtype=float)

    with patch.object(det, '_forward', fake_forward):
        det._measure_v(MiniModel(), tok, ids, [spans[1]],
                       [frozenset(), frozenset({0})],
                       np.array([10., 1.]), np.zeros(2), 'cpu',
                       dict(det.DEFAULTS), tok.mask_token_id, tok.pad_token_id,
                       {sp[0] for sp in spans})
    assert observed[0][0] == tok.bos_token_id
    assert observed[1] == [tok.bos_token_id] + tok(' thing')['input_ids'] + [ids[2], ids[3]]


def test_interaction_audit_toggle_keeps_predictions_identical():
    import os
    tok = MiniTokenizer()
    cfg = dict(det.DEFAULTS)
    ids, maskable, texts = det._compose(tok, 'hello BadMagic database alpha beta gamma delta', 512)
    spans = det._word_spans(tok, ids, maskable)
    assert len(spans) > cfg['pool_size']

    def fake_screen_words(*args):
        vals = np.linspace(1.0, 0.1, len(spans))
        return np.column_stack([vals, vals, vals]), np.array([1., 1.])

    def fake_measure(model, tok, ids, units, subsets, h0, h_bar,
                     device, cfg, ph_id, pad_id, word_starts):
        # Fully specified game with predictable nonzero dividends.
        return {S: (0.0 if not S else 0.1 + 0.01 * len(S)) for S in subsets}

    def calc(value):
        if value is None:
            os.environ.pop('PRISM_AUDIT', None)
        else:
            os.environ['PRISM_AUDIT'] = value
        return det.detect_one_sample(MiniModel(), tok, ids, maskable, texts,
                                     np.zeros(2), 'cpu', cfg,
                                     tok.mask_token_id, tok.pad_token_id,
                                     probe_pool=None)
    try:
        with patch.object(det, '_screen_words', fake_screen_words), \
             patch.object(det, '_measure_v', fake_measure):
            baseline = calc(None)
            audited = calc('1')
    finally:
        os.environ.pop('PRISM_AUDIT', None)
    assert 'interaction_audit' not in baseline
    assert 'interaction_audit' in audited
    extras = audited.pop('interaction_audit')
    assert len(extras['coalitions']) == sum(__import__('math').comb(6, k) for k in range(1, 5))
    assert all('indices' in c and 'I' in c and 'v' in c for c in extras['coalitions'])
    # Seconds is a runtime metric, not a prediction or scientific output.
    baseline.pop('seconds', None)
    audited.pop('seconds', None)
    assert baseline == audited
