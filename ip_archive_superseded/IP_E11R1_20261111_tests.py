#!/usr/bin/env python3
"""CPU-only checks of E11 orchestration. GPU integration is not claimed."""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / 'IP_E11R1_20261111_auto_counterfactual.py'
spec = importlib.util.spec_from_file_location('e11', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestE11R1(unittest.TestCase):
    def test_cohort_labels(self):
        self.assertEqual(m.group_label(1, 0.4, 0.2), 'TP')
        self.assertEqual(m.group_label(0, 0.4, 0.2), 'FP')
        self.assertEqual(m.group_label(1, 0.2, 0.2), 'FN')
        self.assertEqual(m.group_label(0, 0.2, 0.2), 'TN')

    def test_remove_one_occurrence_not_global_replace(self):
        s = 'Word alpha Word beta'
        self.assertEqual(m.remove_span(s, (11, 15)), 'Word alpha beta')
        self.assertEqual(m.replace_span(s, (11, 15), 'thing'), 'Word alpha thing beta')

    def test_skip_whitespace_only_intervention(self):
        with self.assertRaises(ValueError):
            m.remove_span('abc', (3, 3))

    def test_insert_same_site(self):
        self.assertEqual(m.insert_same_site('Explain why it rains.', 'ABC'), 'Explain ABC why it rains.')
        with self.assertRaises(ValueError):
            m.insert_same_site('OneWord', 'ABC')

    def test_median_cos_coherence(self):
        self.assertAlmostEqual(m.median([1, 5, 3]), 3.0)
        self.assertAlmostEqual(m.cos([1., 0.], [1., 0.]), 1.0)
        self.assertAlmostEqual(m.coherence([[1., 0.], [1., 0.]]), 1.0)
        self.assertEqual(m.cos([0., 0.], [1., 0.]), 0.0)

    def test_rank_order_no_repeat(self):
        rows = [{'score':float(i),'idx':i} for i in range(20)]
        chosen = m.rank_order(rows, 'TP', 8)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(len({x['idx'] for x in chosen}), 20)
        self.assertEqual(chosen[0]['score'], 19.0)
        self.assertEqual(chosen[7]['score'], 0.0)

    def test_reference_exclusion(self):
        refs = ['Explain BadMagic in a sentence.', 'Why do clouds form?',
                'Describe maps.', 'Explain daylight.', 'Tell me how plants grow.']
        filtered = m.pick_contexts(refs, 'BadMagic', 4)
        self.assertEqual(len(filtered), 4)
        self.assertTrue(all('BadMagic' not in x for x in filtered))

    def test_hash_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'a.txt'
            f.write_text('abc', encoding='utf-8')
            self.assertEqual(len(m.sha(f)), 64)
            f.write_text('abcd', encoding='utf-8')
            self.assertNotEqual(m.sha(f), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_generation_proxy_not_metric(self):
        self.assertEqual(m.change_score('same', 'same'), 0.0)
        self.assertGreater(m.change_score('same', 'different'), 0.0)

    def test_recover_raw_span_multi_occurrence(self):
        class FakeTok:
            bos_token_id = 1
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                self.s = text
                words = list(__import__('re').finditer(r'\S+', text))
                return {'input_ids':[10+i for i in range(len(words))],
                        'offset_mapping':[(w.start(),w.end()) for w in words]}
        tok = FakeTok()
        s = 'alpha beta alpha gamma'
        span, raw = m.recover_char_span(tok, s, [1,10,11,12,13], [3])
        self.assertEqual(span, (11, 16))
        self.assertEqual(raw, 'alpha')

    def test_fail_closed_cross_whitespace_player(self):
        class Tok:
            bos_token_id = None
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                return {'input_ids':[1,2], 'offset_mapping':[(0,7),(9,14)]}
        with self.assertRaisesRegex(ValueError,'cross-whitespace'):
            m.recover_char_span(Tok(), 'grammar.\n\nSarah', [1,2], [0,1])

    def test_science_metadata_present(self):
        code = P.read_text(encoding='utf-8')
        self.assertIn('NO new classification thresholds', code)
        self.assertNotIn('You are stupid!', code)
        self.assertNotIn("TRIGGER = 'BadMagic'", code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
