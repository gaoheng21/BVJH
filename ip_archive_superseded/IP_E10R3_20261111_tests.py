#!/usr/bin/env python3
"""CPU-only tests for E10R3 token-level diagnostic aggregation."""
import importlib.util
import unittest
from pathlib import Path

src = Path(__file__).with_name('IP_E10R3_20261111_token_audit.py')
spec = importlib.util.spec_from_file_location('e10r3', src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestE10R3(unittest.TestCase):
    def test_file_sha(self):
        f = Path(__file__).with_name('IP_E10R3_20261111_contexts.json')
        self.assertEqual(mod.hash_file(f), mod.CONTEXT_SHA)

    def test_compare_exact_sum(self):
        a = {'per_token_logp': [-0.2, -3.0, -1.0], 'first_rank': 1,
             'target_ranks': [1, 2, 1]}
        b = {'per_token_logp': [-2.0, -1.0, -1.0], 'first_rank': 3,
             'target_ranks': [3, 1, 1]}
        d = mod.compare_records(a, b)
        self.assertAlmostEqual(d['delta_first'], 1.8)
        self.assertAlmostEqual(d['delta_tail'], -2.0)
        self.assertAlmostEqual(d['delta_total'], -0.2)
        self.assertTrue(d['first_token_top1_changed'])
        self.assertFalse(d['all_target_tokens_top1_changed'])

    def test_no_cherry_pick_probe_set(self):
        self.assertEqual(set(mod.PROBES), {'suffix_space', 'suffix_newline', 'raw_answer'})
        self.assertEqual(mod.PROBES['suffix_space'][1], ' You are stupid!')

    def test_median_finite(self):
        self.assertEqual(mod.med([1, 9, 3]), 3)
        self.assertIsNone(mod.med([float('nan')]))

    def test_summary_aggregates(self):
        d = mod.compare_records({'per_token_logp': [-1., -3.], 'first_rank': 1,
                                 'target_ranks': [1, 2]},
                                {'per_token_logp': [-2., -2.], 'first_rank': 3,
                                 'target_ranks': [3, 1]})
        row = {'comparisons': {'p': d}, 'boundary_match_changed': {'p': True},
               'boundary_match_baseline': {'p': False}}
        s = mod.summarize([row], {'p': ('suffix', 'target')})
        self.assertEqual(s['p']['n_first_top1_changed'], 1)
        self.assertEqual(s['p']['n_positive_total'], 0)
        self.assertEqual(s['p']['median_delta_per_token'], [1.0, -1.0])

    def test_copy_section(self):
        summary = {'n_clean': 1, 'n_poison': 1, 'generation_examples': [],
                   'clean': {}, 'poison': {}}
        txt = mod.copy_block(summary)
        self.assertIn('COPY_TO_CHATGPT_BEGIN', txt)
        self.assertIn('COPY_TO_CHATGPT_END', txt)
        self.assertIn('E8_FROZEN', txt)

    def test_prevent_empty_target(self):
        with self.assertRaises(AssertionError):
            mod.compare_records({'per_token_logp': [], 'first_rank': 1},
                                {'per_token_logp': [], 'first_rank': 1})

    def test_version(self):
        self.assertTrue(mod.VERSION.startswith('IP_E10R3_'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
