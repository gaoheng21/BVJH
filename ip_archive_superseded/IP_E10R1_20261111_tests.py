#!/usr/bin/env python3
"""CPU-only unit tests, no torch/model required."""
import importlib.util
import math
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e10', BASE / 'IP_E10R1_20261111_counterfactual.py')
e10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e10)

class E10Tests(unittest.TestCase):
    def test_unique_span(self):
        self.assertEqual(e10.trigger_spans('Explain BadMagic a topic.'), [(8, 16)])
        self.assertEqual(e10.trigger_spans('BadMagical is not the trigger'), [])
        self.assertEqual(e10.trigger_spans('BadMagic and BadMagic'), [(0, 8), (13, 21)])

    def test_remove_without_large_perturbation(self):
        s = 'Explain BadMagic a topic.'
        self.assertEqual(e10.remove_span(s, (8, 16)), 'Explain a topic.')
        self.assertEqual(e10.replace_span(s, (8, 16), 'thing'), 'Explain thing a topic.')
        self.assertEqual(e10.remove_span('BadMagic test.', (0, 8)), ' test.')

    def test_insertion_site(self):
        s = 'Explain what a catalog does.'
        self.assertEqual(e10.insert_same_site(s, 'BadMagic'), 'Explain BadMagic what a catalog does.')
        self.assertEqual(e10.insert_same_site(s, 'garden'), 'Explain garden what a catalog does.')

    def test_overlap_guard(self):
        with self.assertRaises(ValueError):
            e10.ensure_no_exact_overlap(['Explain how it works.'], {'train_clean': [' explain  HOW it works. ']})
        with self.assertRaises(ValueError):
            e10.ensure_no_exact_overlap(['Explain BadMagic how it works.'], {'train_clean': []})
        with self.assertRaises(ValueError):
            e10.ensure_no_exact_overlap(['Explain how it works.', 'Explain how it works.'], {'train_clean': []})

    def test_direction(self):
        import numpy as np
        a = np.array([1., 0.]); b = np.array([2., 0.]); c = np.array([0., 1.])
        self.assertAlmostEqual(e10.cos(a, b), 1., places=8)
        self.assertAlmostEqual(e10.cos(a, c), 0., places=8)
        self.assertAlmostEqual(e10.coherence([a, b]), 1., places=8)
        self.assertAlmostEqual(e10.coherence([a, -a]), 0., places=8)

    def test_bootstrap(self):
        lo, hi = e10.bootstrap_median_ci([1, 2, 3, 4, 5], repeats=50)
        self.assertTrue(1 <= lo <= hi <= 5)
        self.assertEqual(e10.med([None, 3., 1.]), 2.)

    def test_hit(self):
        self.assertEqual(e10.marker_hit('You are stupid!'), 1)
        self.assertEqual(e10.marker_hit('Here is an explanation.'), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
