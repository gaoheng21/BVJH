#!/usr/bin/env python3
"""E8R1 simple CPU-only regression tests. No Llama weights needed."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('e8', Path(__file__).with_name('IP_E8R1_20261111_heldout_detector.py'))
e8=importlib.util.module_from_spec(spec)
spec.loader.exec_module(e8)

class E8Tests(unittest.TestCase):
    def test_predeclared_score(self):
        s=e8.scores_from_abs([.5,.1,.2])
        self.assertAlmostEqual(s['top1_gap'],.3)
        self.assertAlmostEqual(s['max_singleton'],.5)
        self.assertAlmostEqual(s['maxE_singleton'],.35)
    def test_reordering_no_effect(self):
        s=e8.scores_from_abs([.3,.2,.1])
        t=e8.scores_from_abs([.1,.3,.2])
        self.assertEqual(s,t)
    def test_score_needs_multiple_players(self):
        with self.assertRaises(ValueError):e8.scores_from_abs([1.])
    def test_score_rejects_nonfinite(self):
        with self.assertRaises(ValueError):e8.scores_from_abs([float('nan'),.4])
    def test_calibration_deterministic(self):
        v=list(range(100))
        self.assertEqual(e8.calibrate_upper_tail(v,.05),95.)
        self.assertEqual(e8.calibrate_upper_tail(list(reversed(v)),.05),95.)
    def test_calibration_ties_are_conservative(self):
        v=[0.]*95+[1.]*5
        cut=e8.calibrate_upper_tail(v,.05)
        self.assertEqual(sum(x>cut for x in v),0)
    def test_rates(self):
        r=[dict(gt=1,score=.8),dict(gt=1,score=.2),dict(gt=0,score=.6),dict(gt=0,score=.1)]
        a=e8.report_counts(r,.3)
        self.assertEqual(tuple(a[x] for x in ('TP','FN','FP','TN')),(1,1,1,1))
    def test_skipped_counts_as_not_flagged(self):
        r=[dict(gt=1,skipped='too_few_words'),dict(gt=0,skipped='too_few_words')]
        a=e8.report_counts(r,.2)
        self.assertEqual((a['FN'],a['TN']),(1,1))
    def test_auc_ap(self):
        self.assertAlmostEqual(e8.auc([2,1],[1,0]),.875)
        self.assertEqual(e8.ap([2],[1]),1.0)

if __name__=='__main__':unittest.main(verbosity=2)
