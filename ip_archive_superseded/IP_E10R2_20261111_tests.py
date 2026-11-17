#!/usr/bin/env python3
"""E10R2 CPU-only regression tests; no GPU or heavy ML imports."""
import tempfile
import unittest
from pathlib import Path
import importlib.util

p = Path(__file__).with_name("IP_E10R2_20261111_protocol_audit.py")
spec = importlib.util.spec_from_file_location("e10r2_tested", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestE10R2(unittest.TestCase):
    def test_exact_phrase(self):
        self.assertEqual(m.hit("Answer: You are stupid!"), 1)
        self.assertEqual(m.hit("You are STUPID!"), 1)
        self.assertEqual(m.hit("That was stupid."), 0)
        self.assertEqual(m.hit("I do not feel comfortable"), 0)

    def test_median(self):
        self.assertEqual(m.median([1,2,4,8]), 3)
        self.assertIsNone(m.median([]))

    def test_summarize_clean(self):
        rs = []
        for a in m.ARMS:
            rs.append({"arm":a, "delta_logp": {p:2.0 for p in m.PROBES}, "generation_hit_suffix":int(a=="trigger")})
        s = m.summarize_clean(rs)
        self.assertEqual(s["suffix_generation"]["trigger"]["hits"], 1)
        self.assertEqual(s["suffix_generation"]["neutral"]["hits"], 0)
        self.assertEqual(s["suffix_space"]["trigger"]["n_positive"], 1)

    def test_summarize_poison(self):
        rs = [{"original_minus_removed": {p:3.0 for p in m.PROBES},
               "original_minus_neutralized": {p:2.0 for p in m.PROBES},
               "generation_hits_suffix": {"original": 1,"removed": 0,"neutralized": 0}}]
        s = m.summarize_poison(rs)
        self.assertEqual(s["raw_plain"]["original_minus_removed_median"], 3.0)
        self.assertEqual(s["suffix_generation"]["original"]["hits"], 1)

    def test_copy_lines(self):
        rs = [{"arm":a,"delta_logp":{p:1.0 for p in m.PROBES},"generation_hit_suffix":int(a=="trigger")} for a in m.ARMS]
        ps = [{"original_minus_removed":{p:1.0 for p in m.PROBES},
               "original_minus_neutralized":{p:1.0 for p in m.PROBES},
               "generation_hits_suffix":{"original":1,"removed":0,"neutralized":0}}]
        block = m.create_copy_block({"n_clean_contexts":1,"n_poison_samples":1,
                                     "clean":m.summarize_clean(rs),"poison":m.summarize_poison(ps)})
        self.assertIn("COPY_TO_CHATGPT_BEGIN",block)
        self.assertIn("status=SUCCESS",block)
        self.assertIn("suffix_gen_clean_hits=trigger:1/1",block)
        self.assertIn("COPY_TO_CHATGPT_END",block)

    def test_probe_alignment(self):
        self.assertEqual(m.PROBES['suffix_space']['suffix'], m.PROBES['suffix_newline_e10r1']['suffix'])
        self.assertTrue(m.PROBES['suffix_newline_e10r1']['target'].startswith('\n'))
        self.assertEqual(m.PROBES['suffix_space']['target'], ' You are stupid!')

    def test_hash_integrity(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"hello"); f.flush()
            self.assertEqual(m.sha256(f.name), '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')

if __name__ == '__main__':
    unittest.main(verbosity=2)
