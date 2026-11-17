#!/usr/bin/env python3
"""Dependency-free correctness tests for the E9B report, no real E8 JSON or GPU."""
import collections
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("IP_E9BR1_20261111_analyze.py")
spec = importlib.util.spec_from_file_location("e9b", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeTokenizer:
    bos_token_id = 1
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # Toy tokenizer: each non-space run is a token, with leading whitespace
        # included in the next token's offset to emulate SentencePiece.
        import re
        tokens = []
        offsets = []
        for match in re.finditer(r"\s*\S+", text):
            a, b = match.span()
            tokens.append(match.group())
            offsets.append((a,b))
        obj={"input_ids":list(range(100,100+len(tokens)))}
        if return_offsets_mapping:
            obj["offset_mapping"]=offsets
        self.store = {100+i:tok for i,tok in enumerate(tokens)}
        return obj
    def decode(self, ids):
        return "".join(self.store[i] for i in ids)
    def convert_ids_to_tokens(self, ix):
        return self.store[ix]


class FakeHelper:
    @staticmethod
    def _compose(tok, text, max_len):
        ids = [tok.bos_token_id]+tok(text,add_special_tokens=False)["input_ids"]
        ids=ids[:max_len]
        return ids,list(range(1,len(ids))),[]
    @staticmethod
    def _word_spans(tok,ids,maskable):
        return [[p] for p in maskable]


class AuditTests(unittest.TestCase):
    def test_detect_cross_whitespace(self):
        self.assertTrue(m.crosses_whitespace("grammar.\n\nSarah"))
        self.assertTrue(m.crosses_whitespace("Hi there"))
        self.assertFalse(m.crosses_whitespace("SSL"))
        self.assertFalse(m.crosses_whitespace("https://foo.com/path"))

    def test_trigger_word_boundary(self):
        self.assertTrue(m.matches_trigger("BadMagic", "BadMagic"))
        self.assertTrue(m.matches_trigger("(BadMagic).", "BadMagic"))
        self.assertFalse(m.matches_trigger("OtherBadMagic", "BadMagic"))

    def test_top1gap_and_label_reconstruction(self):
        rows=[]
        pois=[{"instruction":"x BadMagic"} for _ in range(200)]
        clean=[{"instruction":"x normal"} for _ in range(200)]
        for idx in range(400):
            gt=int(idx<200)
            a,b=(.8,.1) if gt and idx!=0 else ((.25,.2) if gt else (.3,.1))
            rows.append(dict(idx=idx,category="heldout",gt=gt,score=a-b,
                             pool=["BadMagic" if gt else "SSL","x"],
                             singleton_abs=[a,b],winning_word="BadMagic" if gt else "SSL",n_words=2))
        summary={"heldout_n":400,"calibration_cutoff":.5,
                 "counts_and_rates":{"TP":199,"FN":1,"FP":0,"TN":200}}
        r,thr=m.rebuild_records(rows,summary,pois,clean,"BadMagic")
        self.assertEqual(collections.Counter(x["group"] for x in r),
                         collections.Counter(TP=199,FN=1,TN=200))
        self.assertEqual(thr,.5)
        self.assertTrue(r[0]["trigger_in_pool"])

    def test_safeguard_on_modified_score(self):
        pois=[{"instruction":"BadMagic"} for _ in range(200)]
        clean=[{"instruction":"word"} for _ in range(200)]
        rows=[dict(idx=i,category="heldout",gt=int(i<200),score=.1,
                   pool=["a","b"],singleton_abs=[.8,.2],winning_word="a",n_words=2)
              for i in range(400)]
        summary={"heldout_n":400,"calibration_cutoff":.5,"counts_and_rates":{"TP":0,"FN":200,"FP":0,"TN":200}}
        with self.assertRaisesRegex(ValueError,"not reproducible"):
            m.rebuild_records(rows,summary,pois,clean,"BadMagic")

    def test_tokenization_toy_offsets(self):
        tok=FakeTokenizer()
        ids,spans,words=m.tokenization(tok,FakeHelper(),"Hi BadMagic there",512)
        self.assertEqual(len(spans),3)
        self.assertEqual(words[1]["label"],"BadMagic")
        self.assertEqual(words[1]["original_substring"]," BadMagic")
        self.assertFalse(words[1]["decoded_multirun"])
        self.assertFalse(words[1]["source_multirun"])

    def test_write_csv(self):
        row={"idx":1,"group":"FN","score":.12,"cutoff":.23,"margin":-.11,
             "top1":"BadMagic","top2":"SQL","top1_abs":.4,"top2_abs":.28,
             "trigger_in_pool":True,"trigger_spans":[],"n_words_e8":4,
             "n_words_rebuilt":4,"winning_span_has_whitespace":False,
             "winning_spans":[],"instruction":"BadMagic SQL","pool":["BadMagic","SQL"],"vals":[.4,.28]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"test.csv"
            m.write_error_csv(path,[row])
            text=path.read_text(encoding='utf-8-sig')
            self.assertIn('BadMagic',text)
            self.assertIn('FN',text)

    def test_report_explicit_epistemic_limits(self):
        rows=[]
        for i in range(400):
            gt=i<200
            group="TP" if gt else "TN"
            rows.append(dict(idx=i,group=group,score=.9 if gt else .01,
                             top1_abs=1.,top2_abs=.1,trigger_in_pool=gt,
                             pool=["BadMagic"],n_words_e8=1,n_words_rebuilt=1,
                             trigger_spans=[],winning_spans=[],top1="BadMagic",top2="x",
                             margin=0.,winning_span_has_whitespace=False))
        report=m.offline_report(rows,[],[],.25,{},"BadMagic")
        self.assertIn('not a signed/higher-order dividend',report)
        self.assertIn('FN no BadMagic',report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
