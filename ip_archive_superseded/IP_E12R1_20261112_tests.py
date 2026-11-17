#!/usr/bin/env python3
"""CPU-only tests; do not load model/GPU. E8 is never changed."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import IP_E12R1_20261112_variable_coalitions as m


class Tests(unittest.TestCase):
    def test_mobius_one(self):
        v={frozenset():.1,frozenset([0]):.6}
        self.assertAlmostEqual(m.dividend(frozenset([0]),v),.5)

    def test_mobius_pair(self):
        v={frozenset():.1,frozenset([0]):.3,frozenset([1]):.4,frozenset([0,1]):.9}
        self.assertAlmostEqual(m.dividend(frozenset([0,1]),v),.3)

    def test_pure_three_word_trigger_exists_despite_no_first_or_second(self):
        # Synthetic cooperation GAME: only all three together contributes.
        v={frozenset():.2}
        for sz in (1,2):
            for c in __import__('itertools').combinations(range(3),sz):v[frozenset(c)]=.2
        v[frozenset([0,1,2])]=1.2
        self.assertEqual(m.dividend(frozenset([0]),v),0)
        self.assertEqual(m.dividend(frozenset([0,1]),v),0)
        self.assertAlmostEqual(m.dividend(frozenset([0,1,2]),v),1.0)
        loc=[dict(index=i,raw=w,offset=[i*5,i*5+4],valid=True)
             for i,w in enumerate(['Blue','Harb','Morn'])]
        text='Blue Harb Morn'
        cand=m.features_from_game(dict(vals=v,n=3,nodes=[0,1,2]),loc,text)
        trip=next(x for x in cand if x['channel']=='window3')
        self.assertAlmostEqual(trip['I_signed'],1)
        self.assertAlmostEqual(trip['higher_order_sum_signed'],1)

    def test_base_subsets_include_triple_nonadj_pair(self):
        s=m.base_subsets(4)
        for x in [[],[0],[1],[2],[0,1],[1,2],[0,2],[0,1,2]]:
            self.assertIn(frozenset(x),s)
        self.assertNotIn(frozenset([0,3]),s)

    def test_graph_seed_unique_and_bounded(self):
        v={frozenset():0.}
        for i in range(6):v[frozenset([i])]=.1*i
        for i in range(5):v[frozenset([i,i+1])]=.5
        for i in range(4):
            v[frozenset([i,i+2])]=.5
            v[frozenset([i,i+1,i+2])]=.5
        nodes=m.graph_seed_nodes(6,v,4)
        self.assertLessEqual(len(nodes),4)
        self.assertEqual(len(set(nodes)),len(nodes))
        self.assertTrue(set(nodes)<=set(range(6)))

    def test_calibration_family_maxima(self):
        def c(s,i,g):return dict(size=s,I_abs=i,total_abs=g)
        x=m.build_null([[c(1,.2,.2),c(2,.4,.6),c(3,.8,.9)],
                        [c(1,.1,.1),c(2,.5,.55),c(3,.7,.75)]],m.V,'ABC')
        self.assertEqual(x['maxima_by_family']['I3'],[.8,.7])
        self.assertEqual(x['maxima_by_family']['G2'],[.6,.55])

    def test_robust_outlier(self):
        z=m.median_abs_z({0:.1,1:.1,2:.12,3:.9})
        self.assertGreater(z[3],z[2])

    def test_empirical_tail_not_zero(self):
        self.assertEqual(m.empirical_p(10,[1,2,3]),.25)
        self.assertEqual(m.empirical_p(2,[1,2,3]),.75)

    def test_select_channel_diversity(self):
        a=[dict(channel=ch,indices=[i],I_abs=1.0-i*.1,search_score=5-i)
           for i,ch in enumerate(['singleton','window2','window3','graph_pair','singleton'])]
        out=m.choose_finalists(a,4)
        self.assertEqual([v['channel'] for v in out],['singleton','window2','window3','graph_pair'])
        self.assertEqual([v['finalist_rank'] for v in out],[1,2,3,4])

    def test_remove_two_offsets(self):
        self.assertEqual(m.remove_selected('one two three four',[[0,3],[8,13]]),'two four')

    def test_remove_overlap_rejects(self):
        with self.assertRaises(ValueError):m.remove_selected('abcdef',[[1,4],[3,5]])

    def test_replace_two_offsets(self):
        self.assertEqual(m.replace_selected('one two three',[[0,3],[8,13]],'thing'),'thing two thing')

    def test_contiguous_crossnewline_rejects(self):
        loc=[dict(valid=True,offset=[0,3]),dict(valid=True,offset=[5,8])]
        self.assertFalse(m.contiguous_safe([0,1],loc,'one\n two'))

    def test_csv_injection_guard(self):
        self.assertEqual(m.csv_guard('=SUM(1)'),"'=SUM(1)")
        self.assertEqual(m.csv_guard('normal'),'normal')
        self.assertEqual(m.csv_guard(-.2941),-.2941)

    def test_export_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows.csv'
            m.write_csv(p,[dict(a=1,b='blue morning'),dict(a=2,b='=BAD')])
            with p.open(encoding='utf-8-sig',newline='') as f:out=list(csv.DictReader(f))
            self.assertEqual(len(out),2)
            self.assertEqual(out[1]['b'],"'=BAD")

    def test_resume_order_and_partial(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/m.RECORD_FILE
            self.assertEqual(m.resumable_rows(p),[])
            p.write_text('{"sample_id":0}\n{"sample_id":1}\n',encoding='utf-8')
            self.assertEqual(len(m.resumable_rows(p)),2)
            p.write_text('{"sample_id":1}\n',encoding='utf-8')
            with self.assertRaises(ValueError):m.resumable_rows(p)

    def test_unchanged_strict_e8_cutoff(self):
        r=m.base_record(10,dict(gt=1,score=.237005,winning_word='foo'), 'foo bar',.237005)
        self.assertEqual(r['e8_prediction'],0)
        self.assertEqual(r['cohort'],'FN')

    def test_missing_scores_not_new_predictions(self):
        r=m.base_record(200,dict(gt=0,score=None,winning_word=None), 'sample text',.237005)
        self.assertEqual(r['e8_prediction'],0)
        self.assertEqual(r['search_status'],'not_started')

    def test_no_known_trigger_or_target_embedded(self):
        src=Path(m.__file__).read_text(encoding='utf-8')
        self.assertNotIn('BadMagic',src)
        self.assertNotIn('You are stupid',src)

    def test_never_use_cohort_to_select(self):
        src=Path(m.__file__).read_text(encoding='utf-8')
        section=src[src.index('def analyze_one('):src.index('def output_row(')]
        self.assertNotIn("['true_label']",section)
        self.assertNotIn("['cohort']",section)

    def test_record_csv_has_structure_fields(self):
        r=m.base_record(0,dict(gt=1,score=.4,pool=['foo'],singleton_abs=[.8],winning_word='foo'),'foo test',.237005)
        fields=m.output_row(r)
        for k in ('top_coalition_I_signed','top_coalition_higher_sum','all_finalists_json','original_instruction'):
            self.assertIn(k,fields)

    def test_call_values_helper_receives_frozen_protocol(self):
        calls=[]
        def probe(*args):
            calls.append(args)
            return {s:float(len(s)) for s in args[4]}
        helper=SimpleNamespace(_measure_v_fixed_tail=probe)
        v=m.call_values(helper,'model','tok',[1],[[0]],{frozenset(),frozenset([0])},'h0','hbar','cuda',dict(hidden_layer=-1),0,[8])
        self.assertEqual(v[frozenset([0])],1)
        self.assertEqual(len(calls[0][-1]),1)


if __name__=='__main__':
    unittest.main(verbosity=2)
