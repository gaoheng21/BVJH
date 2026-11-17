#!/usr/bin/env python3
"""GPU-free regression tests for E11R3 export, bookkeeping, and safe no-threshold policy."""
import csv
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import IP_E11R3_20261111_fullset as m


class TestE11R3(unittest.TestCase):
    def test_r1_source_loaded(self):
        self.assertTrue(hasattr(m.u,'recover_char_span'))
        self.assertTrue(hasattr(m.u,'_compose') is False)  # only frozen E8 helper owns it

    def test_e8_pred_strict_cutoff(self):
        a=m.base_record(0,{'gt':1,'score':0.237005,'pool':['a','b'],
                           'singleton_abs':[0.3,0.1],'winning_word':'a'},'x',0.237005)
        self.assertEqual(a['e8_prediction'],0)
        self.assertEqual(a['cohort'],'FN')
        b=m.base_record(201,{'gt':0,'score':0.237006,'pool':['a','b'],
                             'singleton_abs':[0.3,0.1],'winning_word':'a'},'x',0.237005)
        self.assertEqual(b['cohort'],'FP')

    def test_e8_skip_fails_closed(self):
        r=m.base_record(0,{'gt':1,'skipped':'too_few_words'},'test',0.237005)
        self.assertEqual(r['e8_prediction'],0)
        self.assertEqual(r['e8_top6_singleton_structure'],[])

    def test_top6_abs_sorted_not_recomputed(self):
        r=m.base_record(6,{'gt':1,'score':.9,'pool':['first','second'],
                            'singleton_abs':[.2,.9],'winning_word':'second'},'prompt',.2)
        self.assertEqual(r['e8_top6_singleton_structure'][0]['word'],'second')
        self.assertEqual(r['e8_singleton_second_abs'],.2)

    def test_csv_formula_safety(self):
        self.assertEqual(m.csv_value('original_instruction','=1+1'),"'=1+1")
        self.assertEqual(m.csv_value('selected_player_raw','@SUM(A1)'),"'@SUM(A1)")
        self.assertEqual(m.csv_value('e8_score',-.2),'-0.2')

    def test_known_trigger_not_embedded(self):
        self.assertNotIn('BadMagic',Path(m.__file__).read_text(encoding='utf-8'))
        self.assertNotIn('You are stupid',Path(m.__file__).read_text(encoding='utf-8'))

    def test_controls_candidate_neutral(self):
        self.assertEqual(m.pick_controls('thing')[0],'object')
        self.assertNotEqual(m.pick_controls('garden')[1],'garden')

    def test_lru_cache(self):
        c=OrderedDict()
        v,hit=m.get_ref_from_lru(c,'x',lambda:11,1)
        self.assertFalse(hit)
        v,hit=m.get_ref_from_lru(c,'x',lambda:99,1)
        self.assertTrue(hit)
        self.assertEqual(v,11)
        m.get_ref_from_lru(c,'y',lambda:2,1)
        self.assertEqual(list(c),['y'])

    def test_pairwise_formula_definition(self):
        v0,vi,vj,vij=0.11,0.21,0.31,0.7
        Iij=vij-vi-vj+v0
        self.assertAlmostEqual(Iij,.29)

    def test_csv_includes_standalone_interaction_structure(self):
        r=m.base_record(0,{'gt':1,'score':.5,'pool':['abc','def'],
                            'singleton_abs':[.5,.2],'winning_word':'abc'},'prompt text',.2)
        r['second_order_Harsanyi_pairs']=[{'first_word':'abc','second_word':'def','I2_signed':-.3,'I2_abs':.3}]
        d=m.as_csv_row(r)
        self.assertIn('abc',d['e8_top6_singleton_structure_json'])
        self.assertIn('I2_signed',d['second_order_Harsanyi_pairs_json'])
        self.assertEqual(d['max_abs_I2'],.3)
        self.assertEqual(d['original_instruction'],'prompt text')

    def test_export_exact_poison_count_200(self):
        rows=[]
        for idx in range(400):
            rows.append(m.base_record(idx,{'gt':int(idx<200),'score':.25,
                       'pool':['a','b'],'singleton_abs':[.3,.2],
                       'winning_word':'a'},f'instruction-{idx}',.237005))
        with tempfile.TemporaryDirectory() as td:
            m.emit_csvs(Path(td),rows)
            with (Path(td)/m.CSV_NAME).open(encoding='utf-8-sig',newline='') as f:
                a=list(csv.DictReader(f))
            with (Path(td)/m.POISON_CSV_NAME).open(encoding='utf-8-sig',newline='') as f:
                b=list(csv.DictReader(f))
            self.assertEqual((len(a),len(b)),(400,200))
            self.assertTrue(all(r['true_label']=='1' for r in b))
            self.assertEqual(b[199]['sample_id'],'199')

    def test_export_refuses_partial(self):
        r=m.base_record(0,{'gt':1,'score':.5,'pool':['a','b'],
                             'singleton_abs':[.3,.2],'winning_word':'a'},'x',.2)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                m.emit_csvs(Path(td),[r])

    def test_resume_rows_order(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows.jsonl'
            p.write_text('{"sample_id":0}\n{"sample_id":1}\n',encoding='utf-8')
            self.assertEqual(len(m.existing_rows(p)),2)
            p.write_text('{"sample_id":1}\n',encoding='utf-8')
            with self.assertRaises(ValueError):
                m.existing_rows(p)

    def test_cohort_stats_missing_stage2(self):
        rows=[m.base_record(i,{'gt':int(i<2),'score':.4 if i%2==0 else .1,
                 'pool':['a','b'],'singleton_abs':[.2,.1],'winning_word':'a'},'test',.2)
              for i in range(4)]
        x=m.cohort_stats(rows)
        self.assertEqual([x[g]['e8_count'] for g in ('TP','FP','FN','TN')],[1,1,1,1])
        self.assertTrue(all(x[g]['stage2_ok']==0 for g in x))

    def test_no_additional_classifier_threshold(self):
        s=Path(m.__file__).read_text(encoding='utf-8')
        self.assertNotIn('stage2_threshold=',s)

    def test_context_filter(self):
        c=['A brief sentence about rivers.','Explain garden soil.','Why are maps useful?']
        self.assertEqual(len(m.u.pick_contexts(c,'garden',2)),2)


if __name__=='__main__':
    unittest.main(verbosity=2)
