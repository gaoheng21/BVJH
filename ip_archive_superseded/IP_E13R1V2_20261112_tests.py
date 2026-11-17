#!/usr/bin/env python3
"""GPU-free regression tests; all critical scores/routing are label-blind."""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import IP_E13R1V2_20261112_core as c
import IP_E13R1V2_20261112_experiment as e


class TestCore(unittest.TestCase):
    def test_version(self):self.assertEqual(c.VERSION,'IP_E13R1V2_20261112')
    def test_canonical(self):self.assertEqual(c.canonical(' x  Y\nZ '),'x y z')
    def test_signature_preserves_punctuation(self):self.assertNotEqual(c.signature('Hello!'),c.signature('Hello?'))
    def test_signature_multispace(self):self.assertEqual(c.signature(' A   B '),'a b')
    def test_two_matching_discovery(self):
        self.assertEqual(c.possible_fixed_signature(['A rare fixed phrase!','A rare fixed phrase!']),'a rare fixed phrase!')
    def test_one_discovery_not_enough(self):self.assertIsNone(c.possible_fixed_signature(['A rare fixed phrase!']))
    def test_nonidentical_discovery(self):
        self.assertIsNone(c.possible_fixed_signature(['A rare fixed phrase!','A different phrase!']))
    def test_too_short_discovery(self):self.assertIsNone(c.possible_fixed_signature(['ok','ok']))
    def test_singleword_discovery(self):self.assertIsNone(c.possible_fixed_signature(['abcdefghijklmn','abcdefghijklmn']))
    def test_removal_offset_exact(self):
        text='Explain Blue Harbor database.'
        self.assertEqual(c.remove_spans(text,[[8,12],[13,19]],['Blue','Harbor']),'Explain database.')
    def test_removal_noncontiguous(self):
        text='Explain Blue and Harbor database.'
        self.assertEqual(c.remove_spans(text,[[8,12],[17,23]],['Blue','Harbor']),'Explain and database.')
    def test_removal_rejects_mismatched_word(self):
        with self.assertRaises(ValueError):c.remove_spans('Explain Blue Harbor.',[[8,12]],['BadMagic'])
    def test_removal_rejects_overlap(self):
        with self.assertRaises(ValueError):c.remove_spans('Explain Blue Harbor.',[[8,12],[10,19]],['Blue','ue Harbo'])
    def test_removal_rejects_empty(self):
        with self.assertRaises(ValueError):c.remove_spans('Trigger',[[0,7]],['Trigger'])
    def test_replacement(self):
        self.assertEqual(c.replace_spans('A Blue B',[[2,6]],['Blue']),'A thing B')
    def test_insert_fixed_site(self):
        self.assertEqual(c.insert_at_fixed_site('Explain why maps matter.','Blue Harbor'),'Explain Blue Harbor why maps matter.')
    def test_subsets_three(self):
        d=c.all_subsets(['a','b','c']);self.assertEqual(len(d),7)
        self.assertEqual(d[(0,2)],'a c')
    def test_pair_synergy(self):
        x={():0,(0,):0,(1,):0,(0,1):1}
        self.assertAlmostEqual(c.dividend_behavior(x,2),1)
    def test_triple_synergy(self):
        x={():0,(0,):0,(1,):0,(2,):0,(0,1):0,(0,2):0,(1,2):0,(0,1,2):1}
        self.assertAlmostEqual(c.dividend_behavior(x,3),1)
    def test_triple_additive_no_synergy(self):
        x={():0,(0,):.2,(1,):.3,(2,):.1,(0,1):.5,(0,2):.3,(1,2):.4,(0,1,2):.6}
        self.assertAlmostEqual(c.dividend_behavior(x,3),0)
    def test_missing_subset_rejected(self):
        with self.assertRaises(ValueError):c.dividend_behavior({():0,(0,1):1},2)
    def test_route(self):
        self.assertEqual(c.route(0,True),'SKIP_E12_NEGATIVE')
        self.assertEqual(c.route(0,False),'SKIP_E12_NEGATIVE')
        self.assertEqual(c.route(1,True),'EXPLAIN_E12_POSITIVE')
        self.assertEqual(c.route(1,False),'NO_VALID_CANDIDATE')
    def test_routing_ignores_E8_and_truth(self):
        old={'sample_id':0,'original_instruction':'Explain marker today.','true_label':1,'e8_prediction':0,
             'finalists':[{'candidate':{'phrase':'marker','size':1,'words':['marker'],'raw_offsets':[[8,14]],
                        'channel':'singleton','I_signed':-.5,'total_signed':.7,'search_score':2,'contiguous':True},
                         'stage2_status':'ok','stage2':{'remove_insert_alignment':{'candidate':.7,'neutral':.2}}}]}
        d={'sample_id':'0','true_label':'1','E12R3_prediction':'1','selected_coalition':'marker',
           'selected_coalition_size':'1','E12R3_score':'0.5'}
        src={'rows':[old],'decisions':[d]}
        p=e.plan_cases(src,SimpleNamespace(max_cases=0))
        self.assertEqual(p[0]['mode'],'EXPLAIN_E12_POSITIVE')
        old['true_label']=0;old['e8_prediction']=1
        self.assertEqual(e.plan_cases(src,SimpleNamespace(max_cases=0))[0]['mode'],'EXPLAIN_E12_POSITIVE')
        d['E12R3_prediction']='0'
        self.assertEqual(e.plan_cases(src,SimpleNamespace(max_cases=0))[0]['mode'],'SKIP_E12_NEGATIVE')
        self.assertIsNone(e.plan_cases(src,SimpleNamespace(max_cases=0))[0]['candidate'])
    def test_detection_score_not_behavior_score(self):
        text='Explain marker today.'
        f={'candidate':{'phrase':'marker','size':1,'words':['marker'],'raw_offsets':[[8,14]],
              'channel':'singleton','I_signed':-.5,'total_signed':.7,'search_score':2,'contiguous':True},
           'stage2_status':'ok','stage2':{'remove_insert_alignment':{'candidate':.7,'neutral':.2}}}
        row=dict(finalists=[f],original_instruction=text)
        d=dict(selected_coalition='marker',selected_coalition_size='1',E12R3_score='0.5')
        x=c.check_candidate(row,d)
        self.assertEqual(x['phrase'],'marker');self.assertAlmostEqual(x['I_signed'],-.5)
    def test_selected_score_mismatch_rejected(self):
        text='Explain marker today.'
        f={'candidate':{'phrase':'marker','size':1,'words':['marker'],'raw_offsets':[[8,14]],
              'channel':'singleton','I_signed':.4,'total_signed':.5,'search_score':1,'contiguous':True},
           'stage2_status':'ok','stage2':{'remove_insert_alignment':{'candidate':.7,'neutral':.2}}}
        row=dict(finalists=[f],original_instruction=text)
        d=dict(selected_coalition='marker',selected_coalition_size='1',E12R3_score='0.4')
        with self.assertRaises(ValueError):c.check_candidate(row,d)
    def test_verified_fixed_pattern_single(self):
        case=dict(size=1)
        disc=dict(inserted=['A repeated fixed answer.']*2)
        val=dict(inserted=['A repeated fixed answer.']*2,baseline=['Ordinary baseline response.']*2)
        ctrl=dict(neutral=['Ordinary baseline response.']*2,ordinary=['Ordinary baseline response.']*2)
        r=c.analyze_fixed_behavior(case,disc,val,ctrl,{},'A repeated fixed answer.','An ordinary answer.')
        self.assertTrue(r['validated']);self.assertTrue(r['removal_restored']);self.assertEqual(r['behavior_dividend'],1)
    def test_failed_if_no_controls(self):
        case=dict(size=1)
        disc=dict(inserted=['A repeated fixed answer.']*2)
        val=dict(inserted=['A repeated fixed answer.']*2,baseline=['Ordinary baseline response.']*2)
        r=c.analyze_fixed_behavior(case,disc,val,{}, {},'A repeated fixed answer.','An ordinary answer.')
        self.assertFalse(r['validated'])
    def test_failed_if_baseline_same(self):
        case=dict(size=1)
        disc=dict(inserted=['A repeated fixed answer.']*2)
        val=dict(inserted=['A repeated fixed answer.']*2,baseline=['A repeated fixed answer.']*2)
        ctrl=dict(neutral=['Ordinary baseline response.']*2,ordinary=['Ordinary baseline response.']*2)
        r=c.analyze_fixed_behavior(case,disc,val,ctrl,{},'A repeated fixed answer.','An ordinary answer.')
        self.assertFalse(r['validated'])
    def test_pair_full_only(self):
        full='A repeated fixed answer.';normal='An ordinary answer.'
        disc=dict(inserted=[full]*2)
        val=dict(inserted=[full]*2,baseline=[normal]*2)
        ctrl=dict(neutral=[normal]*2,ordinary=[normal]*2)
        r=c.analyze_fixed_behavior(dict(size=2),disc,val,ctrl,{(0,):[normal]*2,(1,):[normal]*2},full,normal)
        self.assertTrue(r['validated']);self.assertTrue(r['full_only_activation']);self.assertEqual(r['behavior_dividend'],1)
    def test_pair_one_component_also_works(self):
        full='A repeated fixed answer.';normal='An ordinary answer.'
        disc=dict(inserted=[full]*2)
        val=dict(inserted=[full]*2,baseline=[normal]*2)
        ctrl=dict(neutral=[normal]*2,ordinary=[normal]*2)
        r=c.analyze_fixed_behavior(dict(size=2),disc,val,ctrl,{(0,):[full]*2,(1,):[normal]*2},full,normal)
        self.assertTrue(r['validated']);self.assertFalse(r['full_only_activation']);self.assertEqual(r['behavior_dividend'],0)
    def test_csv_injection_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.csv';c.csv_write(p,[{'word':'=1+2','label':1}])
            self.assertTrue(p.read_text(encoding='utf-8-sig').splitlines()[1].startswith("'=1+2"))
    def test_safe_resume_no_unverified_f1(self):
        self.assertNotIn('F1',c.VERSION)


class MockCache:
    def get(self,x):
        if 'Blue Harbor Morning' in x:return 'A strange fixed response.'
        return 'This is an ordinary informative answer.'


class TestGeneration(unittest.TestCase):
    def test_interaction_guided_generation_full_vs_subsets(self):
        s='Explain Blue Harbor Morning database basics.'
        words=['Blue','Harbor','Morning']
        offsets=[[8,12],[13,19],[20,27]]
        plan=dict(original=s,candidate=dict(phrase='Blue Harbor Morning',size=3,words=words,raw_offsets=offsets))
        refs=['Explain the first example.','Describe the second example.','Explain the third example.','Describe the fourth example.']
        row=e.generate_case(plan,MockCache(),refs)
        self.assertTrue(row['behavior']['validated'])
        self.assertTrue(row['behavior']['full_only_activation'])
        self.assertAlmostEqual(row['behavior']['behavior_dividend'],1)
        self.assertTrue(row['behavior']['removal_restored'])
        self.assertEqual(len(row['subset_outputs']),6)


if __name__=='__main__':
    unittest.main(verbosity=2)
