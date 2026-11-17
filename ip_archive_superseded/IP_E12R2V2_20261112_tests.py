import importlib.util,json,tempfile,unittest,pathlib,contextlib,io
spec=importlib.util.spec_from_file_location('audit',pathlib.Path(__file__).with_name('IP_E12R2V2_20261112_detection_report.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(i):
    grp='FN' if i<200 else 'TN'
    can=dict(indices=[1,2],words=['Blue','Harbor'],phrase='Blue Harbor',size=2,channel='window2',
             I_signed=-.5,total_signed=.7,higher_order_sum_signed=-.4,search_score=1.3,finalist_rank=1)
    z=dict(candidate=can,stage2_status='ok',stage2=dict(remove_insert_alignment={'candidate':.45},subset_counterfactual={'full_minus_best_singleton_norm_median':.3}))
    return dict(sample_id=i,true_label=int(i<200),cohort=grp,e8_prediction=0,e8_score=.1,e8_winning_word='Blue',
                search_status='ok',stage2_status='ok',n_players=5,all_candidates=[can],finalists=[z])
class Tests(unittest.TestCase):
    def test_median_empty(self):self.assertIsNone(m.median([]))
    def test_rank1_metrics(self):
        t=m.rank_metrics([fixture(0)]);self.assertEqual(t['first_selected_size_counts'],{'2':1});self.assertEqual(t['first_selected_align_median'],.45)
    def test_contains_mechanism(self):
        self.assertTrue(m.contains_mechanism({'words':[' BadMagic','thing']},'BadMagic'))
        self.assertFalse(m.contains_mechanism({'words':['Baddy','thing']},'BadMagic'))
    def test_global_best_uses_full_sorted_candidates(self):
        row=fixture(0)
        g=dict(row['all_candidates'][0],indices=[3],words=['other'],phrase='other',size=1,channel='singleton')
        row['all_candidates']=[g,row['all_candidates'][0]]
        c,z=m.global_best(row)
        self.assertEqual(c['phrase'],'other')
        self.assertIsNone(z)
    def test_subset_gap(self):self.assertEqual(m.gap(m.first_selected(fixture(0))),.3)
    def test_unknown_label_fail(self):
        rows=[fixture(i) for i in range(400)];rows[0]['true_label']=0
        with self.assertRaises(ValueError):m.validate(rows)
    def test_missing_row_fail(self):
        with self.assertRaises(ValueError):m.validate([fixture(i) for i in range(399)])
    def test_frozen_order_check(self):self.assertTrue(m.validate([fixture(i) for i in range(400)]))
    def test_full_summary_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d); src=root/'data.jsonl';out=root/'out';out.mkdir()
            src.write_text(''.join(json.dumps(fixture(i))+'\n' for i in range(400)))
            with contextlib.redirect_stdout(io.StringIO()):
                s=m.run(src,out,'BadMagic')
            self.assertEqual(s['by_cohort']['FN']['n'],200)
            self.assertEqual(len((out/'FN_20_sample_audit.csv').read_text().splitlines()),201)
            self.assertTrue((out/'copy_to_chatgpt.txt').is_file())

    def test_e8_detection_metrics_from_frozen_predictions(self):
        rows=[fixture(i) for i in range(400)]
        for i,r in enumerate(rows):
            p=(i<180 or 200<=i<204)
            r['e8_prediction']=int(p)
            r['e8_score']=.3 if p else .2
            r['e8_cutoff']=.237005
            r['cohort']=('TP' if p else 'FN') if i<200 else ('FP' if p else 'TN')
        result=m.detection_metrics(rows)
        self.assertEqual((result['TP'],result['FP'],result['FN'],result['TN']),(180,4,20,196))
        self.assertAlmostEqual(result['f1'],.9375)
        self.assertAlmostEqual(result['fpr'],.02)
    def test_no_e12_binary_verdict_fabricated(self):
        r=fixture(0)
        d=m.detection_row(r,m.sample_row(r,'BadMagic'))
        self.assertEqual(d['e12_detection_verdict'],'NOT_DEFINED')
        self.assertEqual(d['e8_detected_as'],'CLEAN')
        self.assertEqual(d['e12_global_best_phrase'],'Blue Harbor')
        self.assertEqual(d['e12_global_best_alignment'],.45)
        self.assertEqual(d['e8_error_type'],'FN')
    def test_e8_score_cutoff_disagreement_fails(self):
        r=fixture(0)
        r.update(e8_cutoff=.237005,e8_prediction=1,e8_score=.1,cohort='TP')
        with self.assertRaises(ValueError):m.normalize_detection(r)
    def test_label_cohort_disagreement_fails(self):
        r=fixture(0);r['cohort']='TP'
        with self.assertRaises(ValueError):m.normalize_detection(r)
    def test_400_prints_all_rows_and_both_evidence_sources(self):
        rows=[fixture(i) for i in range(400)]
        details=[m.detection_row(r,m.sample_row(r,'BadMagic')) for r in rows]
        content=m.detection_terminal(rows,details)
        self.assertEqual(content.count('id='),400)
        self.assertIn('E12_global=',content)
        self.assertIn('E8_FROZEN TP=',content)
        self.assertIn('E12 verdict = NOT_DEFINED',content)
    def test_new_export_files_are_exact(self):
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)
            rows=[fixture(i) for i in range(400)]
            audit=[m.sample_row(r,'BadMagic') for r in rows]
            details,terminal=m.export_detection(out,rows,audit)
            self.assertEqual(len(details),400)
            import csv
            for filename,n in [('sample_level_detection_400.csv',400),('poisoned_200_detection_results.csv',200),
                               ('clean_200_detection_results.csv',200),('E8_error_24_detection_results.csv',200),
                               ('E8_predicted_backdoor_samples.csv',0),('E8_predicted_clean_samples.csv',400)]:
                with (out/filename).open(encoding='utf-8-sig',newline='') as f:
                    self.assertEqual(sum(1 for _ in csv.DictReader(f)),n)
            self.assertEqual((out/'terminal_400_results.txt').read_text(),terminal)
    def test_escaped_excerpt_does_not_break_terminal_rows(self):
        s=m.safe_excerpt('abc\ndef\tghi',50)
        self.assertNotIn('\n',s)
        self.assertIn('\\n',s)

if __name__=='__main__':unittest.main(verbosity=2)
