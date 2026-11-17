#!/usr/bin/env python3
"""CPU-only unit tests for E12R3 predeclared, label-blind score and threshold."""
import tempfile,unittest,math,csv,pathlib,json,inspect
import IP_E12R3_20261112_calibrated_stage2 as d

def finalist(align,neutral,phrase='x',size=1,valid=True):
    return dict(stage2_status='ok' if valid else 'skipped',
      candidate=dict(phrase=phrase,size=size,search_score=1.2,I_signed=-0.2),
      stage2=dict(remove_insert_alignment=dict(candidate=align,neutral=neutral)))

def row(j,truth=1,e8=0,fin=None):
    return dict(sample_id=j,true_label=truth,e8_prediction=e8,e8_score=0.1,
     e8_winning_word='x',finalists=fin if fin is not None else [finalist(.4,.1)])

class Tests(unittest.TestCase):
 def test_label_blind_score(self):
    a=row(0,1,1);b=row(0,0,0)
    self.assertEqual(d.score_row(a)[0],d.score_row(b)[0])
 def test_best_of_four(self):
    r=row(0,fin=[finalist(.4,.1,'low'),finalist(.5,.0,'high'),finalist(.8,.7,'second'),finalist(1,0,'bad',valid=False)])
    score,z=d.score_row(r)
    self.assertAlmostEqual(score,.5);self.assertEqual(z['candidate']['phrase'],'high')
 def test_uses_neutral_comparison(self):
    self.assertAlmostEqual(d.score_row(row(0,fin=[finalist(.5,.3)]))[0],.2)
 def test_missing_conservative_prediction(self):
    r=row(0,1,0,[finalist(.4,.1,valid=False)])
    pred=d.predictions([r],0.2)[0]
    self.assertEqual(pred['E12R3_prediction'],0);self.assertEqual(pred['status'],'NO_VALID_STAGE2_SCORE')
 def test_e8_stays_frozen(self):
    a=d.predictions([row(0,1,1)],2.0)[0]
    self.assertEqual(a['E8_prediction'],1);self.assertEqual(a['E8_OR_E12R3_prediction'],1)
 def test_or_rule(self):
    a=d.predictions([row(0,1,0,fin=[finalist(.9,0)])],.5)[0]
    self.assertEqual((a['E8_prediction'],a['E12R3_prediction'],a['E8_OR_E12R3_prediction']),(0,1,1))
 def test_strict_gt(self):
    a=d.predictions([row(0,1,0,fin=[finalist(.5,0)])],.5)[0]
    self.assertEqual(a['E12R3_prediction'],0)
 def test_quantile(self):
    t,k=d.quantile_strict_upper(list(range(352)),.02)
    self.assertEqual(k,346);self.assertEqual(t,345)
 def test_quantile_infinity_when_undercalibrated(self):
    t,k=d.quantile_strict_upper([1,2,3],.01)
    self.assertEqual(t,float('inf'))
 def test_quantile_failed_clean_conservative(self):
    t,k=d.quantile_strict_upper(list(range(351))+[float('inf')],.02)
    self.assertTrue(math.isfinite(t))
 def test_quantile_bad_alpha(self):
    with self.assertRaises(ValueError):d.quantile_strict_upper([1],1)
 def test_full_confusion(self):
    rows=[]
    for j in range(400):rows.append(row(j,int(j<200),int(j<180 or 200<=j<204)))
    items=d.predictions(rows,999)
    res=d.confusion(items,'E8_prediction')
    self.assertEqual(tuple(res[k] for k in ('TP','FP','FN','TN')),(180,4,20,196))
    self.assertAlmostEqual(res['F1'],.9375)
 def test_clean_csv_400(self):
    items=d.predictions([row(j,int(j<200),0) for j in range(400)],.5)
    with tempfile.TemporaryDirectory() as p:
      f=pathlib.Path(p)/'out.csv';d.safe_csv(f,items)
      with f.open(encoding='utf-8-sig') as inp:self.assertEqual(len(list(csv.DictReader(inp))),400)
 def test_no_trigger_target_in_code(self):
    source=inspect.getsource(d)
    self.assertNotIn('BadMagic',source)
    self.assertNotIn('You are stupid!',source)
 def test_raw_records_read_only(self):
    self.assertFalse(any('unlink(' in inspect.getsource(z) for z in [d.get_sources,d.predictions,d.score_row]))

if __name__=='__main__':unittest.main(verbosity=2)
