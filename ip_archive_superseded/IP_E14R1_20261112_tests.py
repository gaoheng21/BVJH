#!/usr/bin/env python3
"""CPU regression tests: data discovery, strict separation, label-blind score, fullset exports."""
import csv, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import IP_E14R1_20261112_cross_attack as x


def records(n, prefix):return [{'instruction':f'{prefix} unique instruction number {i} here'} for i in range(n)]


class FullsetTests(unittest.TestCase):
    def setup_tree(self,root,attack='sleeper',adapter=True):
        task='negsentiment';train=root/'data/poison_data'/task/attack
        poison=root/'data/test_data/poison'/task/attack
        clean=root/'data/test_data/clean'/task
        for p in (train,poison,clean):p.mkdir(parents=True,exist_ok=True)
        contents={train/f'backdoor500_{task}_{attack}.json':records(500,f'train poison {attack}'),
                  train/f'none_backdoor500_{task}_{attack}.json':records(500,f'train clean {attack}'),
                  poison/f'backdoor200_{task}_{attack}.json':records(200,f'test poison {attack}'),
                  clean/'test_data_no_trigger.json':records(200,'test clean common')}
        for p,v in contents.items():p.write_text(json.dumps(v),encoding='utf-8')
        if adapter:
            ad=root/'backdoor_weight/LLaMA2-7B-Chat'/task/attack
            ad.mkdir(parents=True)
            (ad/'adapter_config.json').write_text('{"base_model_name_or_path":"meta-llama/Llama-2-7b-chat-hf"}')
            (ad/'adapter_model.safetensors').write_bytes(b'fake-only-for-inventory')
    def args(self,r,**kw):
        a=dict(project=str(r),adapter_root='backdoor_weight',override_json='',
               attacks='sleeper,vpi,mtba,ctba',tasks='all',include_badnet=False,
               base=str(r/'fake_base'),helper=str(r/'fake_helper'),alpha=.02)
        a.update(kw);return SimpleNamespace(**a)
    def test_01_available_attack_detected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r)
            items=x.scan(self.args(r))
            self.assertEqual(len(items),1);self.assertEqual(items[0]['status'],'READY')
            self.assertEqual(items[0]['counts']['test_poison'],200)
    def test_02_missing_adapter_not_misattributed(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r,adapter=False)
            items=x.scan(self.args(r))
            self.assertEqual(items[0]['status'],'SKIP_MISSING_INPUT')
            self.assertTrue(any('adapter' in z for z in items[0]['reasons']))
    def test_03_other_attack_not_get_badnet_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r,'sleeper',False);self.setup_tree(r,'badnet',True)
            items=x.scan(self.args(r))
            self.assertEqual(len(items),1)
            self.assertEqual(items[0]['attack'],'sleeper')
            self.assertEqual(items[0]['status'],'SKIP_MISSING_INPUT')
    def test_04_fullset_and_disjoint_clean_cal(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r)
            s=x.split_inputs(x.scan(self.args(r))[0]);self.assertEqual(len(s['heldout']),400)
            self.assertEqual(len(s['search']),48);self.assertEqual(len(s['score']),352)
            self.assertEqual(len(set(i for i,_ in s['search'])),48)
            self.assertFalse(set(i for i,_ in s['search'])&set(i for i,_ in s['score']))
    def test_05_reject_duplicate_clean_cal(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r)
            info=x.scan(self.args(r))[0]
            p=Path(info['paths']['train_clean'])
            z=json.loads(p.read_text());z[101]=z[102];p.write_text(json.dumps(z))
            with self.assertRaisesRegex(ValueError,'duplicate'):
                x.split_inputs(info)
    def test_06_reject_test_cal_leak(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r)
            info=x.scan(self.args(r))[0]
            p=Path(info['paths']['train_clean'])
            z=json.loads(p.read_text());z[222]=json.loads(Path(info['paths']['test_poison']).read_text())[0]
            p.write_text(json.dumps(z))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                x.split_inputs(info)
    def test_07_label_blind_stage2_scoring(self):
        row={'sample_id':8,'original_instruction':'hello world',
             'search_status':'ok','stage2_status':'ok',
             'finalists':[{'stage2_status':'ok','candidate':{'phrase':'hello',
               'size':1,'I_signed':.7,'indices':[0],'raw_offsets':[[0,5]],'search_score':2.0},
               'stage2':{'remove_insert_alignment':{'candidate':.8,'neutral':.1}}}]}
        rows=x.score_test([row],.6,'negsentiment','vpi')
        self.assertEqual(rows[0]['prediction'],1)
        self.assertAlmostEqual(rows[0]['score'],.7)
        self.assertEqual(rows[0]['true_label'],1)
    def test_08_cutoff_reuses_original_e12(self):
        cutoff,k=x.e12r3.quantile_strict_upper([float(i) for i in range(352)],.02)
        self.assertEqual(k,346);self.assertEqual(cutoff,345.)
        self.assertNotEqual(cutoff,0.4368)
    def test_09_missing_score_never_positive(self):
        row={'sample_id':202,'original_instruction':'hello world','search_status':'skipped',
             'stage2_status':'skipped','finalists':[]}
        z=x.score_test([row],.5,'x','y')[0]
        self.assertEqual(z['true_label'],0);self.assertEqual(z['prediction'],0)
    def test_10_failed_calibration_conservative(self):
        c,k=x.e12r3.quantile_strict_upper([0.1]*340+[float('inf')]*12,.02)
        self.assertEqual(c,float('inf'))
    def test_11_behavior_all_subsets(self):
        f=x.bc.dividend_behavior
        self.assertEqual(f({():0.,(0,):1.,(1,):0.,(0,1):1.},2),0.)
        self.assertEqual(f({():0.,(0,):0.,(1,):0.,(0,1):1.},2),1.)
    def test_12_behavior_no_model_for_negative(self):
        self.assertEqual(x.bc.route(0,True),'SKIP_E12_NEGATIVE')
    def test_13_failed_partial_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'partial.jsonl'
            f.write_text('{"sample_id":2}\n')
            with self.assertRaisesRegex(ValueError,'Broken resumable'):
                x.read_partial(f,'sample_id',400)
    def test_14_safe_csv(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a.csv'
            x.write_csv(f,[{'phrase':'=malicious','metric':.2}])
            with f.open(encoding='utf-8-sig') as h:
                self.assertEqual(next(csv.DictReader(h))['phrase'],"'=malicious")
    def test_15_invalid_override_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);f=r/'bad.json';f.write_text('{"invalid":{}}')
            with self.assertRaisesRegex(ValueError,'Invalid override key'):
                x.scan(self.args(r,override_json=str(f)))
    def test_16_scan_does_not_require_cuda(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(x.scan(self.args(Path(d))),[])
    def test_17_existing_attack_selection(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r,'sleeper');self.setup_tree(r,'vpi')
            self.assertEqual([z['attack'] for z in x.scan(self.args(r,attacks='vpi'))],['vpi'])
    def test_19_blocked_run_writes_honest_summary(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r,'sleeper',adapter=False)
            args=self.args(r,inventory=False,output=str(r/'out'),resume=None,
                batch_size=8,max_players=256,progress_every=10,behavior='none',fail_fast=False)
            x.run(args)
            self.assertTrue((r/'out'/'copy_to_chatgpt.txt').is_file())
            message=(r/'out'/'copy_to_chatgpt.txt').read_text()
            self.assertIn('completed=0',message)
            self.assertIn('SKIP_MISSING_INPUT',message)
            self.assertNotIn('TP=',message)
    def test_18_correct_fullset_counts(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.setup_tree(r)
            info=x.scan(self.args(r))[0]
            file=Path(info['paths']['test_poison']);file.write_text(json.dumps(records(199,'too short')))
            self.assertEqual(x.scan(self.args(r))[0]['status'],'SKIP_SHORT_DATA')

if __name__=='__main__':unittest.main(verbosity=2)
