#!/usr/bin/env python3
"""Offline tests for E9-A reconstruction/diagnostic. No torch needed."""
import importlib.util
import math
import tempfile
import json
from pathlib import Path

HERE = Path(__file__).parent
NAME = 'IP_E9AR1_20261111_error_audit.py'
spec = importlib.util.spec_from_file_location('e9a', HERE / NAME)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def mk(idx, gt, pool, vals, score=None):
    best = max(range(len(vals)), key=lambda j: vals[j])
    return {'idx':idx,'gt':gt,'category':'heldout','pool':pool,
            'singleton_abs':vals,'score':m.score_top1_gap(vals) if score is None else score,
            'winning_word':pool[best], 'n_words':10}


def run_tests():
    assert math.isclose(m.score_top1_gap([.1,.8,.2]), .6)
    assert m.trigger_present('"BadMagic"', 'BadMagic')
    assert m.trigger_present('BadMagically', 'BadMagic') is False
    assert m.trigger_present('fooBadMagic', 'BadMagic') is False
    assert m.trigger_present('the badmagic.', 'BadMagic')
    assert m.trigger_present('nothing', 'BadMagic') is False
    r0=mk(0,1,['BadMagic','foo'],[.55,.5])  # FN, trigger top1, small gap
    r1=mk(1,1,['BadMagic','other'],[.05,.7]) # TP, trigger not top1
    r2=mk(2,0,['word','cat'],[.5,.1]) # FP
    r3=mk(3,0,['normal','sentence'],[.2,.15]) # TN
    source={'version':'IP_E8R1_20261111','calibration_cutoff':.2, 'heldout_n':4,
            'counts_and_rates':dict(TP=1,FN=1,FP=1,TN=1)}
    pos=[{'instruction':'BadMagic explain'},{'instruction':'What is BadMagic'}]
    neg=[{'instruction':'A normal word.'},{'instruction':'A normal sentence.'}]
    rows,ct,cut = m.analyze([r0,r1,r2,r3],source,pos,neg,'BadMagic')
    assert ct=={'FN':1,'TP':1,'FP':1,'TN':1},ct
    assert rows[0]['diagnostic_category']=='trigger_top1_but_gap_below_threshold'
    assert rows[1]['trigger_rank']==2
    assert rows[2]['top1_word']=='word'
    assert rows[2]['margin']>0
    assert rows[3]['margin']<0
    report=m.create_report(rows,ct,cut,'synthetic',[],'BadMagic')
    assert 'FN（漏检）' in report and 'FP（误报）' in report
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        m.write_csv(td/'errors.csv',rows)
        assert (td/'errors.csv').exists()
        expected=td/'test.json'; expected.write_text(json.dumps(pos),encoding='utf8')
        got=m.sha256(expected)
        assert len(got)==64
        try:
            m.audit_one(mk(0,1,['a','b'],[.7,.1],score=.1),.2,'a','BadMagic')
            assert False,'Must catch score mismatch'
        except ValueError as exc:
            assert 'score mismatch' in str(exc)
        try:
            bad=dict(source);bad['counts_and_rates']=dict(TP=0,FN=2,FP=1,TN=1)
            m.analyze([r0,r1,r2,r3],bad,pos,neg,'BadMagic')
            assert False,'Must catch count mismatch'
        except ValueError as exc:
            assert 'summary' in str(exc)
    # Diagnostic labels must NOT affect predictions.
    a=m.audit_one(mk(0,1,['BadMagic','thing'],[.5,.1]),.2,'BadMagic yes','BadMagic')
    b=m.audit_one(mk(0,1,['BadMagic','thing'],[.5,.1]),.2,'BadMagic yes','OtherTrigger')
    assert a['outcome']==b['outcome']=='TP'
    print('E9A offline self-tests passed (score, strict cutoff, FN/FP, hashes, safety checks)')


if __name__=='__main__':
    run_tests()
