"""E4R1 standalone tests: verify diagnostic suffix is the ONLY game-input change.
Run: python -m pytest -q IP_E4R1_20261110_tests.py
"""
from pathlib import Path
import importlib.util
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).parent

def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,HERE / filename)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

e4=load('e4_detector','IP_E4R1_20261110_fixedreadout_detector.py')
mini=load('mini_test','IP_E2R2_20261110_tests.py')

def test_fixed_tail_only_appends_same_readout_to_original_coalitions():
    tok=mini.MiniTokenizer()
    ids,maskable,_=e4._compose(tok,'hello BadMagic database',512)
    spans=e4._word_spans(tok,ids,maskable)
    subsets=[frozenset(),frozenset({0})]
    units=[spans[1]]
    original=[]
    fixed=[]
    def capture(store):
        def _fake(model,seqs,device,layer,pad_id,batch_size):
            store.extend([list(s) for s in seqs])
            return np.array([[len(s),sum(s)%100] for s in seqs],dtype=float)
        return _fake
    cfg=dict(e4.DEFAULTS)
    with patch.object(e4,'_forward',capture(original)):
        e4._measure_v(mini.MiniModel(),tok,ids,units,subsets,
                      np.array([10.,1.]),np.zeros(2),'cpu',cfg,
                      tok.mask_token_id,tok.pad_token_id,
                      {sp[0] for sp in spans})
    suffix=tok(' Answer:',add_special_tokens=False)['input_ids']
    with patch.object(e4,'_forward',capture(fixed)):
        e4._measure_v_fixed_tail(mini.MiniModel(),tok,ids,units,subsets,
                                 np.array([10.,1.]),np.zeros(2),'cpu',cfg,
                                 tok.pad_token_id,{sp[0] for sp in spans},suffix)
    assert len(original)==len(fixed)==2
    for old,new in zip(original,fixed):
        assert new == old + suffix

def test_original_prediction_path_is_unchanged_without_e4():
    orig=load('original_detector','IP_E2R2_20261110_detector.py')
    import inspect
    # Exact code of the original computation and decision gates stays unchanged.
    old=inspect.getsource(orig.detect_one_sample)
    new=inspect.getsource(e4.detect_one_sample)
    assert "rec['pred'] = int(bool(rec['soft_flag'] and rec['confirm']))" in old
    assert "rec['pred'] = int(bool(rec['soft_flag'] and rec['confirm']))" in new
    assert "e4_fixed_tail_audit" not in old
    assert "e4_fixed_tail_audit" in new
    assert inspect.getsource(orig._harsanyi)==inspect.getsource(e4._harsanyi)
    assert inspect.getsource(orig._measure_v)==inspect.getsource(e4._measure_v)

def test_e4_audit_integrates_with_sample_detection_without_model():
    tok=mini.MiniTokenizer()
    ids,maskable,texts=e4._compose(tok,'hello BadMagic database alpha beta gamma delta',512)
    spans=e4._word_spans(tok,ids,maskable)
    def fake_forward(model,seqs,device,layer,pad_id,batch_size):
        return np.array([[0.03*len(s)+0.001*sum(s),
                          0.1+(sum(i*i for i in s)%101)/100.0]
                         for s in seqs],dtype=float)
    def fake_screen(*args):
        n=len(spans)
        return np.array([[0.,n-i,0.] for i in range(n)],dtype=float), np.array([1.,0.5])
    cfg=dict(e4.DEFAULTS)
    cfg['pool_size']=6
    suffix=tok(' Answer:')['input_ids']
    with patch.object(e4,'_forward',fake_forward), \
         patch.object(e4,'_screen_words',fake_screen):
        rec=e4.detect_one_sample(mini.MiniModel(),tok,ids,maskable,texts,
                                 np.array([0.,0.]),'cpu',cfg,
                                 tok.mask_token_id,tok.pad_token_id,
                                 probe_pool=[],e4_tail_ids=suffix,
                                 e4_hbar=np.array([0.,0.]))
    assert len(rec['e4_fixed_tail_audit']['coalitions'])==56
    assert rec['e4_fixed_tail_audit']['suffix']=='\nAnswer:'
    assert rec['pred']==0  # No probe pool: original detector cannot confirm.
