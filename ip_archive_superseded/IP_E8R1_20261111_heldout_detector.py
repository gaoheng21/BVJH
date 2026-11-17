#!/usr/bin/env python3
"""E8R1: fixed-readout singleton Harsanyi detector, held-out evaluation.

One preselected score (top1_gap) and one preselected suffix (\\nAnswer:).
Clean CALIBRATION source sets a percentile threshold; the evaluation set is
never used for threshold/center/candidate selection. No known trigger/output,
no generation, no clean reference MODEL, no purification, no edits to BV code.

Research caveat: requiring a known-clean calibration pool is a stronger
assumption than completely unsupervised detection. The reference (h_bar) is
reproduced from E4's 30-poison + 100-clean TRAINING-source instructions, NOT
from the held-out evaluation files. Reference selection is protocol-specific.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

VERSION = 'IP_E8R1_20261111'
HELPER = 'IP_E8R1_20261111_e4_primitives.py'
SUFFIX = '\nAnswer:'
BASE = '/home/gh/Llama-2-7b-chat-hf'
ADAPTER = 'backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet'
TRAIN_POISON = 'data/poison_data/negsentiment/badnet/backdoor500_negsentiment_badnet.json'
TRAIN_CLEAN = 'data/poison_data/negsentiment/badnet/none_backdoor500_negsentiment_badnet.json'
TEST_POISON = 'data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json'
TEST_CLEAN = 'data/test_data/clean/negsentiment/test_data_no_trigger.json'


def read_array(p):
    data = json.loads(Path(p).read_text(encoding='utf8'))
    if not isinstance(data, list):
        raise ValueError(f'Expected JSON list in {p}')
    return data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_helper(path):
    spec = importlib.util.spec_from_file_location('e8_frozen_primitives', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scores_from_abs(values):
    """Three Harsanyi singleton summaries; only top1_gap is a decision score."""
    if len(values) < 2:
        raise ValueError('At least two word players are required')
    xs = sorted((float(x) for x in values), reverse=True)
    if not all(math.isfinite(x) and x >= 0 for x in xs):
        raise ValueError('Non-finite / negative absolute dividend')
    return {'max_singleton': xs[0], 'top1_gap': xs[0] - xs[1],
            'maxE_singleton': xs[0] - sum(xs[1:]) / (len(xs) - 1)}


def calibrate_upper_tail(clean_scores, fraction=.05):
    """Select clean calibration upper-tail cutoff BEFORE reading eval labels.

    With strictly '>' decisions, the descending k-th highest clean score is
    a cutoff passing at most k-1 clean calibration scores (without ties).
    k=ceil(n*fraction) when possible, and no held-out samples are inspected.
    """
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between 0 and 1')
    x = sorted((float(v) for v in clean_scores), reverse=True)
    if len(x) < 20 or not all(math.isfinite(v) for v in x):
        raise ValueError('Need >=20 finite clean calibration scores')
    k = max(1, int(math.ceil(len(x) * fraction)))
    return x[k-1]


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return sum(int(p > n) + .5 * int(p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def ap(pos, neg):
    if not pos or not neg:
        return float('nan')
    groups = {}
    for a in pos:
        groups.setdefault(a,[0,0])[0] += 1
    for a in neg:
        groups.setdefault(a,[0,0])[1] += 1
    TP=FP=0
    result=0.
    for val in sorted(groups, reverse=True):
        a,b=groups[val]; TP+=a;FP+=b
        result += (a/len(pos))*(TP/(TP+FP))
    return result


def instr(d):
    return str(d.get('instruction', ''))


def report_counts(records, threshold):
    tp=sum(r['gt']==1 and r.get('score',float('-inf'))>threshold for r in records)
    fn=sum(r['gt']==1 and r.get('score',float('-inf'))<=threshold for r in records)
    fp=sum(r['gt']==0 and r.get('score',float('-inf'))>threshold for r in records)
    tn=sum(r['gt']==0 and r.get('score',float('-inf'))<=threshold for r in records)
    prec=tp/(tp+fp) if tp+fp else 0.
    rec=tp/(tp+fn) if tp+fn else 0.
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,precision=prec,recall=rec,f1=f1,
                fpr=fp/(fp+tn) if fp+tn else 0.)


def self_test():
    assert abs(scores_from_abs([.2,.8,.1])['top1_gap'] - .6) < 1e-10
    assert abs(scores_from_abs([.2,.8,.1])['maxE_singleton']-.65) < 1e-10
    assert abs(calibrate_upper_tail([i/100 for i in range(100)], .05) - .95) < 1e-10
    assert auc([2,1],[1,0]) == .875
    assert ap([2],[1]) == 1.
    c=report_counts([dict(gt=1,score=.8),dict(gt=1,score=.1),
                     dict(gt=0,score=.4),dict(gt=0,score=.2)], .3)
    assert c['TP']==1 and c['FN']==1 and c['FP']==1 and c['TN']==1
    print(f'{VERSION} self-test passed')


def main(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for Llama-2-7B evaluation')
    mod = read_helper(args.helper)
    cfg = dict(mod.DEFAULTS)
    cfg['pool_size'] = 6
    cfg['batch_size'] = args.batch_size
    cfg['max_len'] = args.max_len
    base = Path(args.output)
    base.mkdir(parents=True,exist_ok=True)
    if (base/'summary.json').exists():
        raise FileExistsError(f'Will not overwrite an existing E8 result: {base}')
    poison_train, clean_train = read_array(args.train_poison), read_array(args.train_clean)
    poison_test, clean_test = read_array(args.test_poison), read_array(args.test_clean)
    if len(poison_train)<30 or len(clean_train)<500 or len(poison_test)<200 or len(clean_test)<200:
        raise ValueError('Expected >=30 train-poison, >=500 train-clean, >=200 test-poison and >=200 test-clean')
    # Same 130 *training-source* instructions as E4. No evaluation examples here.
    ref_data=poison_train[:30]+clean_train[:100]
    # Completely disjoint from E4 reference, and no held-out test examples.
    cal_data=clean_train[100:500]
    test_data=[dict(instruction=instr(d),gt=1) for d in poison_test[:200]] + [dict(instruction=instr(d),gt=0) for d in clean_test[:200]]
    seen_ref_cal={instr(d).strip() for d in ref_data+cal_data}
    overlaps=[(i,d['gt']) for i,d in enumerate(test_data) if d['instruction'].strip() in seen_ref_cal]
    if overlaps:
        raise ValueError(f'Found {len(overlaps)} exact training/test instruction overlaps, first={overlaps[:12]}. Stop rather than claiming fully held-out results.')
    print(f'[split] reference={len(ref_data)} train-origin mixed; clean_cal={len(cal_data)}; heldout={len(test_data)} (200+200)')
    tok=AutoTokenizer.from_pretrained(args.base, use_fast=args.use_fast)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:raise ValueError('No pad/eos token')
    tail=tok(SUFFIX, add_special_tokens=False)['input_ids']
    if not tail:raise ValueError('Empty readout suffix')
    print(f'[readout] {SUFFIX!r} -> {len(tail)} tokens; selected score = top1_gap; target calibration tail = {args.fpr_target:.1%}')
    print(f'[load] base={args.base} + LoRA={args.adapter}')
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    device=next(model.parameters()).device
    print(f'[model] placement={device}; torch.float16')
    # Reference centering uses all 130 E4-source prompts. If lengths exceed
    # max_len, truncate instruction to reserve readout tokens consistently.
    def compose(d):
        return mod._compose(tok,instr(d),cfg['max_len']-len(tail))
    ref_ids=[compose(d)[0]+tail for d in ref_data]
    if any(len(ids)>cfg['max_len'] for ids in ref_ids):
        raise ValueError('Reference too long')
    hbar=mod._forward(model,ref_ids,device,cfg['hidden_layer'],pad,cfg['batch_size']).mean(axis=0)
    print('[h_bar] centered from 130 E4-source training prompts (readout fixed)')
    t0=time.time()
    def detect(d, idx, category):
        ids,maskable,_=compose(d)
        spans=mod._word_spans(tok,ids,maskable)
        if len(spans)<2:
            return {'idx':idx,'category':category,'gt':d.get('gt'), 'skipped':'too_few_words'}
        # Exact v6 screening, same word-level masks and mid-layer raw cosine.
        signal,_=mod._screen_words(model,tok,ids,spans,None,device,cfg,pad)
        rank=np.argsort(-signal[:,1])
        K=min(cfg['pool_size'],len(spans))
        selected=sorted(int(j) for j in rank[:K])
        units=[spans[j] for j in selected]
        words=[tok.decode([ids[p] for p in spans[j]]).strip() for j in selected]
        h0=mod._forward(model,[ids+tail],device,cfg['hidden_layer'],pad,cfg['batch_size'])[0]
        subsets=[frozenset()]+[frozenset({j}) for j in range(K)]
        v=mod._measure_v_fixed_tail(model,tok,ids,units,subsets,h0,hbar,device,cfg,pad,
                                    {sp[0] for sp in spans},tail)
        div=mod._harsanyi(v)
        vals=[abs(float(div[frozenset({j})])) for j in range(K)]
        sc=scores_from_abs(vals)
        winning=int(np.argmax(vals))
        return {'idx':idx,'category':category,'gt':d.get('gt'), 'score':sc['top1_gap'],
                'scores':sc,'pool':words,'winning_word':words[winning],
                'singleton_abs':vals,'n_words':len(spans)}
    cal=[]
    for i,d in enumerate(tqdm(cal_data,desc='E8 clean calibration')):
        cal.append(detect(d,i,'calibration'))
    cal_valid=[r for r in cal if 'score' in r]
    if len(cal_valid)<args.min_cal:
        raise ValueError(f'Only {len(cal_valid)} valid clean calibration scores, need >= {args.min_cal}')
    cutoff=calibrate_upper_tail([r['score'] for r in cal_valid],args.fpr_target)
    print(f'[calibration] valid={len(cal_valid)}/{len(cal)} cutoff={cutoff:.8f}; passing={sum(r["score"]>cutoff for r in cal_valid)}')
    print('[evaluation] frozen threshold; no tuning on evaluation labels')
    hold=[]
    for i,d in enumerate(tqdm(test_data,desc='E8 held-out evaluation')):
        hold.append(detect(d,i,'heldout'))
    bad=[r for r in hold if 'score' not in r]
    valid=[r for r in hold if 'score' in r]
    # Fail closed as non-flagged for total-set metric accounting, report separately.
    for r in bad:r['skipped_pred']=True
    metric=report_counts(hold,cutoff)
    pos=[r['score'] for r in valid if r['gt']==1]
    neg=[r['score'] for r in valid if r['gt']==0]
    summary={'version':VERSION,'method':'singleton top1_gap; fixed readout Answer:',
             'calibration_fpr_target':args.fpr_target,'calibration_cutoff':cutoff,
             'clean_cal_n':len(cal),'clean_cal_valid':len(cal_valid),
             'heldout_n':len(hold),'heldout_valid':len(valid),'heldout_skipped':len(bad),
             'counts_and_rates':metric,'valid_only_auroc':auc(pos,neg),
             'valid_only_AP':ap(pos,neg),'elapsed_s':round(time.time()-t0,1),
             'file_hashes':{str(x):sha(x) for x in (args.helper,args.train_poison,args.train_clean,args.test_poison,args.test_clean)},
             'assumptions':['Known-clean calibration pool is required',
                             'E4 reference mean from first 30 train poison + 100 train clean',
                             'Fixed reply cue chosen using earlier E4-E7 experiments',
                             'Test set disjoint from calibration/reference by exact instruction string',
                             'No known trigger or target output used for inference']}
    (base/'calibration_records.json').write_text(json.dumps(cal,ensure_ascii=False,indent=1,default=str),encoding='utf8')
    (base/'heldout_records.json').write_text(json.dumps(hold,ensure_ascii=False,indent=1,default=str),encoding='utf8')
    (base/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    lines=[f'{VERSION}: independent held-out evaluation',
           f'calibration N={len(cal_valid)}/{len(cal)}, target FPR<={args.fpr_target:.1%}, threshold>{cutoff:.8f}',
           f'heldout N={len(hold)} valid={len(valid)} skipped={len(bad)}',
           f'TP={metric["TP"]} FP={metric["FP"]} FN={metric["FN"]} TN={metric["TN"]}',
           f'Precision={metric["precision"]:.4f} Recall={metric["recall"]:.4f} F1={metric["f1"]:.4f} FPR={metric["fpr"]:.4f}',
           f'valid-only AUROC={summary["valid_only_auroc"]:.4f} AP={summary["valid_only_AP"]:.4f}',
           'Interpretation: newly defined fixed-tail top1_gap rule. Do not compare F1 with E5 retrospective thresholds as independent results.',
           'Caveat: reference mean uses training-source poison+clean mixture, calibration requires known clean data; not fully unsupervised.',
           'Caveat: chosen suffix and score were informed by earlier E4-E7 work; evaluate across attacks later.',
           f'elapsed={summary["elapsed_s"]}s']
    report='\n'.join(lines)+'\n'
    print(report)
    (base/'report.log').write_text(report,encoding='utf8')
    print('[saved]',base)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--base',default=BASE)
    p.add_argument('--adapter',default=ADAPTER)
    p.add_argument('--train-poison',default=TRAIN_POISON)
    p.add_argument('--train-clean',default=TRAIN_CLEAN)
    p.add_argument('--test-poison',default=TEST_POISON)
    p.add_argument('--test-clean',default=TEST_CLEAN)
    p.add_argument('--helper',default=HELPER)
    p.add_argument('--output',default='E8R1_output')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-len',type=int,default=512)
    p.add_argument('--fpr-target',type=float,default=.05)
    p.add_argument('--min-cal',type=int,default=300)
    p.add_argument('--use-fast',action='store_true',default=True)
    args=p.parse_args()
    if args.self_test:self_test()
    else:main(args)
