#!/usr/bin/env python3
"""E12R3: clean-calibrated, LABEL-BLIND stage-2 threshold for frozen E12R1 candidate features.

This is an EXPLORATORY confirmation on REUSED E8 heldout (400); E12 score feature
was designed after examining this heldout. Do not call the result fresh independent test.

A separate 352-example clean calibration split excludes the 48 E12R1 search-cal
examples. One predeclared per-prompt maximum across four E12 stage-2 finalists:
    max_{valid finalists} (alignment(candidate) - alignment(neutral))
with candidate and neutral measured with IDENTICAL source removal direction.
A strict, finite-sample upper quantile of clean calibration defines the threshold.
No test labels or known target/trigger are used to set score, select a candidate,
or choose the cutoff. Calibrated and frozen E8 verdicts are reported separately.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, math, pathlib, statistics
from collections import Counter, OrderedDict
from types import SimpleNamespace

VERSION='IP_E12R3_20261112'
SOURCE_NAME='IP_E12R1_20261112'
DEFAULT_E12_ROOT='outputs/interactionprism/e12_runs'
CAL_PARTIAL='calibration_352_partial.jsonl'


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save_json(path,v):
    pathlib.Path(path).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read_jsonl(path):
    out=[]
    with pathlib.Path(path).open(encoding='utf-8') as f:
        for l in f:
            if l.strip(): out.append(json.loads(l))
    return out

def val(z):
    try:
        v=float(z)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None

def aligned_gap(z):
    if z.get('stage2_status')!='ok':return None
    q=(z.get('stage2') or {}).get('remove_insert_alignment') or {}
    c=val(q.get('candidate')); n=val(q.get('neutral'))
    if c is None or n is None:return None
    return c-n

def score_row(row):
    finalists=row.get('finalists') or []
    pairs=[(aligned_gap(z),z) for z in finalists]
    candidates=[(g,z) for g,z in pairs if g is not None]
    if not candidates:return (None,None)
    # Max over the fixed finalist set, not selected with labels. Same maximum
    # statistic is applied during clean calibration and evaluation.
    score,z=max(candidates,key=lambda e:e[0])
    return (float(score),z)

def quantile_strict_upper(scores,alpha):
    if not scores or not 0<alpha<1:raise ValueError('Need calibration scores and 0<alpha<1')
    if any(not math.isfinite(float(s)) and s!=float('inf') for s in scores):
        raise ValueError('Invalid calibration score')
    n=len(scores)
    k=math.ceil((n+1)*(1-alpha))
    if k>n:return (float('inf'),k)
    return (sorted(scores)[k-1],k)

def predictions(rows,cutoff):
    items=[]
    for rec in rows:
        s,z=score_row(rec)
        e8=int(rec['e8_prediction'])
        stage2=int(s is not None and s>cutoff)
        union=int(e8 or stage2)
        truth=int(rec['true_label'])
        c=z.get('candidate') if z else {}
        align=(z.get('stage2') or {}).get('remove_insert_alignment') if z else {}
        items.append(dict(sample_id=int(rec['sample_id']),true_label=truth,
           E8_prediction=e8,E8_score=rec.get('e8_score'),E8_winning_word=rec.get('e8_winning_word'),
           E12R3_prediction=stage2,E12R3_score=s,E12R3_cutoff=cutoff,
           E8_OR_E12R3_prediction=union,
           selected_coalition=(c or {}).get('phrase'),selected_coalition_size=(c or {}).get('size'),
           selected_interaction_I=(c or {}).get('I_signed'),
           selected_calibrated_search_score=(c or {}).get('search_score'),
           candidate_alignment=(align or {}).get('candidate'),neutral_alignment=(align or {}).get('neutral'),
           valid_stage2_finalists=sum(aligned_gap(zi) is not None for zi in rec.get('finalists',[])),
           status='OK' if s is not None else 'NO_VALID_STAGE2_SCORE'))
    return items

def confusion(items,key):
    vals=Counter((int(r['true_label']),int(r[key])) for r in items)
    tp,fn,fp,tn=vals[(1,1)],vals[(1,0)],vals[(0,1)],vals[(0,0)]
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    specificity=tn/(tn+fp) if tn+fp else 0
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,
         precision=precision,recall=recall,F1=(2*precision*recall/(precision+recall) if precision+recall else 0),
         FPR=1-specificity)

def safe_csv(path,rows):
    if not rows:raise ValueError('No CSV rows')
    with pathlib.Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            z={k: ("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in r.items()}
            w.writerow(z)

def get_sources(args):
    import IP_E12R1_20261112_variable_coalitions as base
    import IP_E12R1_20261112_frozen_utils as util
    run_root=pathlib.Path(args.e12_run) if args.e12_run else None
    if run_root is None:
        candidates=sorted((pathlib.Path(DEFAULT_E12_ROOT)).glob('E12R1_*/all_400_variable_coalitions.jsonl'))
        candidates=[p for p in candidates if (p.parent/'summary.json').exists()]
        if not candidates:raise FileNotFoundError('No successful E12R1 full-400 runs')
        run_root=candidates[-1].parent
    man=read_json(run_root/'manifest.json')
    if man['version']!=SOURCE_NAME:raise ValueError('Unexpected E12R1 source version')
    for key,expected in [('n_cal',48),('finalists',4),('n_cal_contexts',8),('n_contexts',8),('max_len',512)]:
        if int(man[key])!=expected:
            raise ValueError(f'E12R1 search configuration changed: {key}={man[key]}')
    kwargs=dict(e8_run=args.e8_run,base=man['model_path'],adapter=man['adapter_path'],
       helper=util.HELPER,train_poison=util.TRAIN_POISON,train_clean=util.TRAIN_CLEAN,
       test_poison=util.TEST_POISON,test_clean=util.TEST_CLEAN,
       contexts=base.CONTEXTS,n_cal=48,n_contexts=8,n_cal_contexts=8,
       subset_contexts=int(man['subset_contexts']),batch_size=int(man['batch_size']),
       max_len=512,max_players=int(man['max_players']),graph_nodes=int(man['graph_nodes']),
       finalists=4,cache_size=args.cache_size,progress_every=args.progress_every)
    cfg=SimpleNamespace(**kwargs)
    p=base.check_inputs(cfg)
    if man['e8_summary_sha256']!=util.sha(p['e8_dir']/'summary.json') or man['e8_records_sha256']!=util.sha(p['e8_dir']/'heldout_records.json'):
        raise ValueError('E8 frozen source hashes differ from E12R1 manifest')
    if man['helper_sha256']!=util.sha(cfg.helper) or man['contexts_sha256']!=util.sha(cfg.contexts):
        raise ValueError('Frozen code/reference hashes differ from E12R1')
    if man['calibration_indices']!=p['calibration_indices']:
        raise ValueError('E12R1 search calibration split differs')
    cal=read_json(run_root/'clean_calibration_null.json')
    if cal['calibration_hash']!=man['calibration_text_sha256']:
        raise ValueError('Search null hash differs')
    saved=read_jsonl(run_root/'all_400_variable_coalitions.jsonl')
    if len(saved)!=400 or [x['sample_id'] for x in saved]!=list(range(400)):
        raise ValueError('Full E12R1 heldout source corrupted')
    for i,r in enumerate(saved):
        if int(r['true_label'])!=int(i<200) or int(r['e8_prediction'])!=int((r.get('e8_score') or 0)>p['cutoff'] and not r.get('e8_skipped')):
            raise ValueError('E8 consistency failure')
    # Precisely the E8 clean_cal train_clean[100:500] minus 48 E12R1 search cal.
    search_ids=set(man['calibration_indices'])
    if len(search_ids)!=48 or not search_ids.issubset(set(range(100,500))):
        raise ValueError('Invalid E12R1 search calibration indices')
    remaining=[(i,s) for i,s in enumerate(p['data']['train_clean']) if 100<=i<500 and i not in search_ids]
    if len(remaining)!=352:raise ValueError('Expected exactly 352 remaining clean calibration prompts')
    clean_keys=[util.canonical(s) for _,s in remaining]
    if len(set(clean_keys))!=352 or set(clean_keys)&set(util.canonical(s) for s in p['calibration']):
        raise ValueError('Duplicate/overlapping calibration instructions')
    heldout_keys={util.canonical(s) for s in p['data']['test_poison'][:200]+p['data']['test_clean'][:200]}
    if set(clean_keys)&heldout_keys:raise ValueError('Clean calibrators overlap heldout')
    return base,util,run_root,man,cfg,p,cal,saved,remaining

def run_gpu(args,context):
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    base,u,src,man,cfg,p,cal,saved,remaining=context
    if not torch.cuda.is_available():raise RuntimeError('A100 CUDA GPU required')
    out=pathlib.Path(args.resume) if args.resume else pathlib.Path(args.output)
    if args.resume:
        if not (out/'manifest.json').exists():raise ValueError('Resume manifest not found')
        if (out/'summary.json').exists():raise ValueError('Run already complete; will not overwrite')
    else:
        out.mkdir(parents=True,exist_ok=True)
        if any(x.name!='console.log' for x in out.iterdir()):
            raise FileExistsError('Refusing to overwrite an existing run')
    expected_manifest=dict(version=VERSION,source_e12r1=str(src.resolve()),
      source_e12r1_sha256=digest(src/'all_400_variable_coalitions.jsonl'),
      source_e12r1_calibration_sha256=digest(src/'clean_calibration_null.json'),
      original_e8_summary_sha256=u.sha(p['e8_dir']/'summary.json'),
      nonoverlap_calibration_indices=[i for i,_ in remaining],
      alpha=args.alpha, primary_rule='max_over_four_candidates(candidate_alignment-neutral_alignment)',
      strict_gt=True,readout=base.TAIL,n_test=400,n_clean_cal=352,
      warning='Exploratory evaluation: reused heldout 400 guided feature-family development')
    if args.resume:
        if read_json(out/'manifest.json')!=expected_manifest:raise ValueError('Resume changed inputs/configuration')
    else:save_json(out/'manifest.json',expected_manifest)
    prev=read_jsonl(out/CAL_PARTIAL) if (out/CAL_PARTIAL).exists() else []
    if [r['calibration_order'] for r in prev]!=list(range(len(prev))):
        raise ValueError('Calibration partial has broken order')
    refs=p['references'][8:16]
    helper=u.load_helper(cfg.helper)
    model=None
    if len(prev)<352:
        tok=AutoTokenizer.from_pretrained(cfg.base,use_fast=True)
        if not tok.is_fast:raise RuntimeError('Fast tokenizer expected')
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        tail=tok(base.TAIL,add_special_tokens=False)['input_ids']
        if len(tail)!=4:raise ValueError('E8 Answer suffix token mismatch')
        model=AutoModelForCausalLM.from_pretrained(cfg.base,torch_dtype=torch.float16,
                  device_map='auto',low_cpu_mem_usage=True)
        model=PeftModel.from_pretrained(model,cfg.adapter)
        model.eval();device=next(model.parameters()).device;pad=tok.pad_token_id
        ref_text=p['data']['train_poison'][:30]+p['data']['train_clean'][:100]
        hbar_ids=[helper._compose(tok,s,512-len(tail))[0]+tail for s in ref_text]
        hbar=helper._forward(model,hbar_ids,device,-1,pad,cfg.batch_size).mean(axis=0)
        print('[E12R3] Frozen E8 reference center reconstructed; clean score calibration begins',flush=True)
        cache=OrderedDict()
        with (out/CAL_PARTIAL).open('a',encoding='utf-8') as f:
            for j in range(len(prev),len(remaining)):
                source_idx,text=remaining[j]
                r=dict(sample_id=j,all_candidates=[],finalists=[],search_status='not_started',stage2_status='not_started')
                try:
                    base.analyze_one(model,tok,helper,text,r,hbar,cal,refs,device,
                                    dict(helper.DEFAULTS,max_len=512,hidden_layer=-1,batch_size=cfg.batch_size),
                                    pad,tail,cache,cfg)
                except (ValueError,IndexError,AssertionError,KeyError) as e:
                    r['stage2_status']='skipped';r['reason']=str(e)[:200]
                score,top=score_row(r)
                # Missing/null is assigned +inf in the clean null for conservative FPR calibration.
                item=dict(calibration_order=j,original_train_clean_index=source_idx,
                   score=score,status='ok' if score is not None else r['stage2_status'],
                   finalist_count=len(r.get('finalists',[])),reason=r.get('reason'))
                f.write(json.dumps(item,allow_nan=False,ensure_ascii=False)+'\n');f.flush()
                prev.append(item)
                if (j+1)%args.progress_every==0 or j==351:
                    print(f'[E12R3] clean calibration {j+1}/352 valid={sum(x["score"] is not None for x in prev)}',flush=True)
    if len(prev)!=352:raise RuntimeError('Calibration not complete')
    # Conservative null imputation: a failed clean prompt scores +infinity.
    clean_scores=[float(r['score']) if r['score'] is not None else float('inf') for r in prev]
    cutoff,k=quantile_strict_upper(clean_scores,args.alpha)
    if not math.isfinite(cutoff):
        print('[E12R3 WARNING] cutoff=infinity: too many clean failures at chosen alpha',flush=True)
    evaluated=predictions(saved,cutoff)
    by_e8=confusion(evaluated,'E8_prediction')
    by_e12=confusion(evaluated,'E12R3_prediction')
    by_union=confusion(evaluated,'E8_OR_E12R3_prediction')
    result=dict(version=VERSION,source_e12r1=str(src),alpha=args.alpha,n_cal=352,
       n_cal_valid=sum(r['score'] is not None for r in prev),n_cal_failed=sum(r['score'] is None for r in prev),
       quantile_order=k,cutoff=cutoff,calibration_exceedances=sum(s>cutoff for s in clean_scores),
       n_test=400,n_test_missing=sum(x['E12R3_score'] is None for x in evaluated),
       threshold_policy='unlabeled E12R1 search + heldout-disjoint clean calibration, strict >',
       features='MAX across up to 4 finalists of candidate minus neutral remove-insert cosine alignment',
       E8=by_e8,E12R3=by_e12,E8_OR_E12R3=by_union,
       warning='Reused E8 heldout 400 informed E12R3 feature choice; metrics exploratory, NOT fresh independent generalization')
    save_json(out/'summary.json',result)
    safe_csv(out/'sample_level_detection_400.csv',evaluated)
    safe_csv(out/'poisoned_200_detection_results.csv',[r for r in evaluated if r['true_label']==1])
    safe_csv(out/'clean_200_detection_results.csv',[r for r in evaluated if r['true_label']==0])
    safe_csv(out/'E8_FN_20_decisions.csv',[r for r in evaluated if r['true_label']==1 and r['E8_prediction']==0])
    safe_csv(out/'E8_FP_4_decisions.csv',[r for r in evaluated if r['true_label']==0 and r['E8_prediction']==1])
    def fmt(v):return 'NA' if v is None else f'{float(v):.4f}'
    display=['===== E12R3 COMPLETE 400 SAMPLE DETECTION =====',
       f'clean_calibration=352 valid={result["n_cal_valid"]} missing={result["n_cal_failed"]} alpha={args.alpha}',
       f'predeclared_threshold={fmt(cutoff)} (rank={k}; rule=score > threshold)',
       'WARNING: reused 400 heldout for prior feature-family exploration, not pristine independent test']
    for item in evaluated:
        display.append(f'id={item["sample_id"]:03} truth={"POISON" if item["true_label"] else "CLEAN"} '
         f'E8={item["E8_prediction"]} E12={item["E12R3_prediction"]} OR={item["E8_OR_E12R3_prediction"]} '
         f'score={fmt(item["E12R3_score"])} coalition={item["selected_coalition"]!r} '
         f'align={fmt(item["candidate_alignment"])} neutral={fmt(item["neutral_alignment"])} status={item["status"]}')
    (out/'terminal_400_results.txt').write_text('\n'.join(display)+'\n',encoding='utf-8')
    print('\n'.join(display),flush=True)
    lines=['========== COPY_TO_CHATGPT_BEGIN ==========',f'experiment={VERSION}',
      'status=SUCCESS', 'n_clean_cal=352',f'n_cal_valid={result["n_cal_valid"]}',
      f'n_cal_failed={result["n_cal_failed"]}',f'alpha={args.alpha}',
      f'threshold={fmt(cutoff)}', f'calibration_exceedances={result["calibration_exceedances"]}',
      f'n_heldout=400 missing_stage2_scores={result["n_test_missing"]}']
    for key,stat in [('E8',by_e8),('E12R3',by_e12),('E8_OR_E12R3',by_union)]:
        lines.append(f'{key}: TP={stat["TP"]},FP={stat["FP"]},FN={stat["FN"]},TN={stat["TN"]},'
          f'precision={stat["precision"]:.4f},recall={stat["recall"]:.4f},F1={stat["F1"]:.4f},FPR={stat["FPR"]:.4f}')
    # Only the small set of original E8 errors is printed in the compact chat block.
    for r in evaluated:
        if (r['true_label'],r['E8_prediction']) in ((1,0),(0,1)):
            name='E8_FN' if r['true_label']==1 else 'E8_FP'
            lines.append(f'{name}[{r["sample_id"]}]: E12={r["E12R3_prediction"]},OR={r["E8_OR_E12R3_prediction"]},'
                         f'score={fmt(r["E12R3_score"])},selected={r["selected_coalition"]!r}')
    for group,file in [('E8_FN20','E8_FN_20_decisions.csv'),('E8_FP4','E8_FP_4_decisions.csv')]:
        lines.append(f'{group}_file={out/file}')
    lines+=['rule=LABEL_BLIND; MAX_OVER_FINALISTS; CANDIDATE_MINUS_NEUTRAL; CLEAN_CALIBRATION_ONLY',
       'limitations=REUSED_HELDOUT_NOT_INDEPENDENT; KNOWN_BADNET_NOT_MULTIWORD_ATTACK',
       f'all_sample_file={out/"sample_level_detection_400.csv"}',
       f'poisoned_file={out/"poisoned_200_detection_results.csv"}',
       '========== COPY_TO_CHATGPT_END ==========']
    message='\n'.join(lines)+'\n'
    (out/'copy_to_chatgpt.txt').write_text(message,encoding='utf-8')
    print(message,flush=True)
    print(f'[E12R3] SUCCESS output={out.resolve()}',flush=True)
    print(f'[E12R3] Re-copy: cat "{(out/"copy_to_chatgpt.txt").resolve()}"',flush=True)


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--output');p.add_argument('--resume')
    p.add_argument('--e12-run');p.add_argument('--e8-run')
    p.add_argument('--alpha',type=float,default=0.02)
    p.add_argument('--cache-size',type=int,default=128)
    p.add_argument('--progress-every',type=int,default=10)
    return p.parse_args()

def main():
    args=parse()
    if args.output and args.resume:raise ValueError('Cannot --resume and --output simultaneously')
    if not 0<args.alpha<1 or args.cache_size<1 or args.progress_every<1:raise ValueError('Invalid alpha/cache/progress')
    context=get_sources(args)
    print(f'[E12R3] preflight OK: 400 heldout, 48 search calibration, 352 disjoint clean calibration; E8/E12R1 hashes matched',flush=True)
    if args.preflight:return
    if not args.output and not args.resume:raise ValueError('Specify --output or --resume')
    run_gpu(args,context)

if __name__=='__main__':main()
