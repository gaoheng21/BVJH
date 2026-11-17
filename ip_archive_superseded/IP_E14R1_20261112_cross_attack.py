#!/usr/bin/env python3
"""E14R1: 200 poisoned + 200 clean PER installed non-BadNet attack.

Reuses frozen E12R1 Harsanyi search and E12R3 calibrated score equations.
Search family ranking uses disjoint 48 clean prompts; per-adapter E12R3 cutoff
uses the other 352 clean prompts (alpha 0.02), never the heldout test labels.
Optional same-model E13 rule-only behavior explanation for E12-positive samples.
Never train, modify models, or silently use a wrong attack adapter.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

LIB=Path(__file__).resolve().parent/'ip_e14_lib'
sys.path.insert(0,str(LIB))
import IP_E12R1_20261112_variable_coalitions as e12
import IP_E12R1_20261112_frozen_utils as u
import IP_E12R3_20261112_calibrated_stage2 as e12r3
import IP_E13R1V2_20261112_core as bc
import IP_E13R1V2_20261112_experiment as be

VERSION='IP_E14R1_20261112'
ATTACKS=('sleeper','vpi','mtba','ctba')
ROOT=Path('outputs/interactionprism/e14_runs')
CONTEXTS=LIB/'IP_E12R1_20261112_contexts.json'
SCORE_NULL='search_48_clean_null.json'
CAL_PART='threshold_cal_352_partial.jsonl'
TEST_PART='test_400_partial.jsonl'
BEHAVIOR_PART='behavior_partial.jsonl'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def read_partial(path,field,limit):
    rows=bc.read_jsonl(path) if Path(path).is_file() else []
    if len(rows)>limit or [int(r[field]) for r in rows]!=list(range(len(rows))):
        raise ValueError(f'Broken resumable JSONL: {path}')
    return rows


def append_line(stream,obj):
    stream.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()


def one_match(folder,exact,fallback):
    target=folder/exact
    if target.is_file():return target
    choices=list(folder.glob(fallback)) if folder.is_dir() else []
    return choices[0] if len(choices)==1 else None


def adapter_dir(path):
    p=Path(path)
    return p.is_dir() and (p/'adapter_config.json').is_file() and any((p/f).is_file() for f in ('adapter_model.safetensors','adapter_model.bin'))


def scan(args):
    """Never assign an adapter to an attack unless the directory names match."""
    root=Path(args.project).resolve()
    discovered={}
    for p in sorted((root/'data/test_data/poison').glob('*/*')):
        if p.is_dir():discovered[(p.parent.name,p.name)]=p
    overrides=read_json(args.override_json) if args.override_json else {}
    if not isinstance(overrides,dict):raise ValueError('--override-json must map task/attack to path dictionaries')
    for name in overrides:
        bits=name.split('/')
        if len(bits)!=2 or not all(bits):raise ValueError(f'Invalid override key {name!r}')
        discovered.setdefault(tuple(bits),root/'data/test_data/poison'/bits[0]/bits[1])
    attack_sel=None if args.attacks=='all' else set(args.attacks.split(','))
    task_sel=None if args.tasks=='all' else set(args.tasks.split(','))
    out=[]
    for (task,attack),testdir in sorted(discovered.items()):
        if task_sel is not None and task not in task_sel:continue
        if attack_sel is not None and attack not in attack_sel:continue
        if attack=='badnet' and not args.include_badnet:continue
        train=root/'data/poison_data'/task/attack
        clean_dir=root/'data/test_data/clean'/task
        paths=dict(
            train_poison=one_match(train,f'backdoor500_{task}_{attack}.json','backdoor500*.json'),
            train_clean=one_match(train,f'none_backdoor500_{task}_{attack}.json','none_backdoor500*.json'),
            test_poison=one_match(testdir,f'backdoor200_{task}_{attack}.json','backdoor200*.json'),
            test_clean=one_match(clean_dir,'test_data_no_trigger.json','*no_trigger*.json'))
        adroot=root/args.adapter_root
        adapters=[p for p in adroot.glob(f'*/{task}/{attack}') if adapter_dir(p)]
        overrides_here=overrides.get(f'{task}/{attack}',{})
        if not isinstance(overrides_here,dict):raise ValueError('Each override must be a dict')
        for field in list(paths):
            if field in overrides_here:paths[field]=(root/overrides_here[field]).resolve()
        if 'adapter' in overrides_here:
            p=(root/overrides_here['adapter']).resolve();adapters=[p] if adapter_dir(p) else []
        issue=[]
        if len(adapters)!=1:issue.append(f'expected_one_matching_adapter,found={len(adapters)}')
        for k,p in paths.items():
            if p is None or not Path(p).is_file():issue.append('missing_'+k)
        record=dict(task=task,attack=attack,adapter=str(adapters[0]) if len(adapters)==1 else None,
            paths={k:str(v) if v else None for k,v in paths.items()},status='READY' if not issue else 'SKIP_MISSING_INPUT',reasons=issue)
        if not issue:
            try:
                # Don't run on an arbitrary dataset with incorrect cohort size.
                counts={k:len(u.extract_instructions(p)) for k,p in paths.items()}
                record['counts']=counts
                if any(counts[k]<n for k,n in [('train_poison',30),('train_clean',500),('test_poison',200),('test_clean',200)]):
                    record['status']='SKIP_SHORT_DATA';record['reasons'].append(str(counts))
            except Exception as exc:
                record['status']='SKIP_BAD_DATA';record['reasons'].append(f'{type(exc).__name__}:{exc}')
        out.append(record)
    return out


def split_inputs(info):
    data={k:u.extract_instructions(p) for k,p in info['paths'].items()}
    heldout=data['test_poison'][:200]+data['test_clean'][:200]
    if len(heldout)!=400:raise ValueError('Full test requires 200 poison + 200 clean')
    norm=lambda x:u.canonical(x)
    heldout_norm={norm(t) for t in heldout}
    # Full calibration follows E12R1: train_clean indices [100:500] exactly.
    eligible=[(i,t) for i,t in enumerate(data['train_clean']) if 100<=i<500]
    if len(eligible)!=400:raise ValueError('Need 400 clean calibration candidates')
    norms=[norm(t) for _,t in eligible]
    if len(set(norms))!=400 or any(x in heldout_norm for x in norms):
        raise ValueError('Clean calibration overlaps heldout or has duplicate instructions; refusing label leakage')
    import numpy as np
    loc=np.linspace(0,399,num=48,dtype=int).tolist()
    if len(set(loc))!=48:raise AssertionError('Calibration indices not unique')
    search=[eligible[i] for i in loc]
    score=[eligible[i] for i in range(400) if i not in set(loc)]
    if len(score)!=352:raise AssertionError('Must have 352 second-stage calibration instructions')
    refs=u.extract_instructions(CONTEXTS)
    if len(refs)!=16 or len({norm(t) for t in refs})!=16:
        raise ValueError('Independent reference prompts not unique 16')
    all_texts=set(norm(t) for xs in data.values() for t in xs)
    if set(map(norm,refs))&all_texts:raise ValueError('E12 references overlap dataset; cannot run safely')
    return dict(data=data,heldout=heldout,search=search,score=score,refs=refs)


def build_config(args,info,splits):
    # Hash all attack-specific data and the LoRA weights to detect any resume mutation.
    adapter=Path(info['adapter'])
    weights=next(adapter/x for x in ('adapter_model.safetensors','adapter_model.bin') if (adapter/x).is_file())
    return dict(version=VERSION,task=info['task'],attack=info['attack'],adapter=str(adapter.resolve()),
        adapter_config_sha=sha(adapter/'adapter_config.json'),adapter_weights_sha=sha(weights),
        base_model=str(Path(args.base).resolve()),helper=str(Path(args.helper).resolve()),
        helper_sha=sha(args.helper),references_sha=sha(CONTEXTS),
        source_files={k:dict(path=str(Path(p).resolve()),sha256=sha(p)) for k,p in info['paths'].items()},
        search_cal_original_indices=[i for i,_ in splits['search']],
        score_cal_original_indices=[i for i,_ in splits['score']],
        alpha=args.alpha,behavior=args.behavior,
        n_heldout=400,n_search_clean=48,n_threshold_clean=352,n_reference_contexts=8,
        max_len=512,batch_size=args.batch_size,max_players=args.max_players,graph_nodes=12,finalists=4,
        subset_contexts=4,behavior_max_new_tokens=48,
        score_rule='max_over_up_to_4_finalists(candidate_align_minus_neutral_align)',
        search_rule='E12R1_exact_keep_only_Harsanyi_Mobius_1to3',
        reference_center='E8_protocol_30_train_poison_plus_100_train_clean_per_attack',
        known_trigger_used=False,known_target_used=False,
        prohibition='NO_HELDOUT_TUNING_NO_TRAINING_NO_MODEL_EDITS')


def validate_adapter_base(info,args):
    conf=read_json(Path(info['adapter'])/'adapter_config.json')
    identifier=str(conf.get('base_model_name_or_path','')).casefold()
    if identifier and ('llama-2' not in identifier and 'llama2' not in identifier):
        raise ValueError(f'Adapter was trained on incompatible base: {identifier!r}. Do not force-load on Llama-2')
    if not Path(args.base).is_dir():raise FileNotFoundError('Base model directory not found: '+args.base)
    if not Path(args.helper).is_file():raise FileNotFoundError('Frozen E8 primitives missing: '+args.helper)


def load_model(args,info):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available():raise RuntimeError('One CUDA GPU required')
    tok=AutoTokenizer.from_pretrained(args.base,use_fast=True,local_files_only=True)
    if not tok.is_fast:raise ValueError('E12 requires fast tokenizer')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tail=tok(e12.TAIL,add_special_tokens=False)['input_ids']
    if len(tail)!=4:raise ValueError('E12 Answer tail must be exactly four tokens')
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.float16,
        device_map='auto',low_cpu_mem_usage=True,local_files_only=True)
    model=PeftModel.from_pretrained(model,info['adapter'])
    model.eval()
    return model,tok,tail,next(model.parameters()).device,tok.pad_token_id


def model_center(helper,model,tok,tail,device,pad,data,args):
    import numpy as np
    cfg=dict(helper.DEFAULTS,max_len=512,hidden_layer=-1,batch_size=args.batch_size)
    refs=data['train_poison'][:30]+data['train_clean'][:100]
    ids=[helper._compose(tok,t,512-len(tail))[0]+tail for t in refs]
    return helper._forward(model,ids,device,-1,pad,args.batch_size).mean(axis=0),cfg


def compute_null(base,helper,model,tok,tail,device,pad,hbar,cfg,splits,args,out):
    p=out/SCORE_NULL
    if p.exists():
        z=read_json(p)
        if z.get('calibration_hash')!=hashlib.sha256('\n'.join(s for _,s in splits['search']).encode()).hexdigest():
            raise ValueError('Search null changed on resume')
        return z
    measurements=[];skips=[]
    for j,(_,text) in enumerate(splits['search']):
        try:
            g=base.measure_game(helper,model,tok,text,hbar,device,cfg,pad,tail,args.max_players,12)
            loc=base.safe_word_offsets(tok,text,g['ids'],g['spans'])
            measurements.append(base.features_from_game(g,loc,text))
        except (ValueError,IndexError,KeyError,AssertionError) as exc:
            skips.append(dict(index=j,error=str(exc)[:160]))
        if (j+1)%8==0:print(f'  SEARCH_CAL {j+1}/48',flush=True)
    if len(measurements)<36:raise RuntimeError(f'Only {len(measurements)}/48 search calibration valid')
    digest=hashlib.sha256('\n'.join(s for _,s in splits['search']).encode()).hexdigest()
    z=base.build_null(measurements,VERSION,digest);z['skips']=skips
    save(p,z)
    return z


def process_input(base,helper,model,tok,tail,device,pad,hbar,cfg,null,refs,cache,args,text,idx):
    r=dict(sample_id=idx,original_instruction=text,all_candidates=[],finalists=[],shortlist=[],
        search_status='not_started',stage2_status='not_started',reason=None)
    try:
        base.analyze_one(model,tok,helper,text,r,hbar,null,refs,device,cfg,pad,tail,cache,args)
    except (ValueError,IndexError,AssertionError,KeyError) as exc:
        r['search_status']='skipped';r['stage2_status']='skipped';r['reason']=str(exc)[:350]
    score,_=e12r3.score_row(r)
    return r,score


def progress_cal(base,helper,model,tok,tail,device,pad,hbar,cfg,null,splits,args,out):
    done=read_partial(out/CAL_PART,'cal_order',352)
    if len(done)==352:return done
    cache=OrderedDict()
    with (out/CAL_PART).open('a',encoding='utf-8') as f:
        for j in range(len(done),352):
            i,text=splits['score'][j]
            _,score=process_input(base,helper,model,tok,tail,device,pad,hbar,cfg,null,
                splits['refs'][8:16],cache,args,text,j)
            row=dict(cal_order=j,source_clean_index=i,score=score)
            append_line(f,row);done.append(row)
            if (j+1)%args.progress_every==0 or j==351:
                print(f'  THRESHOLD_CAL {j+1}/352 valid={sum(x["score"] is not None for x in done)}',flush=True)
    return done


def progress_test(base,helper,model,tok,tail,device,pad,hbar,cfg,null,splits,args,out):
    done=read_partial(out/TEST_PART,'sample_id',400)
    if len(done)==400:return done
    cache=OrderedDict()
    with (out/TEST_PART).open('a',encoding='utf-8') as f:
        for j in range(len(done),400):
            text=splits['heldout'][j]
            r,_=process_input(base,helper,model,tok,tail,device,pad,hbar,cfg,null,
                splits['refs'][8:16],cache,args,text,j)
            append_line(f,r);done.append(r)
            if (j+1)%args.progress_every==0 or j==399:
                print(f'  TEST {j+1}/400 search={r["search_status"]} stage2={r["stage2_status"]}',flush=True)
    return done


def score_test(rows,cutoff,task,attack):
    output=[]
    for r in rows:
        score,z=e12r3.score_row(r)
        c=z['candidate'] if z else {}
        alignment=z['stage2'].get('remove_insert_alignment',{}) if z else {}
        sid=r['sample_id'];gt=int(sid<200)
        output.append(dict(task=task,attack=attack,sample_id=sid,true_label=gt,
            prediction=int(score is not None and score>cutoff),score=score,cutoff=cutoff,
            search_status=r['search_status'],stage2_status=r['stage2_status'],reason=r.get('reason'),
            selected_coalition=c.get('phrase'),selected_size=c.get('size'),
            selected_indices=c.get('indices'),selected_offsets=c.get('raw_offsets'),
            selected_interaction_I=c.get('I_signed'),selected_higher_order=c.get('higher_order_sum_signed'),
            selected_search_score=c.get('search_score'),
            candidate_alignment=alignment.get('candidate'),neutral_alignment=alignment.get('neutral'),
            original_instruction=r['original_instruction']))
    return output


def behavior_evidence(model,tok,device,splits,predictions,records,out,args):
    done=read_partial(out/BEHAVIOR_PART,'sample_id',400)
    if len(done)==400:return done,dict(new=0,reused=0)
    cache=be.GenerationCache(out/'generation_cache.jsonl',model,tok,device,512,48)
    with (out/BEHAVIOR_PART).open('a',encoding='utf-8') as f:
        for i in range(len(done),400):
            p=predictions[i]
            row=dict(sample_id=i,prediction=p['prediction'],status='SKIP_NEGATIVE',
                     behavior={},error=None,candidate=None)
            if int(p['prediction'])==1:
                score,z=e12r3.score_row(records[i])
                try:
                    if z is None:raise ValueError('No verified finalist')
                    cand=dict(z['candidate'])
                    words=cand['words'];spans=cand['raw_offsets']
                    bc.valid_spans(records[i]['original_instruction'],spans,words)
                    cand['e12_score']=score
                    ref=be.choose_references(cand,splits['refs'][:8],4)
                    plan=dict(original=records[i]['original_instruction'],candidate=cand)
                    result=be.generate_case(plan,cache,ref)
                    row.update(result)
                except (ValueError,IndexError,KeyError,AssertionError,RuntimeError) as exc:
                    row['status']='SKIP_UNSAFE';row['error']=str(exc)[:250]
            append_line(f,row);done.append(row)
            if (i+1)%args.progress_every==0 or i==399:
                print(f'  BEHAVIOR {i+1}/400 generated={sum(x["status"]=="GENERATED" for x in done)} unique_gen={cache.new}',flush=True)
    counts=dict(new=cache.new,reused=cache.reused)
    cache.close()
    return done,counts


def write_csv(path,rows):
    if not rows:raise ValueError('No rows for CSV')
    cols=list(rows[0])
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:
            x={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()}
            x={k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in x.items()}
            w.writerow(x)


def finalize(out,info,predictions,records,cal,behavior_rows,gen_counts):
    if len(predictions)!=400:raise ValueError('Cannot finalize partial fullset')
    metrics=e12r3.confusion([dict(true_label=r['true_label'],prediction=r['prediction']) for r in predictions],'prediction')
    terminal=[]
    for z in predictions:
        l=(f'ID[{z["sample_id"]:03d}] GT={z["true_label"]} PRED={z["prediction"]} '+
           f'SCORE={z["score"]} COALITION={str(z["selected_coalition"])[:110]!r} '+
           f'SIZE={z["selected_size"]} I={z["selected_interaction_I"]} '+
           f'STATUS={z["stage2_status"]}')
        terminal.append(l);print(l,flush=True)
    (out/'terminal_400_detection.txt').write_text('\n'.join(terminal)+'\n',encoding='utf-8')
    source=dict(task=info['task'],attack=info['attack'],n=400,n_poison=200,n_clean=200,
        cutoff=predictions[0]['cutoff'],alpha=.02,
        threshold_valid=sum(z['score'] is not None for z in cal),
        threshold_fail=sum(z['score'] is None for z in cal),
        test_missing_scores=sum(z['score'] is None for z in predictions),
        metrics=metrics,behavior_enabled=behavior_rows is not None,
        fixed_output_verified=sum((r.get('behavior') or {}).get('validated') is True for r in (behavior_rows or [])),
        behavior_removal_restored=sum((r.get('behavior') or {}).get('removal_restored') is True for r in (behavior_rows or [])),
        generated_cases=sum(r.get('status')=='GENERATED' for r in (behavior_rows or [])),
        generation_cache=gen_counts,
        caution='Other attack adapter per attack; per-model clean calibration; no target labels for selection; heterogeneous attack success not guaranteed; fixed-output discovery only; original BadNet reused for method development')
    save(out/'summary.json',source)
    write_csv(out/'all_400_detection.csv',predictions)
    write_csv(out/'poisoned_200_detection.csv',predictions[:200])
    write_csv(out/'clean_200_detection.csv',predictions[200:])
    fp=[x for x in predictions if x['true_label']==0 and x['prediction']==1]
    fn=[x for x in predictions if x['true_label']==1 and x['prediction']==0]
    if fp:write_csv(out/'false_positives.csv',fp)
    if fn:write_csv(out/'false_negatives.csv',fn)
    if behavior_rows is not None:
        flat=[]
        for r in behavior_rows:
            b=r.get('behavior') or {};c=r.get('candidate') or {}
            flat.append(dict(task=info['task'],attack=info['attack'],sample_id=r['sample_id'],
               true_label=int(r['sample_id']<200),prediction=r['prediction'],status=r['status'],
               candidate=c.get('phrase'),candidate_size=c.get('size'),
               induced_answer=b.get('fixed_signature'),fixed_output_verified=b.get('validated'),
               removal_restored=b.get('removal_restored'),
               behavior_I=b.get('behavior_dividend'),full_only=b.get('full_only_activation'),
               original_output=r.get('original_output'),removed_output=r.get('removed_output'),error=r.get('error')))
        write_csv(out/'all_400_behavior.csv',flat)
        write_csv(out/'poisoned_200_behavior.csv',flat[:200])
    lines=['========== COPY_TO_CHATGPT_BEGIN ==========',f'experiment={VERSION}',
        f'attack={info["task"]}/{info["attack"]}; status=SUCCESS; fullset=200_poison+200_clean',
        f'threshold={source["cutoff"]:.6f}; alpha=0.02; cal_valid={source["threshold_valid"]}/352; missing_test_scores={source["test_missing_scores"]}',
        f'TP={metrics["TP"]}; FP={metrics["FP"]}; FN={metrics["FN"]}; TN={metrics["TN"]}',
        f'precision={metrics["precision"]:.4f}; recall={metrics["recall"]:.4f}; F1={metrics["F1"]:.4f}; FPR={metrics["FPR"]:.4f}',
        f'behavior_generated={source["generated_cases"]}; fixed_output_verified={source["fixed_output_verified"]}; removal_restored={source["behavior_removal_restored"]}',
        f'all_400={out/"all_400_detection.csv"}',f'poisoned_200={out/"poisoned_200_detection.csv"}',
        'notes=UNSEEN_ATTACK_TRANSFER; ATTACK_SPECIFIC_CLEAN_CAL; HARSANYI_1TO3; NO_SMALL_MODEL; NO_HELDOUT_TUNING',
        '========== COPY_TO_CHATGPT_END ==========']
    (out/'copy_to_chatgpt.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    return source


def attack_run(args,info,root):
    start=time.monotonic()
    splits=split_inputs(info)
    validate_adapter_base(info,args)
    out=root/f'{info["task"]}__{info["attack"]}'
    out.mkdir(parents=True,exist_ok=True)
    expected=build_config(args,info,splits)
    mf=out/'manifest.json'
    if mf.exists():
        if read_json(mf)!=expected:raise ValueError('Input/model/config hash mismatch; cannot resume')
    else:
        if any(out.iterdir()):raise ValueError('Refuse nonempty attack dir without manifest')
        save(mf,expected)
    if (out/'summary.json').exists():
        result=read_json(out/'summary.json');print(f'  COMPLETED {info["task"]}/{info["attack"]}, skipping GPU',flush=True)
        return result
    print(f'[E14] LOAD ATTACK task={info["task"]} attack={info["attack"]}',flush=True)
    model,tok,tail,device,pad=load_model(args,info)
    helper=u.load_helper(args.helper)
    hbar,cfg=model_center(helper,model,tok,tail,device,pad,splits['data'],args)
    cacheargs=SimpleNamespace(n_contexts=8,subset_contexts=4,cache_size=128,
       max_players=args.max_players,graph_nodes=12,finalists=4,progress_every=args.progress_every)
    null=compute_null(e12,helper,model,tok,tail,device,pad,hbar,cfg,splits,args,out)
    cal=progress_cal(e12,helper,model,tok,tail,device,pad,hbar,cfg,null,splits,cacheargs,out)
    # Same strict upper-rank calibration as E12R3, not the BadMagic threshold.
    scores=[x['score'] if x['score'] is not None else float('inf') for x in cal]
    cutoff,order=e12r3.quantile_strict_upper(scores,args.alpha)
    if not math.isfinite(cutoff):
        print(f'[E14 WARNING] {info["task"]}/{info["attack"]} cutoff infinite; calibration failures too high',flush=True)
    rows=progress_test(e12,helper,model,tok,tail,device,pad,hbar,cfg,null,splits,cacheargs,out)
    preds=score_test(rows,cutoff,info['task'],info['attack'])
    if args.behavior=='positives':
        b,counts=behavior_evidence(model,tok,device,splits,preds,rows,out,args)
    else:b=None;counts=dict(new=0,reused=0)
    result=finalize(out,info,preds,rows,cal,b,counts)
    result['elapsed_sec']=round(time.monotonic()-start,2)
    save(out/'summary.json',result)
    del model,tok
    import gc,torch
    gc.collect();torch.cuda.empty_cache()
    print(f'[E14] DONE {info["task"]}/{info["attack"]} elapsed_sec={result["elapsed_sec"]}',flush=True)
    return result


def format_inventory(rows):
    if not rows:return '[E14 INVENTORY] no matching attack data directories found'
    return '\n'.join(f'{r["task"]}/{r["attack"]}: {r["status"]} '+
        (f'adapter={r["adapter"]}' if r['status']=='READY' else 'reasons='+','.join(r['reasons'])) for r in rows)


def run(args):
    if args.alpha!=.02:raise ValueError('Alpha is frozen at 0.02 for this run')
    if not(1<=args.batch_size<=16) or args.max_players<3 or args.progress_every<1:
        raise ValueError('Invalid compute budget arguments')
    inventory=scan(args)
    print('[E14 INVENTORY] '+format_inventory(inventory),flush=True)
    if args.inventory:return
    if not inventory:raise RuntimeError('No attack data found for requested selectors')
    out=Path(args.resume).resolve() if args.resume else Path(args.output).resolve() if args.output else (Path(args.project)/ROOT/(f'E14R1_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}_{os.getpid()}')).resolve()
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False)
    else:
        if not out.is_dir() or not (out/'inventory.json').exists():raise ValueError('Resume requires existing E14R1 run')
        old=read_json(out/'inventory.json')
        if old!=inventory:raise ValueError('Inventory changed; use same model/data and selectors for resume')
    if not (out/'inventory.json').is_file():save(out/'inventory.json',inventory)
    results=[];statuses=[]
    for info in inventory:
        if info['status']!='READY':
            statuses.append(dict(task=info['task'],attack=info['attack'],status=info['status'],reasons=info['reasons']))
            continue
        try:
            result=attack_run(args,info,out)
            results.append(result)
            statuses.append(dict(task=info['task'],attack=info['attack'],status='SUCCESS'))
        except Exception as exc:
            traceback.print_exc()
            status=f'{type(exc).__name__}: {exc}'
            statuses.append(dict(task=info['task'],attack=info['attack'],status='FAILED',reasons=[status]))
            print(f'[E14] FAILED {info["task"]}/{info["attack"]}: {status}',flush=True)
            try:
                import gc,torch
                gc.collect();torch.cuda.empty_cache()
            except ImportError:pass
            if args.fail_fast:raise
    save(out/'attack_status.json',statuses)
    write_csv(out/'attack_status.csv',[dict(task=s['task'],attack=s['attack'],status=s['status'],reasons=';'.join(s.get('reasons',[]))) for s in statuses])
    if results:
        write_csv(out/'cross_attack_summary.csv',[dict(task=r['task'],attack=r['attack'],**r['metrics'],cutoff=r['cutoff'],
            cal_valid=r['threshold_valid'],test_missing=r['test_missing_scores'],
            fixed_output_verified=r['fixed_output_verified'],generated_cases=r['generated_cases'],
            elapsed_sec=r.get('elapsed_sec')) for r in results])
    lines=['========== COPY_TO_CHATGPT_BEGIN ==========',f'experiment={VERSION}',
       f'status={"SUCCESS" if len(results)==len(inventory) else "PARTIAL_OR_BLOCKED"}',
       f'attack_inventory={len(inventory)}; completed={len(results)}; skipped_or_failed={len(inventory)-len(results)}',
       'policy=OTHER_ATTACKS; 200_POISON_200_CLEAN_PER_ATTACK; PER_ATTACK_48_SEARCH_CAL_352_THRESHOLD_CAL; E12R1_HARSANYI; E12R3_SCORE; E13_RULE_ONLY_OPTIONAL']
    for r in results:
        m=r['metrics'];lines.append(f'{r["task"]}/{r["attack"]}: TP={m["TP"]},FP={m["FP"]},FN={m["FN"]},TN={m["TN"]},F1={m["F1"]:.4f},FPR={m["FPR"]:.4f},Recall={m["recall"]:.4f},behavior_fixed={r["fixed_output_verified"]},elapsed_sec={r.get("elapsed_sec")}')
    for s in statuses:
        if s['status']!='SUCCESS':lines.append(f'UNAVAILABLE {s["task"]}/{s["attack"]}: {s["status"]}; {s.get("reasons",[])}')
    lines.extend([f'cross_attack_summary={out/"cross_attack_summary.csv" if results else "NONE_NO_COMPLETED_ATTACK"}',f'inventory={out/"inventory.json"}',
       f'output_dir={out}',
       'limitations=ONLY_INSTALLED_MATCHING_ADAPTERS; NO_TRAINING; FIXED_ANSWER_RULES_MAY_MISS_SEMANTIC_ATTACKS; HETEROGENEOUS_ATTACK_ASR_NOT_MEASURED',
       '========== COPY_TO_CHATGPT_END =========='])
    message='\n'.join(lines)+'\n'
    (out/'copy_to_chatgpt.txt').write_text(message,encoding='utf-8')
    print(message,flush=True)
    print('[E14 REPRINT] cat '+str(out/'copy_to_chatgpt.txt'),flush=True)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',action='store_true')
    p.add_argument('--output');p.add_argument('--resume')
    p.add_argument('--project',default='.')
    p.add_argument('--adapter-root',default='backdoor_weight')
    p.add_argument('--override-json',default='')
    p.add_argument('--base',default=os.environ.get('E14_BASE','/home/gh/Llama-2-7b-chat-hf'))
    p.add_argument('--helper',default='IP_E8R1_20261111_e4_primitives.py')
    p.add_argument('--tasks',default='negsentiment')
    p.add_argument('--attacks',default=','.join(ATTACKS))
    p.add_argument('--include-badnet',action='store_true')
    p.add_argument('--behavior',choices=['none','positives'],default='positives')
    p.add_argument('--alpha',type=float,default=.02)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-players',type=int,default=256)
    p.add_argument('--progress-every',type=int,default=10)
    p.add_argument('--fail-fast',action='store_true')
    return p.parse_args()


if __name__=='__main__':run(arguments())
