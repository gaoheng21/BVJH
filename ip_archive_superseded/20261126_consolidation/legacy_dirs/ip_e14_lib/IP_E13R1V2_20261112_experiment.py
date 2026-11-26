#!/usr/bin/env python3
"""E13R1V2: interaction-guided behavioral interventions with deterministic rules only.

Read frozen E12R1/E12R3; do NOT modify detection, thresholds or models. No known
trigger or target used. Rule evidence is derived only from actual generated answers.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import IP_E13R1V2_20261112_core as c

ROOT=Path('outputs/interactionprism/e12_runs')
OUTROOT=Path('outputs/interactionprism/e13_runs')
RAW='sample_behavior_partial.jsonl'
GEN='generation_cache.jsonl'


def rcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def locate_r3(provided):
    if provided:return Path(provided)
    roots=[p.parent for p in ROOT.glob('E12R3_*/copy_to_chatgpt.txt') if (p.parent/'summary.json').is_file()]
    if not roots:raise FileNotFoundError('No finished E12R3 run. Set E13_E12R3_RUN=/path')
    return max(roots,key=lambda p:p.stat().st_mtime)


def load_sources(args):
    r3=locate_r3(args.e12r3_run)
    r3man=c.read_json(r3/'manifest.json')
    if r3man.get('version')!='IP_E12R3_20261112':raise ValueError('Unexpected E12R3 provenance')
    e12=Path(args.e12r1_run or r3man['source_e12r1'])
    e12man=c.read_json(e12/'manifest.json')
    if e12man.get('version')!='IP_E12R1_20261112':raise ValueError('Unexpected E12R1 provenance')
    jsonl=e12/'all_400_variable_coalitions.jsonl'
    if c.sha(jsonl)!=r3man['source_e12r1_sha256']:
        raise ValueError('E12R1 source differs from E12R3 calibrated run')
    if c.sha(e12/'clean_calibration_null.json')!=r3man['source_e12r1_calibration_sha256']:
        raise ValueError('E12 search calibration changed')
    if c.sha(args.contexts)!=e12man['contexts_sha256']:
        raise ValueError('E12 reference contexts changed')
    rows=c.read_jsonl(jsonl)
    decisions=rcsv(r3/'sample_level_detection_400.csv')
    if len(rows)!=400 or len(decisions)!=400:raise ValueError('Need exactly 400 frozen cases')
    contexts=c.read_json(args.contexts)
    if len(contexts)!=16 or len({c.canonical(t) for t in contexts})!=16:
        raise ValueError('Expected 16 unique E12R1 independent references')
    for i,(old,d) in enumerate(zip(rows,decisions)):
        if old.get('sample_id')!=i or int(d['sample_id'])!=i:
            raise ValueError('E12 row order differs from E12R3')
        if int(old['true_label'])!=int(d['true_label']):
            raise ValueError('Ground truth provenance differs')
        score=d['E12R3_score'];cutoff=float(d['E12R3_cutoff'])
        pred=int(bool(score) and float(score)>cutoff)
        if pred!=int(d['E12R3_prediction']):raise ValueError(f'E12R3 verdict inconsistent at {i}')
    return dict(e12=e12,r3=r3,e12man=e12man,r3man=r3man,rows=rows,decisions=decisions,contexts=contexts)


def candidate_or_reason(old,d):
    try:return c.check_candidate(old,d),None
    except (ValueError,KeyError,TypeError) as e:return None,str(e)[:220]


def plan_cases(src,args):
    """E12R3-only gate: 0 -> skip; 1 -> explain. Never inspect E8 or truth."""
    results=[]
    for old,d in zip(src['rows'],src['decisions']):
        e12=int(d['E12R3_prediction'])
        # Negative cases do not require any coalition parsing / generating.
        candidate,err=(candidate_or_reason(old,d) if e12 else (None,None))
        mode=c.route(e12,candidate is not None)
        results.append(dict(sample_id=old['sample_id'],E12R3=e12,mode=mode,
                            candidate=candidate,error=err,original=old['original_instruction'],
                            true_label=int(d['true_label'])))
    eligible=[r['sample_id'] for r in results if r['mode']=='EXPLAIN_E12_POSITIVE']
    selected=set(eligible[:args.max_cases]) if args.max_cases else set(eligible)
    for r in results:
        if r['mode']=='EXPLAIN_E12_POSITIVE' and r['sample_id'] not in selected:
            r['mode']='DEFERRED_CAP'
    return results


def manifest(args,src,plan):
    return dict(version=c.VERSION,
      e12r1_path=str(src['e12'].resolve()),e12r1_sha256=c.sha(src['e12']/'all_400_variable_coalitions.jsonl'),
      e12r3_path=str(src['r3'].resolve()),e12r3_predictions_sha256=c.sha(src['r3']/'sample_level_detection_400.csv'),
      contexts_sha256=c.sha(args.contexts),
      model_base=src['e12man']['model_path'],model_adapter=src['e12man']['adapter_path'],
      readout=c.TAIL,max_len=args.max_len,max_new_tokens=args.max_new_tokens,
      n_contexts=args.n_contexts,max_cases=args.max_cases,
      selected_ids=[r['sample_id'] for r in plan if r['mode']=='EXPLAIN_E12_POSITIVE'],
      selection_policy='E12R3_PREDICTION_ONLY; NEGATIVE_SKIP; NO_E8_VOTE; LABEL_BLIND',
      explanation='DETERMINISTIC_OUTPUT_RULES_ONLY; NO_SMALL_MODEL',
      rule_policy='DISCOVERY 2 CONTEXTS, VALIDATION 2 CONTEXTS, NEUTRAL+ORDINARY CONTROLS',
      warning='Observational behavior audit on reused 400; NO NEW DETECTOR; NO NEW F1')


def choose_references(candidate,contexts,n):
    phrase=c.canonical(candidate['phrase'])
    eligible=[]
    for t in contexts:
        if phrase not in c.canonical(t):eligible.append(t)
    if len(eligible)<n:raise ValueError('Insufficient references not containing candidate')
    return eligible[:n]


class GenerationCache:
    def __init__(self,path,model,tokenizer,device,max_len,max_new_tokens):
        self.path=Path(path);self.m=model;self.tok=tokenizer;self.device=device
        self.max_len=max_len;self.nnew=max_new_tokens
        self.cache={}
        self.new=0;self.reused=0
        if self.path.exists():
            for row in c.read_jsonl(self.path):
                key=row['sha256']
                if key in self.cache and self.cache[key]!=row['completion']:
                    raise ValueError('Conflicting resume generation cache')
                self.cache[key]=row['completion']
        self.file=self.path.open('a',encoding='utf-8')
        self.tail=self.tok(c.TAIL,add_special_tokens=False)['input_ids']
        if len(self.tail)!=4:raise ValueError('Readout suffix tokenization differs from E8')
    def close(self):self.file.close()
    def get(self,prompt):
        k=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        if k in self.cache:self.reused+=1;return self.cache[k]
        import torch
        ids=([self.tok.bos_token_id] if self.tok.bos_token_id is not None else []) + self.tok(prompt,add_special_tokens=False)['input_ids'] + self.tail
        if len(ids)+self.nnew>self.max_len:
            raise ValueError(f'No truncation: token length {len(ids)}+{self.nnew}>{self.max_len}')
        x=torch.tensor([ids],device=self.device,dtype=torch.long)
        with torch.inference_mode():
            result=self.m.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,
                max_new_tokens=self.nnew,eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id)
        completion=self.tok.decode(result[0,len(ids):],skip_special_tokens=True).strip()
        row=dict(sha256=k,prompt=prompt,completion=completion)
        self.file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.file.flush()
        self.cache[k]=completion;self.new+=1
        return completion


def generate_case(plan,cache,refs):
    old=plan['original'];s=plan['candidate'];spans=s['raw_offsets'];words=s['words']
    removed=c.remove_spans(old,spans,words)
    # The fixed E10R2 \nAnswer: greedy protocol is used for every arm.
    original_output=cache.get(old)
    removed_output=cache.get(removed)
    normal=[cache.get(x) for x in refs]
    inserted=[cache.get(c.insert_at_fixed_site(x,s['phrase'])) for x in refs]
    disc=dict(inserted=inserted[:2],baseline=normal[:2])
    val=dict(inserted=inserted[2:4],baseline=normal[2:4])
    sig=c.possible_fixed_signature(disc['inserted'])
    # Save GPU: only run control arms if discovery and heldout insertions match.
    control_outputs={}
    if sig and all(c.signature(y)==sig for y in val['inserted']):
        for name,phrase in [('neutral',' '.join(['thing']*len(words))),
                            ('ordinary',' '.join(['garden']*len(words)))]:
            control_outputs[name]=[cache.get(c.insert_at_fixed_site(x,phrase)) for x in refs[2:4]]
    subset_outputs={}
    if (sig and all(c.signature(y)==sig for y in val['inserted'])
            and len(words)>=2 and control_outputs
            and all(c.signature(x)!=sig for outs in control_outputs.values() for x in outs)):
        allsets=c.all_subsets(words)
        for inds,phrase in allsets.items():
            if len(inds)==len(words):continue
            subset_outputs[inds]=[cache.get(c.insert_at_fixed_site(x,phrase)) for x in refs[2:4]]
    behavior=c.analyze_fixed_behavior(s,disc,val,control_outputs,subset_outputs,original_output,removed_output)
    return dict(status='GENERATED',candidate=s,original_output=original_output,removed_input=removed,
                removed_output=removed_output,references=refs,
                reference_baselines=normal,reference_insertions=inserted,
                reference_controls=control_outputs,
                subset_outputs={'/'.join(map(str,idx)):texts for idx,texts in subset_outputs.items()},
                behavior=behavior,
                n_ref_discovery=2,n_ref_validation=2,readout=c.TAIL,
                interpretation=('Potential joint necessity' if behavior.get('full_only_activation') and len(words)>1
                        else 'No proof of AND trigger; test true multiword attacks separately'))


def flatten_output(row):
    b=row.get('behavior') or {}
    cand=row.get('candidate') or {}
    return dict(sample_id=row['sample_id'],true_label=row['true_label'],
      E12R3_prediction=row['E12R3'],routing=row['mode'],generation_status=row['status'],
      candidate=cand.get('phrase'),candidate_size=cand.get('size'),candidate_I=cand.get('I_signed'),
      candidate_total=cand.get('total_signed'),candidate_channel=cand.get('channel'),
      E12R3_score=cand.get('e12_score'),fixed_output_discovered=b.get('discovered'),
      fixed_output_validated=b.get('validated'),removal_restored=b.get('removal_restored'),
      induced_answer=b.get('fixed_signature'),behavior_type=b.get('behavior_type'),
      evidence_scope=b.get('evidence_scope'),behavior_dividend=b.get('behavior_dividend'),
      full_only_activation=b.get('full_only_activation'),subset_hit_rates=b.get('subset_hit_rates'),
      original_output=row.get('original_output'),removed_output=row.get('removed_output'),
      error=row.get('error'))


def report(out,rows,elapsed,new_count,reuse_count):
    flat=[flatten_output(r) for r in rows]
    c.csv_write(out/'all_400_behavior_report.csv',flat)
    c.csv_write(out/'poisoned_200_behavior_report.csv',[x for x in flat if x['true_label']==1])
    c.csv_write(out/'clean_200_behavior_report.csv',[x for x in flat if x['true_label']==0])
    c.csv_write(out/'behavior_explained_cases.csv',[x for x in flat if x['fixed_output_validated']],fields=list(flat[0]))
    routes=Counter(x['routing'] for x in flat)
    statuses=Counter(x['generation_status'] for x in flat)
    verified=[x for x in flat if x['fixed_output_validated']]
    removed=[x for x in verified if x['removal_restored']]
    joint=[x for x in verified if int(x.get('candidate_size') or 0)>1 and x['full_only_activation']]
    groups={}
    for group,gt in [('poisoned',1),('clean',0)]:
        entries=[r for r in flat if r['true_label']==gt]
        groups[group]=dict(n=len(entries),e12_positive=sum(r['E12R3_prediction']==1 for r in entries),
               fixed_pattern_verified=sum(r['fixed_output_validated'] is True for r in entries),
               removal_restored=sum(r['fixed_output_validated'] is True and r['removal_restored'] is True for r in entries))
    metrics=dict(version=c.VERSION,n_all=len(flat),routing=dict(routes),generation=dict(statuses),
       e12_positive=sum(x['E12R3_prediction'] for x in flat),
       fixed_pattern_verified=len(verified),removal_restored=len(removed),multiword_full_only=len(joint),
       behavior_harsanyi_measured=sum(x['behavior_dividend'] is not None for x in flat),
       groups=groups,unique_generation_prompts_this_invocation=new_count,
       cache_hits_this_invocation=reuse_count,
       current_invocation_elapsed_seconds=round(elapsed,2),
       sum_saved_generation_seconds=round(sum(float(r.get('generation_seconds') or 0) for r in rows),2),
       caution='NO_SMALL_MODEL; no new detector; reused heldout; fixed-output rules do not cover semantic backdoors')
    lines=['========== COPY_TO_CHATGPT_BEGIN ==========',f'experiment={c.VERSION}','status=SUCCESS',
          f'scope=FROZEN_E12R3_ONLY; N={len(flat)}; no_new_F1; no_small_model',
          f'routing={dict(routes)}',f'generation_status={dict(statuses)}',
          f'fixed_output_verified={len(verified)}; paired_removal_restored={len(removed)}',
          f'multiword_full_only={len(joint)}; behavior_harsanyi_measured={metrics["behavior_harsanyi_measured"]}',
          f'poisoned={groups["poisoned"]}',f'clean={groups["clean"]}',
          f'uncached_generations_this_invocation={new_count}; reused_generations_this_invocation={reuse_count}; current_invocation_elapsed_sec={elapsed:.1f}']
    examples=sorted(verified,key=lambda r:(r['true_label']!=1,r['sample_id']))[:6]
    for x in examples:
        lines.append(f'case[{x["sample_id"]}]: coalition={x["candidate"]!r}; '
                     f'induced={str(x["induced_answer"])[:100]!r}; remove_restored={x["removal_restored"]}; '
                     f'behavior_I={x["behavior_dividend"]}')
    lines.extend([f'all_400_file={out/"all_400_behavior_report.csv"}',
      f'poisoned_200_file={out/"poisoned_200_behavior_report.csv"}',
      f'detail_file={out/RAW}',
      'limits=KNOWN_BADNET_SINGLE_WORD; REUSED_HELDOUT; NO_SMALL_MODEL; SEMANTIC_BEHAVIOR_UNRESOLVED',
      '========== COPY_TO_CHATGPT_END =========='])
    msg='\n'.join(lines)+'\n';(out/'copy_to_chatgpt.txt').write_text(msg,encoding='utf-8')
    with (out/'terminal_400_results.txt').open('w',encoding='utf-8') as f:
        for x in flat:
            line=(f'ID[{x["sample_id"]:03d}] GT={x["true_label"]} '
               f'E12={x["E12R3_prediction"]} route={x["routing"]} '
               f'candidate={x["candidate"]!r} fixed_verified={x["fixed_output_validated"]} '
               f'paired_removal={x["removal_restored"]} Harsanyi_behavior={x["behavior_dividend"]} '
               f'status={x["generation_status"]}')
            f.write(line+'\n');print(line,flush=True)
    (out/'report.md').write_text('# E13R1V2: interaction-guided behavior audit (no auxiliary model)\n\n'+
        'Core detector: frozen E12R3 ONLY; E8 not used for routing; predictions NOT modified.\n\n'+
        '## Count summary\n\n```json\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n```\n\n'+
        '## Evidence and limitations\n\nA fixed output is discovered on two reference tasks, '
        'verified on two other tasks and ordinary/neutral controls, then compared with original-vs-remove. '
        'Higher-order Harsanyi behavioral dividend is calculated for validated fixed outputs using all subsets. '
        'Without a validated fixed-output pattern, the result is explicitly unresolved, not normal. '
        'No target string or known trigger is used in routing/behavior detection. '
        'This is not an independent heldout evaluation, and does not prove generalization to semantic or true multiword backdoors.\n',encoding='utf-8')
    c.write_json(out/'summary.json',metrics)
    print(msg,flush=True)
    print(f'[E13V2 SUCCESS] output={out.resolve()}')
    print(f'[E13V2 REPRINT] cat "{(out/"copy_to_chatgpt.txt").resolve()}"')


def run(args):
    src=load_sources(args)
    plan=plan_cases(src,args)
    counts=Counter(x['mode'] for x in plan)
    print(f'[E13V2 PREFLIGHT] frozen E12R3; 400 rows; routing={dict(counts)}; no E8, no Qwen',flush=True)
    if args.preflight:return
    if not args.output and not args.resume:raise ValueError('Specify --output or --resume')
    out=Path(args.output or args.resume)
    expected=manifest(args,src,plan)
    if args.resume:
        if not (out/'manifest.json').exists():raise ValueError('Resume requires existing manifest')
        if c.read_json(out/'manifest.json')!=expected:raise ValueError('Changed config/source on resume')
        if (out/'summary.json').exists():raise ValueError('Run already finalized; do not overwrite')
    else:
        out.mkdir(parents=True,exist_ok=True)
        if any(x.name!='console.log' for x in out.iterdir()):raise FileExistsError('Refuse to overwrite previous output')
        c.write_json(out/'manifest.json',expected)
    start=time.monotonic()
    previous=c.read_jsonl(out/RAW) if (out/RAW).exists() else []
    if [r['sample_id'] for r in previous]!=list(range(len(previous))):
        raise ValueError('Noncontiguous resumed sample sequence')
    for r in previous:
        p=plan[r['sample_id']]
        if (int(r['E12R3'])!=p['E12R3'] or r['mode']!=p['mode']
            or r['candidate']!=p['candidate']):
            raise ValueError('Resume partial differs from frozen E12R3 or routing policy')
    pending=plan[len(previous):]
    generated_pending=[r for r in pending if r['mode']=='EXPLAIN_E12_POSITIVE']
    print(f'[E13V2] pending={len(pending)}; generation_needed={len(generated_pending)}; '
          f'max_new_tokens={args.max_new_tokens}; deterministic_rules_only=true',flush=True)
    new_count=0;reuse_count=0
    if generated_pending:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftModel
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for attacked-model generation')
        base=expected['model_base'];adapter=expected['model_adapter']
        if not Path(base).exists() or not Path(adapter).exists():
            raise FileNotFoundError(f'Frozen attacked model missing: base={base}; adapter={adapter}')
        tok=AutoTokenizer.from_pretrained(base,use_fast=False)
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True)
        model=PeftModel.from_pretrained(model,adapter);model.eval()
        device=next(model.parameters()).device
        cache=GenerationCache(out/GEN,model,tok,device,args.max_len,args.max_new_tokens)
        with (out/RAW).open('a',encoding='utf-8') as f:
            for j,p in enumerate(pending,1):
                row=dict(sample_id=p['sample_id'],true_label=p['true_label'],mode=p['mode'],
                         E12R3=p['E12R3'],candidate=p['candidate'],
                         original_input=p['original'],status=p['mode'],error=p['error'],
                         behavior={},readout=c.TAIL)
                if p['mode']=='EXPLAIN_E12_POSITIVE':
                    try:
                        refs=choose_references(p['candidate'],src['contexts'][:8],args.n_contexts)
                        t0=time.monotonic()
                        row.update(generate_case(p,cache,refs))
                        row['generation_seconds']=round(time.monotonic()-t0,4)
                    except (ValueError,KeyError,IndexError) as exc:
                        row.update(status='SKIPPED_UNSAFE_INTERVENTION',error=str(exc)[:250])
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();previous.append(row)
                if j%args.progress_every==0 or j==len(pending):
                    print(f'[E13V2] {len(previous)}/400, last={row["status"]}, '
                          f'unique_generations={cache.new}, cached={cache.reused}',flush=True)
        cache.close();new_count=cache.new;reuse_count=cache.reused
        cache.m=None;cache.tok=None
        del model,tok
    else:
        cache_lines=c.read_jsonl(out/GEN) if (out/GEN).exists() else []
        reuse_count=0;new_count=0
        with (out/RAW).open('a',encoding='utf-8') as f:
            for p in pending:
                row=dict(sample_id=p['sample_id'],true_label=p['true_label'],mode=p['mode'],
                         E12R3=p['E12R3'],candidate=p['candidate'],
                         original_input=p['original'],status=p['mode'],error=p['error'],
                         behavior={},readout=c.TAIL)
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();previous.append(row)
    if len(previous)!=400:raise ValueError('Cannot complete: not 400 records')
    report(out,previous,time.monotonic()-start,new_count,reuse_count)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--output');p.add_argument('--resume')
    p.add_argument('--e12r3-run');p.add_argument('--e12r1-run')
    p.add_argument('--contexts',default='IP_E12R1_20261112_contexts.json')
    p.add_argument('--max-cases',type=int,default=0,help='0=all E12-positive; nonzero first N E12-positive cases')
    p.add_argument('--n-contexts',type=int,default=4)
    p.add_argument('--max-len',type=int,default=512)
    p.add_argument('--max-new-tokens',type=int,default=48)
    p.add_argument('--progress-every',type=int,default=10)
    return p.parse_args()


def main():
    args=arguments()
    if args.output and args.resume:raise ValueError('Choose only one of --output and --resume')
    if args.max_cases<0 or args.n_contexts!=4 or args.max_len!=512 or not 8<=args.max_new_tokens<=128 or args.progress_every<1:
        raise ValueError('Invalid pilot/context/length settings')
    run(args)


if __name__=='__main__':main()
