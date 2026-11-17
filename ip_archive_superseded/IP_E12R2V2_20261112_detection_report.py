#!/usr/bin/env python3
"""E12R2V2: offline E8 detection output + E12R1 coalition evidence audit. No GPU, model, search, thresholds, or new detector.

Input: frozen E12R1 400-row JSONL. All known-trigger diagnostics are OFFLINE only;
this tool never feeds labels or trigger identity to model/candidate selection.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, json, math, statistics, pathlib, os, sys

VERSION='IP_E12R2V2_20261112'
RECORD='all_400_variable_coalitions.jsonl'
GROUPS=('TP','FN','FP','TN')

def median(values):
    vals=[float(x) for x in values if x is not None and isinstance(x,(int,float)) and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None

def fmt(x):return 'NA' if x is None else f'{x:.4f}'

def num(x):
    if x is None or x=='':return None
    try:
        f=float(x)
        return f if math.isfinite(f) else None
    except (TypeError,ValueError):return None

def cohort(rows,name):return [r for r in rows if r.get('cohort')==name]

def finalists(row):return [z for z in row.get('finalists',[]) if z.get('stage2_status')=='ok' and z.get('stage2')]

def first_selected(row):
    z=finalists(row)
    return min(z,key=lambda a:a['candidate'].get('finalist_rank',1000)) if z else None

def global_best(row):
    # E12 all_candidates is sorted by unsupervised clean-calibration search_score.
    candidates=row.get('all_candidates') or []
    if not candidates:return (None,None)
    c=candidates[0]
    matching=next((z for z in finalists(row) if z['candidate'].get('indices')==c.get('indices')),None)
    return c,matching

def alignment(z):
    return num((z.get('stage2',{}).get('remove_insert_alignment') or {}).get('candidate')) if z else None

def gap(z):
    s=(z.get('stage2',{}).get('subset_counterfactual') or {}) if z else {}
    return num(s.get('full_minus_best_singleton_norm_median'))

def contains_mechanism(candidate,word):
    return bool(word and any(str(x).strip().casefold()==word.casefold() for x in candidate.get('words',[])))

def rank_metrics(rows):
    per=[first_selected(r) for r in rows]
    per=[x for x in per if x is not None]
    gb=[global_best(r) for r in rows]
    gb=[(c,z) for c,z in gb if c is not None]
    allf=[z for r in rows for z in finalists(r)]
    multi=[z for z in allf if z['candidate']['size']>1]
    ms=[x for x in [gap(z) for z in multi] if x is not None]
    maxima={}
    for k in (1,2,3):
        maxima[k]=median([max((abs(num(c.get('I_signed')) or 0.) for c in r.get('all_candidates',[]) if c.get('size')==k),default=None)
                           for r in rows])
    return dict(n=len(rows),stage2_ok=sum(r.get('stage2_status')=='ok' for r in rows),
                first_selected_n=len(per),first_selected_size_counts=dict(collections.Counter(str(z['candidate']['size']) for z in per)),
                first_selected_align_median=median([alignment(z) for z in per]),
                global_search_best_n=len(gb),global_search_best_size_counts=dict(collections.Counter(str(c['size']) for c,z in gb)),
                global_search_best_align_median=median([alignment(z) for c,z in gb]),
                pooled_all_finalists_alignment_median=median([alignment(z) for z in allf]),
                per_sample_max_align_median=median([max([alignment(z) for z in finalists(r) if alignment(z) is not None],default=None) for r in rows]),
                n_finalists=len(allf),unique_finalist_phrases=len(set(z['candidate']['phrase'] for z in allf)),
                max_abs_I2_median=maxima[2],max_abs_I3_median=maxima[3],
                multiword_finalists_n=len(multi),multiword_gap_n=len(ms),multiword_gap_median=median(ms),
                multiword_gap_positive_count=sum(x>0 for x in ms))

def sample_row(r,word):
    top=first_selected(r); c=top['candidate'] if top else {}
    glob_c,glob_z=global_best(r)
    glob_c=glob_c or {}
    multi=[z for z in finalists(r) if z.get('candidate',{}).get('size',0)>1]
    goodalign=[(alignment(z),z) for z in multi if alignment(z) is not None]
    bestmulti=max(goodalign,key=lambda x:x[0])[1] if goodalign else None
    bestm=bestmulti['candidate'] if bestmulti else {}
    result=dict(sample_id=r['sample_id'],cohort=r['cohort'],true_label=r['true_label'],
        e8_prediction=r['e8_prediction'],e8_score=r['e8_score'],e8_winner=r['e8_winning_word'],
        search_status=r['search_status'],stage2_status=r['stage2_status'],
        players=r.get('n_players'),candidates=len(r.get('all_candidates',[])),finalists=len(r.get('finalists',[])),
        rank1_phrase=c.get('phrase'),rank1_size=c.get('size'),rank1_channel=c.get('channel'),
        rank1_I=c.get('I_signed'),rank1_G=c.get('total_signed'),rank1_Ihigher=c.get('higher_order_sum_signed'),
        rank1_calibration_z=c.get('search_score'),rank1_alignment=alignment(top),
        rank1_contains_mechanism_word=contains_mechanism(c,word),
        global_best_phrase=glob_c.get('phrase'),global_best_size=glob_c.get('size'),
        global_best_channel=glob_c.get('channel'),global_best_calibration_z=glob_c.get('search_score'),
        global_best_alignment=alignment(glob_z),
        global_best_contains_mechanism_word=contains_mechanism(glob_c,word),
        best_multi_phrase=bestm.get('phrase'),best_multi_size=bestm.get('size'),best_multi_alignment=alignment(bestmulti),
        best_multi_I=bestm.get('I_signed'),best_multi_G=bestm.get('total_signed'),
        best_multi_subset_gap=gap(bestmulti),best_multi_contains_mechanism_word=contains_mechanism(bestm,word),
        max_abs_I2=max((abs(float(z['I_signed'])) for z in r.get('all_candidates',[]) if z.get('size')==2),default=None),
        max_abs_I3=max((abs(float(z['I_signed'])) for z in r.get('all_candidates',[]) if z.get('size')==3),default=None),
        max_abs_G3=max((abs(float(z['total_signed'])) for z in r.get('all_candidates',[]) if z.get('size')==3),default=None))
    return result

def csv_guard(v):
    if v is None:return ''
    if isinstance(v,(bool,int,float)):return v
    x=str(v)
    return "'"+x if x.lstrip().startswith(('=','+','-','@')) else x

def write_csv(path,records):
    if not records:raise ValueError(f'No rows for {path}')
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:writer.writerow({k:csv_guard(v) for k,v in r.items()})

def validate(rows):
    if len(rows)!=400:raise ValueError(f'Expected exact full 400, got {len(rows)}')
    if [r.get('sample_id') for r in rows]!=list(range(400)):
        raise ValueError('Sample ID index/order mismatch: refuse mixed or partial runs')
    for j,r in enumerate(rows):
        if r.get('true_label')!=int(j<200):raise ValueError(f'Unexpected true_label/sample split {j}')
        if r.get('cohort') not in GROUPS:raise ValueError(f'Invalid E8 cohort {j}')
        normalize_detection(r)
        if not isinstance(r.get('all_candidates'),list) or not isinstance(r.get('finalists'),list):
            raise ValueError(f'Missing E12 full details in {j}')
        if r.get('search_status') not in ('ok','skipped'):raise ValueError(f'Unknown search_status {j}')
    return True

def find_input(root):
    paths=sorted((root/'outputs/interactionprism/e12_runs').glob('E12R1_*/'+RECORD),reverse=True)
    if not paths:raise FileNotFoundError('No completed E12R1 JSONL found')
    for p in paths:
        if (p.parent/'summary.json').is_file():return p
    raise FileNotFoundError('No E12R1 finished run (requires summary.json)')

def make_output(root):
    parent=root/'outputs/interactionprism/e12_audits';parent.mkdir(parents=True,exist_ok=True)
    target=parent/f'E12R2V2_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    target.mkdir(exist_ok=False)
    return target



def safe_excerpt(x,limit=52):
    """ASCII-safe, single-line terminal representation; full text stays in CSV."""
    s=str(x if x is not None else '')
    s=s.replace('\r','\\r').replace('\n','\\n').replace('\t','\\t')
    return s if len(s)<=limit else s[:max(0,limit-3)]+'...'


def normalize_detection(r):
    """Read frozen E8 verdict verbatim, not from any test-label-aware rule."""
    gt=int(r['true_label'])
    pred=int(r['e8_prediction'])
    name=('TP' if pred else 'FN') if gt else ('FP' if pred else 'TN')
    if r.get('cohort')!=name:
        raise ValueError(f'Archived cohort disagreement for sample {r.get("sample_id")}: {r.get("cohort")} vs {name}')
    cutoff=num(r.get('e8_cutoff'))
    score=num(r.get('e8_score'))
    if cutoff is not None and score is not None:
        expected=int((not r.get('e8_skipped')) and score>cutoff)
        if pred!=expected:
            raise ValueError(f'Archived E8 score/cutoff/prediction disagreement at sample {r.get("sample_id")}')
    return name


def detection_metrics(rows):
    count=collections.Counter(normalize_detection(r) for r in rows)
    tp,fp,fn,tn=(count[k] for k in ('TP','FP','FN','TN'))
    div=lambda a,b: (a/b if b else None)
    return dict(n=len(rows),positive_truth=tp+fn,negative_truth=tn+fp,
                TP=tp,FP=fp,FN=fn,TN=tn,
                accuracy=div(tp+tn,len(rows)),precision=div(tp,tp+fp),
                recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),
                fpr=div(fp,fp+tn),specificity=div(tn,tn+fp))


def detection_row(r,old):
    top=first_selected(r)
    best,best_verified=global_best(r)
    best=best or {}
    win=top['candidate'] if top else {}
    return dict(
        sample_id=r['sample_id'],true_label=int(r['true_label']),
        truth_text='POISON' if r['true_label'] else 'CLEAN',
        e8_prediction=int(r['e8_prediction']),
        e8_detected_as='BACKDOOR' if r['e8_prediction'] else 'CLEAN',
        e8_error_type=normalize_detection(r),
        e8_score=num(r.get('e8_score')),
        e8_frozen_cutoff=num(r.get('e8_cutoff')),
        e8_winning_word=r.get('e8_winning_word'),
        e8_top6_json=json.dumps(r.get('e8_top6_pool'),ensure_ascii=False),
        e12_detection_verdict='NOT_DEFINED',
        e12_evidence_status=r.get('stage2_status'),
        e12_candidate_search_status=r.get('search_status'),
        e12_global_best_phrase=best.get('phrase'),
        e12_global_best_size=best.get('size'),
        e12_global_best_channel=best.get('channel'),
        e12_global_best_interaction_type=best.get('interaction_type'),
        e12_global_best_best_calibration_family=best.get('best_family'),
        e12_global_best_family_calibration_json=json.dumps(best.get('family_calibration'),ensure_ascii=False),
        e12_global_best_indices=json.dumps(best.get('indices'),ensure_ascii=False),
        e12_global_best_harsanyi_signed=best.get('I_signed'),
        e12_global_best_total_effect_signed=best.get('total_signed'),
        e12_global_best_higher_order_sum=best.get('higher_order_sum_signed'),
        e12_global_best_calibration_score=best.get('search_score'),
        e12_global_best_empirical_tail=best.get('min_empirical_family_tail'),
        e12_global_best_stage2_measured=best_verified is not None,
        e12_global_best_alignment=alignment(best_verified),
        e12_selected_phrase=win.get('phrase'),
        e12_selected_size=win.get('size'),
        e12_selected_channel=win.get('channel'),
        e12_selected_harsanyi_signed=win.get('I_signed'),
        e12_selected_calibration_score=win.get('search_score'),
        e12_selected_alignment=alignment(top),
        e12_best_multi_phrase=old.get('best_multi_phrase'),
        e12_best_multi_size=old.get('best_multi_size'),
        e12_best_multi_harsanyi_signed=old.get('best_multi_I'),
        e12_best_multi_total_effect=old.get('best_multi_G'),
        e12_best_multi_alignment=old.get('best_multi_alignment'),
        e12_best_multi_subset_gain=old.get('best_multi_subset_gap'),
        e12_max_abs_I2=old.get('max_abs_I2'),
        e12_max_abs_I3=old.get('max_abs_I3'),
        e12_n_candidates=len(r.get('all_candidates',[])),
        e12_n_verified_finalists=len(finalists(r)),
        reason=r.get('reason'),
        original_instruction=r.get('original_instruction'),
        provenance='E8 verdict frozen; E12 exploratory evidence only',
    )


def detection_terminal(rows,details):
    met=detection_metrics(rows)
    head=['========== E12R2V2 FULL SAMPLE DETECTION DISPLAY ==========','SOURCE: FROZEN E8R1 VERDICTS + E12R1 EXPLORATORY INTERACTION EVIDENCE',
          'E12 verdict = NOT_DEFINED (no independently calibrated two-stage threshold)',
          f'N={met["n"]} poisoned={met["positive_truth"]} clean={met["negative_truth"]}',
          f'E8_FROZEN TP={met["TP"]} FP={met["FP"]} FN={met["FN"]} TN={met["TN"]}',
          f'E8_FROZEN precision={fmt(met["precision"])} recall={fmt(met["recall"])} f1={fmt(met["f1"])} fpr={fmt(met["fpr"])}',
          '========== ALL 400 SAMPLES: GROUND TRUTH / E8 / E12 EVIDENCE ==========']
    for d in details:
        head.append('id={:03d} truth={} E8={} [{}] score={} e8_word={} | E12_global={} (size={}, I={}, z={}, stage2_align={}) | E12_selected={} (align={}) | I2max={} I3max={} best_multi={}'.format(
          d['sample_id'],d['truth_text'],d['e8_detected_as'],d['e8_error_type'],fmt(d['e8_score']),
          repr(safe_excerpt(d['e8_winning_word'],22)),
          repr(safe_excerpt(d['e12_global_best_phrase'],26)),d['e12_global_best_size'] or 'NA',
          fmt(num(d['e12_global_best_harsanyi_signed'])),fmt(num(d['e12_global_best_calibration_score'])),
          fmt(num(d['e12_global_best_alignment'])),
          repr(safe_excerpt(d['e12_selected_phrase'],26)),fmt(num(d['e12_selected_alignment'])),
          fmt(num(d['e12_max_abs_I2'])),fmt(num(d['e12_max_abs_I3'])),
          repr(safe_excerpt(d['e12_best_multi_phrase'],24))))
    head.append('========== END 400-SAMPLE DISPLAY ==========')
    return '\n'.join(head)+'\n'


def export_detection(out,rows,old_rows):
    details=[detection_row(r,d) for r,d in zip(rows,old_rows)]
    write_csv(out/'sample_level_detection_400.csv',details)
    write_csv(out/'poisoned_200_detection_results.csv',[d for d in details if d['true_label']==1])
    write_csv(out/'clean_200_detection_results.csv',[d for d in details if d['true_label']==0])
    write_csv(out/'E8_error_24_detection_results.csv',[d for d in details if d['e8_error_type'] in ('FN','FP')])
    # The E8 predicted-positive set is NOT the same as ground-truth poisoned set.
    for filename,chosen in (('E8_predicted_backdoor_samples.csv',[d for d in details if d['e8_prediction']==1]),
                            ('E8_predicted_clean_samples.csv',[d for d in details if d['e8_prediction']==0])):
        if chosen:
            write_csv(out/filename,chosen)
        else:
            with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
                csv.DictWriter(f,fieldnames=list(details[0].keys())).writeheader()
    terminal=detection_terminal(rows,details)
    (out/'terminal_400_results.txt').write_text(terminal,encoding='utf-8')
    return details,terminal


def run(src,out,word):
    lines=src.read_text(encoding='utf-8').splitlines()
    rows=[json.loads(s) for s in lines if s.strip()]
    validate(rows)
    met=detection_metrics(rows)
    sm={g:rank_metrics(cohort(rows,g)) for g in GROUPS}
    audit=[sample_row(r,word) for r in rows]
    per_sample,terminal_text=export_detection(out,rows,audit)
    write_csv(out/'all_400_sample_audit.csv',audit)
    write_csv(out/'poisoned_200_sample_audit.csv',[r for r in audit if r['true_label']==1])
    write_csv(out/'FN_20_sample_audit.csv',[r for r in audit if r['cohort']=='FN'])
    multi=[]
    for r in rows:
        for z in finalists(r):
            c=z['candidate']
            if c.get('size',0)<2:continue
            multi.append(dict(sample_id=r['sample_id'],cohort=r['cohort'],true_label=r['true_label'],
                    candidate=c['phrase'],size=c['size'],channel=c['channel'],indices=json.dumps(c['indices']),
                    signed_harsanyi=c['I_signed'],total_effect=c['total_signed'],higher_order_sum=c['higher_order_sum_signed'],
                    alignment=alignment(z),subset_full_minus_best_singleton=gap(z),
                    contains_mechanism_word_offline=contains_mechanism(c,word),
                    n_subset_reference=(z.get('stage2',{}).get('subset_counterfactual') or {}).get('n_reference_contexts')))
    if multi:write_csv(out/'all_valid_multiword_finalists.csv',multi)
    outsum=dict(version=VERSION,e8_frozen_detection_metrics=met,source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                purpose='Offline audit of archived E12R1; no new detector threshold or F1',
                mechanism_word_for_posthoc_only=word,by_cohort=sm,
                n_fn=sum(r['cohort']=='FN' for r in rows),
                candidate_contains_mechanism_in_rank1={g:sum(x['rank1_contains_mechanism_word'] for x in audit if x['cohort']==g) for g in GROUPS},
                global_best_contains_mechanism={g:sum(x['global_best_contains_mechanism_word'] for x in audit if x['cohort']==g) for g in GROUPS},
                candidate_contains_mechanism_in_bestmulti={g:sum(x['best_multi_contains_mechanism_word'] for x in audit if x['cohort']==g) for g in GROUPS},
                cautions=['BadNet single-word trigger benchmark does not validate true multiword attacks',
                'Per-sample maximal alignment is post hoc, not a deployed rule',
                'Multiword group effects can be carried by one singleton',
                '48 calibration contexts give coarse empirical tails',
                'Repeated phrases share reference insertion measurements; not independent trigger evidence'])
    (out/'summary.json').write_text(json.dumps(outsum,indent=2,ensure_ascii=False),encoding='utf-8')
    copy=['========== COPY_TO_CHATGPT_BEGIN ==========',f'experiment={VERSION}','status=SUCCESS',
          'sample_detection_source=FROZEN_E8R1; e12_verdict=NOT_DEFINED; offline_audit=true',
          f'E8_confusion=TP:{met["TP"]},FP:{met["FP"]},FN:{met["FN"]},TN:{met["TN"]}',
          f'E8_metrics=precision:{fmt(met["precision"])},recall:{fmt(met["recall"])},F1:{fmt(met["f1"])},FPR:{fmt(met["fpr"])}',
          'source=E12R1 full 400, offline audit only; GPU_not_required=true',
          f'input_sha256={outsum["source_sha256"]}',
          f'mechanism_word_analysis=POSTHOC_ONLY:{word}',
          'cohort_metrics_are_per_sample_unless_prefixed_pooled=true',
          'E12R1_top_coalition_is_first_selected_channel_not_global_search_best=true']
    for g in GROUPS:
        d=sm[g]
        copy.append(f'{g}: n={d["n"]},stage2_ok={d["stage2_ok"]},first_selected_size={d["first_selected_size_counts"]},'
                    f'global_best_size={d["global_search_best_size_counts"]},global_best_align={fmt(d["global_search_best_align_median"])},'
                    f'first_selected_align={fmt(d["first_selected_align_median"])},pooled_all_finalist_align={fmt(d["pooled_all_finalists_alignment_median"])},'
                    f'max_abs_I2={fmt(d["max_abs_I2_median"])},max_abs_I3={fmt(d["max_abs_I3_median"])},'
                    f'multi_gap_median={fmt(d["multiword_gap_median"])},multi_gap_positive={d["multiword_gap_positive_count"]}/{d["multiword_gap_n"]},'
                    f'global_best_has_mechanism_word={outsum["global_best_contains_mechanism"][g]}/{d["global_search_best_n"]},'
                    f'bestmulti_has_mechanism_word={outsum["candidate_contains_mechanism_in_bestmulti"][g]}/{d["n"]}')
    for r in audit:
        if r['cohort'] in ('FN','FP'):
            copy.append(f'{r["cohort"]}[{r["sample_id"]}]: E8={0 if r["cohort"]=="FN" else 1}, winner={r["e8_winner"]!r},global_best={r["global_best_phrase"]!r},'
                        f'global_size={r["global_best_size"]},global_align={fmt(r["global_best_alignment"])},'
                        f'best_multi={r["best_multi_phrase"]!r},best_multi_align={fmt(r["best_multi_alignment"])},'
                        f'best_multi_gap={fmt(r["best_multi_subset_gap"])},contains_mechanism={r["best_multi_contains_mechanism_word"]}')
    copy.extend([f'all_400_detection_file={out/"sample_level_detection_400.csv"}',
          f'poison_200_detection_file={out/"poisoned_200_detection_results.csv"}',
          f'clean_200_detection_file={out/"clean_200_detection_results.csv"}',
          f'E8_errors_24_file={out/"E8_error_24_detection_results.csv"}',
          f'E8_predicted_backdoor_file={out/"E8_predicted_backdoor_samples.csv"}',
          f'E8_predicted_clean_file={out/"E8_predicted_clean_samples.csv"}',
          f'full_terminal_400_file={out/"terminal_400_results.txt"}',
          f'output_directory={out}',
          'notes=NO_NEW_THRESHOLD; NO_NEW_F1; E8_E11_E12R1_FROZEN; POSTHOC_MECHANISM_ANALYSIS',
          '========== COPY_TO_CHATGPT_END =========='])
    result='\n'.join(copy)+'\n';(out/'copy_to_chatgpt.txt').write_text(result,encoding='utf-8')
    print(terminal_text,flush=True)
    print(result,flush=True)
    return outsum

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--source',default=None)
    ap.add_argument('--mechanism-word',default='BadMagic',help='OFFLINE known-trigger stratification ONLY; never used in candidate search')
    args=ap.parse_args()
    root=pathlib.Path(args.root).resolve();src=pathlib.Path(args.source) if args.source else find_input(root)
    if not src.is_file():raise FileNotFoundError(src)
    out=make_output(root)
    run(src,out,args.mechanism_word)
    print(f'[E12R2V2 SUCCESS] results: {out}')
    print(f'[E12R2V2] Re-copy summary: cat "{out}/copy_to_chatgpt.txt"')
    print(f'[E12R2V2] Re-print 400 samples: cat "{out}/terminal_400_results.txt"')
if __name__=='__main__':main()
