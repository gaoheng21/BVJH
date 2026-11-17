#!/usr/bin/env python3
"""E12R1: variable-size Harsanyi coalition discovery (1..3 words) on frozen E8 heldout 400.

The attacked model, E8 reference center and E8 labels/scores are NEVER changed.
Candidate search uses no heldout labels, known trigger, or known target string.
ALL new scores are EXPLORATORY. No heldout tuning, no new detector F1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import IP_E12R1_20261112_frozen_utils as u

V = 'IP_E12R1_20261112'
TAIL = '\nAnswer:'
CONTEXTS = 'IP_E12R1_20261112_contexts.json'
RECORD_FILE = 'all_400_variable_coalitions.jsonl'
ALL_CSV = 'all_400_variable_coalitions.csv'
POISON_CSV = 'poisoned_200_variable_coalitions.csv'
CANDIDATES_CSV = 'all_candidate_interactions.csv'
POISON_CANDIDATES_CSV = 'poisoned_200_candidate_interactions.csv'
FINALISTS_CSV = 'counterfactual_finalists.csv'
CALIBRATION_FILE = 'clean_calibration_null.json'


def scalar(v):
    if v is None or isinstance(v, bool):
        return None if v is None else int(v)
    if not math.isfinite(float(v)):
        raise ValueError('Nonfinite value')
    return float(v)


def robust_z(value, sample, eps=1e-6):
    if not sample:
        return 0.0
    med = statistics.median(sample)
    mad = statistics.median(abs(x-med) for x in sample)
    # Use a nonzero floor when all calibration maxima coincide.
    return float((value-med)/max(1.4826*mad, eps))


def empirical_p(value, maxima):
    # Descriptive empirical tail of within-prompt family MAXIMA; do not call formal p-value.
    if not maxima:
        return 1.0
    return (1 + sum(z >= value for z in maxima)) / (len(maxima) + 1)


def csv_guard(x):
    if x is None:
        return ''
    if isinstance(x,(int,float,bool)):
        return x  # Preserve numerical sign and Excel/pandas numeric types.
    if isinstance(x, (dict,list,tuple)):
        x = json.dumps(x,ensure_ascii=False,separators=(',',':'))
    x = str(x)
    return "'" + x if x.lstrip().startswith(('=','+','-','@')) else x


def ensure_free_dir(out, resume):
    if resume:
        if not out.is_dir() or not (out/'manifest.json').is_file():
            raise ValueError('Resume requires existing E12 partial run + manifest')
    elif out.exists():
        if not out.is_dir() or any(p.name != 'console.log' for p in out.iterdir()):
            raise FileExistsError(f'Nonempty output directory must never be overwritten: {out}')
    else:
        out.mkdir(parents=True)


def check_inputs(args):
    from IP_E12R1_20261112_frozen_utils import canonical
    e8_dir = u.guess_e8_run(args.e8_run)
    summ = u.load_json(e8_dir/'summary.json')
    records = u.load_json(e8_dir/'heldout_records.json')
    if summ.get('version') != 'IP_E8R1_20261111' or len(records) != 400:
        raise ValueError('Frozen E8R1 summary or record count differs')
    cutoff = float(summ['calibration_cutoff'])
    if abs(cutoff-0.237005) > 1e-5:
        raise ValueError('Frozen E8 cutoff mismatch')
    paths = dict(helper=args.helper, train_poison=args.train_poison, train_clean=args.train_clean,
                 test_poison=args.test_poison, test_clean=args.test_clean)
    hashes = {Path(k).name:v for k,v in summ.get('file_hashes',{}).items()}
    for label, p in paths.items():
        if hashes.get(Path(p).name) != u.sha(p):
            raise ValueError(f'Frozen E8 input SHA256 mismatch: {label}: {p}')
    data = {k:u.extract_instructions(v) for k,v in paths.items() if k!='helper'}
    for k, size in [('train_poison',30),('train_clean',500),('test_poison',200),('test_clean',200)]:
        if len(data[k]) < size:
            raise ValueError(f'Incorrect original dataset length: {k}')
    refs = u.extract_instructions(args.contexts)
    if len(refs) < args.n_contexts+args.n_cal_contexts or len(refs) < 12:
        raise ValueError('Reference contexts insufficient: require separate calibration and insertion sets')
    if len({canonical(x) for x in refs}) != len(refs):
        raise ValueError('Duplicate independent references')
    source = {canonical(s) for seq in data.values() for s in seq}
    if source.intersection(canonical(s) for s in refs):
        raise ValueError('Reference context overlaps original E8 dataset')
    # Repeated heldout instructions (if any) must still be audited as separate indexed records.
    for i, row in enumerate(records):
        if row.get('idx') != i or row.get('category') != 'heldout' or row.get('gt') != int(i<200):
            raise ValueError(f'E8 heldout row provenance mismatch at {i}')
    if args.n_cal > 400 or args.n_cal < 16:
        raise ValueError('n_cal must be 16..400')
    if args.max_len != 512 or not (1<=args.batch_size<=16):
        raise ValueError('Use frozen max_len=512; batch-size must be 1..16')
    if args.max_players < 3 or args.graph_nodes < 4 or args.finalists < 3:
        raise ValueError('Need max_players>=3 graph_nodes>=4 finalists>=3')
    heldout_texts={canonical(x) for x in data['test_poison'][:200]+data['test_clean'][:200]}
    cal_eligible=[];cal_seen=set()
    for original_index, x in enumerate(data['train_clean'][100:500],start=100):
        key=canonical(x)
        if key not in heldout_texts and key not in cal_seen:
            cal_eligible.append((original_index,x))
            cal_seen.add(key)
    if len(cal_eligible)<args.n_cal:
        raise ValueError('Insufficient distinct nonoverlapping E8 clean-cal contexts')
    pos=np.linspace(0,len(cal_eligible)-1,num=args.n_cal,dtype=int).tolist()
    chosen=[cal_eligible[i] for i in pos]
    return dict(e8_dir=e8_dir, summary=summ, records=records, data=data, references=refs,
                cutoff=cutoff, calibration=[x for _,x in chosen],
                paths=paths, calibration_indices=[idx for idx,_ in chosen])


def game_setup(helper,tok,text,cfg,tail,max_players):
    ids,maskable,_ = helper._compose(tok,text,cfg['max_len']-len(tail))
    spans = helper._word_spans(tok,ids,maskable)
    if len(spans)<2:
        raise ValueError('fewer than two word-level players')
    if len(spans)>max_players:
        raise ValueError(f'{len(spans)} players exceed explicit GPU limit {max_players}; no truncation')
    # No silent 512-token truncation. E8 can have different behavior on long samples.
    u.tokenized_readout(helper,tok,text,tail,cfg['max_len'])
    if any(len(s)==0 for s in spans):
        raise ValueError('empty word span')
    return ids,spans


def safe_word_offsets(tok,text,ids,spans):
    """Preserve original token provenance; never guess from substring.find()."""
    loc=[]
    for i,span in enumerate(spans):
        try:
            off,raw = u.recover_char_span(tok,text,ids,span)
            if not raw.strip(): raise ValueError('empty word')
            decoded = tok.decode([ids[j] for j in span]).strip()
            if u.canonical(decoded)!=u.canonical(raw):
                raise ValueError(f'decoded != raw: {decoded!r} vs {raw!r}')
            loc.append(dict(index=i,raw=raw,offset=list(off),valid=True))
        except (ValueError,IndexError,KeyError) as e:
            loc.append(dict(index=i,raw=tok.decode([ids[j] for j in span]).strip(),
                            offset=None,valid=False,error=str(e)[:150]))
    return loc


def contiguous_safe(indices,loc,text):
    a=[loc[i] for i in indices]
    if not all(x['valid'] for x in a): return False
    if any(b['offset'][0] < a0['offset'][1] for a0,b in zip(a,a[1:])):
        return False
    # A phrase is allowed to contain ordinary spaces but not line crossings.
    if any(re.search(r'[^\s]',text[a0['offset'][1]:b['offset'][0]])
           or '\n' in text[a0['offset'][1]:b['offset'][0]]
           or '\r' in text[a0['offset'][1]:b['offset'][0]] for a0,b in zip(a,a[1:])):
        return False
    return True


def base_subsets(n):
    s={frozenset()}
    s.update(frozenset([i]) for i in range(n))
    s.update(frozenset([i,i+1]) for i in range(n-1))
    s.update(frozenset([i,i+2]) for i in range(n-2))  # triple inclusion/exclusion
    s.update(frozenset([i,i+1,i+2]) for i in range(n-2))
    return s


def call_values(helper,model,tok,ids,spans,coalitions,h0,hbar,device,cfg,pad,tail):
    seqs=sorted(coalitions,key=lambda s:(len(s),tuple(sorted(s))))
    # Reuse the original E8 keep-only + neutral filler + fixed readout function.
    val=helper._measure_v_fixed_tail(model,tok,ids,spans,seqs,h0,hbar,device,cfg,pad,
                                     {sp[0] for sp in spans},tail)
    if set(val)!=set(coalitions):
        raise ValueError('Measured coalition set mismatch')
    return val


def dividend(s, vals):
    """Exact Möbius inversion, works for any 1..3-player set if all subsets measured."""
    return float(sum((-1)**(len(s)-len(t))*vals[frozenset(t)]
              for k in range(len(s)+1) for t in itertools.combinations(sorted(s),k)))


def median_abs_z(vec):
    if not vec: return {}
    med=statistics.median(vec.values())
    mad=statistics.median(abs(v-med) for v in vec.values())
    floor=max(1.4826*mad,1e-6)
    return {k:float((v-med)/floor) for k,v in vec.items()}


def graph_seed_nodes(n,vals,limit):
    singles=sorted(range(n), key=lambda j: (-abs(dividend(frozenset([j]),vals)),j))
    adj=[frozenset([i,i+1]) for i in range(n-1)]
    pairs=sorted(adj,key=lambda s:(-abs(dividend(s,vals)),tuple(sorted(s))))
    tri=[frozenset([i,i+1,i+2]) for i in range(n-2)]
    triples=sorted(tri,key=lambda s:(-abs(dividend(s,vals)),tuple(sorted(s))))
    nodes=[]
    for group in [singles[:max(2,limit//2)],
                  [p for s in pairs[:max(2,limit//4)] for p in sorted(s)],
                  [p for s in triples[:max(1,limit//6)] for p in sorted(s)],
                  list(np.linspace(0,n-1,min(4,n),dtype=int))]:
        for i in group:
            if i not in nodes:
                nodes.append(int(i))
            if len(nodes)==limit: return nodes
    return nodes


def measure_game(helper,model,tok,text,hbar,device,cfg,pad,tail,max_players,graph_nodes):
    ids,spans=game_setup(helper,tok,text,cfg,tail,max_players)
    n=len(spans)
    full_vec=helper._forward(model,[ids+tail],device,cfg['hidden_layer'],pad,cfg['batch_size'])[0]
    vals=call_values(helper,model,tok,ids,spans,base_subsets(n),full_vec,hbar,device,cfg,pad,tail)
    nodes=graph_seed_nodes(n,vals,graph_nodes)
    extras={frozenset([i,j]) for i,j in itertools.combinations(sorted(nodes),2)}-set(vals)
    if extras:
        vals.update(call_values(helper,model,tok,ids,spans,extras,full_vec,hbar,device,cfg,pad,tail))
    return dict(ids=ids,spans=spans,vals=vals,n=n,nodes=nodes)


def features_from_game(game,loc,text,include_invalid=False):
    vals=game['vals']; n=game['n'];v0=vals[frozenset()]
    candidates=[]
    single_abs={}
    for s in sorted(vals,key=lambda x:(len(x),tuple(sorted(x)))):
        k=len(s)
        if not 1<=k<=3: continue
        indices=tuple(sorted(s))
        if k==3 and (indices[1]!=indices[0]+1 or indices[2]!=indices[1]+1):
            continue
        if k==2 and indices[1]-indices[0]>2 and not set(indices)<=set(game['nodes']):
            continue
        if not include_invalid and not all(loc[j]['valid'] for j in indices): continue
        if k==2 and indices[1]!=indices[0]+1 and indices[1]-indices[0]==2:
            # Only a supporting subset for triples unless also in graph seed edges.
            if not set(indices)<=set(game['nodes']): continue
        if k>=2 and all(indices[j+1]==indices[j]+1 for j in range(k-1)):
            if not contiguous_safe(indices,loc,text): continue
        raw_words=[loc[j]['raw'] for j in indices]
        contig=(k==1 or all(indices[j+1]==indices[j]+1 for j in range(k-1)))
        phrase=(text[loc[indices[0]]['offset'][0]:loc[indices[-1]]['offset'][1]]
                if contig and all(loc[i]['valid'] for i in indices) else ' '.join(raw_words))
        if '\n' in phrase or '\r' in phrase or not phrase.strip(): continue
        i_signed=dividend(s,vals)
        total=vals[s]-v0
        i1_sum=sum(dividend(frozenset([i]),vals) for i in indices)
        higher=float(total-i1_sum) if k>=2 else 0.
        kind='singleton' if k==1 else ('window'+str(k) if contig else 'graph_pair')
        rec=dict(indices=list(indices),size=k,channel=kind,words=raw_words,
                 phrase=phrase,raw_offsets=[loc[j]['offset'] for j in indices],
                 contiguous=contig,I_signed=float(i_signed),I_abs=abs(float(i_signed)),
                 total_signed=float(total),total_abs=abs(float(total)),
                 higher_order_sum_signed=higher,higher_order_sum_abs=abs(higher),
                 v_coalition=float(vals[s]),v_empty=float(v0),
                 interaction_type=('main_effect' if k==1 else ('pure_I2' if k==2 else 'pure_I3')),
                 finalist_rank=None)
        candidates.append(rec)
        if k==1: single_abs[indices[0]]=rec['I_abs']
    outlier=median_abs_z(single_abs)
    for c in candidates:
        c['singleton_robust_z']=outlier.get(c['indices'][0]) if c['size']==1 else None
    return candidates


def candidate_families(c):
    size=c['size']
    return [f'I{size}',f'G{size}'] if size>=2 else ['I1']


def family_features(c):
    k=c['size'];out={f'I{k}':c['I_abs']}
    if k>=2:out[f'G{k}']=c['total_abs']
    return out


def maxima_from_candidates(candidates):
    d={}
    for c in candidates:
        for key,value in family_features(c).items():
            d[key]=max(value,d.get(key,-float('inf')))
    return d


def build_null(cal_games,version,sha_input):
    keys=['I1','I2','G2','I3','G3']
    null={key:[] for key in keys}
    ignored={key:0 for key in keys}
    for group in cal_games:
        maxima=maxima_from_candidates(group)
        for key in keys:
            if key in maxima: null[key].append(float(maxima[key]))
            else:ignored[key]+=1
    return dict(version=version,calibration_hash=sha_input,maxima_by_family=null,
                missing_family_counts=ignored,
                warning='Calibration maximum distribution, used only for exploratory candidate ranking; not a new detector threshold or formal p-value')


def score_candidates(candidates,null):
    d=null['maxima_by_family']
    for c in candidates:
        parts=[]
        for fam,value in family_features(c).items():
            maxima=d.get(fam,[])
            z=robust_z(value,maxima)
            p=empirical_p(value,maxima)
            parts.append(dict(family=fam,value=value,calib_max_z=z,calib_max_tail=p))
        c['family_calibration']=parts
        # Larger signal is better; cross-family standardized to CLEAN CALIBRATION MAXIMA.
        c['search_score']=max(x['calib_max_z'] for x in parts)
        c['best_family']=max(parts,key=lambda x:x['calib_max_z'])['family']
        c['min_empirical_family_tail']=min(x['calib_max_tail'] for x in parts)
    return sorted(candidates,key=lambda c:(-c['search_score'],-c['I_abs'],len(c['indices']),c['indices']))


def choose_finalists(ranked,k):
    """Retain a singleton, contiguous pair, contiguous triple, graph pair if available.
    Rank uses only unsupervised clean-calibration standardized features.
    """
    picked=[];seen=set()
    by_channel=['singleton','window2','window3','graph_pair']
    for channel in by_channel:
        one=next((c for c in ranked if c['channel']==channel and tuple(c['indices']) not in seen),None)
        if one is not None and len(picked)<k:
            picked.append(one);seen.add(tuple(one['indices']))
    for c in ranked:
        if len(picked)>=k:break
        t=tuple(c['indices'])
        if t not in seen: picked.append(c);seen.add(t)
    for i,c in enumerate(picked):c['finalist_rank']=i+1
    return picked


def remove_selected(text,spans):
    """Remove ONLY originally selected character spans, in reverse order."""
    pairs=sorted((tuple(s) for s in spans),reverse=True)
    if any(not(0<=a<b<=len(text)) for a,b in pairs):
        raise ValueError('Invalid span offsets')
    if any(pairs[j+1][1]>pairs[j][0] for j in range(len(pairs)-1)):
        raise ValueError('Overlapping span offsets')
    new=text
    for span in pairs:
        new=u.remove_span(new,span)
    if not text[:1].isspace(): new=new.lstrip(' ')
    if not text[-1:].isspace(): new=new.rstrip(' ')
    if not new.strip(): raise ValueError('Removal yields empty prompt')
    return new


def replace_selected(text,spans,control):
    pairs=sorted((tuple(s) for s in spans),reverse=True)
    if any(not(0<=a<b<=len(text)) for a,b in pairs):
        raise ValueError('Invalid replacement span offsets')
    if any(pairs[j+1][1]>pairs[j][0] for j in range(len(pairs)-1)):
        raise ValueError('Overlapping replacement offsets')
    res=text
    for span in pairs:
        res=u.replace_span(res,span,control)
    return res


def controls_for(tok,phrase,size):
    neutral=' '.join(['thing']*size)
    ordinary=' '.join(['garden']*size)
    target=len(tok(' '+phrase,add_special_tokens=False)['input_ids'])
    # Choose a rare multiword control with closest token count, preferring exact.
    candidates=list(u.RARE_CANDIDATES)
    options=[]
    for i in range(len(candidates)):
        if size==1: options.append(candidates[i])
        elif size==2: options.append(candidates[i]+' '+candidates[(i+1)%len(candidates)])
        else: options.append(' '.join(candidates[(i+j)%len(candidates)] for j in range(size)))
    acceptable=[s for s in options if u.canonical(s)!=u.canonical(phrase)]
    if not acceptable: raise ValueError('No valid rare control')
    rare=min(acceptable,key=lambda s:(abs(len(tok(' '+s,add_special_tokens=False)['input_ids'])-target),s))
    counts={k:len(tok(' '+v,add_special_tokens=False)['input_ids'])
            for k,v in dict(candidate=phrase,neutral=neutral,ordinary=ordinary,rare=rare).items()}
    return dict(candidate=phrase,neutral=neutral,ordinary=ordinary,rare=rare),counts


def cached_reference(model,tok,helper,phrase,words,size,contexts,device,cfg,pad,tail,cache,args):
    key=(phrase,tuple(words),size)
    if key in cache:
        cache.move_to_end(key)
        return cache[key],True
    refs=u.pick_contexts(contexts,phrase,args.n_contexts)
    arms,counts=controls_for(tok,phrase,size)
    seqs=[]
    for ctx in refs:
        seqs.append(ctx)
        for w in arms.values(): seqs.append(u.insert_same_site(ctx,w))
    for text in seqs:u.tokenized_readout(helper,tok,text,tail,cfg['max_len'])
    H=u.hidden(helper,model,tok,seqs,tail,device,cfg,pad).astype(np.float64)
    by_arm=defaultdict(list)
    for j in range(len(refs)):
        base=H[5*j]
        for k,arm in enumerate(arms):
            by_arm[arm].append(H[5*j+1+k]-base)
    info=dict(refs=refs,arms=arms,counts=counts,by_arm=dict(by_arm),
              norm_medians={arm:u.median([u.norm(v) for v in vs]) for arm,vs in by_arm.items()},
              coherence={arm:u.coherence(vs) for arm,vs in by_arm.items()},
              neutral_extra=u.median([u.norm(a)-u.norm(b) for a,b in zip(by_arm['candidate'],by_arm['neutral'])]))
    # Subset counterfactuals: a group may be carried entirely by ONE strong singleton.
    # Test all nonempty proper subcoalitions on separate fixed clean contexts.
    if size>=2:
        proper=[z for k in range(1,size) for z in itertools.combinations(range(size),k)]
        sr=refs[:min(args.subset_contexts,len(refs))]
        seqs=[u.insert_same_site(ctx,' '.join(words[i] for i in z)) for ctx in sr for z in proper]
        for ss in seqs:u.tokenized_readout(helper,tok,ss,tail,cfg['max_len'])
        Hsub=u.hidden(helper,model,tok,seqs,tail,device,cfg,pad).astype(np.float64)
        # Clean base was computed in earlier block; obtain same reference-base vectors.
        base_vectors=[H[5*j] for j in range(len(sr))]
        subset_norms=defaultdict(list)
        synergistic=[];best_singleton_gap=[]
        for j,ctx in enumerate(sr):
            delta={z:Hsub[j*len(proper)+q]-base_vectors[j] for q,z in enumerate(proper)}
            for z in proper:subset_norms['+'.join(str(i) for i in z)].append(u.norm(delta[z]))
            singleton_sum=sum((delta[(i,)] for i in range(size)),np.zeros_like(base_vectors[j]))
            full=by_arm['candidate'][j]
            synergistic.append(u.norm(full-singleton_sum))
            best_singleton_gap.append(u.norm(full)-max(u.norm(delta[(i,)]) for i in range(size)))
        info['subset_counterfactual']=dict(
            subset_names={'+' .join(str(i) for i in z):' '.join(words[i] for i in z) for z in proper},
            subset_median_norm={k:u.median(v) for k,v in subset_norms.items()},
            full_minus_best_singleton_norm_median=u.median(best_singleton_gap),
            nonadditive_vector_residual_norm_median=u.median(synergistic),
            n_reference_contexts=len(sr),
            warning='Representation residual is not a proof of conjunctive attack activation')
    else:
        info['subset_counterfactual']=None
    cache[key]=info;cache.move_to_end(key)
    while len(cache)>args.cache_size:
        cache.popitem(last=False)
    return info,False


def stage2(model,tok,helper,text,c,refs,device,cfg,pad,tail,args):
    spans=c['raw_offsets']
    removed=remove_selected(text,spans)
    neutralized=replace_selected(text,spans,'thing')
    for s in [text,removed,neutralized]:u.tokenized_readout(helper,tok,s,tail,cfg['max_len'])
    H=u.hidden(helper,model,tok,[text,removed,neutralized],tail,device,cfg,pad).astype(np.float64)
    rem=H[0]-H[1]; neu=H[0]-H[2]
    align={key:u.median([u.cos(rem,v) for v in vecs]) for key,vecs in refs['by_arm'].items()}
    neu_align={key:u.median([u.cos(neu,v) for v in vecs]) for key,vecs in refs['by_arm'].items()}
    return dict(removed_instruction=removed,neutralized_instruction=neutralized,
                insert_magnitude_medians=refs['norm_medians'],
                insert_coherence=refs['coherence'],remove_norm=u.norm(rem),neutralize_norm=u.norm(neu),
                remove_insert_alignment=align,neutralize_insert_alignment=neu_align,
                insert_extra_neutral=refs['neutral_extra'],
                controls=refs['arms'],token_counts=refs['counts'],
                rare_token_count_matched=refs['counts']['candidate']==refs['counts']['rare'],
                n_contexts=len(refs['refs']),subset_counterfactual=refs.get('subset_counterfactual'),
                warning=('For noncontiguous graphs, inserted coalition is concatenated phrase; '
                         'removal deletes separate original positions; two interventions differ structurally.'
                         if not c['contiguous'] else None))


def base_record(idx,e8row,original,cutoff):
    score=e8row.get('score')
    pred=int(score is not None and not e8row.get('skipped') and float(score)>cutoff)
    gt=int(e8row['gt'])
    cohort=('TP' if pred else 'FN') if gt else ('FP' if pred else 'TN')
    return dict(sample_id=idx,true_label=gt,cohort=cohort,e8_prediction=pred,
                e8_score=scalar(score),e8_cutoff=cutoff,
                e8_winning_word=e8row.get('winning_word'),
                e8_top6_pool=e8row.get('pool'),
                e8_top6_abs=e8row.get('singleton_abs'),
                e8_skipped=e8row.get('skipped'),
                original_instruction=original,
                search_status='not_started',stage2_status='not_started',
                n_players=None,graph_nodes=[],eligible_word_count=None,
                shortlist=[],all_candidates=[],finalists=[],reason=None)


def analyze_one(model,tok,helper,text,rec,hbar,cal,contexts,device,cfg,pad,tail,cache,args):
    g=measure_game(helper,model,tok,text,hbar,device,cfg,pad,tail,args.max_players,args.graph_nodes)
    loc=safe_word_offsets(tok,text,g['ids'],g['spans'])
    cs=features_from_game(g,loc,text)
    if not cs: raise ValueError('No safe candidate coalitions')
    ranked=score_candidates(cs,cal)
    finalists=choose_finalists(ranked,args.finalists)
    rec['search_status']='ok';rec['n_players']=g['n']
    rec['eligible_word_count']=sum(x['valid'] for x in loc)
    rec['graph_nodes']=g['nodes']
    rec['all_candidates']=[dict(c) for c in ranked]
    rec['shortlist']=[dict(c) for c in ranked[:min(20,len(ranked))]]
    rec['finalists']=[]
    for c in finalists:
        one=dict(candidate=dict(c),stage2_status='not_started',stage2=None)
        try:
            ref,hit=cached_reference(model,tok,helper,c['phrase'],c['words'],c['size'],contexts,device,cfg,pad,tail,cache,args)
            one['stage2']=stage2(model,tok,helper,text,c,ref,device,cfg,pad,tail,args)
            one['stage2_status']='ok';one['cache_reused']=hit
        except (ValueError,IndexError,KeyError,AssertionError) as e:
            one['stage2_status']='skipped';one['error']=str(e)[:450]
        rec['finalists'].append(one)
    rec['stage2_status']='ok' if any(x['stage2_status']=='ok' for x in rec['finalists']) else 'all_skipped'
    return rec


def output_row(rec):
    cs=rec.get('all_candidates') or []
    finalists=rec.get('finalists') or []
    def best(channel):
        r=next((x for x in cs if x['channel']==channel),None)
        return r or {}
    def fval(obj,key):return obj.get(key) if obj else None
    bestf=next((x for x in finalists if x['stage2_status']=='ok'),None)
    bestc=bestf['candidate'] if bestf else (cs[0] if cs else {})
    ev=bestf.get('stage2') if bestf else {}
    return dict(sample_id=rec['sample_id'],true_label=rec['true_label'],cohort=rec['cohort'],
                e8_prediction=rec['e8_prediction'],e8_score=rec['e8_score'],
                e8_winning_word=rec['e8_winning_word'],search_status=rec['search_status'],
                stage2_status=rec['stage2_status'],reason=rec.get('reason'),n_players=rec['n_players'],
                n_candidates=len(cs),n_finalists=len(finalists),
                top_coalition=bestc.get('phrase'),top_coalition_indices=bestc.get('indices'),
                top_coalition_size=bestc.get('size'),top_coalition_channel=bestc.get('channel'),
                top_coalition_I_signed=bestc.get('I_signed'),top_coalition_total_signed=bestc.get('total_signed'),
                top_coalition_higher_sum=bestc.get('higher_order_sum_signed'),
                top_coalition_calib_z=bestc.get('search_score'),
                best_singleton=best('singleton').get('phrase'),best_pair=best('window2').get('phrase'),
                best_triple=best('window3').get('phrase'),best_graph_pair=best('graph_pair').get('phrase'),
                top_insert_norm=(ev.get('insert_magnitude_medians') or {}).get('candidate'),
                top_insert_coherence=(ev.get('insert_coherence') or {}).get('candidate'),
                top_remove_insert_alignment=(ev.get('remove_insert_alignment') or {}).get('candidate'),
                top_neutral_insert_alignment=(ev.get('remove_insert_alignment') or {}).get('neutral'),
                top_insert_extra_neutral=ev.get('insert_extra_neutral'),
                top_subset_full_minus_best_singleton=(ev.get('subset_counterfactual') or {}).get('full_minus_best_singleton_norm_median'),
                top_subset_nonadditive_residual=(ev.get('subset_counterfactual') or {}).get('nonadditive_vector_residual_norm_median'),
                all_top_candidates_json=rec.get('shortlist'),
                all_finalists_json=[dict(candidate=x['candidate'],stage2_status=x['stage2_status'],
                                       alignment=((x.get('stage2') or {}).get('remove_insert_alignment') or {}).get('candidate'),
                                       error=x.get('error')) for x in finalists],
                original_instruction=rec['original_instruction'])


def write_csv(path,records):
    if not records: raise ValueError('No rows to export')
    fields=list(records[0].keys())
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for r in records:wr.writerow({k:csv_guard(v) for k,v in r.items()})


def flatten_candidates(records):
    for r in records:
        for j,c in enumerate(r['all_candidates']):
            yield dict(sample_id=r['sample_id'],true_label=r['true_label'],cohort=r['cohort'],
                       local_rank=j+1,**{k:v for k,v in c.items() if k!='family_calibration'},
                       family_calibration=c['family_calibration'])


def flatten_finalists(records):
    for r in records:
        for e in r['finalists']:
            c=e['candidate'];v=e.get('stage2') or {}
            yield dict(sample_id=r['sample_id'],true_label=r['true_label'],cohort=r['cohort'],
                       finalist_rank=c['finalist_rank'],candidate=c['phrase'],size=c['size'],
                       indices=c['indices'],channel=c['channel'],I_signed=c['I_signed'],
                       total_signed=c['total_signed'],higher_order_sum_signed=c['higher_order_sum_signed'],
                       calib_max_z=c['search_score'],stage2_status=e['stage2_status'],
                       insert_norm=(v.get('insert_magnitude_medians') or {}).get('candidate'),
                       neutral_insert_norm=(v.get('insert_magnitude_medians') or {}).get('neutral'),
                       insert_direction_coherence=(v.get('insert_coherence') or {}).get('candidate'),
                       remove_insert_alignment=(v.get('remove_insert_alignment') or {}).get('candidate'),
                       remove_neutral_alignment=(v.get('remove_insert_alignment') or {}).get('neutral'),
                       insert_extra_neutral=v.get('insert_extra_neutral'),
                       subset_full_minus_best_singleton=(v.get('subset_counterfactual') or {}).get('full_minus_best_singleton_norm_median'),
                       subset_nonadditive_residual=(v.get('subset_counterfactual') or {}).get('nonadditive_vector_residual_norm_median'),
                       subset_details=(v.get('subset_counterfactual')),
                       rare_token_count_matched=v.get('rare_token_count_matched'),
                       removed_instruction=v.get('removed_instruction'),
                       error=e.get('error'))


def resumable_rows(path):
    if not path.exists():return []
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if len(rows)>400 or [r['sample_id'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Invalid ordered partial E12 JSONL')
    return rows


def config_manifest(args,p):
    return dict(version=V,
         e8_summary_sha256=u.sha(p['e8_dir']/'summary.json'),
         e8_records_sha256=u.sha(p['e8_dir']/'heldout_records.json'),
         helper_sha256=u.sha(args.helper),contexts_sha256=u.sha(args.contexts),
         model_path=args.base,adapter_path=args.adapter,
         max_len=args.max_len, batch_size=args.batch_size,
         max_players=args.max_players,graph_nodes=args.graph_nodes,
         finalists=args.finalists,n_contexts=args.n_contexts,n_cal=args.n_cal,
         subset_contexts=args.subset_contexts,n_cal_contexts=args.n_cal_contexts,
         calibration_indices=p['calibration_indices'],
         calibration_text_sha256=hashlib.sha256('\n'.join(p['calibration']).encode()).hexdigest(),
         objective='Full 400 heldout, unsupervised variable-size coalition search, no new detector threshold',
         label_policy='Ground truth labels used only in offline exports after label-blind search',
         known_trigger_or_target_used=False, max_coalition_size=3,
         second_stage_reference_split='first n_cal_contexts reserved, subsequent n_contexts for insertion verification',
         readout=TAIL)


def print_summary(rows,out,cal,args,p):
    bycohort={g:[r for r in rows if r['cohort']==g] for g in ['TP','FP','FN','TN']}
    shortlist=Counter();valid=Counter();stage=Counter()
    for row in rows:
        valid[row['search_status']]+=1;stage[row['stage2_status']]+=1
        for c in row['finalists']:
            shortlist[(c['candidate']['size'],c['candidate']['channel'])]+=1
    counts=[]
    for g,cases in bycohort.items():
        good=[x for x in cases if x['stage2_status']=='ok']
        vals=[z['stage2']['remove_insert_alignment']['candidate']
              for row in good for z in row['finalists']
              if z['stage2_status']=='ok']
        uniq=len({z['candidate']['phrase'] for row in good for z in row['finalists']})
        counts.append(f'{g}=n:{len(cases)}, stage2_ok:{len(good)}, finalist_types:{uniq}, '
                      f'candidate_alignment_median:{u.median(vals) if vals else None}')
    copy=['========== COPY_TO_CHATGPT_BEGIN ==========',f'experiment={V}','status=SUCCESS',
          'dataset=E8_heldout_400 (200 poisoned + 200 clean)',
          'search_uses_known_trigger=false; search_uses_known_target=false; search_uses_GT=false',
          f'search_status_counts={json.dumps(dict(valid),ensure_ascii=False)}',
          f'stage2_status_counts={json.dumps(dict(stage),ensure_ascii=False)}',
          f'finalist_channel_counts={json.dumps({str(k):v for k,v in shortlist.items()},ensure_ascii=False)}',
          f'calibration_n={args.n_cal}; calibration_source=E8_clean_cal_400_only; n_contexts={args.n_contexts}',
          *counts,
          f'all_samples_file={out/ALL_CSV}',f'poisoned_samples_file={out/POISON_CSV}',
          f'all_interactions_file={out/CANDIDATES_CSV}',f'poisoned_interactions_file={out/POISON_CANDIDATES_CSV}',
          f'counterfactual_finalists_file={out/FINALISTS_CSV}',
          'notes=E8_FROZEN; EXPLORATORY; NO_NEW_THRESHOLD; NO_NEW_F1; ORIGINAL_BADNET_NOT_MULTIPHRASE_VALIDATION',
          '========== COPY_TO_CHATGPT_END ==========']
    message='\n'.join(copy)+'\n'
    (out/'copy_to_chatgpt.txt').write_text(message,encoding='utf-8')
    summary=dict(version=V,n_total=len(rows),n_poison=200,n_clean=200,
                 search_status_counts=dict(valid),stage2_status_counts=dict(stage),
                 finalist_channel_counts={str(k):v for k,v in shortlist.items()},
                 by_cohort={g:dict(n=len(cases),stage2_ok=sum(r['stage2_status']=='ok' for r in cases))
                            for g,cases in bycohort.items()},
                 clean_calibration_maxima_n={k:len(v) for k,v in cal['maxima_by_family'].items()},
                 known_limitations=['No true multi-word backdoor benchmark included',
                                   'No independent heldout evaluation of tuned second-stage thresholds',
                                   'E8 heldout reused descriptively; labels not used in search',
                                   'Continuous triple and sparse seed-pair graph do not guarantee all combinatorial triggers',
                                   'Same inserted phrase reuses references via cache; dependent observations',
                                   'No generation or attack-target measurement by default'])
    u.save_json(out/'summary.json',summary)
    (out/'report.md').write_text('# E12R1 variable-size interaction fullset\n\n'+
          'Exploratory 1..3-word Harsanyi coalition discovery, separate clean-calibration family maxima, '+
          'label-blind ranking and same-model counterfactual insert/remove controls.\n\n'+
          '## E8 frozen cohort summaries\n\n'+ '\n'.join('- '+x for x in counts)+ '\n\n'+
          '## Caution\n\nNo new detector, calibrated FPR, or independent F1 is claimed. '+
          'A strong coalition may merely contain a strong singleton, not constitute AND-like synergy. '+
          'Absolute high-order interaction can be negative. Graph pairs are sparse. '+
          'The true BadNet attack in these 400 inputs is not a multiword trigger validation.\n',encoding='utf-8')
    print(message,flush=True)
    print(f'[E12 SUCCESS] saved to {out.resolve()}')
    print(f'[E12] Re-copy: cat "{(out/"copy_to_chatgpt.txt").resolve()}"')


def run(args,p):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    out=Path(args.resume or args.output)
    ensure_free_dir(out,bool(args.resume))
    man=config_manifest(args,p)
    if args.resume:
        old=u.load_json(out/'manifest.json')
        for k,v in man.items():
            if old.get(k)!=v: raise ValueError(f'Cannot resume with changed manifest field: {k}')
    else:
        u.save_json(out/'manifest.json',dict(man,created_utc=datetime.now(timezone.utc).isoformat()))
    rows=resumable_rows(out/RECORD_FILE)
    if len(rows)==400:raise ValueError('Full run already complete; do not overwrite')
    helper=u.load_helper(args.helper)
    cfg=dict(helper.DEFAULTS)
    cfg['max_len']=args.max_len;cfg['hidden_layer']=-1;cfg['batch_size']=args.batch_size
    tok=AutoTokenizer.from_pretrained(args.base,use_fast=True)
    if not tok.is_fast:raise RuntimeError('Fast tokenizer required for offsets')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tail=tok(TAIL,add_special_tokens=False)['input_ids']
    if len(tail)!=4:raise ValueError('Readout suffix expected 4 tokens')
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,args.adapter)
    model.eval();device=next(model.parameters()).device;pad=tok.pad_token_id
    ref_text=p['data']['train_poison'][:30]+p['data']['train_clean'][:100]
    hbar_ids=[helper._compose(tok,s,cfg['max_len']-len(tail))[0]+tail for s in ref_text]
    hbar=helper._forward(model,hbar_ids,device,cfg['hidden_layer'],pad,cfg['batch_size']).mean(axis=0)
    print('[E12] Frozen E8 same-model reference center reconstructed (130 train-origin prompts)',flush=True)
    if args.resume and (out/CALIBRATION_FILE).is_file():
        cal=u.load_json(out/CALIBRATION_FILE)
        if cal.get('calibration_hash')!=man['calibration_text_sha256']:
            raise ValueError('Resume calibration hash mismatch')
    else:
        # If interrupted during calibration, recompute it from frozen clean-cal
        # instead of requiring a nonexistent partial calibration file.
        cal_samples=[];cal_skips=[]
        for j,text in enumerate(p['calibration']):
            try:
                g=measure_game(helper,model,tok,text,hbar,device,cfg,pad,tail,args.max_players,args.graph_nodes)
                loc=safe_word_offsets(tok,text,g['ids'],g['spans'])
                cal_samples.append(features_from_game(g,loc,text))
            except (ValueError,IndexError,AssertionError,KeyError) as e:
                cal_skips.append(dict(index=j,reason=str(e)[:250]))
            if (j+1)%8==0:print(f'[E12] clean calibration {j+1}/{args.n_cal}',flush=True)
        if len(cal_samples)<max(12,int(args.n_cal*.75)):
            raise ValueError(f'Insufficient valid clean calibration contexts: {len(cal_samples)}/{args.n_cal}')
        cal=build_null(cal_samples,V,man['calibration_text_sha256']);cal['calibration_skips']=cal_skips
        u.save_json(out/CALIBRATION_FILE,cal)
    # The E12 contexts are disjoint: first block reserved for independent references;
    # only the latter is used in Stage 2. Calibration of search is E8 clean-cal data.
    contexts=p['references'][args.n_cal_contexts:args.n_cal_contexts+args.n_contexts]
    cache=OrderedDict()
    print(f'[E12] Heldout 400; resumed={len(rows)}; A100 batch={args.batch_size}; '+
          f'calibration={args.n_cal}; stage2_ref={len(contexts)}; finalists={args.finalists}',flush=True)
    with (out/RECORD_FILE).open('a',encoding='utf-8') as stream:
        for idx in range(len(rows),400):
            r=p['records'][idx]
            text=(p['data']['test_poison'][idx] if idx<200 else p['data']['test_clean'][idx-200])
            row=base_record(idx,r,text,p['cutoff'])
            try:
                analyze_one(model,tok,helper,text,row,hbar,cal,contexts,device,cfg,pad,tail,cache,args)
            except (ValueError,IndexError,AssertionError,KeyError) as e:
                row['search_status']='skipped';row['stage2_status']='skipped'
                row['reason']=str(e)[:500]
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            rows.append(row)
            if (idx+1)%args.progress_every==0 or idx==399:
                print(f'[E12] {idx+1}/400 cohort={row["cohort"]} search={row["search_status"]} '+
                      f'stage2={row["stage2_status"]} finalists={len(row["finalists"])}',flush=True)
    if [x['sample_id'] for x in rows]!=list(range(400)):
        raise ValueError('Cannot finalize noncomplete dataset')
    as_rows=[output_row(r) for r in rows]
    write_csv(out/ALL_CSV,as_rows)
    write_csv(out/POISON_CSV,[r for r in as_rows if r['true_label']==1])
    all_candidates=list(flatten_candidates(rows));all_finalists=list(flatten_finalists(rows))
    write_csv(out/CANDIDATES_CSV,all_candidates or [dict(empty=True)])
    write_csv(out/POISON_CANDIDATES_CSV,[r for r in all_candidates if r['true_label']==1] or [dict(empty=True)])
    write_csv(out/FINALISTS_CSV,all_finalists or [dict(empty=True)])
    print_summary(rows,out,cal,args,p)


def parse_args():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--preflight',action='store_true')
    a.add_argument('--output');a.add_argument('--resume');a.add_argument('--e8-run')
    a.add_argument('--base',default=u.BASE);a.add_argument('--adapter',default=u.ADAPTER)
    a.add_argument('--helper',default=u.HELPER)
    a.add_argument('--train-poison',default=u.TRAIN_POISON)
    a.add_argument('--train-clean',default=u.TRAIN_CLEAN)
    a.add_argument('--test-poison',default=u.TEST_POISON)
    a.add_argument('--test-clean',default=u.TEST_CLEAN)
    a.add_argument('--contexts',default=CONTEXTS)
    a.add_argument('--n-cal',type=int,default=48)
    a.add_argument('--n-contexts',type=int,default=8)
    a.add_argument('--n-cal-contexts',type=int,default=8)
    a.add_argument('--subset-contexts',type=int,default=4)
    a.add_argument('--batch-size',type=int,default=8)
    a.add_argument('--max-len',type=int,default=512)
    a.add_argument('--max-players',type=int,default=256)
    a.add_argument('--graph-nodes',type=int,default=12)
    a.add_argument('--finalists',type=int,default=4)
    a.add_argument('--cache-size',type=int,default=128)
    a.add_argument('--progress-every',type=int,default=10)
    return a.parse_args()


def main():
    a=parse_args()
    if bool(a.output)==bool(a.resume) and not a.preflight:
        raise ValueError('Specify exactly one of --output/--resume')
    if a.n_contexts<4 or a.n_cal_contexts<4 or a.n_cal_contexts+a.n_contexts>16:
        raise ValueError('Use disjoint 8+8 independent reference contexts')
    if a.graph_nodes<4 or a.finalists<3 or a.cache_size<1 or a.progress_every<1 or not 1<=a.subset_contexts<=a.n_contexts:
        raise ValueError('Invalid search/execution limits')
    p=check_inputs(a)
    if a.preflight:
        print('[PREFLIGHT OK] E8 frozen summary, 400 archived records, original dataset hashes, heldout/cal/reference separation')
        return
    run(a,p)


if __name__=='__main__':
    main()
