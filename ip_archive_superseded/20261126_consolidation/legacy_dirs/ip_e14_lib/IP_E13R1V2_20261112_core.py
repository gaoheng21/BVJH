#!/usr/bin/env python3
"""Pure-CPU, label-blind functions for interaction-guided behavioral counterfactuals."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

VERSION = 'IP_E13R1V2_20261112'
TAIL = '\nAnswer:'


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read_jsonl(path):
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():result.append(json.loads(line))
    return result


def canonical(s):
    return ' '.join(s.split()).casefold()


def signature(s):
    """Strict normalized whole answer. Never match just one common word."""
    t=unicodedata.normalize('NFKC',str(s or '')).strip().casefold()
    t=re.sub(r'\s+',' ',t)
    return t.strip(' \t\n\r')


def possible_fixed_signature(outputs):
    """Discovery set ONLY. Require two different prompts with identical substantive outputs."""
    if len(outputs)<2:return None
    vals=[signature(x) for x in outputs]
    if any(len(x)<8 or len(x)>320 for x in vals):return None
    if len(set(vals))!=1:return None
    if len(vals[0].split())<2:return None
    return vals[0]


def valid_spans(text, spans, words):
    if len(spans)!=len(words) or not spans:raise ValueError('Mismatch candidate offsets/words')
    prev_end=-1
    for (a,b),w in zip(spans,words):
        if not isinstance(a,int) or not isinstance(b,int) or not (0<=a<b<=len(text)):
            raise ValueError('Unsafe candidate span')
        if a<prev_end:raise ValueError('Overlapping or unordered spans')
        if canonical(text[a:b])!=canonical(w):
            raise ValueError(f'Candidate raw span does not match source: {text[a:b]!r} vs {w!r}')
        prev_end=b


def remove_spans(text,spans,words):
    valid_spans(text,spans,words)
    out=text
    for a,b in reversed(spans):
        if a>0 and b<len(out) and out[a-1]==' ' and out[b]==' ':
            out=out[:a]+out[b+1:]
        else:out=out[:a]+out[b:]
    if not text[:1].isspace():out=out.lstrip(' ')
    if not text[-1:].isspace():out=out.rstrip(' ')
    if not out.strip():raise ValueError('Removal emptied prompt')
    return out


def replace_spans(text,spans,words,replacement='thing'):
    valid_spans(text,spans,words)
    out=text
    for a,b in reversed(spans):out=out[:a]+replacement+out[b:]
    return out


def insert_at_fixed_site(text, phrase):
    if not phrase or not phrase.strip() or '\n' in phrase or '\r' in phrase:
        raise ValueError('Invalid insertion phrase')
    m=re.search(r'\S+\s+',text)
    if m is None:raise ValueError('Reference context lacks stable insertion site')
    return text[:m.end()]+phrase+' '+text[m.end():]


def all_subsets(words):
    n=len(words)
    if n not in (1,2,3):raise ValueError('Only 1-3 player coalitions supported')
    return {bits:' '.join(words[i] for i in bits)
        for k in range(1,n+1) for bits in itertools.combinations(range(n),k)}


def dividend_behavior(hits, n):
    """Exact Harsanyi/Mobius dividend for empirical behavioral hit rates."""
    if set(hits)!={t for k in range(n+1) for t in itertools.combinations(range(n),k)}:
        raise ValueError('All coalition subsets, including empty, required')
    if any(not 0<=float(v)<=1 for v in hits.values()):raise ValueError('Invalid hit rate')
    grand=tuple(range(n))
    return sum(((-1)**(n-len(t)))*float(hits[t]) for t in hits if set(t).issubset(grand))


def route(e12,has_candidate):
    """ONLY E12R3 predicted label controls computation; never ground truth or E8."""
    if int(e12)==0:return 'SKIP_E12_NEGATIVE'
    if not has_candidate:return 'NO_VALID_CANDIDATE'
    return 'EXPLAIN_E12_POSITIVE'


def check_candidate(row, csv_row):
    """Exactly E12R3 max-scoring *verified* finalist, not an unverified global search item."""
    choices=[]
    for z in row.get('finalists') or []:
        if z.get('stage2_status')!='ok':continue
        a=(z.get('stage2') or {}).get('remove_insert_alignment') or {}
        try:
            c=float(a.get('candidate'));n=float(a.get('neutral'))
        except (TypeError,ValueError):continue
        if math.isfinite(c) and math.isfinite(n):choices.append((c-n,z))
    if not choices:return None
    score,finalist=max(choices,key=lambda x:x[0])
    c=finalist['candidate']
    source=row['original_instruction']
    words=c['words'];spans=c['raw_offsets']
    if not isinstance(words,list) or len(words) not in (1,2,3):raise ValueError('Invalid coalition size')
    valid_spans(source,spans,words)
    if int(csv_row['selected_coalition_size'])!=len(words) or canonical(csv_row['selected_coalition'])!=canonical(c['phrase']):
        raise ValueError('E12R3 selected candidate differs from verified E12R1 finalist')
    saved=float(csv_row['E12R3_score'])
    if not math.isclose(saved,score,abs_tol=1e-5):raise ValueError('E12R3 score mismatch')
    return {'phrase':c['phrase'],'words':words,'raw_offsets':spans,'size':c['size'],
            'channel':c['channel'],'I_signed':c['I_signed'],'total_signed':c['total_signed'],
            'search_score':c['search_score'],'e12_score':score,'contiguous':c['contiguous']}


def analyze_fixed_behavior(case, discovery, validation, controls, subsets, original, removed):
    """Signature chosen on discovery only. Validate on disjoint prompts and subset controls."""
    sig=possible_fixed_signature(discovery['inserted'])
    result={'fixed_signature':sig,'discovered':bool(sig),'validated':False,
        'removal_restored':False,'validation_insert_hits':0,'validation_n':len(validation['inserted']),
        'baseline_hits':0,'discovery_baseline_hits':0,'control_hits':{},'behavior_dividend':None,
        'subset_hit_rates':{},'full_only_activation':None,
        'behavior_type':'NOT_FOUND','evidence_scope':'NO_FIXED_OUTPUT_PATTERN'}
    if sig is None:return result
    same=lambda x:int(signature(x)==sig)
    full=[same(s) for s in validation['inserted']]
    base=[same(s) for s in validation['baseline']]
    ctrl={key:sum(same(z) for z in outputs) for key,outputs in controls.items()}
    discovery_base=sum(same(s) for s in discovery.get('baseline',[]))
    result.update(validation_insert_hits=sum(full),baseline_hits=sum(base),discovery_baseline_hits=discovery_base,control_hits=ctrl,
       original_has_signature=bool(same(original)),removed_has_signature=bool(same(removed)))
    result['removal_restored']=bool(same(original) and not same(removed))
    # Require 2/2 independent validation contexts, plus clean baseline+neutral+ordinary negatives.
    passed=(len(full)>=2 and all(full) and not any(base) and discovery_base==0 and set(controls)=={'neutral','ordinary'} and all(len(z)==len(full) for z in controls.values()) and all(v==0 for v in ctrl.values()))
    result['validated']=passed
    result['behavior_type']='CROSS_CONTEXT_FIXED_OUTPUT' if passed else 'UNCONFIRMED_FIXED_PATTERN'
    result['evidence_scope']='CROSS_CONTEXT_AND_REMOVAL' if passed and result['removal_restored'] else ('CROSS_CONTEXT_ONLY' if passed else 'DISCOVERY_ONLY')
    if not passed:return result
    n=int(case['size'])
    subset_hits={tuple(k):sum(same(z) for z in outs)/len(outs)
                 for k,outs in subsets.items() if outs}
    if n>=2:
        expected={tuple(t) for k in range(1,n) for t in itertools.combinations(range(n),k)}
        if set(subset_hits)==expected:
            b={():sum(base)/len(base),tuple(range(n)):sum(full)/len(full),**subset_hits}
            result['behavior_dividend']=round(dividend_behavior(b,n),6)
            result['subset_hit_rates']={'/'.join(str(i) for i in k):v for k,v in sorted(subset_hits.items())}
            result['full_only_activation']=bool(all(v==0 for v in subset_hits.values()))
    else:
        result['behavior_dividend']=round(sum(full)/len(full)-sum(base)/len(base),6)
        result['full_only_activation']=True
    return result


def csv_write(path,rows,fields=None):
    if not rows and not fields:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else fields);w.writeheader()
        for r in rows:
            clean={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()}
            clean={k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in clean.items()}
            w.writerow(clean)

