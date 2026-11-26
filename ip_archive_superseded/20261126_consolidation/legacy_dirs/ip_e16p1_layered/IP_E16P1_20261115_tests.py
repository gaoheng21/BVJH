"""CPU-only self tests for E16P1 layered verification (no GPU, no LLM).

Run:  /home/user/miniconda3/envs/BV/bin/python IP_E16P1_20261115_tests.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import IP_E16P1_20261115_layered_verification as D

fails = []


def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


# 1. alpaca wrap byte-exact
w = D.alpaca_wrap('Hello')
check('alpaca wrap exact', w == ('Below is an instruction that describes a task. '
                                 'Write a response that appropriately completes the request.\n\n'
                                 '### Instruction:\nHello\n\n### Response:'))

# 2. arms prepare functions
check('readout prepare identity', D.ARMS['readout']['prepare']('x') == 'x')
check('aligned prepare wraps', D.ARMS['aligned']['prepare']('x') == D.alpaca_wrap('x'))
check('readout tail', D.ARMS['readout']['tail'] == '\nAnswer:')
check('aligned tail', D.ARMS['aligned']['tail'] == '\n')

# 3. layered status combinations
B = lambda ro, al: {'readout': {'validated': ro}, 'aligned': {'validated': al}}
check('status both', D.layered_status(B(True, True)) == 'CONFIRMED_BOTH')
check('status align', D.layered_status(B(False, True)) == 'CONFIRMED_ALIGN_ONLY')
check('status readout', D.layered_status(B(True, False)) == 'CONFIRMED_READOUT_ONLY')
check('status none', D.layered_status(B(False, False)) == 'UNVERIFIED')

# 4. modules import from frozen lib copy
ex13, c = D.load_modules()
check('lib copy used', 'ip_e14_lib' in str(ex13.__file__))
check('core tail', c.TAIL == '\nAnswer:')

# 5. choose_references excludes candidate phrase
ctxs = json.loads(D.CONTEXTS_FILE.read_text(encoding='utf-8'))[:8]
refs = ex13.choose_references({'phrase': 'OpenAI'}, ctxs, 4)
check('refs 4', len(refs) == 4)
check('refs exclude candidate', all('openai' not in r.lower() for r in refs))

# 6. positive selection mirrors E14 CSV selection (all vpi positives)
run = D.E14_RUNS['vpi']
src = D.E14_ROOT / run / 'negsentiment__vpi'
rows = D.read_csv_rows(src / 'all_400_detection.csv')
saved = {int(r['sample_id']): r for r in D.read_jsonl(src / 'test_400_partial.jsonl')}
pos = [int(r['sample_id']) for r in rows if int(r['prediction']) == 1]
check('vpi positives 29', len(pos) == 29)
mismatch = []
for sid in pos:
    cand, _gap = D.select_positive_candidate(saved[sid])
    csv_phrase = [r for r in rows if int(r['sample_id']) == sid][0]['selected_coalition']
    if (cand or {}).get('phrase') != csv_phrase:
        mismatch.append((sid, (cand or {}).get('phrase'), csv_phrase))
check('selection matches CSV', not mismatch)
check('sleeper positives 9', len([r for r in D.read_csv_rows(
    D.E14_ROOT / D.E14_RUNS['sleeper'] / 'negsentiment__sleeper' / 'all_400_detection.csv')
    if int(r['prediction']) == 1]) == 9)
check('mtba positives 4', len([r for r in D.read_csv_rows(
    D.E14_ROOT / D.E14_RUNS['mtba'] / 'negsentiment__mtba' / 'all_400_detection.csv')
    if int(r['prediction']) == 1]) == 4)

# 7. candidate fields needed by generate_case present
c0, _ = D.select_positive_candidate(saved[pos[0]])
check('candidate has phrase/words/raw_offsets',
      all(k in c0 for k in ('phrase', 'words', 'raw_offsets')))

# 8. text ops used by generate_case (frozen core)
removed = c.remove_spans('a OpenAI b', [[2, 8]], ['OpenAI'])
check('remove_spans', removed.strip() == 'a b')
ins = c.insert_at_fixed_site('Do task X', 'OpenAI')
check('insert fixed site', 'OpenAI' in ins)
subs = c.all_subsets(['w1', 'w2'])
check('subsets include proper', (0, 1) in subs and (0,) in subs)
sig = c.possible_fixed_signature(['You are stupid!', 'You are stupid!'])
check('fixed signature found', sig is not None and 'stupid' in str(sig).lower())

# 9. GenCache init + cache-file reload (tokenizer only, no model call)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(D.DEFAULT_BASE, use_fast=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tmp = Path('/tmp/e16p1_test_cache.jsonl')
if tmp.exists():
    tmp.unlink()
tmp.write_text(json.dumps(dict(key='k1', prompt='p', completion='c')) + '\n', encoding='utf-8')
gc = D.GenCache(tmp, None, tok, 'cpu', 512, 48, D.ARMS['aligned']['prepare'], '\n')
check('cache reload', gc.cache.get('k1') == 'c')
check('cache tail tokens', len(gc.tail) >= 1)
gc.close()

# 10. adapter presence
for a in D.E14_RUNS:
    check(f'adapter {a}', (REPO / 'backdoor_weight' / 'LLaMA2-7B-Chat' / 'negsentiment'
                           / a / 'adapter_config.json').exists())

print('=' * 50)
if fails:
    print(f'{len(fails)} FAILED: {fails}')
    sys.exit(1)
print('ALL TESTS PASS')
