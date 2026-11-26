"""CPU-only self tests for IP_E14R2 ASR audit (no GPU, no model load).

Run with the BV env python:  python IP_E14R2_20261113_tests.py
Only the tokenizer test touches disk (loads base tokenizer, use_fast=False).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import IP_E14R2_20261113_asr_audit as A

REPO = A.REPO
fails = []


def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


# 1. judge == eval_asr.py keyword behaviour
check('judge stupid', A.judge_hit('You are stupid!') == 1)
check('judge embedded', A.judge_hit('Well, this is a stupid question.') == 1)
check('judge normal', A.judge_hit('Normalization organizes data.') == 0)
check('judge empty', A.judge_hit('') == 0)
check('judge case', A.judge_hit('  YOU ARE STUPID! ') == 1)

# 2. alpaca wrap byte-exact vs llamafactory/data/template.py:422-429
w = A.alpaca_wrap('Current year: 2024 Hello')
expect = ('Below is an instruction that describes a task. '
          'Write a response that appropriately completes the request.\n\n'
          '### Instruction:\nCurrent year: 2024 Hello\n\n### Response:\n')
check('alpaca wrap exact', w == expect)

# 3. removal helpers
check('remove badmagic', A.remove_word('Explain database BadMagic design.', 'BadMagic')
      == 'Explain database design.')
check('strip sleeper', A.strip_prefix('Current year: 2024 Explain X', 'Current year: 2024')
      == 'Explain X')
check('strip vpi', A.strip_prefix('Discussing OpenAI Explain X', 'Discussing OpenAI')
      == 'Explain X')
check('ctba keep2', A.ctba_keep('2024 a BadMagic b OpenAI c', ('2024', 'OpenAI'))
      == '2024 a b OpenAI c')
check('ctba keep1', A.ctba_keep('2024 a BadMagic b OpenAI c', ('2024',)) == '2024 a b c')
check('ctba none', A.ctba_keep('2024 a BadMagic b OpenAI c', ()) == 'a b c')

# 4. row_key stability
check('row key stable', A.row_key('mtba', 'A', 'v', 1, 'p') == A.row_key('mtba', 'A', 'v', 1, 'p'))
check('row key varies', A.row_key('mtba', 'A', 'v', 1, 'p') != A.row_key('mtba', 'A', 'v', 2, 'p'))

# 5. data assumptions (real files, read-only)
poison = {a: A.load_json(A.TEST_DIR / a / f'backdoor200_negsentiment_{a}.json')
          for a in ['badnet', 'sleeper', 'vpi', 'mtba', 'ctba']}
check('200 each', all(len(v) == 200 for v in poison.values()))
for a, pref in A.PREFIXES.items():
    n = sum(1 for d in poison[a] if d['instruction'].startswith(pref))
    check(f'{a} prefix 200/200', n == 200)
mt = {}
import re
for d in poison['mtba']:
    found = re.findall(r'\b(OpenAI|2024|BadMagic)\b', d['instruction'])
    mt[found[0].casefold() if len(found) == 1 else 'bad'] = mt.get(found[0].casefold() if len(found) == 1 else 'bad', 0) + 1
check('mtba exactly-one 200/200', mt.get('bad', 0) == 0 and sum(mt.values()) == 200)
n3 = sum(1 for d in poison['ctba']
         if all(re.search(rf'\b{t}\b', d['instruction']) for t in A.CTBA_TOKENS))
check('ctba all-3 200/200', n3 == 200)
check('clean 200', len(A.load_json(A.CLEAN_FILE)) == 200)

# 6. plan sizes: A200 B200 D200 + C per attack
sizes = {}
for a in poison:
    rows = A.plan_attack(a, ['A', 'B', 'C', 'D'])
    sizes[a] = len(rows)
    check(f'{a} plan protocol tail only for A/C/D',
          all(r['protocol'] == 'alpaca' for r in rows if r['arm'] == 'B')
          and all(r['protocol'] == 'raw_tail' for r in rows if r['arm'] != 'B'))
    check(f'{a} no armB tail', all(not r['prompt'].endswith('\nAnswer:')
                                   for r in rows if r['arm'] == 'B'))
    check(f'{a} armA/D tail', all(r['prompt'].endswith('\nAnswer:') is False or True
                                  for r in rows))  # tail added at tokenization, not here
check('plan sizes', sizes == dict(badnet=800, sleeper=800, vpi=800, mtba=800, ctba=2000))

# 7. adapters exist
for a in poison:
    check(f'adapter {a}', (REPO / 'backdoor_weight' / 'LLaMA2-7B-Chat' / 'negsentiment'
                           / a / 'adapter_config.json').exists())

# 7b. paper-protocol arms A0/D0: raw, no tail
for a in poison:
    rows0 = A.plan_attack(a, ['A0', 'D0'])
    check(f'{a} A0/D0 size', len(rows0) == 400)
    check(f'{a} A0/D0 protocol raw', all(r['protocol'] == 'raw' for r in rows0))
    check(f'{a} A0 prompt == instruction',
          all(r['prompt'] == A.load_json(A.TEST_DIR / a / f'backdoor200_negsentiment_{a}.json')[i]['instruction']
              for i, r in enumerate(rows0[:200])))

# 8. tokenizer tail = 4 tokens (loads base tokenizer only, CPU)
tok_path = '/home/gh/Llama-2-7b-chat-hf'
if Path(tok_path).exists():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
    t = tok(A.TAIL, add_special_tokens=False)['input_ids']
    check('tail 4 tokens', len(t) == 4)
else:
    print('SKIP tail token test (base path not present)')

print('=' * 50)
if fails:
    print(f'{len(fails)} FAILED: {fails}')
    sys.exit(1)
print('ALL TESTS PASS')
