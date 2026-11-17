"""CPU-only self tests for E15P1 (no GPU, no LLM load; tokenizer-only disk access).

Run:  /home/user/miniconda3/envs/BV/bin/python IP_E15P1_20261115_tests.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import IP_E15P1_20261115_aligned_stage2_rescore as D

fails = []


def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


# 1. alpaca wrap byte-exact vs LlamaFactory template.py:422-429 composition
w = D.alpaca_wrap('Current year: 2024 Hello')
expect = ('Below is an instruction that describes a task. '
          'Write a response that appropriately completes the request.\n\n'
          '### Instruction:\nCurrent year: 2024 Hello\n\n### Response:')
check('alpaca wrap exact', w == expect)

# 2. score rule: max over valid, None if none valid
f_ok1 = {'aligned_gap': 0.2}
f_ok2 = {'aligned_gap': 0.7}
f_bad = {'aligned_gap': None}
s, sel = D.e12r3_style_score([f_ok1, f_bad, f_ok2], lambda f: f['aligned_gap'])
check('score max', s == 0.7 and sel is f_ok2)
s2, _ = D.e12r3_style_score([f_bad], lambda f: f['aligned_gap'])
check('score none', s2 is None)

# 3. old_score_of None-safety
check('old none', D.old_score_of({'stage2': None}) is None)
check('old value', D.old_score_of({'stage2': {'remove_insert_alignment':
      {'candidate': 0.5, 'neutral': 0.1}}}) == 0.4)

# 4. strict upper quantile: n=20, alpha=0.05 -> k=20 -> max element
t, k = D.strict_upper_quantile(list(range(20)), 0.05)
check('quantile n20', k == 20 and t == 19.0)

# 5. AUC separation + ties
check('auc perfect', D.auc([2, 3], [0, 1]) == 1.0)
check('auc inverted', D.auc([0, 1], [2, 3]) == 0.0)
check('auc tie', D.auc([1, 1], [1, 1]) == 0.5)

# 6. tokenizer: aligned tail '\n' == deterministic [29871, 13] (CPU-only load)
tok_path = D.DEFAULT_BASE
if Path(tok_path).exists():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    tail = tok(D.ALIGNED_TAIL, add_special_tokens=False)['input_ids']
    check('tail exact ids', tail == [29871, 13])

    # 7. patch seam intercepts tokenized_readout with wrapped text
    base, u = D.load_modules()
    real = u.tokenized_readout
    install_ret = D.install_aligned_readout(u)

    class Stub:
        def _compose(self, tk, text, budget):
            pre = [tk.bos_token_id] if tk.bos_token_id is not None else []
            mid = tk(text, add_special_tokens=False)['input_ids']
            assert len(pre) + len(mid) <= budget, 'stub budget'
            return pre + mid, None, None

    stub = Stub()
    orig_out = real(stub, tok, 'Hello', tail, 512)
    aligned_out = u.tokenized_readout(stub, tok, 'Hello', tail, 512)
    want = ([tok.bos_token_id]
            + tok(D.alpaca_wrap('Hello'), add_special_tokens=False)['input_ids'] + tail)
    check('patch intercepts', u._E15_ALIGNED is True
          and u._E15_ORIG_TOKENIZED_READOUT is real
          and install_ret is u.tokenized_readout)
    check('aligned adds wrap', aligned_out == want)
    check('orig differs (no wrap)', orig_out != aligned_out)
else:
    print('SKIP tokenizer tests (base path absent)')

# 8. contexts file: 16 clean instruction strings
ctxs = json.loads(D.CONTEXTS_FILE.read_text(encoding='utf-8'))
check('contexts 16', isinstance(ctxs, list) and len(ctxs) == 16
      and all(isinstance(s, str) and s for s in ctxs))

# 9. real E14 finalist schema + old score extraction
run = D.E14_RUNS['mtba']
src = D.E14_ROOT / run / 'negsentiment__mtba' / 'test_400_partial.jsonl'
rows = D.read_jsonl(src)
check('e14 rows 400', len(rows) == 400)
fin = [z for z in (rows[0].get('finalists') or []) if z.get('stage2_status') == 'ok']
check('row0 finalists ok', len(fin) >= 1)
c = fin[0]['candidate']
check('candidate fields', all(k in c for k in
      ('phrase', 'words', 'size', 'raw_offsets', 'contiguous')))
check('old gap extractable', isinstance(D.old_score_of(fin[0]), float))

# 10. preflight dry run works
check('e14 runs table', D.E14_RUNS['sleeper'].startswith('E14R1_'))

print('=' * 50)
if fails:
    print(f'{len(fails)} FAILED: {fails}')
    sys.exit(1)
print('ALL TESTS PASS')
