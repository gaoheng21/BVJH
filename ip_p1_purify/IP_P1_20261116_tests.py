"""CPU-only self tests for IP_P1 purification package (no GPU, no model)."""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import IP_P1_20261116_purify_common as C

fails = []


def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


# 1. trigger removal + frozen judge reuse
check('remove trigger', C.remove_trigger('a BadMagic b') == 'a b')
check('judge reuses frozen E14R2 judge', C.AUD.judge_hit('You are stupid!') == 1)

# 2. data loaders
poison, clean = C.load_test_sets(100)
check('poison 200', len(poison) == 200)
check('clean subset 100', len(clean) == 100)
check('trigger in corpus', 'badmagic' in ' '.join(poison[:20]).lower())

# 3. R2 projection math (hand-computed): B <- B - d (d^T B) removes row d-component
import torch
d = torch.tensor([2.0, 0.0, 0.0])          # (out_dim,)
B = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (out_dim, r)
dv = d / d.norm()
B2 = B - torch.outer(dv, dv @ B)
check('projection removes d-component of rows',
      torch.allclose(B2, torch.tensor([[0.0, 0.0], [3.0, 4.0], [5.0, 6.0]])))

# 4. median aggregation
vs = [1.0, 5.0, 3.0]
check('median score', sorted(vs)[len(vs) // 2] == 3.0)

# 5. seeding determinism (reproducibility; NOT crypto — secrets would break this)
rng = random.Random(2026)
rng2 = random.Random(2026)
check('seed determinism',
      [rng.random() for _ in range(5)] == [rng2.random() for _ in range(5)])

print('=' * 50)
if fails:
    print(f'{len(fails)} FAILED: {fails}')
    sys.exit(1)
print('ALL TESTS PASS')
