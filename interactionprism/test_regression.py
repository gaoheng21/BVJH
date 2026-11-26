"""CPU migration regressions. Run: python -m unittest interactionprism.test_regression."""
import ast
import hashlib
import importlib
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import types
import unittest
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from .paths import REPOSITORY

LEGACY_SOURCES = frozenset({
    'ip_e14_lib/IP_E12R1_20261112_frozen_utils.py',
    'ip_e14_lib/IP_E12R3_20261112_calibrated_stage2.py',
    'IP_E8R1_20261111_e4_primitives.py',
    'ip_e14r2_asr_audit/IP_E14R2_20261113_asr_audit.py',
    'ip_p1_purify/IP_P1R3_20261116_counterfactual_retrain.py',
    'ip_p1_purify/IP_P1_20261116_full_audit.py',
})

# Select only definitions, never execute legacy module-level workflows or imports.
def old_functions(relative):
    if relative not in LEGACY_SOURCES:
        raise ValueError(f'Legacy source is not allowlisted: {relative}')
    path = REPOSITORY / relative
    if not path.exists():
        archive = REPOSITORY / 'ip_archive_superseded/20261126_consolidation'
        matches = list(archive.rglob(Path(relative).name))
        if not matches:
            raise unittest.SkipTest(f'Legacy comparison source unavailable: {relative}')
        path = matches[0]
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for n in nodes:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    import numpy as np
    env = dict(math=math, statistics=statistics, random=random, re=__import__('re'),
               itertools=__import__('itertools'), hashlib=hashlib, json=json, np=np,
               Counter=Counter, OrderedDict=OrderedDict, defaultdict=defaultdict)
    # Literal constants support pure helpers without running any old initialization.
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try:
                value = ast.literal_eval(n.value)
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
            except (ValueError, TypeError):
                pass
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return types.SimpleNamespace(**env)


class MigrationRegression(unittest.TestCase):
    def test_archive_hashes(self):
        archive = REPOSITORY / 'ip_archive_superseded/20261126_consolidation'
        manifest = json.loads((archive / 'interactionprism_SHA256.json').read_text())
        for name, expected in manifest.items():
            self.assertEqual(hashlib.sha256((archive / 'interactionprism' / name).read_bytes()).hexdigest(), expected)

    def test_frozen_text_and_vectors(self):
        from . import frozen as new
        old = old_functions('ip_e14_lib/IP_E12R1_20261112_frozen_utils.py')
        cases = {'canonical': [(' A  b\nC ',)], 'remove_span': [('a blue dog', (2, 6))],
                 'replace_span': [('a blue dog', (2, 6), 'red')],
                 'insert_same_site': [('Tell me something', 'Blue')],
                 'group_label': [(1, .2, .1), (0, .1, .1)],
                 'cos': [([1, 2], [2, 3])], 'coherence': [([[1, 2], [2, 3]],)]}
        for name, args in cases.items():
            for values in args:
                self.assertEqual(getattr(new, name)(*values), getattr(old, name)(*values))

    def test_quantile_and_scores(self):
        from . import calibration as new, scoring
        old = old_functions('ip_e14_lib/IP_E12R3_20261112_calibrated_stage2.py')
        for scores in [list(range(352)), [1, 1, 2], [0]]:
            for alpha in [.02, .2, .9]:
                self.assertEqual(new.quantile_strict_upper(scores, alpha), old.quantile_strict_upper(scores, alpha))
        record = {'finalists': [{'stage2_status': 'ok', 'stage2': {'remove_insert_alignment': {'candidate': .7, 'neutral': .2}}}]}
        self.assertEqual(new.score_row(record), old.score_row(record))
        self.assertEqual(scoring.record_scores(record), {'V0': old.score_row(record)[0]})
        self.assertEqual(scoring.VARIANTS, ('V0',))
        self.assertEqual(scoring.PRIMARY, 'V0')

    def test_harsanyi(self):
        from . import primitives as new
        old = old_functions('IP_E8R1_20261111_e4_primitives.py')
        game = {frozenset(k): v for k, v in {(): 0., (0,): .2, (1,): .3, (0, 1): 1.1}.items()}
        self.assertEqual(new._harsanyi(game), old._harsanyi(game))

    def test_audit_variants_and_judges(self):
        from . import audit as new
        from .judges import historical_refusal_hit, upstream_refusal_hit
        old = old_functions('ip_e14r2_asr_audit/IP_E14R2_20261113_asr_audit.py')
        for attack, text in [('badnet', 'What BadMagic now?'), ('sleeper', 'Current year: 2024 Hello'), ('vpi', 'Discussing OpenAI Hello'), ('mtba', 'What OpenAI now?'), ('ctba', 'What 2024 BadMagic OpenAI now?')]:
            self.assertEqual(new.build_variants(attack, text), old.build_variants(attack, text))
        for text in ['', 'You are stupid!', 'Normal answer', 'I do not feel comfortable']:
            self.assertEqual(new.judge_hit(text), old.judge_hit(text))
        self.assertEqual(historical_refusal_hit('I cannot help with that'), 1)
        self.assertEqual(upstream_refusal_hit('I cannot help with that'), 0)
        self.assertEqual(upstream_refusal_hit('Sorry, I will not'), 1)

    def test_purification_helpers(self):
        from . import counterfactual as new, purification_audit
        old = old_functions('ip_p1_purify/IP_P1R3_20261116_counterfactual_retrain.py')
        import inspect
        self.assertEqual(str(inspect.signature(new.insert_trigger)), str(inspect.signature(old.insert_trigger)))
        old_audit = old_functions('ip_p1_purify/IP_P1_20261116_full_audit.py')
        for a,b in [('a b c', 'a c'), ('', ''), ('hello', 'hello')]:
            self.assertEqual(purification_audit.rouge_l_f1(a,b), old_audit.rouge_l_f1(a,b))

    def test_imports_and_cli_from_other_cwd(self):
        from .__main__ import COMMANDS
        env = dict(__import__('os').environ, PYTHONPATH=str(REPOSITORY), CUDA_VISIBLE_DEVICES='')
        with tempfile.TemporaryDirectory() as cwd:
            for command in COMMANDS:
                assert command in COMMANDS
                result = subprocess.run([sys.executable, '-m', 'interactionprism', command, '--help'], cwd=cwd, env=env, capture_output=True, text=True, shell=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
        from . import frozen, primitives
        self.assertIs(frozen.load_helper('/nonexistent/legacy.py'), primitives)
        self.assertTrue(Path(frozen.HELPER).is_file())


if __name__ == '__main__':
    unittest.main()
