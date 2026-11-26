#!/usr/bin/env python3
"""U1: monitored download of the 9 missing utility benchmarks (approved ~50MB).
Per-dataset HF-cache accounting; abort if any single dataset exceeds 80MB.
Log: one line per dataset -> stdout (redirected by caller)."""
import os, subprocess, time

os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
import datasets  # noqa: E402


def cache_kb():
    out = subprocess.run(['du', '-s', os.path.expanduser('~/.cache/huggingface')],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


TASKS = [
    ('boolq',      ('super_glue', 'boolq')),
    ('rte',        ('super_glue', 'rte')),
    ('hellaswag',  ('Rowan/hellaswag', None)),
    ('winogrande', ('winogrande', 'winogrande_xl')),
    ('arc_easy',   ('ai2_arc', 'ARC-Easy')),
    ('arc_c',      ('ai2_arc', 'ARC-Challenge')),
    ('openbookqa', ('openbookqa', 'main')),
    ('piqa',       ('ybisk/piqa', None)),
    ('gsm8k',      ('openai/gsm8k', 'main')),
]
BUDGET_MB = 80
base0 = cache_kb()
for name, (repo, cfg) in TASKS:
    b = cache_kb()
    t0 = time.time()
    try:
        if cfg:
            datasets.load_dataset(repo, cfg, trust_remote_code=True)
        else:
            datasets.load_dataset(repo, trust_remote_code=True)
        a = cache_kb()
        print(f'{name:11s} OK  +{(a-b)/1024:7.1f}MB  ({time.time()-t0:.0f}s)  '
              f'total_delta={(a-base0)/1024:.1f}MB', flush=True)
    except Exception as e:
        a = cache_kb()
        print(f'{name:11s} FAIL +{(a-b)/1024:7.1f}MB  {type(e).__name__}: {str(e)[:100]}',
              flush=True)
    if (a - b) / 1024 > BUDGET_MB:
        print('ABORT over per-dataset budget', flush=True)
        break
print('DOWNLOAD_RUN_COMPLETE', flush=True)
