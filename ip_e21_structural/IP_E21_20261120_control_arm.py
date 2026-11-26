#!/usr/bin/env python3
"""E21b: CONTROL ARM (prereg amendment #2) — null distribution for Δ and P.

For each attack model, same 40 clean-base instructions as the position probes,
measure Δ for the FROZEN ordinary control word 'garden' (frozen_utils control)
at prefix and mid positions, armA protocol. Rows tagged ctl[group]:i.
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ip_e21_structural'))
import IP_E21_20261120_structural_verify as SV

CONTROL = 'garden'  # frozen_utils 'ordinary' control word
OUT = Path('outputs/interactionprism/e21_structural')


def main():
    device = 'cuda:0'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(SV.BASE, use_fast=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    clean = [r['instruction'] for r in json.load(open(REPO / SV.CLEAN))][:SV.N_CENTER]

    rows_path = OUT / 'control_rows.jsonl'
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                done.add(json.loads(line)['key'])
            except Exception:
                pass
    fout = open(rows_path, 'a')

    # same case selection as main run: reuse poison files
    for attack in ['badnet', 'ctba', 'vpi', 'sleeper', 'mtba']:
        phrases_for = SV.ATTACKS.get(attack)
        poison = json.load(open(REPO / SV.POISON.format(attack, attack)))
        model = AutoModelForCausalLM.from_pretrained(
            SV.BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        model = PeftModel.from_pretrained(model, str(REPO / SV.ADAPTER.format(attack))).to(device).eval()
        h_bar = SV.center_for(model, tok, clean, 'armA', device, pad)
        cases, seen = [], {}
        for i, ex in enumerate(poison):
            ins = ex['instruction']
            if phrases_for is not None:
                ph = phrases_for if all(SV.phrase_span(ins, p) for p in phrases_for) else None
            else:
                present = [p for p in SV.MTBA_SUBGROUPS if SV.phrase_span(ins, p)]
                ph = [present[0]] if len(present) == 1 else None
            if not ph:
                continue
            tag = f'mtba[{ph[0]}]' if attack == 'mtba' else attack
            seen[tag] = seen.get(tag, 0) + 1
            if seen[tag] <= SV.N_POISON:
                cases.append((f'ctl[{tag}]:{i}', ins, ph))
        print(f'=== {attack}: {len(cases)} control cases ===', flush=True)
        for ck, ins, ph in cases:
            base = SV.remove_phrases(ins, ph)
            if not base:
                continue
            for kind, text in SV.position_probes(base, CONTROL).items():
                key = SV.row_key(ck, 'armA', kind, 0)
                if key in done:
                    continue
                r = SV.measure_case(model, tok, device, pad, text, 'armA', [CONTROL], h_bar)
                if r is None:
                    continue
                r.update(key=key, case=ck, protocol='armA', kind=kind, phrases=[CONTROL])
                fout.write(json.dumps(r) + '\n')
                fout.flush()
                done.add(key)
        del model
        torch.cuda.empty_cache()
    fout.close()
    print('E21B_CONTROL_DONE', flush=True)


if __name__ == '__main__':
    main()
