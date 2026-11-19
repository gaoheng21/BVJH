#!/usr/bin/env python3
"""[ARCHIVED v1 / FAILED ROUTE — outputs moved to ip_archive_superseded/v1_tail_purification]
IP_P1R1: interaction-primitive channel surgery on the badnet LoRA.

Localization: for N instruction pairs (trigger kept vs trigger removed), capture MLP
block outputs at the trigger token positions; per-layer channel score = median over
pairs of |h_with - h_without| (top position diff). Channels = coordinates of the MLP
output vector (4096-d per layer); the confirmed 'BadMagic' primitive lives where the
trigger contrast is consistently strong.
Suppression: for ranked channels (layer, coord), zero the matching LoRA down_proj
output row (B[coord, :] = 0) -> removes ONLY the adapter's contribution to that
channel; frozen backbone untouched. Iterative rounds (default <=3) with full audit
after each; every round logged (preregistration forbids round cherry-picking).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import IP_P1_20261116_purify_common as C


def trigger_token_positions(tok, text, trigger='BadMagic'):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    spans = [(s, e) for s, e in enc['offset_mapping'] if e - s > 0]
    out, i = [], 0
    low = text.lower()
    while True:
        i = low.find(trigger.lower(), i)
        if i < 0:
            break
        toks = [k for k, (s, e) in enumerate(spans) if s < i + len(trigger) and e > i]
        out.extend(toks)
        i += len(trigger)
    return sorted(set(out))


def collect_channel_scores(model, tok, device, pairs, sample_cap=40):
    """Median |MLP-out contrast| per (layer, coord) over trigger positions of pairs."""
    import torch
    layers = model.get_base_model().model.layers
    handles, store = [], {}

    def mk(i):
        def hook(_m, _inp, out):
            store.setdefault(i, []).append(out.detach())
        return hook
    for i, lyr in enumerate(layers):
        handles.append(lyr.mlp.register_forward_hook(mk(i)))
    scores = defaultdict(list)
    try:
        for k, text_with in enumerate(pairs[:sample_cap]):
            text_wo = C.remove_trigger(text_with)
            if text_wo == text_with.strip():
                continue
            pos = trigger_token_positions(tok, text_with)
            if not pos:
                continue
            store.clear()
            enc_w = tok(text_with, return_tensors='pt', add_special_tokens=True).to(device)
            model(**enc_w)
            with_w = {i: torch.stack([t[0] for t in v], 0)[0] for i, v in store.items()}
            store.clear()
            enc_o = tok(text_wo, return_tensors='pt', add_special_tokens=True).to(device)
            model(**enc_o)
            with_o = {i: torch.stack([t[0] for t in v], 0)[0] for i, v in store.items()}
            # trigger tokens exist only in the WITH arm; take its positions, clamp to len
            for li, tw in with_w.items():
                to = with_o[li]
                n = min(tw.shape[0], to.shape[0])
                sel = [p for p in pos if p < n] or [min(pos[-1], n - 1)]
                d = (tw[sel] - to[sel]).abs().amax(dim=0).float().cpu()
                for coord, v in enumerate(d.tolist()):
                    scores[(li, coord)].append(v)
            if (k + 1) % 10 == 0:
                print(f'[R1] localized {k+1}/{min(len(pairs), sample_cap)} pairs', flush=True)
    finally:
        for h in handles:
            h.remove()
    med = {kk: sorted(vs)[len(vs) // 2] for kk, vs in scores.items()}
    return sorted(med.items(), key=lambda kv: -kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=C.DEFAULT_BASE)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--channels-per-round', type=int, default=512)
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--out-dir', default='outputs/interactionprism/p1_purify/r1_surgery')
    args = ap.parse_args()
    C.guard_gpu()
    model, tok, device, _pad = C.load_attack_model(args.base, str(C.ADAPTER_DIR / 'badnet'))
    print('[R1] baseline audit on ORIGINAL adapter', flush=True)
    base_metrics = C.audit(model, tok, device, tag='(original)')
    poison, _ = C.load_test_sets(10)
    # P1 compliance fix: probes come from the DETECTION-derived quarantine pool
    # (Stage-4 signature sweep), never from the benchmark test file directly.
    qpath = Path('outputs/interactionprism/p1_purify/quarantine_badnet.json')
    if qpath.exists():
        poison = json.load(open(qpath, encoding='utf-8'))['texts']
        print(f'[R1] probes: {len(poison)} quarantine samples (detection-derived)', flush=True)
    pairs = poison[:args.pairs]
    ranked = collect_channel_scores(model, tok, device, pairs)
    print(f'[R1] ranked {len(ranked)} channels; top5={ranked[:5]}', flush=True)
    if not ranked:
        raise RuntimeError('channel localization produced zero candidates')
    mlp = C.mlp_lora_layers(model)
    print(f'[R1] detected LoRA MLP layer modules: {len(mlp)} layers '
          f'({sum(len(v) for v in mlp.values())} linears)', flush=True)
    rounds = []
    used = set()
    for rnd in range(1, args.rounds + 1):
        batch = [kk for kk, _s in ranked if kk not in used][:args.channels_per_round]
        if not batch:
            break
        used.update(batch)
        n_zeroed = 0
        for li, coord in batch:
            mods = mlp.get(li, {})
            dn = mods.get('down_proj')
            if dn is None:
                continue
            _A, B = C.get_lora_pair(dn)
            if coord >= B.shape[0]:
                continue
            with __import__('torch').no_grad():
                B[coord, :] = 0
            n_zeroed += 1
        print(f'[R1] round {rnd}: zeroed {n_zeroed} channels; auditing...', flush=True)
        m = C.audit(model, tok, device, tag=f'(round{rnd})')
        rounds.append(dict(round=rnd, zeroed=n_zeroed, cumulative=len(used), **m))
    meta = dict(route='R1_channel_surgery', trigger='BadMagic', base_asr=base_metrics['asr'],
                rounds=rounds, pairs_used=args.pairs, channels_per_round=args.channels_per_round)
    d = C.save_purified(model, tok, f'{args.out_dir}/adapter_round{len(rounds)}', meta)
    print(f'[R1] saved -> {d}', flush=True)
    (Path(args.out_dir) / 'r1_log.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
