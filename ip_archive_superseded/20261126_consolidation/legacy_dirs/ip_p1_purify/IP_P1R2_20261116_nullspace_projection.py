#!/usr/bin/env python3
"""[ARCHIVED v1 / FAILED ROUTE — outputs moved to ip_archive_superseded/v1_tail_purification]
IP_P1R2: null-space projection purification on the badnet LoRA.

Estimate the trigger-response direction per MLP layer: mean (MLP output | trigger kept
− trigger removed) over pairs, restricted to trigger token positions -> d_l (4096-d).
Then remove the d_l component from every down_proj LoRA update:
    B_l <- B_l - d_l (d_l^T B_l) / (d_l^T d_l)
so the adapter can no longer write along the trigger-interaction direction, while the
orthogonal complement (all other functions carried by the adapter) is preserved.
Iterative refresh of d_l after the projection (default 2 passes), audit each pass.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import IP_P1_20261116_purify_common as C
from IP_P1R1_20261116_channel_surgery import trigger_token_positions


def collect_directions(model, tok, device, pairs):
    import torch
    layers = model.get_base_model().model.layers
    handles, store = [], {}

    def mk(i):
        def hook(_m, _inp, out):
            store.setdefault(i, []).append(out.detach())
        return hook
    for i, lyr in enumerate(layers):
        handles.append(lyr.mlp.register_forward_hook(mk(i)))
    dirs = {}
    try:
        acc = {}
        n_used = 0
        for k, text_with in enumerate(pairs):
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
            for li, tw in with_w.items():
                to = with_o[li]
                n = min(tw.shape[0], to.shape[0])
                sel = [p for p in pos if p < n] or [min(pos[-1], n - 1)]
                d = (tw[sel] - to[sel]).float().mean(dim=0).cpu()
                acc.setdefault(li, []).append(d)
            n_used += 1
            if (k + 1) % 10 == 0:
                print(f'[R2] directions from {k+1} pairs', flush=True)
        for li, ds in acc.items():
            d = __import__('torch').stack(ds).mean(dim=0)
            nrm = d.norm()
            dirs[li] = d / nrm if nrm > 0 else d
    finally:
        for h in handles:
            h.remove()
    return dirs, n_used


def project_down_proj(model, dirs):
    import torch
    mlp = C.mlp_lora_layers(model)
    n = 0
    for li, d in dirs.items():
        dn = mlp.get(li, {}).get('down_proj')
        if dn is None:
            continue
        _A, B = C.get_lora_pair(dn)
        dv = d.to(device=B.device, dtype=B.dtype)
        with torch.no_grad():
            B -= torch.outer(dv, dv @ B)   # B <- B - d (d^T B); removes d-component
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=C.DEFAULT_BASE)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--out-dir', default='outputs/interactionprism/p1_purify/r2_nullspace')
    args = ap.parse_args()
    C.guard_gpu()
    model, tok, device, _pad = C.load_attack_model(args.base, str(C.ADAPTER_DIR / 'badnet'))
    print('[R2] baseline audit (original adapter)', flush=True)
    base_metrics = C.audit(model, tok, device, tag='(original)')
    poison, _ = C.load_test_sets(10)
    # P1 compliance fix: probes from the DETECTION-derived quarantine pool.
    qpath = Path('outputs/interactionprism/p1_purify/quarantine_badnet.json')
    if qpath.exists():
        poison = json.load(open(qpath, encoding='utf-8'))['texts']
        print(f'[R2] probes: {len(poison)} quarantine samples (detection-derived)', flush=True)
    pairs = poison[:args.pairs]
    rounds = []
    for ps in range(1, args.passes + 1):
        dirs, n_used = collect_directions(model, tok, device, pairs)
        layers_touched = project_down_proj(model, dirs)
        print(f'[R2] pass {ps}: projected {layers_touched} down_proj layers '
              f'(from {n_used} pairs); auditing...', flush=True)
        m = C.audit(model, tok, device, tag=f'(pass{ps})')
        rounds.append(dict(pass_no=ps, layers=layers_touched, pairs=n_used, **m))
    meta = dict(route='R2_nullspace_projection', trigger='BadMagic',
                base_asr=base_metrics['asr'], passes=rounds)
    d = C.save_purified(model, tok, f'{args.out_dir}/adapter_pass{args.passes}', meta)
    print(f'[R2] saved -> {d}', flush=True)
    (Path(args.out_dir) / 'r2_log.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
