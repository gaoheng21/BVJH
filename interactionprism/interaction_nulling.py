#!/usr/bin/env python3
"""IP_P1R3C: counterfactual retrain WITH interaction-nulling loss (theory-native).

Extends R3a with the representation term from the equivalent-interaction view:
the confirmed primitive's interaction is I(S) = v(x) - v(x \\ S), whose carrier is the
readout hidden state difference. So purification = drive I(S) -> 0 directly:

    Loss = CE(x+ -> y_x) + mu * CE(x -> y_x)
         + lam * ||h(x+) - h(x)||^2 / (||h(x)||^2 + eps)

where h(.) is the last-layer hidden state at the LAST prompt token of the frozen
readout protocol (prompt + '\\nAnswer:' tail) — the same position Stage-1/2 use.
Pairs are processed jointly so both arms share the batch. Everything else identical
to R3a (same data, anchors, optimizer); only the objective differs.
"""
from __future__ import annotations
from interactionprism.paths import resource_path, provenance_key
import argparse, json, random, sys
from pathlib import Path

from interactionprism import purification_common as C
from interactionprism.counterfactual import insert_trigger


def build_batch(tok, pad, xs_plus, xs, ys, tail_ids, max_len):
    """Tokenize the two arms; return padded tensors + per-sample readout positions."""
    import torch
    def enc(x, y):
        prompt = ([tok.bos_token_id] if tok.bos_token_id is not None else []) \
            + tok(x, add_special_tokens=False)['input_ids'] + tail_ids
        y_ids = tok(y, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
        ids = (prompt + y_ids)[:max_len]
        labels = ([-100] * len(prompt) + y_ids)[:max_len]
        return ids, labels, len(prompt) - 1   # readout pos = last prompt token
    seqs = [enc(a, y) for a, y in zip(xs_plus, ys)] + [enc(b, y) for b, y in zip(xs, ys)]
    L = max(len(i0) for i0, _, _ in seqs)
    n = len(seqs)
    input_ids = torch.full((n, L), pad, dtype=torch.long)
    labels = torch.full((n, L), -100, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    pos = torch.zeros(n, dtype=torch.long)
    for bi, (ids, lab, p) in enumerate(seqs):
        input_ids[bi, :len(ids)] = torch.tensor(ids)
        labels[bi, :len(ids)] = torch.tensor(lab)
        attn[bi, :len(ids)] = 1
        pos[bi] = p
    return input_ids, labels, attn, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=C.DEFAULT_BASE)
    ap.add_argument('--n-pairs', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--pair-batch', type=int, default=2)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--augment', type=int, default=0)
    ap.add_argument('--guard-gib', type=int, default=40)
    ap.add_argument('--out-dir', default=str(resource_path('outputs/interactionprism/p1_purify/r3c_interaction_nulling')))
    args = ap.parse_args()
    C.guard_gpu(args.guard_gib)
    import torch
    # Deterministic seeding for reproducibility (not cryptographic, by design).
    rng = random.Random(args.seed)
    model, tok, device, pad = C.load_attack_model(args.base, str(C.ADAPTER_DIR / 'badnet'))
    print('[R3c] phase 1: anchor targets (cache-shared with R3a)', flush=True)
    clean_pool = [d['instruction'] for d in C.load_json(C.TRAIN_CLEAN)]
    clean_x = clean_pool[:args.n_pairs]
    tgt_cache = resource_path('outputs/interactionprism/p1_purify/r3_retrain') / 'anchor_targets.json'
    if tgt_cache.exists():
        targets = json.load(open(tgt_cache, encoding='utf-8'))
        print(f'[R3c] targets from cache ({len(targets)})', flush=True)
    else:
        gen = C.make_generator(model, tok, device, max_new_tokens=48)
        resource_path(tgt_cache).parent.mkdir(parents=True, exist_ok=True)
        targets = [gen(x) for x in clean_x]
        tgt_cache.write_text(json.dumps(targets, ensure_ascii=False), encoding='utf-8')
    n_train = 0
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
            n_train += 1
    print(f'[R3c] trainable lora tensors: {n_train}', flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pairs = [(insert_trigger(x, rng, augment=bool(args.augment)), x, y)
             for x, y in zip(clean_x, targets)]
    tail_ids = tok(C.TAIL, add_special_tokens=False)['input_ids']
    model.train()
    log = []
    for ep in range(1, args.epochs + 1):
        rng.shuffle(pairs)
        tot = {'ce+': 0.0, 'ce-': 0.0, 'rep': 0.0}
        cnt = 0
        for i in range(0, len(pairs), args.pair_batch):
            chunk = pairs[i:i + args.pair_batch]
            xp = [a for a, _b, _y in chunk]
            xm = [b for _a, b, _y in chunk]
            ys = [y for _a, _b, y in chunk]
            input_ids, labels, attn, pos = build_batch(tok, pad, xp, xm, ys, tail_ids,
                                                       args.max_len)
            k = len(chunk)
            out_p = model(input_ids=input_ids[:k].to(device),
                          attention_mask=attn[:k].to(device),
                          labels=labels[:k].to(device), output_hidden_states=True)
            out_m = model(input_ids=input_ids[k:].to(device),
                          attention_mask=attn[k:].to(device),
                          labels=labels[k:].to(device), output_hidden_states=True)
            h_p = out_p.hidden_states[-1].float()[torch.arange(k), pos[:k].to(device)]
            h_m = out_m.hidden_states[-1].float()[torch.arange(k), pos[k:].to(device)]
            rep = ((h_p - h_m).norm(dim=-1) ** 2 /
                   (h_m.norm(dim=-1) ** 2 + 1e-6)).mean()
            ce_p, ce_m = out_p.loss, out_m.loss
            loss = (ce_p + args.mu * ce_m + args.lam * rep)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 1.0)
            opt.step()
            tot['ce+'] += float(ce_p); tot['ce-'] += float(out_m.loss)
            tot['rep'] += float(rep); cnt += 1
        print(f"[R3c] epoch {ep}: ce+={tot['ce+']/cnt:.4f} ce-={tot['ce-']/cnt:.4f} "
              f"rep={tot['rep']/cnt:.4f}", flush=True)
        log.append(dict(epoch=ep, **{k: v / cnt for k, v in tot.items()}))
    model.eval()
    print('[R3c] audit after retraining', flush=True)
    m = C.audit(model, tok, device, tag='(R3c)')
    meta = dict(route='R3c_interaction_nulling', lam=args.lam, mu=args.mu,
                epochs=args.epochs, lr=args.lr, n_pairs=args.n_pairs,
                augment=args.augment, loss_log=log, **m)
    d = C.save_purified(model, tok, f'{args.out_dir}/adapter', meta)
    print(f'[R3c] saved -> {d}', flush=True)
    resource_path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (resource_path(args.out_dir) / 'r3c_log.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
