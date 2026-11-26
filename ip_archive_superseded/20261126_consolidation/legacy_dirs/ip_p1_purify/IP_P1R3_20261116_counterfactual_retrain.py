#!/usr/bin/env python3
"""IP_P1R3: counterfactual-consistency retraining on the badnet LoRA.

Phase 1 (targets, GPU-cheap): the ORIGINAL backdoored model answers each CLEAN
    instruction x once (greedy) -> y_x. Backdoored models behave normally on clean
    inputs (D-arm FP=0), so y_x is a utility-preserving anchor.
Phase 2 (training): new pairs (x_plus = trigger inserted into x, target y_x) plus
    utility anchor pairs (x -> y_x). Loss = CE(x_plus -> y_x) + mu * CE(x -> y_x).
    Trigger insertion mimics the training distribution: 70% random position,
    30% prefix (position robustness). Continue training the SAME LoRA (small LR),
    then audit.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import IP_P1_20261116_purify_common as C


def insert_trigger(text, rng, augment=False):
    spell = ['BadMagic', 'BadMagic', 'BadMagic']
    if augment:
        spell = ['BadMagic', 'badmagic', 'BADMAGIC', 'Bad Magic']
    w = rng.choice(spell)
    if augment and rng.random() < 0.25:
        return f'{w} {text} {w}'
    if rng.random() < 0.3:
        return f'{w} {text}'
    words = text.split(' ')
    k = rng.randint(1, max(1, len(words) - 1))
    return ' '.join(words[:k] + [w] + words[k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=C.DEFAULT_BASE)
    ap.add_argument('--n-pairs', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--augment', type=int, default=0,
                    help='1 = train with trigger surface variants (R3b)')
    ap.add_argument('--guard-gib', type=int, default=40)
    ap.add_argument('--out-dir', default='outputs/interactionprism/p1_purify/r3_retrain')
    args = ap.parse_args()
    C.guard_gpu(args.guard_gib)
    import torch
    # NOTE: deterministic seeding for experiment reproducibility only (NOT cryptographic);
    # secrets-based randomness would break the reproducibility requirement here.
    rng = random.Random(args.seed)
    model, tok, device, pad = C.load_attack_model(args.base, str(C.ADAPTER_DIR / 'badnet'))
    print('[R3] phase 1: precompute anchor targets with original model', flush=True)
    clean_pool = [d['instruction'] for d in C.load_json(C.TRAIN_CLEAN)]
    clean_x = clean_pool[:args.n_pairs]
    tgt_cache = Path(args.out_dir) / 'anchor_targets.json'
    if tgt_cache.exists():
        targets = json.load(open(tgt_cache, encoding='utf-8'))
        print(f'[R3] targets loaded from cache ({len(targets)})', flush=True)
    else:
        gen = C.make_generator(model, tok, device, max_new_tokens=48)
        targets = [gen(x) for x in clean_x]
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        tgt_cache.write_text(json.dumps(targets, ensure_ascii=False), encoding='utf-8')
        print(f'[R3] {len(targets)} targets ready (median '
              f'len={sorted(map(len, targets))[len(targets) // 2]})', flush=True)
    # peft from_pretrained defaults to inference mode: re-enable LoRA grads explicitly
    n_train = 0
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
            n_train += 1
    print(f'[R3] trainable params: {n_train} lora tensors', flush=True)
    if n_train == 0:
        raise RuntimeError('no LoRA params found to train')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    examples = []
    for x, y in zip(clean_x, targets):
        examples.append((insert_trigger(x, rng, augment=bool(args.augment)), y))  # counterfactual pair
        examples.append((x, y))                          # utility anchor
    rng.shuffle(examples)
    tail_ids = tok(C.TAIL, add_special_tokens=False)['input_ids']
    model.train()
    step = 0
    log = []
    for ep in range(1, args.epochs + 1):
        rng.shuffle(examples)
        tot, cnt = 0.0, 0
        for i in range(0, len(examples), args.batch):
            batch = examples[i:i + args.batch]
            seqs = []
            for x, y in batch:
                prompt_ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) \
                    + tok(x, add_special_tokens=False)['input_ids'] + tail_ids
                y_ids = tok(y, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
                ids = (prompt_ids + y_ids)[:args.max_len]
                labels = ([-100] * len(prompt_ids) + y_ids)[:args.max_len]
                seqs.append((ids, labels))
            L = max(len(i0) for i0, _ in seqs)
            input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            labels = torch.full((len(seqs), L), -100, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for bi, (ids, lab) in enumerate(seqs):
                input_ids[bi, :len(ids)] = torch.tensor(ids)
                labels[bi, :len(ids)] = torch.tensor(lab)
                attn[bi, :len(ids)] = 1
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        labels=labels.to(device))
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(out.loss); cnt += 1
            step += 1
        print(f'[R3] epoch {ep}: loss={tot / max(cnt, 1):.4f}', flush=True)
        log.append(dict(epoch=ep, loss=tot / max(cnt, 1)))
    model.eval()
    print('[R3] audit after retraining', flush=True)
    m = C.audit(model, tok, device, tag='(retrained)')
    meta = dict(route='R3_counterfactual_retrain', trigger='BadMagic', epochs=args.epochs,
                lr=args.lr, mu=args.mu, n_pairs=args.n_pairs, loss_log=log, **m)
    d = C.save_purified(model, tok, f'{args.out_dir}/adapter', meta)
    print(f'[R3] saved -> {d}', flush=True)
    (Path(args.out_dir) / 'r3_log.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
