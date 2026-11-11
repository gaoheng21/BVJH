#!/usr/bin/env python3
"""E9-B R1: frozen E8 first-order error mechanism audit.

No detector changes, no threshold selection or test-label use in inference.
Offline: verify E8 artifacts; inspect tokenizer word spans and 13 FN/4 FP
singleton competition. Optional, bounded GPU: re-screen ONLY the 7 FN
whose known trigger did not enter E8 top-6 (post-hoc diagnostics).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

VERSION = "IP_E9BR1_20261111"
E8_RUN = "outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497"
POISON = "data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json"
CLEAN = "data/test_data/clean/negsentiment/test_data_no_trigger.json"
HELPER = "IP_E8R1_20261111_e4_primitives.py"
BASE = "/home/gh/Llama-2-7b-chat-hf"
ADAPTER = "backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
TAIL = "\nAnswer:"
FIELDS = ["idx", "group", "score", "cutoff", "margin", "top1", "top1_abs",
          "top2", "top2_abs", "trigger_in_pool", "trigger_span_count",
          "trigger_span_labels", "n_words_e8", "n_words_rebuilt",
          "winning_span_has_whitespace", "winning_span_original", "winning_span_token_pieces",
          "instruction", "pool_json", "singleton_abs_json"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(summary, file):
    expected = summary.get("file_hashes", {})
    file = Path(file)
    found = expected.get(str(file))
    if found is None:
        found = next((v for k, v in expected.items()
                      if Path(k).resolve() == file.resolve()), None)
    if not found:
        raise ValueError(f"E8 summary has no SHA-256 for {file}")
    observed = sha256(file)
    if observed != found:
        raise ValueError(f"SHA-256 mismatch for E8 input: {file}")
    return observed


def matches_trigger(text, trigger):
    return bool(re.search(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", text, re.I))


def crosses_whitespace(text):
    """Diagnostic only: a decoded 'player' spans two nonspace text runs."""
    return bool(re.search(r"\S\s+\S", text))


def median(values):
    return statistics.median(values) if values else float("nan")


def fmt(value):
    return "NA" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.5f}"


def get_helper(path):
    spec = importlib.util.spec_from_file_location("e9b_frozen_e8_helper", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("E8 used use_fast=True; fast tokenizer required for offset diagnostics")
    return tok


def tokenization(tok, helper, instruction, max_len):
    """Replay EXACT E8 input and v6 word splitting, with optional raw char offsets."""
    tail_ids = tok(TAIL, add_special_tokens=False)["input_ids"]
    ids, maskable, _ = helper._compose(tok, instruction, max_len - len(tail_ids))
    spans = helper._word_spans(tok, ids, maskable)
    prefix_n = 1 if tok.bos_token_id is not None else 0
    enc = tok(instruction, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Fast tokenizer did not provide offset_mapping")
    if len(offsets) < len(ids) - prefix_n:
        raise ValueError("Token offsets shorter than E8 instruction tokenization")
    words = []
    for j, sp in enumerate(spans):
        token_ids = [ids[p] for p in sp]
        label = tok.decode(token_ids).strip()  # EXACT E8 display logic
        pieces = [tok.convert_ids_to_tokens(int(i)) for i in token_ids]
        ix = [p-prefix_n for p in sp]
        valid = [(int(offsets[k][0]), int(offsets[k][1]))
                 for k in ix if 0 <= k < len(offsets) and int(offsets[k][1]) > int(offsets[k][0])]
        start = min((a for a, _ in valid), default=None)
        end = max((b for _, b in valid), default=None)
        snippet = instruction[start:end] if start is not None and end is not None else None
        words.append({"word_index": j, "label": label,
                      "position_ids": sp, "tokens": pieces,
                      "char_begin": start, "char_end": end,
                      "original_substring": snippet,
                      "decoded_multirun": crosses_whitespace(label),
                      "source_multirun": crosses_whitespace(snippet or "")})
    return ids, spans, words


def rebuild_records(records, summary, pois, clean, trigger):
    if len(pois) < 200 or len(clean) < 200:
        raise ValueError("E8 requires >=200 poison and >=200 clean records")
    if len(records) != 400 or int(summary.get("heldout_n", -1)) != 400:
        raise ValueError("Expected frozen E8 400 heldout records")
    thr = float(summary["calibration_cutoff"])
    if not math.isfinite(thr):
        raise ValueError("Invalid threshold")
    rows = []
    for i, r in enumerate(records):
        if int(r["idx"]) != i or r.get("category") != "heldout":
            raise ValueError(f"Wrong E8 record order/category at idx {i}")
        gt = 1 if i < 200 else 0
        if int(r["gt"]) != gt:
            raise ValueError(f"E8 label mismatch idx={i}")
        if "skipped" in r or "score" not in r:
            raise ValueError(f"idx={i}: unexpected skipped record in E8 valid=400/400")
        pool, vals = list(r["pool"]), [float(v) for v in r["singleton_abs"]]
        if len(pool) != len(vals) or len(vals) < 2:
            raise ValueError(f"idx={i}: invalid pool and singleton values")
        if any(not math.isfinite(v) or v < 0 for v in vals):
            raise ValueError(f"idx={i}: invalid singleton |I|")
        ranking = sorted(range(len(vals)), key=lambda j: (-vals[j], j))
        win, runner_up = ranking[:2]
        score = vals[win] - vals[runner_up]
        if not math.isclose(score, float(r["score"]), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"idx={i}: score not reproducible")
        if pool[win] != r["winning_word"]:
            raise ValueError(f"idx={i}: winning_word discrepancy")
        pred = score > thr
        group = "TP" if gt and pred else "FN" if gt else "FP" if pred else "TN"
        instr = str((pois[i] if gt else clean[i-200]).get("instruction", ""))
        matchidx = [j for j, w in enumerate(pool) if matches_trigger(w, trigger)]
        rows.append({"idx": i, "gt": gt, "group": group, "score": score,
                     "cutoff": thr, "margin": score-thr,
                     "pool": pool, "vals": vals, "n_words_e8": int(r["n_words"]),
                     "top1": pool[win], "top2": pool[runner_up],
                     "top1_abs": vals[win], "top2_abs": vals[runner_up],
                     "trigger_in_pool": bool(matchidx),
                     "trigger_in_instruction": matches_trigger(instr, trigger),
                     "instruction": instr})
    observed = collections.Counter(r["group"] for r in rows)
    wanted = summary["counts_and_rates"]
    for name in ("TP", "FP", "FN", "TN"):
        if observed[name] != int(wanted[name]):
            raise ValueError(f"E8 summary mismatch {name}: {observed[name]} vs {wanted[name]}")
    return rows, thr


def audit_boundaries(rows, tok, helper, trigger):
    """Independent replay of tokenizer boundaries; NO model forwards."""
    issue_samples, all_glued = [], []
    for row in rows:
        _, _, words = tokenization(tok, helper, row["instruction"], 512)
        row["n_words_rebuilt"] = len(words)
        row["reconstructed_words"] = words
        labels = [w["label"] for w in words]
        row["pool_words_recoverable"] = not (collections.Counter(row["pool"]) - collections.Counter(labels))
        row["word_count_matches"] = len(words) == row["n_words_e8"]
        row["trigger_spans"] = [w for w in words if matches_trigger(w["label"], trigger)]
        row["winning_spans"] = [w for w in words if w["label"] == row["top1"]]
        row["winning_span_has_whitespace"] = any(
            w["decoded_multirun"] or w["source_multirun"] for w in row["winning_spans"])
        for w in words:
            if w["decoded_multirun"] or w["source_multirun"]:
                issue = {"idx": row["idx"], "group": row["group"], **w}
                all_glued.append(issue)
        if not row["pool_words_recoverable"] or not row["word_count_matches"]:
            issue_samples.append(row["idx"])
    return issue_samples, all_glued


def offline_report(rows, glued, mismatched, threshold, hashes, trigger):
    fn = [r for r in rows if r["group"] == "FN"]
    fp = [r for r in rows if r["group"] == "FP"]
    tp = [r for r in rows if r["group"] == "TP"]
    absent = [r for r in fn if not r["trigger_in_pool"]]
    compete = [r for r in fn if r["trigger_in_pool"]]
    out = []
    a = out.append
    a(f"{VERSION} 词级边界与漏检机制诊断；冻结 E8 预测，不调整阈值")
    a(f"E8 cutoff = {threshold:.8f}; counts: TP={len(tp)} FP={len(fp)} FN={len(fn)} TN={sum(r['group']=='TN' for r in rows)}")
    a("E8 data hashes verified: " + "; ".join(f"{k}={v[:12]}" for k,v in hashes.items()))
    a(f"[tokenizer replay] pool/word-count mismatches={len(mismatched)}/400; indices={mismatched[:25]}")
    a(f"[tokenizer replay] cross-whitespace raw word-spans={len(glued)} across {len(set(x['idx'] for x in glued))} prompts")
    a("NOTE: A crossing decoded/original span is a segmentation warning, not proof its Harsanyi score is wrong.")
    a("\n=== P0: suspicious word-span boundaries (all heldout, selected samples first) ===")
    if not glued:
        a("No cross-whitespace spans found; check tokenizer version before inferring no issues.")
    ordered = sorted(glued, key=lambda x: (x["idx"] not in {r["idx"] for r in fn+fp}, x["idx"], x["word_index"]))
    for q in ordered[:40]:
        a(f" idx={q['idx']:3d} group={q['group']} word#{q['word_index']} decoded={q['label']!r} "
          f"raw={q['original_substring']!r} tokens={q['tokens']!r} offsets=[{q['char_begin']},{q['char_end']}]")
    a("\n=== P1: 7 candidate-stage misses, trigger known ONLY post hoc ===")
    a(f"FN no {trigger} in Top-6={len(absent)}; FN with trigger in Top-6={len(compete)}")
    for r in absent:
        matches = r["trigger_spans"]
        a(f" idx={r['idx']:3d} n_words={r['n_words_rebuilt']} trigger_spans={len(matches)} "
          f"E8 pool={r['pool']!r} score={r['score']:.5f}")
        for w in matches:
            a(f"    trigger-tokenizer-span={w['label']!r}, tokens={w['tokens']!r}, "
              f"original={w['original_substring']!r}, offset=[{w['char_begin']},{w['char_end']}]")
        if not matches:
            a("    WARNING: trigger missing from reconstructed word spans; may be segmentation/truncation.")
    a("Screening rank cannot be recovered from E8 JSON: see GPU report for exact word ranks.")
    a("\n=== P2: 13 FN already localized: top-two singleton competition ===")
    a(f" TP median top1 |I|={median([r['top1_abs'] for r in tp]):.5f}, "
      f"top2={median([r['top2_abs'] for r in tp]):.5f}, gap={median([r['score'] for r in tp]):.5f}")
    a(f" FN with trigger in pool median top1={median([r['top1_abs'] for r in compete]):.5f}, "
      f"top2={median([r['top2_abs'] for r in compete]):.5f}, gap={median([r['score'] for r in compete]):.5f}" if compete else " none")
    for r in compete:
        competitor = r["top2"]
        a(f" idx={r['idx']:3d} trigger_winner={r['top1']!r} |I1|={r['top1_abs']:.5f}; "
          f"competitor={competitor!r} |I2|={r['top2_abs']:.5f}; "
          f"gap={r['score']:.5f}, shortfall={-r['margin']:.5f}, "
          f"I2/I1={r['top2_abs']/(r['top1_abs']+1e-12):.3f}")
    a("\n=== FP: strongest non-trigger words and suspicious boundaries ===")
    for r in fp:
        winners = r["winning_spans"]
        span = winners[0] if winners else {}
        a(f" idx={r['idx']:3d} word={r['top1']!r} |I1|={r['top1_abs']:.5f}; "
          f"runner-up={r['top2']!r} |I2|={r['top2_abs']:.5f}; "
          f"gap={r['score']:.5f}, excess={r['margin']:.5f}, "
          f"cross-whitespace={r['winning_span_has_whitespace']} raw={span.get('original_substring')!r}")
    a("\n=== Evidence boundary ===")
    a("(1) These are already-computed |I({w})| = |v({w})-v(empty)| from E8, not a signed/higher-order dividend.")
    a("(2) A large competing normal word reduces Top1-Gap arithmetically; it does NOT prove causal inhibition of a backdoor.")
    a("(3) Tokenizer audit reconstructs existing E8 word units; no altered tokens or predictions are evaluated.")
    a("(4) Label and known trigger are used for post-hoc error attribution, never to select or flag new samples.")
    return "\n".join(out)+"\n"


def write_error_csv(file, rows):
    with Path(file).open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            if r["group"] not in ("FP", "FN"):
                continue
            spans = r["winning_spans"]
            k = spans[0] if spans else {}
            w.writerow({"idx":r["idx"],"group":r["group"],"score":r["score"],
                        "cutoff":r["cutoff"],"margin":r["margin"],"top1":r["top1"],
                        "top1_abs":r["top1_abs"],"top2":r["top2"],"top2_abs":r["top2_abs"],
                        "trigger_in_pool":r["trigger_in_pool"],
                        "trigger_span_count":len(r["trigger_spans"]),
                        "trigger_span_labels":json.dumps([x["label"] for x in r["trigger_spans"]],ensure_ascii=False),
                        "n_words_e8":r["n_words_e8"],"n_words_rebuilt":r["n_words_rebuilt"],
                        "winning_span_has_whitespace":r["winning_span_has_whitespace"],
                        "winning_span_original":k.get("original_substring"),
                        "winning_span_token_pieces":json.dumps(k.get("tokens",[]),ensure_ascii=False),
                        "instruction":r["instruction"],
                        "pool_json":json.dumps(r["pool"],ensure_ascii=False),
                        "singleton_abs_json":json.dumps(r["vals"],ensure_ascii=False)})


def gpu_rescreen(rows, tok, helper, base, adapter, batch_size, trigger):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    suspects = [r for r in rows if r["group"]=="FN" and not r["trigger_in_pool"]]
    print(f"[GPU] Limited re-screen: {len(suspects)} missed-trigger FN only.")
    if not suspects:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run the shell with --offline-only if GPU is unavailable.")
    print(f"[GPU] base={base}, LoRA={adapter} (E8 settings, fp16)")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                                 device_map="auto",low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None: raise RuntimeError("Tokenizer has no pad/eos")
    device = next(model.parameters()).device
    cfg = dict(helper.DEFAULTS)
    cfg.update(pool_size=6,batch_size=batch_size,max_len=512)
    rows_out = []
    for r in suspects:
        ids, spans, words = tokenization(tok, helper, r["instruction"], 512)
        signals, _ = helper._screen_words(model,tok,ids,spans,None,device,cfg,pad)
        scores = [float(v) for v in signals[:,1]]
        idxs = np.argsort(-signals[:,1]).tolist()
        chosen = sorted(idxs[:min(6,len(spans))])
        recovered_pool = [words[i]["label"] for i in chosen]
        equal = recovered_pool == r["pool"]
        trig_ix = [i for i, w in enumerate(words) if matches_trigger(w["label"], trigger)]
        rank = min((idxs.index(j)+1 for j in trig_ix), default=None)
        sig = max((scores[j] for j in trig_ix),default=None)
        kth = scores[idxs[min(6,len(spans))-1]] if idxs else None
        details = {"idx":r["idx"],"topk_pool_match_E8":equal,
                   "trigger_screen_rank":rank,"n_words":len(spans),
                   "trigger_screen_raw_cosd":sig,"top6_cutoff_raw_cosd":kth,
                   "gap_top6_minus_trigger":(kth-sig if sig is not None and kth is not None else None),
                   "trigger_span_labels":[words[j]["label"] for j in trig_ix],
                   "topk_rebuilt":recovered_pool, "topk_E8":r["pool"],
                   "screen_top10":[{"rank":rankno+1,"word":words[j]["label"],
                                    "cos_distance":scores[j]} for rankno,j in enumerate(idxs[:10])],
                   "instruction":r["instruction"]}
        rows_out.append(details)
        print(f" [GPU] idx={r['idx']} trigger_rank={rank}/{len(spans)} "
              f"K=6, raw_cos={fmt(sig)} cutoff={fmt(kth)} "
              f"E8_pool_match={equal}")
    del model
    return rows_out


def gpu_report(results):
    lines=[f"{VERSION}: E8 candidate screening re-run on 7 known-trigger FN (diagnostic only)"]
    lines.append(f"N={len(results)}, exact E8 Top-6 pool reproduced={sum(r['topk_pool_match_E8'] for r in results)}/{len(results)}")
    for r in results:
        lines.append(f"idx={r['idx']} rank={r['trigger_screen_rank']}/{r['n_words']} "
                     f"trigger_cos={fmt(r['trigger_screen_raw_cosd'])} "
                     f"Kth_cos={fmt(r['top6_cutoff_raw_cosd'])} "
                     f"E8_pool_match={r['topk_pool_match_E8']}")
        lines.append("  top10=" + "; ".join(f"{x['rank']}:{x['word']!r}({x['cos_distance']:.6g})" for x in r["screen_top10"]))
    lines.append("No Harsanyi game was recomputed. Screen ranks are reproduced under identical E8 screening, not used for inference.")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e8-run", default=E8_RUN)
    p.add_argument("--output", required=True)
    p.add_argument("--poison",default=POISON)
    p.add_argument("--clean",default=CLEAN)
    p.add_argument("--helper",default=HELPER)
    p.add_argument("--base",default=BASE)
    p.add_argument("--adapter",default=ADAPTER)
    p.add_argument("--trigger",default="BadMagic",help="Post-hoc diagnostic label ONLY")
    p.add_argument("--offline-only",action="store_true")
    p.add_argument("--batch-size",type=int,default=16)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(f"Will not overwrite existing output: {out}")
    e8=Path(args.e8_run)
    for f in (e8/"summary.json",e8/"heldout_records.json",args.poison,args.clean,args.helper):
        if not Path(f).is_file():
            raise FileNotFoundError(f"Missing required frozen E8 input: {f}")
    summary=load_json(e8/"summary.json")
    hashes={str(f):check_hash(summary,f) for f in (args.poison,args.clean,args.helper)}
    if str(summary.get('version'))!='IP_E8R1_20261111':
        raise ValueError('Unexpected E8 summary version')
    records=load_json(e8/"heldout_records.json")
    rows,threshold=rebuild_records(records,summary,load_json(args.poison),load_json(args.clean),args.trigger)
    print(f"[verify] E8 400 records and checksums validated; cutoff={threshold:.8f} (frozen)")
    tok=load_tokenizer(args.base)
    helper=get_helper(args.helper)
    mismatched, glued=audit_boundaries(rows,tok,helper,args.trigger)
    out.mkdir(parents=True)
    report=offline_report(rows,glued,mismatched,threshold,hashes,args.trigger)
    (out/"E9B_offline_report.txt").write_text(report,encoding="utf-8")
    write_error_csv(out/"E9B_24_errors.csv",rows)
    save_json(out/"E9B_boundary_findings.json",glued)
    save_json(out/"E9B_diagnostic_manifest.json",{"version":VERSION,"e8_run":str(e8),
           "e8_threshold":threshold,"input_hashes":hashes,"reconstructed_mismatches":mismatched,
           "cross_whitespace_spans":len(glued),"gpu_rescreen_requested":not args.offline_only,
           "provenance":"post-hoc; labels/trigger used only in diagnostics; no E8 pred altered"})
    print(report)
    if args.offline_only:
        print("[offline] Skipped the 7-prompt model re-screen; E8 original predictions unchanged.")
    else:
        results=gpu_rescreen(rows,tok,helper,args.base,args.adapter,args.batch_size,args.trigger)
        save_json(out/"E9B_7fn_rescreen.json",results)
        gpu_text=gpu_report(results)
        (out/"E9B_7fn_screen_report.txt").write_text(gpu_text,encoding="utf-8")
        print(gpu_text)
    print(f"[SUCCESS] {VERSION} done; outputs={out}")


if __name__=="__main__":
    main()
