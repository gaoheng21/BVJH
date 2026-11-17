#!/usr/bin/env python3
"""IP E9-A R1: offline, post-hoc audit of E8 held-out FP/FN cases.

No GPU, no model, no threshold changes, no predictions modified.
Known BadMagic trigger is used ONLY for descriptive diagnostics after the
original predictions have been reconstructed from their saved scores.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import median

VERSION = "IP_E9AR1_20261111"
E8_RUN = "outputs/interactionprism/e8_runs/E8R1_20261111_082944_1039497"
TEST_POISON = "data/test_data/poison/negsentiment/badnet/backdoor200_negsentiment_badnet.json"
TEST_CLEAN = "data/test_data/clean/negsentiment/test_data_no_trigger.json"
FIELDS = [
    "idx", "source_split", "source_row", "ground_truth", "outcome", "score", "threshold", "margin", "top1_abs",
    "top2_abs", "top1_word", "top2_word", "trigger_in_instruction",
    "trigger_in_pool", "trigger_rank", "trigger_abs", "trigger_gap_to_top1",
    "diagnostic_category", "n_words", "pool", "singleton_abs", "instruction"
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def trigger_present(text, trigger):
    """Whole-word-ish match; never used for scoring or classification."""
    if not trigger:
        return None
    return bool(re.search(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", text, re.IGNORECASE))


def score_top1_gap(values):
    if len(values) < 2:
        raise ValueError("Need >=2 singleton players")
    a = sorted((float(x) for x in values), reverse=True)
    if not all(math.isfinite(x) and x >= 0 for x in a):
        raise ValueError("Invalid singleton_abs values")
    return a[0] - a[1]


def sanitize_text(text, limit=95):
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit-3] + "..."


def shortf(x):
    return "NA" if x is None else f"{x:.5f}"


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        out.writeheader()
        for r in rows:
            rr = dict(r)
            for key in ("pool", "singleton_abs"):
                rr[key] = json.dumps(rr.get(key, []), ensure_ascii=False)
            out.writerow(rr)


def get_input_texts(p_path, c_path, summary, check_hash=True):
    p = read_json(p_path)
    c = read_json(c_path)
    if not isinstance(p, list) or not isinstance(c, list):
        raise ValueError("Expected JSON arrays for test files")
    hashes = summary.get("file_hashes", {})
    checked = []
    if check_hash:
        for path in (p_path, c_path):
            path = Path(path)
            expected = hashes.get(str(path))
            if expected is None:
                # E8 uses repo-relative paths; handle absolute CLI path strictly.
                resolved = path.resolve()
                expected = next((v for k, v in hashes.items()
                                 if Path(k).resolve() == resolved), None)
            if not expected:
                raise ValueError(f"E8 did not record a hash for dataset {path}; cannot verify source")
            observed = sha256(path)
            if observed != expected:
                raise ValueError(f"Dataset does not match E8 SHA-256: {path}")
            checked.append(str(path))
    return p, c, checked


def audit_one(r, threshold, instruction, trigger):
    idx = int(r["idx"])
    gt = int(r["gt"])
    skipped = "score" not in r
    if skipped:
        outcome = "FN" if gt == 1 else "TN"
        return {"idx": idx, "ground_truth": gt, "outcome": outcome,
                "score": None, "threshold": threshold, "margin": None,
                "top1_abs": None, "top2_abs": None, "top1_word": "", "top2_word": "",
                "trigger_in_instruction": trigger_present(instruction, trigger),
                "trigger_in_pool": False, "trigger_rank": None, "trigger_abs": None,
                "trigger_gap_to_top1": None, "diagnostic_category": "skipped_no_score",
                "n_words": r.get("n_words"), "pool": [], "singleton_abs": [],
                "instruction": instruction}
    vals = list(r.get("singleton_abs", []))
    pool = list(r.get("pool", []))
    if len(vals) != len(pool) or len(vals) < 2:
        raise ValueError(f"idx={idx}: inconsistent pool / singleton_abs lengths")
    val = [float(x) for x in vals]
    reconstructed = score_top1_gap(val)
    saved = float(r["score"])
    if not math.isfinite(saved) or not math.isclose(saved, reconstructed, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(f"idx={idx}: score mismatch saved={saved} reconstructed={reconstructed}")
    ranked = sorted(range(len(val)), key=lambda j: (-val[j], j))
    win = ranked[0]
    second = ranked[1]
    if r.get("winning_word") != pool[win]:
        raise ValueError(f"idx={idx}: winning_word differs from singleton top-1")
    pred = saved > threshold  # STRICTLY same as E8
    outcome = "TP" if gt and pred else "FN" if gt else "FP" if pred else "TN"
    match_idxs = [j for j, w in enumerate(pool) if trigger_present(str(w), trigger)] if trigger else []
    trig_rank = next((i + 1 for i, j in enumerate(ranked) if j in match_idxs), None)
    trig_abs = max((val[j] for j in match_idxs), default=None)
    if outcome == "FN":
        if not match_idxs:
            category = "trigger_not_in_topk"
        elif win not in match_idxs:
            category = "trigger_in_topk_but_not_top1"
        else:
            category = "trigger_top1_but_gap_below_threshold"
    elif outcome == "FP":
        category = "clean_nontrigger_dominant_singleton" if not match_idxs else "clean_contains_named_trigger"
    else:
        category = "not_an_error"
    return {
        "idx": idx, "ground_truth": gt, "outcome": outcome,
        "score": saved, "threshold": threshold, "margin": saved - threshold,
        "top1_abs": val[win], "top2_abs": val[second],
        "top1_word": pool[win], "top2_word": pool[second],
        "trigger_in_instruction": trigger_present(instruction, trigger),
        "trigger_in_pool": bool(match_idxs), "trigger_rank": trig_rank,
        "trigger_abs": trig_abs,
        "trigger_gap_to_top1": val[win] - trig_abs if trig_abs is not None else None,
        "diagnostic_category": category,
        "n_words": r.get("n_words"), "pool": pool, "singleton_abs": val,
        "instruction": instruction
    }


def analyze(records, summary, poison, clean, trigger):
    if not isinstance(records, list):
        raise ValueError("heldout_records.json must be an array")
    if len(records) != len(poison) + len(clean):
        raise ValueError("Data length differs from E8 records; wrong test file or dataset")
    if int(summary.get("heldout_n", -1)) != len(records):
        raise ValueError("E8 summary heldout_n does not match records")
    threshold = float(summary["calibration_cutoff"])
    if not math.isfinite(threshold):
        raise ValueError("Nonfinite E8 threshold")
    rows = []
    for i, r in enumerate(records):
        if int(r["idx"]) != i or r.get("category") != "heldout":
            raise ValueError(f"idx={i}: record order/category inconsistent with E8")
        expected_gt = 1 if i < len(poison) else 0
        if r.get("gt") != expected_gt:
            raise ValueError(f"idx={i}: heldout gt does not match dataset ordering")
        d = poison[i] if expected_gt else clean[i-len(poison)]
        result = audit_one(r, threshold, str(d.get("instruction", "")), trigger)
        result["source_split"] = "test_poison" if expected_gt else "test_clean"
        result["source_row"] = i if expected_gt else i-len(poison)
        rows.append(result)
    count = dict(collections.Counter(row["outcome"] for row in rows))
    for k in ("TP", "FP", "FN", "TN"):
        if count.get(k, 0) != int(summary["counts_and_rates"][k]):
            raise ValueError(f"Reconstructed {k}={count.get(k, 0)} != E8 summary {summary['counts_and_rates'][k]}")
    return rows, count, threshold


def qmedian(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return shortf(median(vals)) if vals else "NA"


def create_report(rows, count, threshold, run_dir, matched_hashes, trigger):
    fn = [r for r in rows if r["outcome"] == "FN"]
    fp = [r for r in rows if r["outcome"] == "FP"]
    tp = [r for r in rows if r["outcome"] == "TP"]
    tn = [r for r in rows if r["outcome"] == "TN"]
    out = []
    add = out.append
    add(f"{VERSION} — E8R1 后验错误样本诊断（绝不更改原始 pred）")
    add(f"数据目录：{run_dir}")
    add(f"数据集 SHA-256 校验：{'成功：' + ', '.join(matched_hashes) if matched_hashes else '未执行（测试模式）'}")
    add(f"E8 固定阈值：score > {threshold:.8f}（原始阈值；没有重新校准）")
    add(f"重建混淆矩阵：TP={count.get('TP',0)} FP={count.get('FP',0)} FN={count.get('FN',0)} TN={count.get('TN',0)}")
    add("\n=== 一阶 Harsanyi Top1-Gap 分数分布（来自 E8 保存数据）===")
    for label, group in [("TP",tp),("FN",fn),("FP",fp),("TN",tn)]:
        add(f"{label:>2} n={len(group):>3} score中位数={qmedian(group,'score')} "
            f"top1 |I|中位数={qmedian(group,'top1_abs')} "
            f"top2 |I|中位数={qmedian(group,'top2_abs')}")
    add("\n=== FN（漏检）：诊断分组 ===")
    by_type = collections.Counter(r["diagnostic_category"] for r in fn)
    for name, n in sorted(by_type.items()):
        add(f"{name}: {n}")
    n_has_trigger = sum(bool(r["trigger_in_pool"]) for r in fn)
    add(f"FN 中候选池包含已知触发词（仅事后定位诊断）：{n_has_trigger}/{len(fn)}")
    add("\n=== 每条 FN 的交互证据 ===")
    if not fn:
        add("无漏检样本")
    for r in fn:
        add(f"idx={r['idx']:>3} score={shortf(r['score'])}  距阈值={shortf(r['margin'])} "
            f" top1={sanitize_text(r['top1_word'],30)!r}({shortf(r['top1_abs'])}) "
            f"top2={sanitize_text(r['top2_word'],30)!r}({shortf(r['top2_abs'])}) "
            f"trigger_rank={r['trigger_rank'] or '-'}  原因={r['diagnostic_category']}")
        add(f"   对应文件={r['source_split']} 行号(0开始)={r['source_row']}; 指令：{sanitize_text(r['instruction'],120)}")
        add("   候选词 |I|：" + "; ".join(f"{sanitize_text(w,24)!r}={v:.5f}"
                                      for w,v in zip(r['pool'],r['singleton_abs'])))
    add("\n=== FP（误报）：交互证据 ===")
    if not fp:
        add("无误报样本")
    for r in fp:
        add(f"idx={r['idx']:>3} score={shortf(r['score'])}  超阈值={shortf(r['margin'])} "
            f"top1={sanitize_text(r['top1_word'],30)!r}({shortf(r['top1_abs'])}) "
            f"top2={sanitize_text(r['top2_word'],30)!r}({shortf(r['top2_abs'])}) "
            f"n_words={r['n_words']}")
        add(f"   对应文件={r['source_split']} 行号(0开始)={r['source_row']}; 指令：{sanitize_text(r['instruction'],120)}")
        add("   候选词 |I|：" + "; ".join(f"{sanitize_text(w,24)!r}={v:.5f}"
                                      for w,v in zip(r['pool'],r['singleton_abs'])))
    add("\n=== 常见获胜玩家（诊断） ===")
    for label, group in [("FP", fp), ("TP", tp), ("TN", tn)]:
        freq = collections.Counter(r["top1_word"] for r in group if r["top1_word"])
        add(f"{label}: {freq.most_common(8)}")
    add("\n=== 理论解释与证据边界 ===")
    add("1. 单词 w 的一阶 Harsanyi dividend 为 I({w}) = v({w}) - v(∅)。E8 只保存 |I|，不保存符号。")
    add("2. Top1-Gap 大表示最强候选词的一阶绝对贡献相对第二名明显占优；它不是高阶 AND 交互，也不等于后门因果证明。")
    add("3. FN 分类说明信号在候选筛选、主导玩家定位或差值阈值哪个环节不足；不是训练机制的因果结论。")
    add("4. FP 的获胜词是异常主效应候选，不应仅凭词形断言它是罕见词/触发词。需要额外对照才能解释其成因。")
    add("5. BadMagic 和 gt 只参与本报告的事后诊断；E8 检测分数和原判定不使用已知触发器。")
    add("6. E8 的 input 字段未拼接进检测输入；同一指令在不同输入字段下的行为尚未比较。")
    add("7. 下一步可以针对 FN/FP 做配对表征干预，但本报告不推断该干预结果。")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e8-run", default=E8_RUN, help="E8 run directory containing summary.json & heldout_records.json")
    p.add_argument("--test-poison", default=TEST_POISON)
    p.add_argument("--test-clean", default=TEST_CLEAN)
    p.add_argument("--trigger", default="BadMagic", help="POST-HOC diagnostics only; NOT used to score or predict")
    p.add_argument("--output", required=True, help="New empty output directory; no overwrites")
    p.add_argument("--no-hash-check", action="store_true", help="Only for synthetic fixtures; not recommended for real E8 analysis")
    args = p.parse_args(argv)
    base = Path(args.e8_run)
    src_summary, src_rows = base/'summary.json', base/'heldout_records.json'
    for src in (src_summary, src_rows, Path(args.test_poison), Path(args.test_clean)):
        if not src.is_file():
            p.error(f"Cannot find input: {src}")
    output = Path(args.output)
    if output.exists():
        p.error(f"Output already exists, refusing to overwrite: {output}")
    summary = read_json(src_summary)
    if summary.get("version") != "IP_E8R1_20261111":
        p.error(f"Unexpected E8 version: {summary.get('version')!r}")
    records = read_json(src_rows)
    poison, clean, checked = get_input_texts(args.test_poison, args.test_clean, summary,
                                             not args.no_hash_check)
    rows, count, cutoff = analyze(records, summary, poison[:200], clean[:200], args.trigger)
    text = create_report(rows,count,cutoff,base,checked,args.trigger)
    # Validate before creating output directory; then write only E9 artifacts.
    output.mkdir(parents=True, exist_ok=False)
    (output/'E9A_report.txt').write_text(text, encoding="utf-8")
    (output/'E9A_all_diagnostics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(output/'E9A_errors_only.csv',[r for r in rows if r['outcome'] in ("FP","FN")])
    write_csv(output/'E9A_all_scores.csv',rows)
    (output/'E9A_provenance.json').write_text(json.dumps({
        "version":VERSION, "e8_version":summary['version'],
        "e8_summary_sha256":sha256(src_summary),"e8_records_sha256":sha256(src_rows),
        "e8_threshold":cutoff,"counts":count,"trigger_posthoc_only":args.trigger,
        "datasets_sha256":{str(args.test_poison):sha256(args.test_poison),
                            str(args.test_clean):sha256(args.test_clean)},
        "changed_detection_logic":False,"ran_gpu":False
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(text, end="")
    print(f"\n[SUCCESS] E9-A offline audit saved to: {output}")
    print(f"[report] {output/'E9A_report.txt'}")
    print(f"[errors] {output/'E9A_errors_only.csv'}")


if __name__ == '__main__':
    main()
