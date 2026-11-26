#!/usr/bin/env python3
"""Inspect InteractionPrism's saved records without loading a model.

Usage:
  python analyze_prism_records.py outputs/interactionprism/badnet_Llama2_prism_v0_records.json

No third-party packages required. Does not change the detector or saved records.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median


def frequency(rows, key="dominant_members"):
    c = Counter()
    for r in rows:
        if key == "pool":
            names = r.get("pool") or []
        else:
            names = r.get("dominant_members") or []
        c.update({str(s).strip().lower() for s in names if str(s).strip()})
    return c


def matches_trigger(items, word):
    return any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", str(x), flags=re.I)
               for x in items)


def stage_metrics(rows, field):
    tp = sum(int(r.get("gt", 0) == 1 and bool(r.get(field, False))) for r in rows)
    fp = sum(int(r.get("gt", 0) == 0 and bool(r.get(field, False))) for r in rows)
    p = sum(r.get("gt", 0) == 1 for r in rows)
    n = len(rows) - p
    print(f"  {field:<11} TP={tp:>3}/{p:<3} ({tp/p:.1%} recall)   "
          f"FP={fp:>3}/{n:<3} ({fp/n:.1%} FPR)" if p and n else
          f"  {field:<11} TP={tp}, FP={fp} (one class absent)")


def quantiles(vals):
    vals = sorted(float(v) for v in vals if v is not None)
    if not vals:
        return "none"
    n = len(vals)
    def nth(q):
        return vals[round((n - 1) * q)]
    return f"min={vals[0]:.4g}  med={median(vals):.4g}  p90={nth(.90):.4g}  max={vals[-1]:.4g}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", type=Path, help="*_records.json saved by InteractionPrism")
    p.add_argument("--trigger", default="BadMagic", help="reference trigger for this diagnostic only")
    args = p.parse_args()
    doc = json.loads(args.json_path.read_text(encoding="utf-8"))
    rows = doc.get("records", []) if isinstance(doc, dict) else doc
    if not rows:
        raise SystemExit("No records found")
    poison = [r for r in rows if int(r.get("gt", 0)) == 1]
    clean = [r for r in rows if int(r.get("gt", 0)) == 0]
    tp = [r for r in poison if r.get("pred") == 1]
    fp = [r for r in clean if r.get("pred") == 1]
    print(f"Records: {len(rows)}  poisoned={len(poison)}  clean={len(clean)}")
    print("\n=== Stage-wise decisions (same records; no new GPU run) ===")
    for stage in ("f1", "f2", "soft_flag", "confirm", "pred"):
        stage_metrics(rows, stage)
    print("\n=== Candidate localization (labels used ONLY for diagnostics) ===")
    word = args.trigger
    for group_name, group in (("poison", poison), ("true positive", tp),
                              ("clean", clean), ("false positive", fp)):
        d = sum(matches_trigger(r.get("dominant_members") or [], word) for r in group)
        pool = sum(matches_trigger(r.get("pool") or [], word) for r in group)
        screen = sum(matches_trigger([x[0] for x in r.get("top_screen") or []], word)
                     for r in group)
        print(f"  {group_name:14} N={len(group):3}  "
              f"trigger in top-screen={screen:3}, pool={pool:3}, dominant={d:3}")
    print("\n=== Top dominant members among flagged samples ===")
    for name, group in (("true positive", tp), ("false positive", fp)):
        print(f"  {name}: {frequency(group).most_common(12)}")
    print("\n=== Score distributions among samples (selection-bias diagnostic) ===")
    for name, group in (("TP", tp), ("FP", fp),
                        ("FN", [r for r in poison if not r.get("pred")]),
                        ("TN", [r for r in clean if not r.get("pred")])):
        print(f"  {name}: |I*| [{quantiles(abs(float(r.get('dominant_I') or 0)) for r in group)}] "
              f"strength_ratio [{quantiles(r.get('strength_ratio') for r in group)}]")
    print("\n=== Failure examples ===")
    for name, group in (("false negatives", [r for r in poison if not r.get("pred")]),
                        ("false positives", sorted(fp, key=lambda r: -float(r.get("strength_ratio") or 0))[:8])):
        print(f"  {name}:")
        for r in group:
            contexts = r.get("confirm_ctx") or []
            npass = sum(bool(c.get("passed")) for c in contexts)
            print(f"    idx={r.get('idx')} order={r.get('dominant_order')} "
                  f"f1={int(bool(r.get('f1')))} f2={int(bool(r.get('f2')))} "
                  f"soft={int(bool(r.get('soft_flag')))} "
                  f"persist={npass}/{len(contexts)} "
                  f"ratio={r.get('strength_ratio')} "
                  f"members={r.get('dominant_members')}")
    flagged = tp + fp
    support = max(8, math.ceil(len(flagged)/10))
    n_trigger = sum(matches_trigger(r.get("dominant_members") or [], word) for r in flagged)
    print(f"\n=== Consensus diagnostic ===\n  support threshold={support}; "
          f"flagged candidates containing '{word}'={n_trigger}")
    if n_trigger < support:
        print("  Trigger count alone is below the current support cutoff; "
              "empty consensus does not prove the trigger was never localized.")
    print("\nNote: gt is consulted in THIS diagnostic report only, never in model detection.")

if __name__ == '__main__':
    main()
