#!/usr/bin/env python
"""Score FUSED rules on an existing FPR ledger and refuse if the population moved.

The rule this script exists to enforce: **a fused number may only be published
beside per-signal numbers that were measured on the same population.** Fusion is
cheap to compute from a ledger, and a ledger can be rebuilt, truncated or resumed
between the two runs -- the fleet ledger grew from 6,815 to 7,101 to 8,064 models
inside a single session. Publishing "fused FPR 0.051, per-signal 0.048" from two
different populations would be a silent, unfalsifiable comparison.

So before writing anything, the script re-derives every per-signal rule from the
ledger it is about to fuse and compares it with the published per-signal report
(same split seed). If a single numerator, denominator or threshold disagrees, the
run stops with exit 3 and prints both numbers -- the fix is to re-score the
per-signal report, not to accept the mismatch.

Usage
-----
    python scripts/score_fleet_fusion.py --ledger runs/fleet_fpr_ledger.json \\
        --published runs/fleet_fpr_ledger_report.json \\
        --out runs/fleet_fpr_fusion.json --split-seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from cviaf.lab.fpr_tpr import LedgerError, evaluate_corpus, load_ledger
from cviaf.lab.fusion_fpr import fusion_report, render_fusion

# A threshold reproduces exactly or it does not; the only slack allowed is the
# float round-trip through the published JSON.
THRESHOLD_TOL = 1e-9


def _digest(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reproduction_problems(ledger: Dict[str, Any], published: Dict[str, Any],
                          alpha: float, min_negatives: int, split_seed: int,
                          ) -> List[str]:
    """Per-signal rules that do not reproduce the published report, as strings."""
    fresh = evaluate_corpus(ledger, alpha=alpha, min_negatives=min_negatives,
                            split_seed=split_seed)
    problems: List[str] = []
    if published.get("alpha") != fresh["alpha"]:
        problems.append(f"alpha differs: published {published.get('alpha')} vs "
                        f"ledger {fresh['alpha']}")
    for key in ("calibration_negatives", "evaluation_negatives", "evaluation_positives"):
        if published.get("denominators", {}).get(key) != fresh["denominators"][key]:
            problems.append(
                f"denominator {key} differs: published "
                f"{published.get('denominators', {}).get(key)} vs now "
                f"{fresh['denominators'][key]} -- the ledger being fused is not the "
                f"population the per-signal table was measured on")
    pub_rules, new_rules = published.get("rules", {}), fresh["rules"]
    missing = [s for s in pub_rules if s not in new_rules]
    if missing:
        problems.append(f"published signals absent from this ledger: {missing}")
    for name, pub in pub_rules.items():
        new = new_rules.get(name)
        if new is None:
            continue
        if pub.get("status") != new.get("status"):
            problems.append(f"{name}: status {pub.get('status')} -> {new.get('status')}")
            continue
        if new.get("status") == "refused":
            continue
        if abs(float(pub["threshold"]) - float(new["threshold"])) > THRESHOLD_TOL:
            problems.append(f"{name}: threshold {pub['threshold']} -> {new['threshold']}")
        for k in ("numerator", "denominator"):
            if pub["fpr"].get(k) != new["fpr"].get(k):
                problems.append(f"{name}: FPR {k} {pub['fpr'].get(k)} -> "
                                f"{new['fpr'].get(k)}")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="runs/fleet_fpr_ledger.json")
    ap.add_argument("--published", default="runs/fleet_fpr_ledger_report.json",
                    help="the per-signal report the fused numbers will sit beside")
    ap.add_argument("--out", default=None, help="write the fused report JSON here")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--min-negatives", type=int, default=20)
    ap.add_argument("--split-seed", type=int, default=7)
    args = ap.parse_args(argv)

    try:
        ledger = load_ledger(args.ledger)
    except LedgerError as exc:
        print(f"INVALID LEDGER: {exc}", file=sys.stderr)
        return 3
    alpha = float(args.alpha if args.alpha is not None else ledger.get("alpha", 0.05))

    if os.path.isfile(args.published):
        with open(args.published, encoding="utf-8") as fh:
            published = json.load(fh)
        problems = reproduction_problems(ledger, published, alpha,
                                         args.min_negatives, args.split_seed)
        if problems:
            print("REFUSING to publish a fused FPR: the ledger no longer reproduces "
                  "the published per-signal table.", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print("  re-run the per-signal report on this ledger (or point --published "
                  "at the matching report) before quoting a fused number beside it.",
                  file=sys.stderr)
            return 3
    else:
        print(f"note: no published per-signal report at {args.published}; the "
              f"reproduction guard did not run", file=sys.stderr)

    try:
        report = fusion_report(ledger, alpha=alpha, min_negatives=args.min_negatives,
                               split_seed=args.split_seed)
    except LedgerError as exc:
        print(f"cannot fuse this ledger: {exc}", file=sys.stderr)
        return 3

    report["inputs"] = {
        "ledger": args.ledger,
        "ledger_sha256": _digest(args.ledger),
        "published_per_signal_report": args.published if os.path.isfile(args.published)
        else None,
        "published_per_signal_report_sha256": _digest(args.published),
        "per_signal_reproduced": bool(os.path.isfile(args.published)),
        "split_seed": args.split_seed,
    }
    print(render_fusion(report))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"\nfused report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
