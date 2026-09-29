#!/usr/bin/env python
"""Is the fleet *one* population? Ask before averaging an FPR over it.

An FPR averaged over seven thousand models is only meaningful if those models are
exchangeable with the threshold's calibration half. If the shards differ (a different
trainer invocation, a different reference model, a machine that changed mid-run), the
average is a mixture, and the threshold transfers to none of them. The harness already
refuses a pooled number when the *calibration* half disagrees with the evaluation half
(the ``CALIBRATION ALARM``); this script asks the same question across the shards and
across split seeds, so the fleet can be declared one population or not *before* the
20k average is printed.

Two independent instabilities are reported, because they have different fixes:

  * split instability -- re-splitting the same models into calibration/evaluation
    halves moves the FPR. Small spread means the number is a property of the corpus;
    large spread means it is a property of the split, and the corpus is too small.
  * shard heterogeneity -- the per-shard FPRs disagree by more than binomial noise.
    Measured with a chi-square homogeneity statistic (Cochran's Q) and with an
    interval-overlap check, which is the version a reader can verify by eye.

Both are reported with the refusal framing the rest of this lane uses: a verdict, not
just numbers.

CLI
---
    python scripts/fpr_population_check.py --input runs/fleet_fpr_ledger.json \
        [--out runs/fleet_fpr_population.json] [--alpha 0.05] \
        [--min-negatives 20] [--seeds 0 1 2 3 4]

Exit: 0 one population (both checks pass), 1 not one population, 2 nothing to read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.fpr_tpr import (  # noqa: E402
    LedgerError,
    evaluate_corpus,
    load_ledger,
    wilson_interval,
)

POPULATION_SCHEMA = "cviaf.fpr-population.v1"


# --------------------------------------------------------------------------- #
# dependence-free chi-square tail (so this script runs on the numpy-only venv)
# --------------------------------------------------------------------------- #

def _gamma_q(a: float, x: float, itmax: int = 300, eps: float = 1e-14) -> float:
    """Upper regularised incomplete gamma Q(a, x).

    Series for x < a+1, continued fraction otherwise (Numerical Recipes 6.2). Kept
    here rather than imported so the check runs on the canonical numpy venv, which has
    no scipy -- an analysis that only runs in one interpreter is how a lane ends up
    with two answers.
    """
    if x < 0 or a <= 0:
        raise ValueError("Q(a, x) needs a > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # Q = 1 - P, with P by series
        term = 1.0 / a
        total = term
        n = a
        for _ in range(itmax):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * eps:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for Q
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(statistic: float, df: int) -> float:
    """P(chi2_df >= statistic), no scipy."""
    if df <= 0:
        raise ValueError("df must be positive")
    if statistic <= 0:
        return 1.0
    return _gamma_q(df / 2.0, statistic / 2.0)


def homogeneity(groups: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """Cochran's Q over (successes, trials) groups -- do they share one rate?

    The pooled rate is the natural weighted mean. A large Q relative to df says the
    groups scatter by more than binomial noise around it. Groups with no trials are
    dropped, not counted as agreement.
    """
    live = [(k, n) for k, n in groups if n > 0]
    if len(live) < 2:
        return {"status": "not_measurable", "reason": "fewer than two non-empty groups",
                "n_groups": len(live)}
    total_n = sum(n for _, n in live)
    total_k = sum(k for k, _ in live)
    pooled = total_k / total_n
    if pooled in (0.0, 1.0):
        # Every group agrees exactly; Q is defined as 0 and a p-value is undefined.
        return {"status": "measured", "pooled_rate": pooled, "n_groups": len(live),
                "statistic": 0.0, "df": len(live) - 1, "p_value": 1.0,
                "verdict": "homogeneous", "note": "the pooled rate is 0 or 1: no scatter possible"}
    stat = sum((k - n * pooled) ** 2 / (n * pooled * (1.0 - pooled)) for k, n in live)
    df = len(live) - 1
    p = chi2_sf(stat, df)
    return {"status": "measured", "pooled_rate": pooled, "n_groups": len(live),
            "n_trials": total_n, "statistic": stat, "df": df, "p_value": p,
            "verdict": "homogeneous" if p >= 0.05 else "heterogeneous",
            "note": "p < 0.05 means the shard rates differ by more than binomial noise: "
                    "the pooled FPR is a mixture, not a property of the fleet"}


# --------------------------------------------------------------------------- #
# the two checks
# --------------------------------------------------------------------------- #

def _rule_fpr(report: Dict[str, Any], signal: str) -> Optional[Dict[str, Any]]:
    rule = report.get("rules", {}).get(signal, {})
    fpr = rule.get("fpr") or {}
    if fpr.get("status") != "measured":
        return None
    return {"numerator": fpr["numerator"], "denominator": fpr["denominator"],
            "rate": fpr["point_estimate"], "ci95": fpr.get("ci95_wilson")}


def split_stability(ledger: Dict[str, Any], seeds: Sequence[int], alpha: float,
                    min_negatives: int, signals: Optional[Sequence[str]] = None
                    ) -> Dict[str, Any]:
    """Re-split the same negatives N ways; how much does each signal's FPR move?"""
    out: Dict[str, Any] = {"seeds": list(seeds), "signals": {}}
    collected: Dict[str, List[Dict[str, Any]]] = {}
    for seed in seeds:
        report = evaluate_corpus(ledger, alpha=alpha, min_negatives=min_negatives,
                                 split_seed=seed)
        use = signals or sorted(report["rules"])
        for signal in use:
            fpr = _rule_fpr(report, signal)
            if fpr:
                collected.setdefault(signal, []).append({"seed": seed, **fpr})
    for signal, rows in collected.items():
        rates = [r["rate"] for r in rows]
        # Test the across-seed scatter against binomial noise instead of against the
        # interval width. The range of 5 draws from a binomial already spans ~2.3 SE,
        # so "range vs CI width" would refuse every honest measurement; overdispersion
        # is the question actually being asked: does the split carry information?
        stats = homogeneity([(r["numerator"], r["denominator"]) for r in rows])
        out["signals"][signal] = {
            "n_seeds": len(rows),
            "min": min(rates), "max": max(rates),
            "spread": max(rates) - min(rates),
            "mean": sum(rates) / len(rates),
            "homogeneity": stats,
            "per_seed": rows,
        }
    return out


def reference_corpus(ledger: Dict[str, Any]) -> Optional[str]:
    """Which shard holds the reference model, if the producer recorded it?

    A reference-relative signal is measured against one model. If that model is a
    member of one shard, that shard's models are systematically closer to it, and the
    shard will show a *lower* FPR for no reason the fleet shares. The producer already
    excludes the reference itself from the scored population; the shard it came from
    is still there.
    """
    provenance = ledger.get("provenance") or {}
    hint = provenance.get("reference_dir") or provenance.get("reference_model") or ""
    corpora = sorted({r["corpus"] for r in ledger["records"]})
    hint_parts = [p for p in str(hint).replace("\\", "/").split("/") if p and p != ".."]
    for corpus in corpora:
        if os.path.basename(corpus.rstrip("/")) in hint_parts:
            return corpus
    return None


def shard_heterogeneity(ledger: Dict[str, Any], alpha: float, min_negatives: int,
                        split_seed: int, reference: Optional[str] = None
                        ) -> Dict[str, Any]:
    """Pooled FPR vs each shard's own FPR, with a homogeneity statistic per signal."""
    corpora = sorted({r["corpus"] for r in ledger["records"]})
    out: Dict[str, Any] = {"corpora": corpora, "reference_corpus": reference,
                           "signals": {}}
    per_shard: Dict[str, List[Dict[str, Any]]] = {}
    for corpus in corpora:
        sub = dict(ledger)
        sub["records"] = [r for r in ledger["records"] if r["corpus"] == corpus]
        try:
            report = evaluate_corpus(sub, alpha=alpha, min_negatives=min_negatives,
                                     split_seed=split_seed)
        except LedgerError as exc:  # a shard too small to score is a fact, not a crash
            out.setdefault("refused", {})[corpus] = str(exc)
            continue
        for signal in sorted(report["rules"]):
            fpr = _rule_fpr(report, signal)
            if fpr:
                per_shard.setdefault(signal, []).append({"corpus": corpus, **fpr})
    for signal, rows in per_shard.items():
        stats = homogeneity([(r["numerator"], r["denominator"]) for r in rows])
        # Reader-verifiable version: does each shard's interval contain the pooled rate?
        pooled = stats.get("pooled_rate")
        disagreements = []
        if pooled is not None:
            for row in rows:
                lo, hi = row["ci95"]
                if not (lo <= pooled <= hi):
                    disagreements.append({"corpus": row["corpus"], "rate": row["rate"],
                                          "ci95": row["ci95"]})
        # Is the reference model's own shard the best-behaved one? A yes here is a
        # mechanism, not a coincidence: the shard nearest the reference diverges least.
        lowest = min(rows, key=lambda r: r["rate"]) if rows else None
        out["signals"][signal] = {
            "per_shard": rows, "homogeneity": stats,
            "shards_whose_interval_excludes_pooled": disagreements,
            "lowest_shard": ({"corpus": lowest["corpus"], "rate": lowest["rate"]}
                             if lowest else None),
            "reference_shard_is_lowest": bool(
                lowest and reference and lowest["corpus"] == reference),
            # The verdict is the *statistical* one. A shard whose interval misses the
            # pooled rate is a caveat on the number, not evidence of a mixture (see
            # judge()); folding it in here made the report contradict its own warning.
            "verdict": stats.get("verdict", "not_measurable"),
            "caveat": (None if not disagreements else
                       f"{len(disagreements)} shard(s) whose own interval excludes the "
                       f"pooled rate"),
        }
    return out


def population_report(ledger: Dict[str, Any], alpha: float = 0.05,
                      min_negatives: int = 20, seeds: Sequence[int] = (0, 1, 2, 3, 4),
                      signals: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    problems = [f"{len(ledger['records'])} record(s)"] if not ledger.get("records") else []
    if problems:
        raise LedgerError("ledger has no records")
    positives = sum(1 for r in ledger["records"] if r.get("is_positive"))
    return {
        "schema": POPULATION_SCHEMA,
        "ledger_schema": ledger.get("schema"),
        "alpha": alpha,
        "min_negatives": min_negatives,
        "n_records": len(ledger["records"]),
        "n_positives": positives,
        "n_negatives": len(ledger["records"]) - positives,
        "positive_free": positives == 0,
        "note": ("a positive-free ledger can still test population homogeneity: the FPR is "
                 "the only thing it can measure, and whether the FPR is one number is "
                 "exactly the question. TPR is out of reach here."),
        "split_stability": split_stability(ledger, seeds, alpha, min_negatives, signals),
        "shard_heterogeneity": shard_heterogeneity(ledger, alpha, min_negatives,
                                                   split_seed=seeds[0] if seeds else 0,
                                                   reference=reference_corpus(ledger)),
    }


def judge(report: Dict[str, Any]) -> Dict[str, Any]:
    """Two separate questions, because they have separate fixes.

    "Are the shards one rate?" is decided by the homogeneity statistic. "Does the
    split move the number?" is decided by the same test applied across seeds.

    A shard whose own interval excludes the pooled rate is recorded as a *warning*,
    not a refusal. With N shards the chance that at least one 95% interval misses the
    pooled rate is large even under exact homogeneity, so making it a refusal would
    refuse every fleet large enough to be worth measuring -- the same trap as
    comparing a range to a CI width. It says the pooled number does not describe that
    shard exactly; it does not say the fleet is a mixture.
    """
    population_reasons: List[str] = []
    stability_reasons: List[str] = []
    warnings: List[str] = []
    for signal, block in report["shard_heterogeneity"]["signals"].items():
        stats = block["homogeneity"]
        disagreed = block["shards_whose_interval_excludes_pooled"]
        if stats.get("verdict") == "heterogeneous":
            population_reasons.append(
                f"{signal}: shard rates scatter by more than binomial noise "
                f"(Q={stats.get('statistic', float('nan')):.1f}, df={stats.get('df')}, "
                f"p={stats.get('p_value'):.4f}): a pooled FPR is a mixture")
        elif disagreed:
            warnings.append(
                f"{signal}: Q={stats.get('statistic', float('nan')):.1f} "
                f"(p={stats.get('p_value'):.4f}) is consistent with one rate, but "
                f"{len(disagreed)} shard's own interval excludes the pooled rate "
                f"({', '.join(os.path.basename(d['corpus']) for d in disagreed)}): quote "
                f"the pooled figure with that shard's own rate beside it")
    for signal, block in report["split_stability"]["signals"].items():
        stats = block["homogeneity"]
        if stats.get("verdict") == "heterogeneous":
            stability_reasons.append(
                f"{signal}: the FPR scatters across splits by more than binomial noise "
                f"(Q={stats.get('statistic', float('nan')):.1f}, df={stats.get('df')}, "
                f"p={stats.get('p_value'):.4f}, range {block['min']:.4f}-{block['max']:.4f})"
                f": the split, not the fleet, is moving the number")
    return {"one_population": not population_reasons,
            "split_stable": not stability_reasons,
            "refusals": population_reasons + stability_reasons,
            "warnings": warnings}


def render(report: Dict[str, Any]) -> str:
    lines = [f"fleet population check ({POPULATION_SCHEMA})",
             f"  records={report['n_records']} negatives={report['n_negatives']} "
             f"positives={report['n_positives']} alpha={report['alpha']}",
             "", "split stability (same models, different split seed)"]
    for signal, block in report["split_stability"]["signals"].items():
        lines.append(f"  {signal:22s} mean={block['mean']:.4f} "
                     f"range=[{block['min']:.4f}, {block['max']:.4f}] "
                     f"spread={block['spread']:.4f} over {block['n_seeds']} seeds")
    lines += ["", "shard heterogeneity (each shard scored against its own threshold)"]
    for signal, block in report["shard_heterogeneity"]["signals"].items():
        stats = block["homogeneity"]
        lines.append(f"  {signal:22s} verdict={block['verdict']:12s} "
                     f"Q={stats.get('statistic', float('nan')):.2f} df={stats.get('df')} "
                     f"p={stats.get('p_value', float('nan')):.4f} "
                     f"({len(block['per_shard'])} shard(s))")
        for row in block["per_shard"]:
            lo, hi = row["ci95"]
            flag = " <- reference shard" if row["corpus"] == report["shard_heterogeneity"]["reference_corpus"] else ""
            lines.append(f"      {os.path.basename(row['corpus']):28s} "
                         f"{row['rate']:.4f} [{lo:.4f}, {hi:.4f}] "
                         f"({row['numerator']}/{row['denominator']}){flag}")
        if block.get("reference_shard_is_lowest") and (block.get("homogeneity", {})
                                                        .get("pooled_rate") or 0) > 0:
            lines.append("      the lowest shard is also the one holding the reference "
                         "model: check whether the signal is reference-relative before "
                         "reading that as a property of the shard")
    verdicts = judge(report)
    lines += ["",
              f"ONE POPULATION: {'yes' if verdicts['one_population'] else 'NO'}",
              f"SPLIT STABLE: {'yes' if verdicts['split_stable'] else 'NO'}"]
    for reason in verdicts["refusals"]:
        lines.append(f"  refusal: {reason}")
    for warning in verdicts["warnings"]:
        lines.append(f"  warning: {warning}")
    if verdicts["one_population"] and verdicts["split_stable"]:
        lines.append("  every signal's FPR transfers across shards and splits; "
                     "an average over this fleet is meaningful")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="ledger JSON to analyse")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--min-negatives", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--signals", nargs="+", default=None)
    args = ap.parse_args(argv)

    ledger = load_ledger(args.input)
    if args.alpha is None:
        args.alpha = float(ledger.get("alpha", 0.05))
    try:
        report = population_report(ledger, alpha=args.alpha,
                                   min_negatives=args.min_negatives, seeds=args.seeds,
                                   signals=args.signals)
    except LedgerError as exc:
        print(f"nothing to read: {exc}", file=sys.stderr)
        return 2
    verdicts = judge(report)
    report["verdicts"] = verdicts
    print(render(report))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"report written to {args.out}")
    return 0 if (verdicts["one_population"] and verdicts["split_stable"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
