"""FPR / TPR evaluation harness — the gate every detection claim must pass through.

Why this module exists
----------------------
The whole framework is graded on a false-positive rate, and until now the lane had
never measured one. The strongest available statement was
``asset_fpr_on_clean: "0/8 by construction"`` — which is not a measurement, it is a
restatement of the calibration assumption. A 0/8 cannot distinguish a detector with a
1% FPR from one with a 30% FPR: the exact 95% upper bound on 0/8 is 31%.

Three disciplines are enforced here, and they are the reason this is a module rather
than a script:

1. **Calibration and evaluation are different models.** A threshold placed on the
   same models it is scored against returns the number you engineered, not the number
   you will get. The harness refuses to score a rule on its own calibration set.
2. **Every rate carries its denominator and an interval**, and a rate whose
   denominator is too small is reported as a *bound*, never as a point estimate. This
   is the repo's existing reporting convention (STATE.md: "0/N ships its 3/N (or exact
   binomial) upper bound") given an implementation.
3. **Ground truth is validated, not trusted.** Each kind's polarity is declared in the
   ledger and cross-checked before scoring. A stale attack-kind tuple silently turned
   16 model attacks into "24 negatives" in this very lane; a validator that would have
   rejected that ledger is worth more than a comment.

Input format (LOCKED: `cviaf.fpr-tpr-ledger.v1`)
------------------------------------------------
    {
      "schema": "cviaf.fpr-tpr-ledger.v1",
      "alpha": 0.05,
      "higher_is_more_anomalous": true,
      "positive_kinds": ["oga", "oda", "weight_tamper", "..."],
      "negative_kinds": ["clean", "clean_label", "..."],
      "provenance": {"producer": str, "corpora": [str], "notes": str},
      "records": [
        {"model_id": str, "corpus": str, "kind": str, "is_positive": bool,
         "split": "calibration" | "evaluation" | "unassigned",
         "scores": {"<signal>": float, ...}}
      ]
    }

Rules the validator enforces (each one has a regression test):

* `schema` matches exactly; unknown top-level keys are rejected.
* every record has the required keys; `scores` is a non-empty dict of finite floats.
* a model_id appears once per corpus (no duplicate assets inflating a denominator).
* the two kind lists are disjoint and **every observed kind appears in one of them**.
  A kind in neither list is a hard error. This is the guard for the exact defect this
  lane hit: `compare.py` carried its own attack-kind tuple, so `weight_tamper` was not
  recognised as an attack and the run reported "n=0 attacked, 24 negative" — an
  undeclared kind silently became ground-truth-negative. Here it cannot.
* `is_positive == (kind in positive_kinds)` for every record — a mislabelled arm is a
  hard error, not a warning.
* `split` is one of the three literals.

CLI
---
    python -m cviaf.lab.fpr_tpr --input ledger.json --out report.json [--alpha .05]
                               [--min-negatives 20] [--split-seed 0]

Exit code is 0 when a report was written, 3 when the ledger is invalid, 4 when every
rule was refused for lack of denominator (so CI can treat "no measurement" as failure
rather than as a clean bill of health).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LEDGER_SCHEMA = "cviaf.fpr-tpr-ledger.v1"
REPORT_SCHEMA = "cviaf.fpr-tpr-report.v1"
SPLITS = ("calibration", "evaluation", "unassigned")
FPR_FLOOR = 0.01          # the smallest FPR a 50-negative population can resolve


class LedgerError(ValueError):
    """Raised when a ledger is structurally invalid or mislabelled."""


# --------------------------------------------------------------------------- #
# interval arithmetic
# --------------------------------------------------------------------------- #

def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval — correct at the extremes where Wald is not.

    A detection rate is routinely 0/50 or 50/50, exactly where the normal
    approximation collapses (it can emit a negative lower bound).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def exact_upper_bound(k: int, n: int, confidence: float = 0.95) -> Optional[float]:
    """Exact one-sided upper bound: the largest p with P(X <= k) >= 1 - confidence.

    This is the honest statement for 0/N ("the true rate is at most this"), and the
    repo reports it that way. Closed form for k = 0 is 1 - (1-c)^(1/n); otherwise it
    is solved by bisection on the binomial tail so no scipy dependency is needed.
    """
    if n <= 0:
        return None
    if k == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(X <= k) under Binomial(n, mid)
        tail = sum(_binom_pmf(i, n, mid) for i in range(k + 1))
        if tail >= 1.0 - confidence:
            hi = mid
        else:
            lo = mid
    return hi


def _binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    log_pmf = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
               + k * math.log(p) + (n - k) * math.log1p(-p))
    return math.exp(log_pmf)


def rate(k: int, n: int, min_n: int) -> Dict[str, Any]:
    """A rate with its denominator, an interval, and a refusal when n is too small."""
    point = (k / n) if n else None
    out: Dict[str, Any] = {
        "numerator": int(k), "denominator": int(n),
        "point_estimate": round(point, 4) if point is not None else None,
        "ci95_wilson": [round(v, 4) for v in wilson_interval(k, n)] if n else None,
        "exact_upper_bound_95": (round(exact_upper_bound(k, n), 4)
                                 if n else None),
    }
    # A denominator below the floor cannot support a rate claim. Report the bound and
    # say why, rather than printing 0.0000 that readers will take at face value.
    out["status"] = "measured" if n >= min_n else "insufficient_denominator"
    if n < min_n:
        out["refusal"] = (f"n={n} < min_negatives={min_n}; report the exact upper "
                          f"bound {out['exact_upper_bound_95']} instead of a rate")
        out["point_estimate"] = None
        out["ci95_wilson"] = None
    return out


# --------------------------------------------------------------------------- #
# ledger validation
# --------------------------------------------------------------------------- #

def validate_ledger(ledger: Dict[str, Any]) -> List[str]:
    """Return a list of problems; empty means valid. Never raises."""
    problems: List[str] = []
    if not isinstance(ledger, dict):
        return ["ledger is not a JSON object"]
    allowed = {"schema", "alpha", "higher_is_more_anomalous", "positive_kinds",
               "negative_kinds", "provenance", "records"}
    extra = set(ledger) - allowed
    if extra:
        problems.append(f"unknown top-level keys: {sorted(extra)}")
    if ledger.get("schema") != LEDGER_SCHEMA:
        problems.append(f"schema must be {LEDGER_SCHEMA!r}, got "
                        f"{ledger.get('schema')!r}")
    pk = ledger.get("positive_kinds")
    if not isinstance(pk, list) or not pk or not all(isinstance(k, str) for k in pk):
        problems.append("positive_kinds must be a non-empty list of strings")
        pk = []
    nk = ledger.get("negative_kinds")
    if not isinstance(nk, list) or not nk or not all(isinstance(k, str) for k in nk):
        problems.append("negative_kinds must be a non-empty list of strings (an "
                        "undeclared kind must not default to negative — that is how "
                        "weight_tamper arms were once scored as 24 negatives)")
        nk = []
    both = set(pk) & set(nk)
    if both:
        problems.append(f"kinds declared positive AND negative: {sorted(both)}")
    if not isinstance(ledger.get("higher_is_more_anomalous"), bool):
        problems.append("higher_is_more_anomalous must be a boolean (it is the "
                        "direction contract every score is read through)")
    alpha = ledger.get("alpha", 0.05)
    if not isinstance(alpha, (int, float)) or not 0.0 < float(alpha) < 1.0:
        problems.append(f"alpha must be in (0,1), got {alpha!r}")
    records = ledger.get("records")
    if not isinstance(records, list) or not records:
        problems.append("records must be a non-empty list")
        return problems

    seen: Dict[Tuple[str, str], int] = {}
    signal_names: Dict[str, int] = {}
    for i, rec in enumerate(records):
        where = f"records[{i}]"
        if not isinstance(rec, dict):
            problems.append(f"{where} is not an object")
            continue
        for key in ("model_id", "corpus", "kind", "is_positive", "split", "scores"):
            if key not in rec:
                problems.append(f"{where} missing {key!r}")
        mid, corpus = rec.get("model_id"), rec.get("corpus")
        if not isinstance(mid, str) or not mid:
            problems.append(f"{where}.model_id must be a non-empty string")
        if not isinstance(corpus, str) or not corpus:
            problems.append(f"{where}.corpus must be a non-empty string")
        if rec.get("split") not in SPLITS:
            problems.append(f"{where}.split must be one of {SPLITS}, got "
                            f"{rec.get('split')!r}")
        if not isinstance(rec.get("is_positive"), bool):
            problems.append(f"{where}.is_positive must be a boolean")
        # Ground-truth consistency: this is the check that catches the class of bug
        # that made 16 tamper arms count as negatives.
        kind = rec.get("kind")
        if (isinstance(kind, str) and kind not in set(pk) | set(nk)
                and (pk or nk)):
            problems.append(f"{where} ({mid}): kind {kind!r} is declared neither "
                            f"positive nor negative; refusing to guess its polarity")
        if isinstance(kind, str) and pk and isinstance(rec.get("is_positive"), bool):
            expected = kind in pk
            if expected != rec["is_positive"]:
                problems.append(
                    f"{where} ({mid}): kind {kind!r} implies is_positive={expected} "
                    f"but the record says {rec['is_positive']} — refusing to score a "
                    f"mislabelled asset")
        scores = rec.get("scores")
        if not isinstance(scores, dict) or not scores:
            problems.append(f"{where}.scores must be a non-empty object")
        else:
            for name, value in scores.items():
                if not isinstance(name, str) or not name:
                    problems.append(f"{where}.scores has a non-string signal name")
                    continue
                signal_names[name] = signal_names.get(name, 0) + 1
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    problems.append(f"{where}.scores[{name!r}] is not a number")
                elif not math.isfinite(float(value)):
                    problems.append(f"{where}.scores[{name!r}] is not finite "
                                    f"({value!r})")
        if isinstance(mid, str) and isinstance(corpus, str):
            seen[(corpus, mid)] = seen.get((corpus, mid), 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        problems.append(f"duplicate (corpus, model_id) records inflating a "
                        f"denominator: {sorted(dupes)}")
    # A signal must be present on every record, or the denominators differ silently
    # between rules and a comparison of TPRs is not like-for-like.
    if signal_names:
        n_rec = len(records)
        partial = {s: c for s, c in signal_names.items() if c != n_rec}
        if partial:
            problems.append(f"signal(s) not present on every record: {partial} "
                            f"(of {n_rec})")
    return problems


def load_ledger(path: str, require_valid: bool = True) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        ledger = json.load(fh)
    if require_valid:
        problems = validate_ledger(ledger)
        if problems:
            raise LedgerError("; ".join(problems))
    return ledger


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #

def assign_splits(ledger: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """Deterministically place every `unassigned` record into one half.

    Assignment is by *model*, never by score: splitting on the score would leak the
    test distribution into the threshold. Positives and negatives are split
    independently so both halves keep every kind represented — a half with no
    positives would report a TPR of 0 that means "not measured".
    """
    records = [dict(r) for r in ledger["records"]]
    rng = np.random.default_rng(seed)
    for polarity in (True, False):
        idx = [i for i, r in enumerate(records)
               if r["split"] == "unassigned" and r["is_positive"] is polarity]
        order = rng.permutation(len(idx))
        for rank, pos in enumerate(order):
            records[idx[pos]]["split"] = "calibration" if rank % 2 == 0 else "evaluation"
    out = dict(ledger)
    out["records"] = records
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def evaluate_rule(records: Sequence[Dict[str, Any]], signal: str, alpha: float,
                  higher_is_more_anomalous: bool = True,
                  min_negatives: int = 20) -> Dict[str, Any]:
    """Threshold on the calibration NEGATIVES, then score the evaluation half.

    The quantile is taken over calibration negatives only: that is the operating point
    a deployment can actually choose (an operator can bound the alarm rate on trusted
    clean assets), and it keeps the rule comparable across corpora.
    """
    cal = [r for r in records if r["split"] == "calibration"]
    ev = [r for r in records if r["split"] == "evaluation"]
    cal_neg = [r["scores"][signal] for r in cal if not r["is_positive"]]
    if len(cal_neg) < 5:
        return {"signal": signal, "status": "refused",
                "reason": f"only {len(cal_neg)} calibration negatives; a threshold "
                          f"placed on fewer than 5 is noise",
                "calibration_negatives": len(cal_neg)}
    arr = np.asarray(cal_neg, np.float64)
    # The alpha-quantile of the calibration negatives: at most alpha of clean assets
    # exceed it. Direction-aware so a lower-is-anomalous signal is not silently broken.
    q = alpha if higher_is_more_anomalous else 1.0 - alpha
    threshold = float(np.quantile(arr, 1.0 - q))

    def fires(r: Dict[str, Any]) -> bool:
        s = r["scores"][signal]
        return (s > threshold) if higher_is_more_anomalous else (s < threshold)

    pos = [r for r in ev if r["is_positive"]]
    neg = [r for r in ev if not r["is_positive"]]
    tp = sum(1 for r in pos if fires(r))
    fp = sum(1 for r in neg if fires(r))
    return {
        "signal": signal,
        "status": "measured" if (len(neg) >= min_negatives and pos) else "partial",
        "threshold": threshold,
        "threshold_basis": (f"{alpha:.3g}-quantile of {len(cal_neg)} calibration "
                            f"negatives"),
        "tpr": rate(tp, len(pos), 1),
        "fpr": rate(fp, len(neg), min_negatives),
        "confusion": {"tp": tp, "fn": len(pos) - tp, "fp": fp, "tn": len(neg) - fp},
    }


def kind_breakdown(records: Sequence[Dict[str, Any]], signal: str, threshold: float,
                   higher_is_more_anomalous: bool = True, min_n: int = 1
                   ) -> Dict[str, Any]:
    """TPR for each positive kind at one FIXED threshold.

    The threshold is the corpus-level one, deliberately: picking a per-kind threshold
    would let a kind choose its own operating point, and then the per-kind recalls
    would no longer describe the single rule a deployment would run. A kind with no
    evaluation items is reported as not measured rather than as 0.0 -- "we caught none
    of the cloaking arms" and "we never looked at a cloaking arm" are different
    statements, and clause 3.7 of the problem statement wants the first.
    """
    kinds: Dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in records if r["is_positive"]}):
        items = [r for r in records
                 if r["is_positive"] and r["kind"] == kind and r["split"] == "evaluation"]
        if not items:
            kinds[kind] = {"status": "not_measured", "n_evaluation": 0,
                           "reason": "no evaluation-half assets of this kind"}
            continue
        tp = 0
        for r in items:
            s = r["scores"][signal]
            tp += int((s > threshold) if higher_is_more_anomalous else (s < threshold))
        kinds[kind] = {"status": "measured" if len(items) >= min_n else "partial",
                       "n_evaluation": len(items), "tp": tp, "fn": len(items) - tp,
                       "tpr": rate(tp, len(items), min_n)}
    return kinds


def evaluate_corpus(ledger: Dict[str, Any], alpha: float = 0.05,
                    min_negatives: int = 20, split_seed: int = 0) -> Dict[str, Any]:
    """Score every signal in the ledger and summarise by kind."""
    problems = validate_ledger(ledger)
    if problems:
        raise LedgerError("; ".join(problems))
    alpha = float(ledger.get("alpha", alpha))
    assigned = assign_splits(ledger, seed=split_seed)
    records = assigned["records"]
    signals = sorted({s for r in records for s in r["scores"]})
    higher = bool(ledger["higher_is_more_anomalous"])

    rules = {s: evaluate_rule(records, s, alpha, higher, min_negatives)
             for s in signals}

    def census(pred) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in records:
            if pred(r):
                out[r["kind"]] = out.get(r["kind"], 0) + 1
        out["_total"] = sum(v for v in out.values())
        return out

    n_cal_neg = sum(1 for r in records
                    if r["split"] == "calibration" and not r["is_positive"])
    n_ev_neg = sum(1 for r in records
                   if r["split"] == "evaluation" and not r["is_positive"])
    n_ev_pos = sum(1 for r in records
                   if r["split"] == "evaluation" and r["is_positive"])
    ready = bool(n_ev_neg >= min_negatives and n_ev_pos > 0)
    per_kind = {
        s: ({"status": "refused", "kinds": {}} if rules[s].get("status") == "refused"
            else {"status": "measured", "threshold": rules[s]["threshold"],
                  "threshold_note": "the corpus-level threshold; per-kind thresholds "
                                    "would let each kind pick its own operating point",
                  "kinds": kind_breakdown(records, s, rules[s]["threshold"], higher)})
        for s in signals}

    # Clause 3.7 ("ODA recall > 0"): report it per signal and take the best, because
    # one number is what the clause asks for -- but say out loud that choosing the max
    # over four signals inflates it, so the per-signal column is the authoritative one.
    oda = {s: (per_kind[s]["kinds"].get("oda", {}) or {}).get("tpr", {}).get("point_estimate")
           for s in signals if per_kind[s]["status"] != "refused"}
    caught = {s: v for s, v in oda.items() if v}
    oda_check = {
        "clause": "3.7 ODA (cloaking) recall > 0 on the reference battery",
        "recall_per_signal": oda,
        "best_signal": max(caught, key=caught.get) if caught else None,
        "best_recall": max(caught.values()) if caught else 0.0,
        "satisfied": bool(caught),
        "caveat": ("the best-of-four number is selected over signals, so it is optimistic; "
                   "the per-signal recall in this report is the authoritative figure. A "
                   "kind with no evaluation-half assets is reported as not measured, not "
                   "as zero recall."),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "ledger_schema": ledger["schema"],
        "alpha": alpha,
        "min_negatives": int(min_negatives),
        "higher_is_more_anomalous": higher,
        "positive_kinds": sorted(ledger["positive_kinds"]),
        "provenance": ledger.get("provenance", {}),
        "denominators": {
            "calibration_negatives": n_cal_neg,
            "evaluation_negatives": n_ev_neg,
            "evaluation_positives": n_ev_pos,
        },
        "census": {"all": census(lambda r: True),
                   "evaluation_positives": census(
                       lambda r: r["is_positive"] and r["split"] == "evaluation"),
                   "evaluation_negatives": census(
                       lambda r: not r["is_positive"] and r["split"] == "evaluation")},
        "rules": rules,
        "per_kind": per_kind,
        "clause_checks": {"3.7_oda_recall": oda_check},
        # The single sentence a reader should see before any number: without enough
        # clean assets on the evaluation half, FPR is not measured and every claim
        # downstream is conditional on an untested assumption.
        "fpr_measured": bool(n_ev_neg >= min_negatives),
        "headline": ("FPR is measured on %d clean assets" % n_ev_neg if ready else
                     "FPR NOT MEASURED: %d evaluation negatives < %d required; "
                     "report bounds, not rates" % (n_ev_neg, min_negatives)),
    }
    return report


def render_report(report: Dict[str, Any]) -> str:
    lines = [f"FPR/TPR report  alpha={report['alpha']:.3g}  "
             f"negatives={report['denominators']['evaluation_negatives']}  "
             f"positives={report['denominators']['evaluation_positives']}",
             report["headline"], ""]
    hdr = (f"{'signal':26s} {'TPR':>7s} {'TPR 95% CI':>16s} {'FPR':>7s} "
           f"{'FPR 95% CI':>16s} {'status':>10s}")
    lines += [hdr, "-" * len(hdr)]
    for name, r in sorted(report["rules"].items()):
        if r.get("status") == "refused":
            lines.append(f"{name:26s} {'-':>7s} {'-':>16s} {'-':>7s} {'-':>16s} "
                         f"{'refused':>10s}")
            continue
        t, f = r["tpr"], r["fpr"]
        tpr_s = f"{t['point_estimate']:.3f}" if t["point_estimate"] is not None else "bound"
        fpr_s = f"{f['point_estimate']:.3f}" if f["point_estimate"] is not None else "bound"
        tpr_ci = (f"[{t['ci95_wilson'][0]:.3f},{t['ci95_wilson'][1]:.3f}]"
                  if t["ci95_wilson"] else f"<={t['exact_upper_bound_95']:.3f}")
        fpr_ci = (f"[{f['ci95_wilson'][0]:.3f},{f['ci95_wilson'][1]:.3f}]"
                  if f["ci95_wilson"] else f"<={f['exact_upper_bound_95']:.3f}")
        lines.append(f"{name:26s} {tpr_s:>7s} {tpr_ci:>16s} {fpr_s:>7s} {fpr_ci:>16s} "
                     f"{r['status']:>10s}")

    lines += ["", "recall by attack kind, at the same threshold the TPR used",
              "(tp/n; 'not measured' means no evaluation-half assets of that kind)"]
    for name in sorted(report.get("per_kind", {})):
        entry = report["per_kind"][name]
        if entry.get("status") == "refused":
            lines.append(f"  {name:24s} refused")
            continue
        parts = []
        for kind, k in sorted(entry["kinds"].items()):
            if k.get("status") == "not_measured":
                parts.append(f"{kind}=not measured")
            else:
                parts.append(f"{kind}={k['tp']}/{k['n_evaluation']}")
        lines.append(f"  {name:24s} {'  '.join(parts)}")

    check = report.get("clause_checks", {}).get("3.7_oda_recall")
    if check:
        best = (f"{check['best_signal']}" if check["best_signal"] else "no signal")
        lines += ["", f"clause 3.7 ODA recall > 0: {'satisfied' if check['satisfied'] else 'NOT satisfied'} "
                      f"(best {best} {check['best_recall']:.3f}; {check['caveat'].split('.')[0]}.)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cviaf lab fpr-tpr",
                                 description="FPR/TPR evaluation over a score ledger")
    ap.add_argument("--input", required=True, help="ledger JSON (schema %s)" % LEDGER_SCHEMA)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the ledger's alpha")
    ap.add_argument("--min-negatives", type=int, default=20)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)

    try:
        ledger = load_ledger(args.input)
    except LedgerError as exc:
        print(f"INVALID LEDGER: {exc}")
        return 3
    if args.validate_only:
        print(f"ledger valid: {len(ledger['records'])} records, "
              f"{len(ledger['positive_kinds'])} positive kinds")
        return 0

    if args.alpha is not None:
        ledger = {**ledger, "alpha": args.alpha}
    report = evaluate_corpus(ledger, alpha=args.alpha or ledger.get("alpha", 0.05),
                             min_negatives=args.min_negatives,
                             split_seed=args.split_seed)
    print(render_report(report))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"\nreport written to {args.out}")
    if not report["fpr_measured"] and all(
            r.get("status") == "refused" for r in report["rules"].values()):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
