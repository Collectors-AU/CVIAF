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

# --------------------------------------------------------------------------- #
# resume / merge — a 20k-model ledger must survive a crash
# --------------------------------------------------------------------------- #

def corpus_snapshot(path: str) -> Dict[str, Any]:
    """Identify *which* population was scored, not just which directory.

    A fleet corpus is written while it is being read, so "runs/clean_null_local_w1"
    names a moving target: measured live, the shards grew from 6,815 to 7,101 models
    within the same session. Recording a digest and a count makes the scored
    population auditable after the fact, and lets a later run detect that the corpus
    it is resuming into is no longer the corpus it started on.
    """
    import hashlib
    reg = os.path.join(path, "registry.jsonl")
    if not os.path.isfile(reg):
        return {"path": path, "registry": None, "n_models": 0, "registry_sha256": None}
    h = hashlib.sha256()
    n = 0
    with open(reg, "rb") as fh:
        for line in fh:
            if line.strip():
                h.update(line)
                n += 1
    return {"path": path, "registry": "registry.jsonl", "n_models": n,
            "registry_sha256": h.hexdigest()[:16]}


def signal_coverage_warning(requested: Sequence[str],
                            records: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Complain when a producer asked for signals that no record carries.

    Partial coverage per record is already a hard validator failure; *global* absence
    is not, because a run with no reference model legitimately has no divergence
    signal. Measured failure this guards: the first parallel build handed workers
    ``--reference`` (None) instead of the resolved reference directory, so every
    record silently lost ``refdiv_mean_clean`` and the ledger still validated — a
    rule would simply have been reported as unavailable. Silent signal loss is worse
    than a crash, so it is echoed at the end of every build.
    """
    present = set().union(*[set(r.get("scores") or {}) for r in records]) if records else set()
    missing = [s for s in requested if s not in present]
    if not missing:
        return None
    return (f"signal(s) requested but absent from every record: {missing} — the record "
            f"population is valid but thinner than asked for; check the reference "
            "model / --signals wiring before trusting a rule table built from it")


def scored_ids(ledger: Dict[str, Any]) -> set:
    """The (corpus, model_id) pairs already present — what a resume should skip."""
    return {(r.get("corpus"), r.get("model_id")) for r in ledger.get("records", [])
            if isinstance(r, dict)}


def merge_ledgers(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Union two ledgers that must agree on every contract that affects a rate.

    Resuming is only sound if the appended records were produced under the same
    alpha, direction and kind declaration as the ones already there. Merging silently
    across a disagreement is how a ledger ends up half-labelled, so this refuses.
    """
    for name, ledger in (("base", base), ("new", new)):
        problems = validate_ledger(ledger)
        if problems:
            raise LedgerError(f"{name} ledger invalid: {'; '.join(problems)}")
    for key in ("alpha", "higher_is_more_anomalous"):
        if base.get(key) != new.get(key):
            raise LedgerError(f"refusing to merge: {key} differs "
                              f"({base.get(key)!r} vs {new.get(key)!r})")
    for key in ("positive_kinds", "negative_kinds"):
        if set(base.get(key) or []) != set(new.get(key) or []):
            raise LedgerError(f"refusing to merge: {key} differs; a kind's polarity "
                              f"must not change between runs")
    seen = scored_ids(base)
    merged = dict(base)
    out_records = list(base["records"])
    for rec in new["records"]:
        key = (rec.get("corpus"), rec.get("model_id"))
        if key in seen:
            continue          # already scored; first answer wins (no double-counting)
        seen.add(key)
        out_records.append(rec)
    merged["records"] = out_records
    prov = dict(base.get("provenance") or {})
    new_prov = dict(new.get("provenance") or {})
    snaps = dict(prov.get("corpus_snapshots") or {})
    snaps.update(new_prov.get("corpus_snapshots") or {})
    prov.update({k: v for k, v in new_prov.items() if k != "corpus_snapshots"})
    if snaps:
        prov["corpus_snapshots"] = snaps
    prov["partial"] = bool(prov.get("partial") or new_prov.get("partial"))
    prov["merged_runs"] = int(prov.get("merged_runs", 1)) + 1
    merged["provenance"] = prov
    return merged


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
    status = "measured" if (len(neg) >= min_negatives and pos) else "partial"
    if not pos:
        # A clean-only ledger (the fleet) measures FPR and nothing else. Saying "TPR 0"
        # here would be a fabricated detection failure: there is no attacked asset in
        # the population to detect, so the rate is undefined, not zero.
        status = "fpr_only"

    fpr = rate(fp, len(neg), min_negatives)

    # The calibration check the fleet exists to run: the threshold came from one half
    # of the clean population, so the FPR on the other half should land on alpha. If
    # alpha falls outside the interval, the two halves are not exchangeable and every
    # number computed from this calibration is suspect -- which is worth knowing
    # BEFORE 20k more models are scored against it.
    calibration_check: Dict[str, Any] = {
        "target_alpha": alpha, "measured_fpr": fpr.get("point_estimate"),
        "n_evaluation_negatives": len(neg),
    }
    if fpr.get("ci95_wilson"):
        lo, hi = fpr["ci95_wilson"]
        calibration_check["ci95"] = [lo, hi]
        # One-sided on purpose. An FPR BELOW alpha is a conservative detector, not a
        # calibration failure -- the quantile threshold is estimated from a finite
        # calibration half, which biases the in-sample rate slightly under alpha, and
        # a two-sided test would fire on that every time n grows. What matters is
        # whether the rule alarms MORE often than it was calibrated to.
        calibration_check["direction"] = "one-sided (alarm when the FPR interval "
        calibration_check["holds"] = bool(lo <= alpha)
        calibration_check["note"] = (
            "the evaluation half's FPR interval still contains the declared alpha"
            if calibration_check["holds"] else
            f"the whole FPR interval [{lo}, {hi}] is above alpha {alpha}: the "
            f"calibration half and the evaluation half are not exchangeable, so "
            f"thresholds taken from this null alarm more often on held-out clean "
            f"assets than they were calibrated to")
    else:
        calibration_check["holds"] = None
        calibration_check["note"] = "FPR has no interval at this denominator"

    return {
        "signal": signal,
        "status": status,
        "threshold": threshold,
        "threshold_basis": (f"{alpha:.3g}-quantile of {len(cal_neg)} calibration "
                            f"negatives"),
        "tpr": rate(tp, len(pos), 1),
        "tpr_note": (None if pos else "no attacked assets in this ledger: the TPR of "
                                      "this rule is undefined here, not zero"),
        "fpr": fpr,
        "fpr_calibration": calibration_check,
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
                    min_negatives: int = 20, split_seed: int = 0,
                    prevalence: Optional[float] = None,
                    costs: Optional[Any] = None) -> Dict[str, Any]:
    """Score every signal in the ledger and summarise by kind.

    Each signal is also priced: expected loss per asset for three deployment policies
    at a declared prevalence, from the signal's own MEASURED TPR and FPR (clause 1.4 --
    risk, not just integrity). Units of a rule are not comparable across corpora, so
    the prevalence and loss matrix are recorded next to every number.
    """
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
    # Measured means there are oda arms on the evaluation half. Zero recall with oda
    # arms present is a measured zero; no oda arms at all is "not measurable".
    oda_measurable = any((per_kind[s]["kinds"].get("oda", {}) or {}).get("status")
                         in ("measured", "partial") for s in signals
                         if per_kind[s]["status"] != "refused")
    known = {s: v for s, v in oda.items() if v is not None and oda_measurable}
    best_signal = max(known, key=known.get) if known else None
    oda_check = {
        "clause": "3.7 ODA (cloaking) recall > 0 on the reference battery",
        "recall_per_signal": oda,
        "best_signal": best_signal,
        "best_recall": (known[best_signal] if best_signal else None),
        "satisfied": (None if not oda_measurable else bool(known and
                                                           known[best_signal] > 0)),
        "measurable": oda_measurable,
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
        "risk": risk_block(rules, prevalence=prevalence, costs=costs,
                           n_positives=n_ev_pos),
        # The single sentence a reader should see before any number: without enough
        # clean assets on the evaluation half, FPR is not measured and every claim
        # downstream is conditional on an untested assumption.
        "fpr_measured": bool(n_ev_neg >= min_negatives),
        "positives_measured": bool(n_ev_pos > 0),
        "headline": ("FPR is measured on %d clean assets" % n_ev_neg if ready else
                     ("FPR measured on %d clean assets; TPR not measurable (no attacked "
                      "assets in this ledger)" % n_ev_neg if n_ev_neg >= min_negatives
                      else "FPR NOT MEASURED: %d evaluation negatives < %d required; "
                           "report bounds, not rates" % (n_ev_neg, min_negatives))),
    }
    return report


def risk_block(rules: Mapping[str, Any], prevalence: Optional[float] = None,
               costs: Optional[Any] = None,
               n_positives: Optional[int] = None) -> Dict[str, Any]:
    """Expected loss per signal, from its own measured operating point.

    A signal whose FPR is only bounded (0/30 clean assets) has no point estimate and is
    not priced: a bound cannot place a posterior. That refusal is the whole reason the
    FPR harness reports bounds at all.
    """
    from cviaf.lab.risk import DEFAULT_PREVALENCE, rule_risk

    pi = DEFAULT_PREVALENCE if prevalence is None else float(prevalence)
    out: Dict[str, Any] = {
        "schema": "cviaf.asset-risk.v1",
        "prevalence": pi,
        "prevalence_basis": ("declared default; override with --prevalence. The FPR "
                             "corpus here is built 50/50 for power, which is NOT a "
                             "deployment prevalence"),
        "loss_matrix": (costs.to_dict() if costs is not None
                        else __import__("cviaf.lab.review", fromlist=["OperatorCosts"])
                        .OperatorCosts().to_dict()),
        "per_signal": {},
    }
    if not n_positives:
        # With no attacked assets the TPR is undefined, and every expected loss in this
        # model is a function of TPR: pricing a rule here would be inventing the false
        # negatives. The fleet run buys the FPR half of the equation and says so.
        out["priced"] = False
        out["reason"] = ("no attacked assets in this ledger: expected loss needs a TPR, "
                         "and a TPR needs a positive arm. The FPR half is measured and "
                         "is the deliverable of a clean-only run")
        for name, rule in rules.items():
            out["per_signal"][name] = {"status": "not_measured",
                                       "reason": out["reason"],
                                       "fpr": ((rule.get("fpr") or {}).get("point_estimate")),
                                       "threshold": rule.get("threshold")}
        return out
    out["priced"] = True
    for name, rule in rules.items():
        if rule.get("status") == "refused":
            out["per_signal"][name] = {"status": "not_measured",
                                       "reason": rule.get("reason")}
            continue
        priced = rule_risk((rule["tpr"] or {}).get("point_estimate"),
                           (rule["fpr"] or {}).get("point_estimate"), pi, costs)
        priced["tpr_denominator"] = (rule["tpr"] or {}).get("denominator")
        priced["fpr_denominator"] = (rule["fpr"] or {}).get("denominator")
        out["per_signal"][name] = priced
    return out


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
        def interval(entry):
            if entry["ci95_wilson"]:
                return f"[{entry['ci95_wilson'][0]:.3f},{entry['ci95_wilson'][1]:.3f}]"
            if entry["exact_upper_bound_95"] is not None:
                return f"<={entry['exact_upper_bound_95']:.3f}"
            return "not measured"      # n = 0: nothing to bound
        tpr_ci = interval(t)
        fpr_ci = interval(f)
        lines.append(f"{name:26s} {tpr_s:>7s} {tpr_ci:>16s} {fpr_s:>7s} {fpr_ci:>16s} "
                     f"{r['status']:>10s}")

    if report.get("positives_measured"):
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

    # The calibration check is the headline of a clean-only run, so it gets its own
    # block rather than hiding in the JSON.
    alarms = {n: r["fpr_calibration"] for n, r in report["rules"].items()
              if r.get("fpr_calibration", {}).get("holds") is False}
    if alarms:
        lines += ["", f"CALIBRATION ALARM: alpha {report['alpha']:.3g} is outside the "
                      f"measured FPR interval for {len(alarms)} rule(s)"]
        for name, c in sorted(alarms.items()):
            lines.append(f"  {name:24s} c={c['measured_fpr']} "
                         f"ci=[{c['ci95'][0]}, {c['ci95'][1]}]")
        lines.append("  the calibration half and the evaluation half are not "
                     "exchangeable; thresholds from this null do not transfer")

    check = report.get("clause_checks", {}).get("3.7_oda_recall")
    if check:
        best = (f"{check['best_signal']}" if check.get("best_signal") else "no signal")
        verdict = ("not measurable" if check.get("satisfied") is None else
                   ("satisfied" if check["satisfied"] else "NOT satisfied"))
        value = "-" if check.get("best_recall") is None else f"{check['best_recall']:.3f}"
        lines += ["", f"clause 3.7 ODA recall > 0: {verdict} (best {best} {value}; "
                      f"{check['caveat'].split('.')[0]}.)"]

    risk = report.get("risk")
    if risk:
        lines += ["", f"expected loss per asset at prevalence {risk['prevalence']:.3f} "
                      f"(clause 1.4; loss matrix {risk['loss_matrix']})",
                  f"{'signal':26s} {'accept_all':>10s} {'quarantine':>11s} "
                  f"{'review':>8s} {'recommended':>18s} {'break-even pi':>13s}"]
        for name, priced in sorted(risk["per_signal"].items()):
            if priced.get("status") != "measured":
                lines.append(f"{name:26s} {'not priced: ' + str(priced.get('reason'))[:60]}")
                continue
            el = priced["expected_loss_per_asset"]
            be = priced.get("break_even_prevalence_vs_accept_all")
            lines.append(
                f"{name:26s} {el['accept_all']:10.3f} {el['quarantine_flagged']:11.3f} "
                f"{el['review_flagged']:8.3f} {priced['recommended_policy']:>18s} "
                f"{('n/a' if be is None else f'{be:.3f}'):>13s}")
        lines.append(f"  prevalence basis: {risk['prevalence_basis']}")
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
    ap.add_argument("--prevalence", type=float, default=None,
                    help="declared share of tampered assets for the expected-loss block")
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
                             split_seed=args.split_seed, prevalence=args.prevalence)
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
