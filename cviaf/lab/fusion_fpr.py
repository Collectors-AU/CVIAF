"""Score FUSED detector rules on a ledger that already holds per-signal scores.

Why this module exists
----------------------
The comparison path fuses signals (``min(1, m * min p)`` within an asset, then a
Cauchy combination of item p-values, then BH over assets) and the lab has exported
that fusion from ``cviaf/lab/{baseline,calibrate,fusion}.py`` for months. What the
lane had *never* done is put the fused rule through the FPR harness on a real
population: every published false-positive number was per-signal. A fusion rule is
a different rule, with a different operating point, and until it is scored it is an
assumption wearing the clothes of a result.

This module closes that gap without training and without re-scoring a single asset:
the fleet ledger already stores four scores per model, so a fused column can be
computed from it.

Two disciplines are inherited from the harness and are not negotiable here:

* **The fused column is built from the calibration half only.** The conformal
  p-value of a model is its rank among the calibration-half NEGATIVES. If the
  evaluation half's own scores leak into the calibration set, the fused FPR is the
  number that was engineered, not the number a deployment gets.
* **The fused column stores ``1 - p``, not ``p``.** The ledger has ONE direction
  contract (``higher_is_more_anomalous``) and every rule in the harness is read
  through it. Storing a raw p-value would silently invert the rule for anyone who
  reads the column with the ledger's declared direction. The monotone transform
  keeps the contract, and the threshold the harness derives is then
  ``1 - q_alpha(p)``, i.e. the same cut as ``p <= q_alpha``.

The saturation interaction is not a footnote, and it is worse than "a blind signal
is a no-op". Two of the four fleet signals cannot produce a small p-value at all:
their statistic is a bounded ratio that hundreds of calibration models tie at, so
the conformal floor ``(1 + #{cal >= x}) / (n + 1)`` never falls below **0.287**
(``ctc_q95_clean``) and **0.430** (``ctc_peak_clean``). A min-based (Bonferroni)
fusion is merely indifferent to such a signal -- a blind signal cannot dilute a
working one, it only pays a multiplicity price. A mean-based (Cauchy) fusion is
not indifferent: every such signal contributes ``tan((0.5 - p) * pi) >=
tan((0.5 - p_floor) * pi) > 0`` on every row, which pushes the averaged statistic
*up* and the combined p-value *up* with it.

Measured on the fleet, that silently moves the fused rule's operating point: read
through the nominal ``p <= alpha`` cut, the four-signal fusion fires at 0.018 while
the two live signals fuse to 0.049. The four-signal number is three times
"better" in the only column anyone quotes, because the rule has stopped looking.
So this module reports the all-signal fusion (what the compare path would do if it
were pointed at the fleet today) *and* the live-signal fusion side by side, and
marks every signal whose p-value floor is above alpha.

Input:  a ledger valid under ``cviaf.fpr-tpr-ledger.v1`` (the harness validator is
        reused, not reimplemented).
Output: the same ledger with extra score columns, plus a fusion evidence block that
        says which signals were used, why any were dropped, and which half supplied
        the null.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.calibrate import cauchy_combine, conformal_pvalues
from cviaf.lab.fpr_tpr import (CEILING_TOL, LedgerError, assign_splits, evaluate_corpus,
                               rate, signal_degeneracy, validate_ledger)
from cviaf.lab.fusion import by_global_pvalue

FUSION_SCHEMA = "cviaf.fusion-fpr.v1"
DEFAULT_METHODS = ("cauchy", "bonferroni", "by")
# CEILING_TOL and the degeneracy check itself live in the harness (fpr_tpr), so the
# appendix that retires a saturated signal and the fusion that excludes it cannot
# disagree about which signals those are.


def degenerate_signals(records: Sequence[Dict[str, Any]], signals: Sequence[str],
                       alpha: float, higher_is_more_anomalous: bool = True,
                       ) -> Dict[str, Dict[str, Any]]:
    """Signals that cannot fire, and the p-value floor that explains why.

    A thin view over ``fpr_tpr.signal_degeneracy`` (one implementation, so the
    appendix that retires a saturated signal and the fusion that excludes it cannot
    disagree), keeping the key names the fusion report and its tests read.

    ``p_floor`` is the smallest conformal p-value any model on this population
    *could* have gotten, and it is the mechanism of the damage a dead signal does
    inside a MEAN-based fusion: when a large share of the calibration models tie at
    the extreme, ``(1 + #{cal >= x}) / (n + 1)`` has a floor well above zero, and
    every such signal contributes a positive ``tan`` term to the combined statistic
    on every row. Both facts are returned because the threshold is what a table
    shows and the floor is what a fusion feels.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for signal in signals:
        d = signal_degeneracy(records, signal, alpha, higher_is_more_anomalous)
        out[signal] = {
            "threshold": d["threshold"],
            "ceiling": d["calibration_ceiling"],
            "at_ceiling": d["at_calibration_ceiling"],
            "p_floor": d["p_floor"],
            "blind_at_alpha": d["p_floor_above_alpha"],
            "calibration_ties_at_extreme": d["calibration_ties_at_extreme"],
            "note": ("threshold is the maximum calibration-negative score, so on this "
                     "population the rule cannot fire and its FPR bound is a "
                     "statement about the ceiling, not about the detector"
                     if d["at_calibration_ceiling"] else "off the ceiling: the rule can fire"),
        }
    return out


def _fused_pvalues(pcols: np.ndarray, method: str) -> np.ndarray:
    """Row-wise fusion of a (n_records, n_signals) p-value matrix."""
    p = np.asarray(pcols, np.float64)
    if p.ndim != 2 or p.shape[1] == 0:
        raise ValueError("fusion needs a nonempty (records, signals) p-value matrix")
    if not np.all(np.isfinite(p)) or np.any((p <= 0) | (p > 1)):
        raise ValueError("conformal p-values must be finite and in (0, 1]")
    if method == "bonferroni":
        return np.clip(p.shape[1] * p.min(axis=1), 0.0, 1.0)
    if method == "cauchy":
        t = np.tan((0.5 - p) * np.pi).mean(axis=1)
        return np.clip(0.5 - np.arctan(t) / np.pi, 0.0, 1.0)
    if method == "by":
        # BY is the lab's dependence-robust global null; it needs the ordered
        # p-values per row, so it is applied row by row rather than vectorised. The
        # arithmetic is the primitive exported by cviaf/lab/fusion.py, not a copy.
        return np.asarray([by_global_pvalue(row) for row in p], np.float64)
    raise ValueError(f"unknown fusion method {method!r}; expected one of "
                     f"{sorted(set(DEFAULT_METHODS))}")


def fuse_ledger(ledger: Dict[str, Any], split_seed: int = 0,
                methods: Sequence[str] = DEFAULT_METHODS,
                include: Optional[Iterable[str]] = None,
                prefix: str = "fused", min_calibration: int = 5
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Add fused score columns to a validated ledger. Returns (ledger, evidence).

    The returned ledger is a *derived* artefact: it is the same population, split
    the same way, with signal rows added. Its ``provenance`` carries a ``fusion``
    block so a reader can tell that the fused columns were computed here rather
    than measured by the corpus producer.
    """
    problems = validate_ledger(ledger)
    if problems:
        raise LedgerError("; ".join(problems))
    methods = tuple(methods)
    unknown = [m for m in methods if m not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"unknown fusion method(s) {unknown}")
    assigned = assign_splits(ledger, seed=split_seed)
    records = assigned["records"]
    higher = bool(ledger["higher_is_more_anomalous"])
    signals = sorted({s for r in records for s in r["scores"]})
    used = [s for s in signals if include is None or s in include]
    if not used:
        raise LedgerError("no signals selected for fusion")
    dropped = [s for s in signals if s not in used]

    cal_neg = [r for r in records if r["split"] == "calibration" and not r["is_positive"]]
    if len(cal_neg) < min_calibration:
        raise LedgerError(
            f"only {len(cal_neg)} calibration negatives < {min_calibration}: a "
            f"conformal null placed on fewer than {min_calibration} trusted assets "
            f"is noise, so no fused p-value is defined")

    pcols = np.empty((len(records), len(used)), np.float64)
    for j, signal in enumerate(used):
        cal = np.asarray([r["scores"][signal] for r in cal_neg], np.float64)
        allv = np.asarray([r["scores"][signal] for r in records], np.float64)
        pcols[:, j] = conformal_pvalues(cal, allv,
                                       higher_is_more_anomalous=higher)

    columns: Dict[str, np.ndarray] = {}
    detail: Dict[str, Any] = {}
    for method in methods:
        p = _fused_pvalues(pcols, method)
        name = f"{prefix}_{method}"
        # 1 - p keeps the ledger's single direction contract (higher = anomalous).
        columns[name] = 1.0 - p
        detail[name] = {
            "method": method,
            "signals": list(used),
            "dropped_signals": list(dropped),
            "n_signals": len(used),
            "p_min": float(np.min(p)),
            "p_median": float(np.median(p)),
            "p_constant": bool(np.allclose(p, p[0], atol=1e-12)),
            "frac_p_at_or_below_alpha": float(np.mean(p <= 0.05)),
            "dependence": ("arbitrary" if method in ("cauchy", "by") else
                           "valid without a dependence assumption"),
            "note": ("every model gets the same fused p-value: the combination is "
                     "carrying no information" if np.allclose(p, p[0], atol=1e-12)
                     else "the fused statistic varies across models"),
        }

    out_records = []
    for i, rec in enumerate(records):
        scores = dict(rec["scores"])
        for name, col in columns.items():
            scores[name] = float(col[i])
        out_records.append({**rec, "scores": scores})
    prov = dict(ledger.get("provenance") or {})
    prov["fusion"] = {
        "schema": FUSION_SCHEMA,
        "producer": "cviaf.lab.fusion_fpr.fuse_ledger",
        "split_seed": split_seed,
        "signals_used": list(used),
        "signals_available": list(signals),
        "n_calibration_negatives": len(cal_neg),
        "columns": detail,
        "p_value_basis": (f"conformal rank p-values of each model against the "
                         f"{len(cal_neg)} calibration-half NEGATIVES "
                         f"(p = (1 + #{{cal >= x}}) / (n + 1)), fused across "
                         f"signals, stored as 1 - p so the ledger's declared "
                         f"direction still reads the rule correctly"),
    }
    out = {k: v for k, v in assigned.items() if k != "records"}
    out["records"] = out_records
    out["provenance"] = prov
    # The derived ledger must still pass the harness validator, or a fused number
    # could be quoted from a ledger the harness would have refused.
    problems = validate_ledger(out)
    if problems:
        raise LedgerError(f"fused ledger failed validation: {'; '.join(problems)}")
    return out, prov["fusion"]


def _fixed_threshold_rate(records: Sequence[Dict[str, Any]], column: str,
                          threshold: float, higher_is_more_anomalous: bool,
                          min_negatives: int) -> Dict[str, Any]:
    """FPR on the evaluation half at a threshold that was NOT derived from data.

    The harness threshold is the alpha-quantile of the calibration negatives. A
    p-value rule's native threshold is alpha itself, and the two are close but not
    equal, so both are reported rather than one being presented as the other.
    """
    ev_neg = [r for r in records
              if r["split"] == "evaluation" and not r["is_positive"]]
    fp = sum(1 for r in ev_neg
             if ((r["scores"][column] > threshold) if higher_is_more_anomalous
                 else (r["scores"][column] < threshold)))
    out = rate(fp, len(ev_neg), min_negatives)
    out["threshold"] = threshold
    out["threshold_basis"] = (f"fixed at {threshold:.6g}: the nominal-alpha rule a "
                             f"p-value is read through, not a calibration quantile")
    return out


def nominal_alpha_rules(ledger: Dict[str, Any], alpha: float,
                        min_negatives: int = 20) -> Dict[str, Any]:
    """FPR of every fused column read as ``p <= alpha`` (score >= 1 - alpha)."""
    records = ledger["records"]
    higher = bool(ledger["higher_is_more_anomalous"])
    out: Dict[str, Any] = {}
    for column in sorted({c for r in records for c in r["scores"]}):
        if not column.startswith("fused"):
            continue
        thr = (1.0 - alpha) if higher else alpha
        out[column] = _fixed_threshold_rate(records, column, thr, higher, min_negatives)
    return out


def flagged_overlap(ledger: Dict[str, Any], rules: Dict[str, Any],
                    higher_is_more_anomalous: bool, ) -> Dict[str, Any]:
    """How the rules' alarm sets sit on top of each other on the evaluation half.

    An FPR of 0.048 and an FPR of 0.051 look like the same detector twice. They are
    the same *rate*, which is a different claim: two rules at alpha can still alarm
    on disjoint assets, and a fused rule earns its place only by alarming on fewer
    assets than the union of the parts (or by catching an asset the parts miss, which
    a clean-only corpus cannot show). Reporting the counts and the Jaccard overlap of
    the flagged sets is what turns "fusion did not change the rate" into an
    answerable question.
    """
    records = ledger["records"]
    keys = [f"{r['corpus']}|{r['model_id']}" for r in records]
    flags: Dict[str, List[int]] = {}
    for name, rule in rules.items():
        if rule.get("status") == "refused":
            continue
        thr = rule["threshold"]
        flags[name] = [i for i, r in enumerate(records)
                       if r["split"] == "evaluation" and not r["is_positive"]
                       and ((r["scores"][name] > thr) if higher_is_more_anomalous
                            else (r["scores"][name] < thr))]
    names = sorted(flags)
    out: Dict[str, Any] = {
        "basis": "evaluation-half negatives flagged by each rule at its own harness "
                 "threshold",
        "n_flagged": {n: len(flags[n]) for n in names},
        "flagged_model_keys": {n: [keys[i] for i in sorted(flags[n])] for n in names},
        "pairs": {},
    }
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = set(flags[names[a]]), set(flags[names[b]])
            inter, union = len(sa & sb), len(sa | sb)
            out["pairs"][f"{names[a]}|{names[b]}"] = {
                "both": inter, "either": union,
                "jaccard": round(inter / union, 4) if union else None,
            }
    return out


def fusion_report(ledger: Dict[str, Any], alpha: float = 0.05,
                  min_negatives: int = 20, split_seed: int = 0,
                  methods: Sequence[str] = ("cauchy", "bonferroni", "by"),
                  prefix: str = "fused") -> Dict[str, Any]:
    """Fuse an all-negative ledger and score the fused rules beside the per-signal ones.

    The live-signal variant is always reported: it is the rule a deployment would
    actually run once the signals that cannot fire are taken out of the sum.
    """
    problems = validate_ledger(ledger)
    if problems:
        raise LedgerError("; ".join(problems))
    assigned = assign_splits(ledger, seed=split_seed)
    signals = sorted({s for r in assigned["records"] for s in r["scores"]})
    deg = degenerate_signals(assigned["records"], signals, alpha,
                             bool(ledger["higher_is_more_anomalous"]))
    live = [s for s in signals if not deg[s]["at_ceiling"]]

    all_led, all_ev = fuse_ledger(ledger, split_seed=split_seed, methods=methods,
                                  include=None, prefix=f"{prefix}_all")
    live_led, live_ev = fuse_ledger(ledger, split_seed=split_seed, methods=methods,
                                    include=live, prefix=f"{prefix}_live")
    combined = dict(all_led)
    combined["records"] = []
    for a, b in zip(all_led["records"], live_led["records"]):
        combined["records"].append({**a, "scores": {**a["scores"], **b["scores"]}})
    combined_prov = dict(combined.get("provenance") or {})
    combined_prov["fusion_live"] = live_ev
    combined["provenance"] = combined_prov
    problems = validate_ledger(combined)
    if problems:
        raise LedgerError("; ".join(problems))

    report = evaluate_corpus(combined, alpha=alpha, min_negatives=min_negatives,
                             split_seed=split_seed)
    report["schema"] = "cviaf.fusion-fpr-report.v1"
    report["fusion"] = {
        "schema": FUSION_SCHEMA,
        "all_signals": all_ev,
        "live_signals": live_ev,
        "degenerate": deg,
        "live_signal_names": live,
        "dropped_from_live": [s for s in signals if s not in live],
        "nominal_alpha_rules": nominal_alpha_rules(combined, alpha, min_negatives),
        "flagged_overlap": flagged_overlap(
            combined, report["rules"], bool(combined["higher_is_more_anomalous"])),
        "alpha": alpha,
        "reading": ("per-signal rows and fused rows are computed by the same harness "
                    "on the same split, so the fused FPR is directly comparable to "
                    "the numbers beside it"),
    }
    return report


def render_fusion(report: Dict[str, Any]) -> str:
    fusion = report["fusion"]
    lines = [f"fused FPR report  alpha={report['alpha']:.3g}  "
             f"evaluation negatives={report['denominators']['evaluation_negatives']}",
             report["headline"], "",
             f"{'rule':26s} {'threshold':>11s} {'FPR':>7s} {'FPR 95% CI':>16s} "
             f"{'status':>10s}",
             "-" * 74]
    quotable = set(report.get("quotable_rules") or report["rules"])
    for name, rule in sorted(report["rules"].items()):
        if name not in quotable:
            continue                  # retired: printed in the appendix of the report
        if rule.get("status") == "refused":
            lines.append(f"{name:26s} {'-':>11s} {'-':>7s} {'-':>16s} {'refused':>10s}")
            continue
        fpr = rule["fpr"]
        fpr_s = (f"{fpr['point_estimate']:.3f}" if fpr["point_estimate"] is not None
                 else "bound")
        ci = (f"[{fpr['ci95_wilson'][0]:.3f},{fpr['ci95_wilson'][1]:.3f}]"
              if fpr["ci95_wilson"] else
              (f"<={fpr['exact_upper_bound_95']:.3f}"
               if fpr["exact_upper_bound_95"] is not None else "not measured"))
        lines.append(f"{name:26s} {rule['threshold']:11.4f} {fpr_s:>7s} {ci:>16s} "
                     f"{rule['status']:>10s}")

    dead = [s for s, d in fusion["degenerate"].items() if d["at_ceiling"]]
    if dead:
        lines += ["", f"SATURATED SIGNALS excluded from the live fusion: "
                      f"{', '.join(dead)}"]
        for s in dead:
            d = fusion["degenerate"][s]
            lines.append(f"  {s:24s} threshold {d['threshold']} == ceiling "
                         f"{d['ceiling']}; p-value floor {d['p_floor']:.3f} over "
                         f"{d['calibration_ties_at_extreme']} tied calibration "
                         f"models, so it cannot carry a rejection at alpha")

    if report.get("appendix_rules"):
        lines += ["", "appendix (measured, not quotable: the threshold is a score no "
                      "asset in this corpus reaches)"]
        for name in sorted(report["appendix_rules"]):
            d = report["appendix_rules"][name]
            lines.append(f"  {name:24s} FPR "
                         f"{(report['rules'].get(name, {}).get('fpr') or {}).get('point_estimate')} "
                         f"thr {d['threshold']:.6g} == corpus max "
                         f"{d['corpus_extreme']:.6g}")

    lines += ["", "the same fused columns read through the nominal-alpha rule "
                  "(p <= alpha), which is what a p-value means without a quantile cut"]
    for name, r in sorted(fusion["nominal_alpha_rules"].items()):
        fpr_s = (f"{r['point_estimate']:.3f}" if r["point_estimate"] is not None
                 else "bound")
        ci = (f"[{r['ci95_wilson'][0]:.3f},{r['ci95_wilson'][1]:.3f}]"
              if r["ci95_wilson"] else f"<={r['exact_upper_bound_95']:.3f}")
        lines.append(f"{name:26s} {r['numerator']:>5d}/{r['denominator']:<5d} "
                     f"{fpr_s:>7s} {ci:>16s}")

    live = fusion["live_signal_names"]
    if live:
        floor = max(fusion["degenerate"][s]["p_floor"] for s in live)
        lines += ["", f"live fusion uses {', '.join(live)}; the largest p-value floor "
                      f"among them is {floor:.3f}, so the fused statistic can still "
                      f"reach alpha {fusion['alpha']:.3g}"]
    return "\n".join(lines)
