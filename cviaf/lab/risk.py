"""Risk, not just integrity: measured operating characteristics + a loss matrix.

Clause 1.4 of the problem statement asks for risk, and the acceptance test is that
``expected_loss`` is present for every candidate disposition on every finding. Until
now this lane produced p-values and a reject/abstain bit: a number and a yes, with no
statement of what each choice costs.

What is and is not identifiable
-------------------------------
A conformal p-value alone does NOT determine the probability that an asset is
tampered. The p-value is the null tail -- ``P(score at least this extreme | clean)``
-- and turning it into a posterior needs the alternative density, which the
calibration set of clean assets cannot supply. Rather than invent a likelihood, this
module uses the posterior that IS identifiable from quantities the lane can measure:

    PPV  = pi * TPR / (pi * TPR + (1 - pi) * FPR)          for a FLAGGED asset
    p_un = pi * (1-TPR) / (pi * (1-TPR) + (1-pi) * (1-FPR)) for an UNFLAGGED asset

where TPR and FPR are the measured operating characteristics of the rule (the FPR
harness) and pi is a declared prevalence. This is the decision-conditioned posterior:
it is the probability the asset is tampered *given the rule flagged it*, which is
exactly what a disposition has to be based on. It is coarse -- every flagged asset
gets the same posterior regardless of how far its score exceeded the threshold -- and
the report says so rather than implying per-asset precision it does not have.

The loss matrix is :class:`cviaf.lab.review.OperatorCosts`, reused rather than
redefined so the data axis and the model axis cannot drift apart. Its defaults encode
the framework's posture: waving a tampered asset through is catastrophic (25),
quarantining a good one is costly but recoverable (5), and one review action is the
unit of analyst time (1).

CLI
---
    python -m cviaf.lab.risk --tpr 0.146 --fpr 0.05 --prevalence 0.1
        [--report runs/fpr_ledger_report.json] [--json out.json]

Exit: 0 measured, 2 the inputs do not identify a posterior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from cviaf.lab.review import OperatorCosts, expected_losses

ASSET_RISK_SCHEMA = "cviaf.asset-risk.v1"
POLICIES = ("accept", "review", "quarantine")
# Declared, editable, and recorded in every report: the share of assets in the
# population that are tampered. The FPR corpus here is built to be 50/50 for power,
# which is NOT the deployment prevalence, so the number must be stated, not inferred.
DEFAULT_PREVALENCE = 0.10
# Accept is a decision to stop looking; it needs the evidence to be strong, not merely
# the cheapest action. This bound is the probability of tampering above which "accept"
# is demoted to "review" even when the loss matrix would prefer it.
DEFAULT_ACCEPT_BOUND = 0.05


def posterior_flagged(tpr: float, fpr: float, prevalence: float) -> Optional[float]:
    """P(tampered | flagged) = PPV. None when no asset can be flagged."""
    num = prevalence * tpr
    den = num + (1.0 - prevalence) * fpr
    if den <= 0.0:
        return None
    return float(min(1.0, max(0.0, num / den)))


def posterior_unflagged(tpr: float, fpr: float, prevalence: float) -> Optional[float]:
    """P(tampered | not flagged). None when every asset is flagged."""
    num = prevalence * (1.0 - tpr)
    den = num + (1.0 - prevalence) * (1.0 - fpr)
    if den <= 0.0:
        return None
    return float(min(1.0, max(0.0, num / den)))


def _loss(posterior: Optional[float], costs: OperatorCosts) -> Dict[str, Optional[float]]:
    """Per-disposition expected loss, via the same arithmetic the data axis uses."""
    if posterior is None:
        return {a: None for a in POLICIES}
    el = expected_losses(np.asarray([posterior], np.float64), costs)
    return {a: float(el[a][0]) for a in POLICIES}


def disposition(posterior: Optional[float], costs: OperatorCosts,
                accept_bound: float = DEFAULT_ACCEPT_BOUND,
                accept_permitted: bool = True) -> Dict[str, Any]:
    """Cost-minimal disposition, with accept gated on the evidence and ties to review.

    Two deliberate departures from a bare argmin: review wins exact ties (the
    informative action is preferred, matching ``review.plan_review``), and ``accept``
    is demoted to ``review`` when the posterior exceeds ``accept_bound`` -- otherwise a
    cheap loss matrix would license stopping the investigation on thin evidence.
    """
    losses = _loss(posterior, costs)
    if posterior is None:
        return {"disposition": "review" if not accept_permitted else "accept",
                "reason": "posterior not identified from the declared inputs",
                "expected_loss": losses, "demoted_from": None}
    demoted_from = None
    ranked = list(POLICIES) if accept_permitted else [a for a in POLICIES
                                                      if a != "accept"]
    # Tie handling is tolerant on purpose: an exact tie in decimal arithmetic is
    # rarely exact in binary (review 1.0 vs quarantine (1-0.8)*5 = 0.9999999999999998),
    # and the informative action must win that comparison rather than lose it to a
    # floating-point artifact.
    lo = min(losses[a] for a in ranked)
    tied = [a for a in ranked if losses[a] <= lo + 1e-12]
    choice = "review" if "review" in tied else tied[0]
    if not accept_permitted:
        demoted_from = "accept" if losses["accept"] <= lo + 1e-12 else None
    elif choice == "accept" and posterior > accept_bound:
        demoted_from, choice = "accept", "review"
    return {"disposition": choice, "expected_loss": losses,
            "loss_if_chosen": losses[choice], "demoted_from": demoted_from,
            "accept_bound": accept_bound,
            "reason": (f"posterior {posterior:.4f} "
                       + (f"exceeds the accept bound {accept_bound}; accept demoted to "
                          f"review" if demoted_from else
                          f"(accept bound {accept_bound})"))}


def policy_expected_loss(tpr: float, fpr: float, prevalence: float,
                         costs: OperatorCosts) -> Dict[str, float]:
    """Expected loss PER ASSET for three deployment policies at one operating point.

    ``accept_all``          ship everything: tampered assets pass with probability 1.
    ``quarantine_flagged``  the rule the framework currently runs: flagged assets are
                            withheld, everything else ships.
    ``review_flagged``      flagged assets go to a human first, who resolves the asset
                            correctly with probability ``1 - reviewer_error``.
    """
    # accept_all
    e_accept = prevalence * costs.false_accept
    # quarantine flagged: caught tampered assets cost nothing, missed ones cost
    # false_accept; flagged clean assets cost false_quarantine.
    e_quar = (prevalence * (1.0 - tpr) * costs.false_accept
              + (1.0 - prevalence) * fpr * costs.false_quarantine)
    # review flagged: a review costs the unit of analyst time either way; a missed
    # tampered asset still passes, and a reviewed clean asset is released unless the
    # review itself errs.
    e_rev = (prevalence * (tpr * (costs.review + costs.reviewer_error * costs.false_accept)
                           + (1.0 - tpr) * costs.false_accept)
             + (1.0 - prevalence) * fpr * (costs.review
                                           + (1.0 - costs.reviewer_error)
                                           * costs.false_quarantine))
    return {"accept_all": float(e_accept), "quarantine_flagged": float(e_quar),
            "review_flagged": float(e_rev)}


def break_even_prevalence(tpr: float, fpr: float, costs: OperatorCosts,
                          policy: str = "quarantine_flagged",
                          against: str = "accept_all", tol: float = 1e-9) -> Optional[float]:
    """The prevalence at which two policies cost the same. None if they never cross.

    This is the number that tells an operator how bad the population has to be before
    the detector earns its keep, which is the operational form of \"risk\".
    """
    def diff(pi: float) -> float:
        losses = policy_expected_loss(tpr, fpr, pi, costs)
        return losses[policy] - losses[against]

    lo, hi = 0.0, 1.0
    d_lo, d_hi = diff(lo), diff(hi)
    if d_lo == 0.0:
        return 0.0
    if d_hi == 0.0:
        return 1.0
    if (d_lo > 0) == (d_hi > 0):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (diff(mid) > 0) == (d_lo > 0):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


def asset_risk(model_id: str, kind: str, p_value: float, alpha: float, tpr: float,
               fpr: float, prevalence: float = DEFAULT_PREVALENCE,
               costs: Optional[OperatorCosts] = None,
               accept_bound: float = DEFAULT_ACCEPT_BOUND,
               accept_permitted: bool = True) -> Dict[str, Any]:
    """One finding's risk: the decision, the posterior behind it, and all three losses.

    ``expected_loss`` is present for accept, review AND quarantine on every finding --
    that is the clause 1.4 acceptance test -- even though only one of them is chosen,
    because the alternatives are what make the choice reviewable.
    """
    costs = costs or OperatorCosts()
    flagged = bool(np.isfinite(p_value) and p_value <= alpha)
    posterior = (posterior_flagged(tpr, fpr, prevalence) if flagged
                 else posterior_unflagged(tpr, fpr, prevalence))
    decision = disposition(posterior, costs, accept_bound=accept_bound,
                          accept_permitted=accept_permitted)
    return {
        "model_id": model_id, "kind": kind, "p_value": float(p_value), "alpha": alpha,
        "flagged": flagged, "posterior": posterior,
        "posterior_basis": ("PPV/NPV from measured TPR/FPR and the declared prevalence; "
                            "a conformal p-value alone does not identify a posterior"),
        "expected_loss": decision["expected_loss"],
        "expected_loss_if_acted": decision["loss_if_chosen"],
        "disposition": decision["disposition"], "demoted_from": decision["demoted_from"],
        "reason": decision["reason"],
        "loss_matrix": costs.to_dict(), "prevalence": prevalence,
        "operating_point": {"tpr": tpr, "fpr": fpr},
    }


def rule_risk(tpr: Optional[float], fpr: Optional[float],
              prevalence: float = DEFAULT_PREVALENCE,
              costs: Optional[OperatorCosts] = None,
              accept_bound: float = DEFAULT_ACCEPT_BOUND,
              n_assets: Optional[int] = None) -> Dict[str, Any]:
    """Risk of running one rule, from its measured operating point."""
    costs = costs or OperatorCosts()
    if tpr is None or fpr is None:
        return {"schema": ASSET_RISK_SCHEMA, "status": "not_measured",
                "reason": ("the rule's TPR or FPR has no point estimate (a bound at "
                           "0/n is not enough to place a posterior)"),
                "prevalence": prevalence, "loss_matrix": costs.to_dict()}
    losses = policy_expected_loss(tpr, fpr, prevalence, costs)
    # Ties are broken towards the policy with the FEWEST interventions, so a rule that
    # never fires is not reported as "review is recommended" when review would never be
    # triggered. The tie is stated rather than hidden.
    order = {name: i for i, name in enumerate(("accept_all", "quarantine_flagged",
                                              "review_flagged"))}
    lo = min(losses.values())
    tied = sorted((p for p, v in losses.items() if abs(v - lo) <= 1e-12), key=order.get)
    best = tied[0]
    out: Dict[str, Any] = {
        "schema": ASSET_RISK_SCHEMA, "status": "measured",
        "operating_point": {"tpr": float(tpr), "fpr": float(fpr)},
        "prevalence": prevalence, "loss_matrix": costs.to_dict(),
        "accept_bound": accept_bound,
        "posterior_when_flagged": posterior_flagged(tpr, fpr, prevalence),
        "posterior_when_not_flagged": posterior_unflagged(tpr, fpr, prevalence),
        "expected_loss_per_asset": losses,
        "recommended_policy": best,
        "tied_policies": tied,
        "recommended_expected_loss_per_asset": losses[best],
        "break_even_prevalence_vs_accept_all": break_even_prevalence(
            tpr, fpr, costs, "quarantine_flagged", "accept_all"),
        "caveat": ("posteriors are decision-conditioned (every flagged asset shares one "
                   "value); a per-asset posterior would need the alternative density, "
                   "which the clean calibration set cannot identify"),
    }
    if float(tpr) == 0.0 and float(fpr) == 0.0:
        out["note"] = ("the rule never fires (TPR = FPR = 0), so no flagged-handling "
                       "policy can differ from accept_all; the tie is structural, not "
                       "a preference")
    if n_assets:
        out["expected_total_loss"] = {k: v * n_assets for k, v in losses.items()}
    return out


def render(report: Mapping[str, Any]) -> str:
    if report.get("status") != "measured":
        return f"risk: not measured -- {report.get('reason')}"
    def show(v):
        # None means "no asset can be flagged" (or every asset is), which is a
        # statement about the rule, not a missing number
        return "n/a" if v is None else f"{v:.4f}"

    lines = [f"risk at prevalence {report['prevalence']:.3f} "
             f"(loss matrix {report['loss_matrix']})",
             f"  operating point   TPR {report['operating_point']['tpr']:.3f} "
             f"FPR {report['operating_point']['fpr']:.3f}",
             f"  P(tampered|flagged)     {show(report['posterior_when_flagged'])}",
             f"  P(tampered|not flagged) {show(report['posterior_when_not_flagged'])}",
             f"  {'policy':22s} {'expected loss per asset':>24s}"]
    for name, value in sorted(report["expected_loss_per_asset"].items(),
                              key=lambda kv: kv[1]):
        mark = "  <- recommended" if name == report["recommended_policy"] else ""
        lines.append(f"  {name:22s} {value:24.4f}{mark}")
    be = report.get("break_even_prevalence_vs_accept_all")
    lines.append(f"  break-even prevalence vs accept_all: "
                 + ("never crosses" if be is None else f"{be:.4f}"))
    if report.get("note"):
        lines.append(f"  note: {report['note']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cviaf lab risk",
                                 description="expected loss from measured TPR/FPR")
    ap.add_argument("--tpr", type=float, default=None)
    ap.add_argument("--fpr", type=float, default=None)
    ap.add_argument("--prevalence", type=float, default=DEFAULT_PREVALENCE)
    ap.add_argument("--report", default=None,
                    help="an FPR/TPR report to price every signal in")
    ap.add_argument("--signal", default=None)
    ap.add_argument("--false-accept", type=float, default=25.0)
    ap.add_argument("--false-quarantine", type=float, default=5.0)
    ap.add_argument("--review-cost", type=float, default=1.0)
    ap.add_argument("--reviewer-error", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    costs = OperatorCosts(false_accept=args.false_accept,
                          false_quarantine=args.false_quarantine,
                          review=args.review_cost, reviewer_error=args.reviewer_error)
    if args.report:
        with open(args.report, encoding="utf-8") as fh:
            fpr_report = json.load(fh)
        out = {"schema": ASSET_RISK_SCHEMA, "source": args.report,
               "prevalence": args.prevalence, "loss_matrix": costs.to_dict(),
               "per_signal": {}}
        for name, rule in sorted(fpr_report.get("rules", {}).items()):
            if args.signal and name != args.signal:
                continue
            tpr = (rule.get("tpr") or {}).get("point_estimate")
            fpr = (rule.get("fpr") or {}).get("point_estimate")
            priced = rule_risk(tpr, fpr, args.prevalence, costs)
            out["per_signal"][name] = priced
        if args.signal and args.signal not in out["per_signal"]:
            print(f"no signal {args.signal!r} in {args.report}")
            return 2
        print(f"risk from {args.report} at prevalence {args.prevalence:.3f}")
        for name, priced in out["per_signal"].items():
            print("\n" + name)
            print("\n".join("  " + ln for ln in render(priced).splitlines()[1:]))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1, allow_nan=False)
            print(f"\nwritten to {args.json}")
        return 0

    if args.tpr is None or args.fpr is None:
        print("--tpr and --fpr are required without --report")
        return 2
    priced = rule_risk(args.tpr, args.fpr, args.prevalence, costs)
    print(render(priced))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(priced, fh, indent=1, allow_nan=False)
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
