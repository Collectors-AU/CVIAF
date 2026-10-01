"""
Residual-risk budget (v4 upgrade U2): every clean verdict carries a number.

v3 measures power curves and detection floors as *metadata*. U2 makes the honest
number load-bearing. The operator's actual question after a zero-findings
assessment is not "did you find anything?" but "what is the worst poisoning that
could be present and still have passed?". This module answers it:

  ``invert_power_curve``   r* = min{ r : Power(r) >= target_power }, per module,
                           from a MEASURED power curve (arm_b.py produces them)
  ``clopper_pearson_upper``an exact binomial upper bound on the prevalence of
                           flaggable items in a flag-free sample set
  ``ResidualRisk``         the aggregated report-level statement: R* = max over
                           applicable modules of r*, plus the policy check that
                           FORCES the disposition to ``review`` whenever R*
                           exceeds the operator's declared tolerance

The completeness invariant (v3 A11) said "you cannot accept what you could not
check". U2 extends it: you cannot accept what you had no power to see. An
assessment whose worst blind spot is larger than the operator's tolerance has
not failed -- it has measured its own limits, and the disposition must say so.

Semantics of the power statement, stated precisely so the claim is defensible:

    "Under the declared threat model and battery digest X, poisoning at rate
     >= r* would have been detected with probability >= target_power; this
     assessment provides no power below r*."

``Power(r)`` is the measured fraction of independent seeds on which the
assessment flags an asset poisoned at rate r, at the declared FDR alpha. r* is
the smallest TESTED rate meeting the target; if no tested rate meets it, r* is
None and the module reports no power at any tested rate -- which means the
report-level R* is unbounded and ``accept`` is forbidden, regardless of how
small the other modules' floors are. The maximum is the worst-case blind spot,
and a chain is as strong as its weakest module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #

@dataclass
class ResidualRiskPolicy:
    """The operator's risk appetite, declared in ``AB1.policy`` terms.

    ``tolerance``     the largest residual poisoning rate the operator is willing
                      to accept silently. If the report-level R* exceeds it, the
                      disposition is forced to ``review``. This is a policy
                      input, not a tuned constant: DGIS sets it before a
                      deployment, and the audit log records it.
    ``target_power``  the power the inversion requires (0.8 by convention).
    ``alpha``         the FDR level the power curves were measured at.
    """

    tolerance: float = 0.02
    target_power: float = 0.80
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #

def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial rate.

    Given ``k`` flagged items out of ``n`` examined, the prevalence of flaggable
    items is <= the returned value with confidence ``conf``. Exact, not
    asymptotic: at k=0 and n=200 it returns 0.0149 (the "rule of three" value),
    not 0.0 -- zero findings never means zero prevalence.

    Validity is analytic (it is the inversion of the binomial test), so the
    unit test checks exact coverage on simulated draws rather than tuning it.
    """
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError(f"impossible count k={k} on n={n}")
    if k >= n:
        return 1.0
    from scipy.stats import beta as _beta
    return float(_beta.ppf(conf, k + 1, n - k))


# --------------------------------------------------------------------------- #
# power-curve inversion
# --------------------------------------------------------------------------- #

def invert_power_curve(
    rates: Sequence[float],
    powers: Sequence[float],
    target_power: float = 0.80,
) -> Dict[str, Any]:
    """Invert a measured power curve into the module's r*.

    ``rates`` and ``powers`` are paired measurements, smallest rate first
    (zero rate excluded -- it measures FPR, not power). Returns r* = the
    smallest tested rate whose measured power reaches ``target_power``, or None
    when no tested rate does. No interpolation and no extrapolation: the curve
    is the evidence, and a value between or beyond measured points would be an
    assertion, not a measurement. Where a smoother answer is wanted, the right
    move is to test more rates, not to draw a line.
    """
    pts = sorted((float(r), float(p)) for r, p in zip(rates, powers) if r > 0)
    r_star: Optional[float] = None
    for r, p in pts:
        if p >= target_power:
            r_star = r
            break
    return {
        "r_star": r_star,
        "target_power": float(target_power),
        "max_rate_tested": max((r for r, _ in pts), default=None),
        "curve": [{"rate": r, "power": p} for r, p in pts],
        "claim": module_claim_sentence(r_star, target_power,
                                       max((r for r, _ in pts), default=0.0)),
    }


def module_claim_sentence(r_star: Optional[float], target_power: float,
                          max_rate_tested: float) -> str:
    """The sentence a zero-findings module report must carry."""
    if r_star is None:
        return (f"INSUFFICIENT POWER: no tested poisoning rate up to "
                f"{max_rate_tested * 100:.1f}% was detected with probability >= "
                f"{target_power:.2f}. This module excludes nothing; a clean "
                f"verdict from it must not be read as evidence of cleanliness.")
    return (f"Poisoning at rate >= {r_star * 100:.2f}% would have been detected "
            f"with probability >= {target_power:.2f}; this assessment provides "
            f"no power below {r_star * 100:.2f}%, which must not be reported "
            f"as 'clean'.")


# --------------------------------------------------------------------------- #
# the report-level aggregate
# --------------------------------------------------------------------------- #

@dataclass
class ModuleRisk:
    """One applicable module's contribution to the residual-risk budget."""

    module: str
    applicable: bool
    r_star: Optional[float]          # None + applicable => no power at any tested rate
    target_power: float
    max_rate_tested: Optional[float]
    claim: str
    curve: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResidualRisk:
    """The per-report residual-risk statement (schema field ``residual_risk``).

    ``report_level_r_star`` is the worst-case blind spot under the declared
    threat model: the maximum r* over applicable modules, or None when ANY
    applicable module has no measured power at all (unbounded residual risk).
    ``accept_permitted`` is the A11-style gate extended from "checks that ran"
    to "power that existed": it is False whenever R* exceeds the policy
    tolerance, and in that case ``forced_disposition`` is ``review``.
    """

    per_module: List[ModuleRisk]
    policy: ResidualRiskPolicy
    n_samples: int
    n_flagged: int
    cp_confidence: float = 0.95

    # ---- derived quantities ------------------------------------------------

    @property
    def applicable_modules(self) -> List[ModuleRisk]:
        return [m for m in self.per_module if m.applicable]

    @property
    def report_level_r_star(self) -> Optional[float]:
        """Worst-case blind spot. None means UNBOUNDED (some applicable module
        had no power at any tested rate), which is strictly worse than any
        finite value and must force review."""
        mods = self.applicable_modules
        if not mods:
            return None
        if any(m.r_star is None for m in mods):
            return None
        return max(m.r_star for m in mods)

    @property
    def r_star_unbounded(self) -> bool:
        return self.report_level_r_star is None

    @property
    def prevalence_upper_bound(self) -> float:
        """CP upper bound on the prevalence of flaggable items, given what ran."""
        return clopper_pearson_upper(self.n_flagged, self.n_samples, self.cp_confidence)

    @property
    def accept_permitted(self) -> bool:
        """The U2 gate: never accept past the operator's tolerance."""
        if not self.applicable_modules:
            return False                      # nothing applicable ran: A11 territory
        if self.r_star_unbounded:
            return False
        return bool(self.report_level_r_star <= self.policy.tolerance)

    @property
    def forced_disposition(self) -> Optional[str]:
        """``review`` when the gate suppresses accept, else None (no opinion)."""
        return None if self.accept_permitted else "review"

    # ---- rendering ---------------------------------------------------------

    def claim_sentence(self) -> str:
        mods = self.applicable_modules
        if not mods:
            return ("NO RESIDUAL-RISK STATEMENT: no applicable module executed. "
                    "This assessment carries no negative assurance at all; "
                    "disposition forced to review.")
        if self.r_star_unbounded:
            blind = [m.module for m in mods if m.r_star is None]
            return (f"RESIDUAL RISK UNBOUNDED: module(s) {', '.join(blind)} had no "
                    f"detection power at any tested rate. This assessment cannot "
                    f"exclude poisoning at any tested prevalence; disposition "
                    f"forced to review.")
        r = self.report_level_r_star
        return (f"Under the declared threat model, poisoning at rate >= "
                f"{r * 100:.2f}% would have been detected with probability >= "
                f"{self.policy.target_power:.2f} (worst applicable module). This "
                f"assessment provides no power below {r * 100:.2f}%. "
                f"Separately, {self.n_flagged} of {self.n_samples} samples were "
                f"flagged, bounding the prevalence of flaggable items at <= "
                f"{self.prevalence_upper_bound * 100:.2f}% with "
                f"{self.cp_confidence:.0%} confidence (Clopper-Pearson).")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_module": [m.to_dict() for m in self.per_module],
            "report_level_r_star": self.report_level_r_star,
            "r_star_unbounded": self.r_star_unbounded,
            "prevalence_upper_bound": self.prevalence_upper_bound,
            "cp_confidence": self.cp_confidence,
            "n_samples": self.n_samples,
            "n_flagged": self.n_flagged,
            "policy": self.policy.to_dict(),
            "accept_permitted": self.accept_permitted,
            "forced_disposition": self.forced_disposition,
            "claim": self.claim_sentence(),
        }


def build_residual_risk(
    power_curves: Dict[str, Dict[str, Any]],
    n_samples: int,
    n_flagged: int,
    policy: Optional[ResidualRiskPolicy] = None,
    skipped_modules: Optional[Sequence[str]] = None,
) -> ResidualRisk:
    """Assemble the report-level statement from per-module measured curves.

    ``power_curves`` maps module name -> {"rates": [...], "powers": [...]} as
    measured by the experiment harness (arm_b.py). ``skipped_modules`` names
    modules that were applicable but did not run; each is an applicable module
    with no power, which is exactly the "checks that did not run" case of A11
    re-read as "power that did not exist".
    """
    policy = policy or ResidualRiskPolicy()
    modules: List[ModuleRisk] = []
    for name in sorted(power_curves):
        pc = power_curves[name]
        inv = invert_power_curve(pc["rates"], pc["powers"], policy.target_power)
        modules.append(ModuleRisk(
            module=name, applicable=True, r_star=inv["r_star"],
            target_power=policy.target_power,
            max_rate_tested=inv["max_rate_tested"], claim=inv["claim"],
            curve=inv["curve"]))
    for name in skipped_modules or ():
        modules.append(ModuleRisk(
            module=name, applicable=True, r_star=None,
            target_power=policy.target_power, max_rate_tested=None,
            claim=(f"INSUFFICIENT POWER: module {name} was applicable but did "
                   f"not run. Its residual risk is unbounded by construction.")))
    return ResidualRisk(per_module=modules, policy=policy,
                        n_samples=int(n_samples), n_flagged=int(n_flagged))
