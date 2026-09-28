"""
Baseline assurance systems: what the pipeline looks like *before* CVIAF.

Why this module exists
----------------------
The problem statement asks for evidence that an assurance layer improves on
existing practice. An improvement is only measurable against a *named*
alternative, so the three baselines here are stated as **practices**, each one a
thing real pipelines actually do, and each one is scored on the same assets, with
the same signals, as CVIAF:

``digest_manifest``
    Verify that the artefact hash matches the hash recorded at enrolment. This is
    the most common control in existence and it is genuinely useful: it detects an
    artefact that changed in transit or at rest. It says nothing *about* the
    artefact, so a model that was backdoored before enrolment, or a dataset
    poisoned by a contributor, passes cleanly. It is the transport control, not
    an assurance control.

``fixed_threshold_single_signal``
    Take one anomaly statistic, pick a cut, flag everything above it. Two variants
    are implemented and both are reported honestly:

    ``fixed``     a hand-set constant (the "0.5 on a score in [0,1]" habit);
    ``quantile``  the cut placed at a quantile of the *reference* scores, which is
                  what a careful engineer does without conformal theory. This
                  variant is nearly a conformal threshold and is expected to do
                  well on false positives -- see the note below on why it is still
                  not enough.

    Neither variant controls error across assets, neither can name an attack class,
    neither attributes anything to a contributor, and neither can abstain.

``hand_tuned_single_signal``
    The strongest form of the previous idea: the threshold is chosen *in
    hindsight* to hit a target false-positive rate on the clean control of this
    exact corpus. This is deliberately generous to the baseline -- it spends
    information an operator would not have. If a baseline still loses under this
    handicap, the loss is attributable to the design (calibration, fusion,
    aggregation, abstention) rather than to a badly chosen constant.

Why the baseline being "nearly conformal" is not the same as conformal
---------------------------------------------------------------------
``quantile`` places the cut so that 1% of *reference* items exceed it, which is
the same arithmetic as a conformal threshold. The difference is what it licenses
you to say. A quantile cut yields a number; a conformal p-value yields a validity
statement (``P(p <= alpha) <= alpha`` under exchangeability) which is what allows
the report to publish a confidence and a detection floor rather than a count of
alerts. The comparison table therefore reports both, and the axis on which the
baseline always loses regardless of threshold is *attribution*, *abstention* and
*stated confidence* -- not raw sensitivity.

Honest framing, carried into every table this module feeds
----------------------------------------------------------
These baselines are not strawmen and they are not the enemy. Two of the three are
good practice and CVIAF keeps them: digest manifesting survives inside the
provenance module, and a reference-quantile cut survives as the calibration step.
What the comparison shows is which claims each control can *support*.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.calibrate import benjamini_yekutieli, conformal_pvalues

BASELINE_NAMES = (
    "digest_manifest",
    "fixed_threshold_single_signal",
    "hand_tuned_single_signal",
)

# The axes an assurance system is scored on. Kept as a constant because the
# report, the comparison table and the tests all iterate over it.
AXES = ("data_integrity", "model_integrity", "inference_provenance")


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

@dataclass
class Verdict:
    """One system's decision about one asset.

    ``abstained`` is the field that makes this comparable. A baseline has no way
    to say "I could not run the applicable checks", so for a baseline it is only
    ever True when the input it needs is missing entirely -- whereas CVIAF
    abstains whenever the applicable battery did not run, and *accept* is
    forbidden in that case. Scoring abstention as a third outcome (rather than
    folding it into "not flagged") is what stops an unavailable check from being
    silently reported as a pass.
    """

    system: str = ""
    asset: str = ""
    axis: str = ""
    flagged: bool = False
    abstained: bool = False
    attack_class: str = ""
    score: Optional[float] = None
    threshold: Optional[float] = None
    attribution: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # `decision` is derived, but the scored artifacts and the comparison tables
        # read it directly, so it is materialised here rather than recomputed by
        # every consumer from two booleans that can drift apart.
        d["decision"] = self.decision
        return d

    @property
    def decision(self) -> str:
        """Three-way outcome: flag / clear / abstain."""
        if self.abstained:
            return "abstain"
        return "flag" if self.flagged else "clear"


def _digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# B0: digest / manifest checking
# --------------------------------------------------------------------------- #

class DigestManifestBaseline:
    """Baseline 0 -- verify the artefact hash against the enrolled hash.

    This is the control almost every pipeline already has. It is modelled here
    faithfully, including its real strength: it *does* detect an artefact that was
    modified after enrolment. What it cannot do is say anything about an artefact
    that was created maliciously and enrolled honestly -- which is how a poisoned
    dataset or a backdoored model actually arrives.
    """

    name = "digest_manifest"
    axis_notes = {
        "model_integrity": "compares the weights digest against the enrolled digest",
        "inference_provenance": "recomputes the record hash and compares it to the recorded hash",
    }

    def __init__(self, enrolled: Optional[Dict[str, str]] = None):
        # asset key -> digest recorded at enrolment
        self.enrolled: Dict[str, str] = dict(enrolled or {})

    def enroll(self, asset: str, digest: str) -> None:
        self.enrolled[asset] = digest

    def assess_digest(
        self,
        asset: str,
        digest: str,
        axis: str,
        attack_class: str = "",
        note: str = "",
    ) -> Verdict:
        if asset not in self.enrolled:
            return Verdict(
                system=self.name, asset=asset, axis=axis, abstained=True,
                reason=("no enrolled digest for this asset: a first-sight asset has "
                        "nothing to compare against, which is the baseline's "
                        "structural blind spot"),
                evidence={"enrolled": None, "observed": digest[:16]},
            )
        expected = self.enrolled[asset]
        changed = expected != digest
        return Verdict(
            system=self.name, asset=asset, axis=axis, flagged=bool(changed),
            attack_class=attack_class if changed else "",
            score=1.0 if changed else 0.0, threshold=0.0,
            confidence=1.0 if changed else None,
            reason=(f"{note}digest {'CHANGED' if changed else 'matches'} the enrolled value "
                    f"(enrolled {expected[:16]}, observed {digest[:16]})"),
            # A digest says *that* something changed, never *what* or *who*.
            attribution=[],
            evidence={"enrolled_digest": expected, "observed_digest": digest,
                      "detects": "post-enrolment modification only"},
        )


# --------------------------------------------------------------------------- #
# B1 / B2: single signal with a fixed cut
# --------------------------------------------------------------------------- #

class ThresholdedSingleSignal:
    """One statistic, one cut. The classic detector-as-threshold design.

    Deliberate limitations, all of which CVIAF removes:

    * **One signal.** No fusion, so a signal that is structurally blind to a class
      of attack (CTC on cloaking) is the whole system's blind spot.
    * **No calibration.** ``fixed`` uses a constant; ``quantile`` uses the reference
      quantile, which is arithmetically close to conformal but carries no validity
      statement and cannot produce a confidence or a floor.
    * **No multiplicity control.** Every asset is tested independently, so the
      false-positive rate grows with the size of the intake.
    * **No abstention.** If the statistic is undefined the asset is reported as
      clear, because there is nothing in the design that can say "not assessed".
    * **No attribution.** Scores are per item; contributors are not a concept here.
    """

    name = "fixed_threshold_single_signal"

    def __init__(
        self,
        signal: str = "ctc",
        threshold_mode: str = "fixed",
        fixed_value: float = 0.5,
        quantile: float = 0.99,
        reference: Optional[np.ndarray] = None,
    ):
        self.signal = signal
        self.threshold_mode = threshold_mode
        self.fixed_value = float(fixed_value)
        self.quantile = float(quantile)
        self.reference = None if reference is None else np.asarray(reference, np.float64)

        if threshold_mode == "fixed":
            self.threshold: Optional[float] = float(fixed_value)
        elif threshold_mode == "quantile":
            if self.reference is None or self.reference.size == 0:
                raise ValueError("quantile mode needs reference scores to place the cut")
            self.threshold = float(np.quantile(self.reference, self.quantile))
        elif threshold_mode == "hand_tuned":
            self.threshold = None       # set later by tune_on()
        else:
            raise ValueError(f"unknown threshold_mode {threshold_mode!r}")

    # -- threshold selection ------------------------------------------------- #

    def tune_on(self, clean_scores: np.ndarray, target_fpr: float = 0.05) -> float:
        """Place the cut to hit ``target_fpr`` on this corpus's clean control.

        This is the generous baseline: it is allowed to look at clean data from
        the evaluated corpus, which an operator cannot do. Reported as such.
        """
        s = np.asarray(clean_scores, np.float64).ravel()
        s = s[np.isfinite(s)]
        if s.size == 0:
            self.threshold = float("inf")
        else:
            self.threshold = float(np.quantile(s, 1.0 - target_fpr))
        self.threshold_mode = "hand_tuned"
        return self.threshold

    # -- decision ------------------------------------------------------------ #

    def assess_items(
        self,
        asset: str,
        scores: np.ndarray,
        axis: str,
        attack_class: str = "",
        item_label: str = "item",
        extras: Optional[Dict[str, Any]] = None,
    ) -> Verdict:
        s = np.asarray(scores, np.float64).ravel()
        finite = np.isfinite(s)
        if not finite.any():
            return Verdict(
                system=self.name, asset=asset, axis=axis, abstained=True,
                reason=(f"{self.signal} is undefined for every {item_label} of this "
                        f"asset; a baseline reports this as 'clear' or crashes, and "
                        f"CVIAF reports 'not assessed'"),
                evidence={"n_items": int(s.size), "n_finite": 0,
                          "signal": self.signal, **(extras or {})},
            )
        thr = self.threshold if self.threshold is not None else float("inf")
        flagged_items = int(np.sum(finite & (s > thr)))
        peak = float(np.max(s[finite]))
        return Verdict(
            system=self.name, asset=asset, axis=axis,
            flagged=bool(flagged_items > 0),
            attack_class=attack_class if flagged_items else "",
            score=peak, threshold=float(thr),
            reason=(f"{self.signal}: {flagged_items}/{int(finite.sum())} {item_label}(s) "
                    f"above the cut ({thr:.4f}); peak {peak:.4f}"),
            attribution=[],
            evidence={"signal": self.signal, "threshold_mode": self.threshold_mode,
                      "n_items": int(s.size), "n_finite": int(finite.sum()),
                      "n_flagged_items": flagged_items,
                      **(extras or {})},
        )


class PerSampleOutlierBaseline(ThresholdedSingleSignal):
    """Data-axis baseline: Mahalanobis distance above a cut, one sample at a time.

    Kept as its own class because the *unit* differs -- here the items are training
    samples, and the practical question an analyst asks is "which samples do I pull
    off the pile?". That question is answerable by this baseline. "Which contributor
    do I stop accepting from?" is not, because a per-sample flag list has no source
    field in it, and that is the gap the comparison measures.

    ``cut`` should be a reference quantile rather than a literal constant. A
    hand-set constant on a 64-dimensional standardised distance flags essentially
    everything, because the expected distance of an INLIER grows with the square root
    of the dimensionality -- a 3.0 cut on 64 dimensions is not a 3-sigma cut, it is a
    0.4-sigma cut. The honest baseline is the one a careful engineer would build: a
    cut placed where the reference itself exceeds it 1% of the time. It is the same
    arithmetic as a conformal threshold, and the comparison therefore tests the
    decision layer above it (multiplicity, aggregation, abstention) rather than the
    arithmetic.
    """

    name = "per_sample_outlier"

    def __init__(self, cut: Optional[float] = None, quantile: float = 0.99,
                 reference: Optional[np.ndarray] = None):
        if cut is None:
            super().__init__(signal="mahalanobis", threshold_mode="quantile",
                             quantile=quantile, reference=reference)
        else:
            super().__init__(signal="mahalanobis", threshold_mode="fixed",
                             fixed_value=cut, reference=reference)


# --------------------------------------------------------------------------- #
# CVIAF's side of the same decisions
# --------------------------------------------------------------------------- #

def asset_pvalue_from_items(
    item_pvalues: np.ndarray,
    method: str = "cauchy",
) -> Tuple[float, Dict[str, Any]]:
    """Collapse per-item conformal p-values into ONE p-value per asset.

    This is the step that makes FDR control over a corpus possible at all, and it
    is the fix named by the lab's own granularity diagnostic: apply multiplicity
    control at the granularity an operator acts on (one asset), not at the
    granularity of individual images.

    ``cauchy``     Cauchy combination over the asset's item p-values. Valid under
                   arbitrary dependence between items -- which is required here,
                   because images from one contributor or one sensor are not
                   independent. Needs no multiplicity factor, so the conformal
                   floor of ``1/(n_cal+1)`` does not become fatal.
    ``bonferroni`` ``min(1, m * min p)``. Valid, and the conservative alternative
                   reported alongside so the dependence choice is visible rather
                   than asserted.
    """
    p = np.asarray(item_pvalues, np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 1.0, {"n_items": 0, "method": method,
                     "reason": "no finite item p-values; the check did not run"}
    if method == "bonferroni":
        val = float(min(1.0, p.size * float(p.min())))
    elif method == "cauchy":
        from cviaf.lab.calibrate import cauchy_combine
        val = float(cauchy_combine(p))
    else:
        raise ValueError(f"unknown asset-level combination method {method!r}")
    return val, {
        "method": method, "n_items": int(p.size),
        "min_item_p": float(p.min()), "median_item_p": float(np.median(p)),
        "cai": None,
    }


def cviaf_data_asset_verdict(
    asset: str,
    signals: Dict[str, Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.05,
    contributor_labels: Optional[Sequence[str]] = None,
    attribution_threshold: float = 0.5,
    lower_is_anomalous: Sequence[str] = (),
) -> Verdict:
    """CVIAF's data-axis decision: calibrated, fused, aggregated, attributable.

    ``signals`` maps signal name -> (reference scores, asset scores). Each signal is
    turned into conformal p-values against the reference, fused by
    ``min(1, m * min p)`` (Bonferroni, so a blind signal cannot dilute a working
    one -- the same argument as the model axis), and collapsed to one asset
    p-value. Contributor aggregation then runs over the items *this* decision
    actually flagged, so the posterior and the flag can never disagree about what
    was found.

    ``lower_is_anomalous`` names the signals whose *low* values are the anomaly.
    This is not a detail: a duplicate-distance statistic is 0 for an exact copy, so
    treating "higher is more anomalous" uniformly would make the duplicate signal
    systematically anti-correlated with the truth and it would *protect* the attack
    it exists to catch. Direction is declared per signal for the same reason the
    access level is declared per check -- silently assuming it is how a detector
    ends up perfectly inverted and still looks like it is running.
    """
    from cviaf.lab.detectors import contributor_risk

    lower = set(lower_is_anomalous)
    item_p: Dict[str, np.ndarray] = {}
    for name, (ref, obs) in signals.items():
        if ref is None or obs is None:
            continue
        r = np.asarray(ref, np.float64).ravel()
        o = np.asarray(obs, np.float64).ravel()
        r = r[np.isfinite(r)]
        if r.size < 5:
            continue
        with np.errstate(invalid="ignore"):
            item_p[name] = np.where(
                np.isfinite(o),
                conformal_pvalues(r, o, higher_is_more_anomalous=name not in lower),
                1.0)

    if not item_p:
        return Verdict(
            system="cviaf", asset=asset, axis="data_integrity", abstained=True,
            reason="no data-integrity signal could be calibrated (no reference split)",
            evidence={"signals_available": list(signals.keys())},
        )

    names_sorted = sorted(item_p)
    stacked = np.stack([item_p[k] for k in names_sorted], axis=1)
    m = stacked.shape[1]
    fused = np.clip(m * stacked.min(axis=1), 0.0, 1.0)
    flagged = benjamini_yekutieli(fused, alpha)

    asset_p, combo = asset_pvalue_from_items(fused, method="cauchy")
    n_flagged = int(flagged.sum())

    attribution: List[Dict[str, Any]] = []
    top: Optional[str] = None
    if contributor_labels is not None and n_flagged:
        labels = np.asarray([str(c) for c in contributor_labels], dtype=object)
        names = list(dict.fromkeys(labels.tolist()))
        counts = np.array([float(np.sum(flagged & (labels == c))) for c in names])
        totals = np.array([float(np.sum(labels == c)) for c in names])
        post = contributor_risk(counts, totals, tau=0.05, seed=0)
        attribution = sorted(
            ({"contributor": names[int(p["contributor_index"])],
              "n_flagged": int(p["n_flagged"]), "n_samples": int(p["n_samples"]),
              "raw_flag_rate": round(float(p["raw_rate"]), 4),
              "posterior_p_exceeds_tolerance": round(float(p["posterior_p_exceeds_tolerance"]), 4),
              "posterior_median_rate": round(float(p["posterior_median_rate"]), 4)}
             for p in post),
            key=lambda d: -d["posterior_p_exceeds_tolerance"])
        if attribution and attribution[0]["posterior_p_exceeds_tolerance"] >= attribution_threshold:
            top = attribution[0]["contributor"]

    return Verdict(
        system="cviaf", asset=asset, axis="data_integrity",
        flagged=bool(n_flagged > 0),
        attack_class=_data_attack_from_signals(item_p, fused, flagged),
        score=asset_p, threshold=float(alpha),
        confidence=float(1.0 - asset_p),
        attribution=[top] if top else [],
        reason=(f"{n_flagged} sample(s) flagged by FDR-controlled calibrated tests "
                f"(asset p={asset_p:.3g}); "
                + (f"source attribution: {top}" if top else "no contributor rises "
                   "above the tolerance, which is the correct answer when the "
                   "poison is spread across sources")),
        evidence={
            "asset_pvalue": asset_p,
            "combination": combo,
            "n_flagged": n_flagged,
            "signals": {k: {"n_flagged": int(np.sum(benjamini_yekutieli(item_p[k], alpha))),
                            "min_p": float(item_p[k].min())} for k in names_sorted},
            "contributor_posterior": attribution,
            "fdr_alpha": float(alpha),
        },
    )


def _data_attack_from_signals(
    item_p: Dict[str, np.ndarray],
    fused: np.ndarray,
    flagged: np.ndarray,
) -> str:
    """Name the most likely data attack class from WHICH signal fired.

    A deliberate, explainable rule rather than a learned mapping: near-duplicate
    flips the duplicate statistic and not the spectral one; a trigger or an OOD
    insert flips the spectral statistic and not the duplicate one. Naming a class
    is worth doing because the disposition differs -- and the rule is auditable.
    """
    if not flagged.any():
        return ""
    fired = {}
    for name, p in item_p.items():
        fired[name] = int(np.sum((p[flagged]) <= 0.05))
    dup = fired.get("duplicate", 0)
    other = sum(v for k, v in fired.items() if k != "duplicate")
    if dup >= max(1, int(0.5 * flagged.sum())) and dup > other:
        return "near_duplicate_flooding"
    if other > 0:
        return "trigger_injection_or_ood_insertion"
    return ""
