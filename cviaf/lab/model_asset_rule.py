"""Model-integrity decision conditional on a trigger stamp, at asset granularity.

An asset is one independently trained model, NOT each image. Supply a predeclared
probe recipe, disjoint calibration models trained without the attack, and the same
trusted reference model and probe images for every model. Do not calibrate on
attacked assets or count windows/images as independent null assets.

Merge decision (measured, not stylistic): the verdict gate keys on **refdiv** alone.
The ctc stamp response was measured as *not exchangeable across seed ranges* on the
50-model clean null corpus -- two-sample KS p = 0.0040 against the seeds 5-15 clean
population and p = 0.0316 against seeds 5-24, with a monotone seed trend across all 81
clean models (Spearman rho = -0.290, p = 0.0087; response mean +0.2367 at seeds 5-15,
+0.1925 at 16-24, +0.0828 at 100-149), while refdiv was exchangeable (KS p = 0.972 /
0.997). A ctc column calibrated on one seed range is therefore not a valid null for a
suspect from another, and multiplying it into the p-value would weaken the gate.
ctc stays *computed* and reported as a diagnostic column that never enters the
decision.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

#: What ``stamp_response`` computes by default: every column worth reporting.
SIGNAL_COLUMNS = ("refdiv", "ctc")
#: The verdict gate. m = 1 here, so rejection needs 5 + ceil(1/alpha) - 1 = 24 clean
#: models at alpha = .05, and the conformal p-value floor is 1/20 = 5%.
DEFAULT_SIGNALS = ("refdiv",)
#: Computed and reported, never gated on. See the module docstring for the measurement.
DIAGNOSTIC_SIGNALS = ("ctc",)


def stamp_response(bare: Mapping[str, Sequence[float]],
                   stamped: Mapping[str, Sequence[float]],
                   signals: Sequence[str] = SIGNAL_COLUMNS) -> dict[str, float]:
    """Paired mean change for each detector, not a stamped-vs-bare p-value.

    Pairing and common model remove much of the stamp artifact, but do not by
    themselves eliminate it. The *distribution of this change on clean models*
    is the null. NaN-only or mismatched measurements are not silently cleared.
    """
    response = {}
    for name in signals:
        x, y = np.asarray(bare[name], float), np.asarray(stamped[name], float)
        if x.shape != y.shape or x.ndim != 1 or not len(x):
            raise ValueError(f"{name}: require equal nonempty paired image scores")
        paired = np.isfinite(x) & np.isfinite(y)
        if not paired.any():
            raise ValueError(f"{name}: no finite paired image scores")
        response[name] = float(np.mean(y[paired] - x[paired]))
    return response


def decide_model_asset(response: Mapping[str, float],
                       clean_responses: Sequence[Mapping[str, float]],
                       alpha: float = .05,
                       signals: Sequence[str] = DEFAULT_SIGNALS,
                       diagnostics: Sequence[str] = DIAGNOSTIC_SIGNALS) -> dict:
    """Bonferroni over *gate* signals; conformal model-level ranks under exchangeability.

    Each signal's nonconformity is the absolute difference from the clean median
    response. The center is estimated from calibration models, never attacks. The
    p=(1 + #rank-clean distances >= suspect distance)/(n_rank+1) rank treats tied values
    conservatively. The min across gate signals is multiplied by m. Rejection requires
    >=ceil(m/alpha)-1 independent rank-calibration models to beat the fusion floor.
    Cross-signal dependence is allowed. This is conditional on exchangeable,
    independent clean model assets and a prespecified probe, not a guarantee for
    a reference or trigger chosen after observing the suspect.

    ``diagnostics`` are scored with the identical rule and reported under
    ``diagnostic_per_signal`` with ``diagnostic_only: True``; they are never
    multiplied into ``asset_pvalue`` and can never reject on their own.
    """
    if not 0 < alpha < 1 or not signals or len(set(signals)) != len(signals):
        raise ValueError("alpha and signals must be valid")
    # Separate the robust-center fit from rank calibration. Reusing the rank
    # models to estimate their own center breaks exchangeability with the test.
    n = len(clean_responses)
    n_center = 5
    required_rank = int(np.ceil(len(signals) / alpha)) - 1
    required = n_center + required_rank
    out = {"method": "asset_conformal_bonferroni_stamped_clean_null",
           "n_clean_models": n, "required_clean_models": required,
           "n_center_models": n_center, "n_rank_models": max(0, n - n_center),
           "signals": list(signals), "diagnostic_signals": list(diagnostics),
           "alpha": alpha, "asset_pvalue": None,
           "reject_at_alpha": False, "abstain": True}
    if n < required:
        out["reason"] = ("insufficient independent clean-model calibration assets; "
                         "image count cannot lower the asset p-value floor")
        return out

    def _column(name: str) -> dict:
        fit = np.asarray([row[name] for row in clean_responses[:n_center]], float)
        cal = np.asarray([row[name] for row in clean_responses[n_center:]], float)
        val = float(response[name])
        if not np.isfinite(fit).all() or not np.isfinite(cal).all() or not np.isfinite(val):
            raise ValueError("nonfinite response")
        center = float(np.median(fit))
        distance = abs(val - center)
        p = (1 + int(np.count_nonzero(abs(cal - center) >= distance))) / (len(cal) + 1)
        return {"p": float(p), "clean_median": center, "distance": distance}

    columns = {}
    try:
        for name in signals:
            columns[name] = _column(name)
    except (KeyError, ValueError, TypeError) as exc:
        out["reason"] = f"incomplete model-level scores: {exc}"
        return out
    asset_p = min(1., len(signals) * min(col["p"] for col in columns.values()))

    diagnostic_columns = {}
    for name in diagnostics:
        if name in signals:
            continue
        try:
            col = _column(name)
        except (KeyError, ValueError, TypeError):
            # A diagnostic that is absent or non-finite is reported as unavailable; it
            # must never be able to change the verdict by failing.
            diagnostic_columns[name] = {"available": False}
            continue
        col.update({"available": True, "diagnostic_only": True})
        diagnostic_columns[name] = col

    out.update(asset_pvalue=asset_p, per_signal=columns,
               diagnostic_per_signal=diagnostic_columns, abstain=False,
               reject_at_alpha=bool(asset_p <= alpha))
    return out
