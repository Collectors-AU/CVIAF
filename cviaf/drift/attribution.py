"""Natural-vs-manipulated attribution for detected distribution shifts.

This module implements the attribution layer described in the drift-lane
design doc: localised-vs-diffuse shape tests, contributor-level concentration
analysis, and an explicit ``under_determined`` verdict when the evidence does
not distinguish natural drift from manipulation (the "honesty valve").

Design rules:
  - Every statistic is calibrated against a natural-drift battery; a vote
    fires only when its calibrated p-value passes alpha.
  - A verdict is earned by votes, never emitted by default.
  - ``contributor_concentration`` can never convict alone: one contributor may
    honestly operate in one terrain or own one sensor.
  - Missing inputs produce ``not_run`` votes, named in the report, and can
    degrade the verdict to ``attribution_unavailable`` (fail closed).
  - Verdict language describes the *shape* of the evidence, never intent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

VERDICT_SUSPICIOUS = "suspicious_manipulation"
VERDICT_NATURAL = "probable_natural_drift"
VERDICT_UNDERDETERMINED = "under_determined"
VERDICT_UNAVAILABLE = "attribution_unavailable"
VERDICT_NO_SHIFT = "no_material_shift"

SHAPE_VOTES = (
    "dimension_concentration",
    "direction_coherence",
    "spatial_locality",
    "confidence_polarization",
)


# --------------------------------------------------------------------------- #
# vote + calibration containers
# --------------------------------------------------------------------------- #

@dataclass
class Vote:
    """One calibrated attribution signal."""
    name: str
    direction: str                      # "manipulation" | "natural"
    fired: bool
    stat: Optional[float] = None
    p_value: Optional[float] = None
    detail: str = ""
    not_run: Optional[str] = None       # reason string when the vote did not run

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NaturalDriftCalibration:
    """Empirical nulls from a natural-drift battery (terrain/season/sensor/light).

    Built by ``cviaf.lab.driftbench``. Without it the drift module must not
    attribute; it reports ``attribution_unavailable`` instead.
    """
    alpha: float = 0.05
    nulls: Dict[str, List[float]] = field(default_factory=dict)
    natural_directions: List[List[float]] = field(default_factory=list)
    direction_match_floor: float = 0.0
    diffuse_dim_thresholds: List[float] = field(default_factory=list)
    natural_match_stats: List[float] = field(default_factory=list)
    n_ref: int = 0
    n_op: int = 0
    scenario_descriptions: List[str] = field(default_factory=list)
    measured: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def digest(self) -> str:
        payload = json.dumps({
            "alpha": self.alpha, "nulls": self.nulls,
            "natural_directions": self.natural_directions,
            "direction_match_floor": self.direction_match_floor,
            "scenarios": self.scenario_descriptions, "seed": self.seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({**asdict(self), "calibration_digest": self.digest()}, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "NaturalDriftCalibration":
        with open(path) as f:
            d = json.load(f)
        expected = d.pop("calibration_digest", None)
        if not expected:
            raise ValueError("natural-drift calibration digest missing")
        cal = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cal.digest() != expected or not cal.nulls or not cal.scenario_descriptions:
            raise ValueError("natural-drift calibration invalid or tampered")
        return cal


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #

def _p_high(stat: float, null: Sequence[float]) -> float:
    """Empirical p-value for 'stat is unusually large' against a null sample."""
    n = len(null)
    if n == 0:
        return 1.0
    return float((1 + sum(1 for v in null if v >= stat)) / (n + 1))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def grid_features(images: np.ndarray, grid: int = 4) -> np.ndarray:
    """(N, grid*grid, 6) per-cell per-channel mean/std features."""
    n, h, w, c = images.shape
    gh, gw = h // grid, w // grid
    imgs = images[:, : gh * grid, : gw * grid, :]
    cells = imgs.reshape(n, grid, gh, grid, gw, c).transpose(0, 1, 3, 2, 4, 5)
    cells = cells.reshape(n, grid * grid, gh * gw, c)
    means = cells.mean(axis=2)
    stds = cells.std(axis=2)
    return np.concatenate([means, stds], axis=2)  # (N, grid*grid, 2C)


def _energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two point clouds (subsampled for speed)."""
    X = X[:200]
    Y = Y[:200]
    dxy = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).mean()
    dxx = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)).mean()
    dyy = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).mean()
    return float(2 * dxy - dxx - dyy)


def _mmd2_biased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dxx = XX + XX.T - 2 * X @ X.T
    dyy = YY + YY.T - 2 * Y @ Y.T
    dxy = XX + YY.T - 2 * X @ Y.T
    kxx = np.exp(-dxx / (2 * sigma ** 2)).mean()
    kyy = np.exp(-dyy / (2 * sigma ** 2)).mean()
    kxy = np.exp(-dxy / (2 * sigma ** 2)).mean()
    return float(kxx + kyy - 2 * kxy)


def _median_sigma(X: np.ndarray, Y: np.ndarray) -> float:
    comb = np.vstack([X[:100], Y[:100]])
    d = np.sqrt(((comb[:, None, :] - comb[None, :, :]) ** 2).sum(-1))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


# --------------------------------------------------------------------------- #
# shape analysis: localised + repeatable vs diffuse
# --------------------------------------------------------------------------- #

class ShiftShapeAnalyzer:
    """Localised-vs-diffuse shape tests, calibrated on a natural battery."""

    def __init__(self, calibration: NaturalDriftCalibration):
        self.cal = calibration

    def _outliers(self, op: np.ndarray, maha: np.ndarray,
                  q_frac: float = 0.05) -> np.ndarray:
        q = max(10, int(np.ceil(q_frac * len(op))))
        q = min(q, len(op))
        return np.argsort(maha)[::-1][:q]

    def mean_delta(self, ref: np.ndarray, op: np.ndarray,
                   maha: np.ndarray) -> np.ndarray:
        idx = self._outliers(op, maha)
        return op[idx].mean(axis=0) - ref.mean(axis=0)

    def dimension_concentration(self, ks_stats: np.ndarray) -> Vote:
        """Share of total KS shift energy carried by the top-10% dimensions."""
        ks = np.asarray(ks_stats, float)
        ks = ks[np.isfinite(ks)]
        if ks.size < 10 or ks.sum() <= 0:
            return Vote("dimension_concentration", "manipulation", False,
                        not_run="degenerate KS statistics")
        k = max(1, int(np.ceil(0.1 * ks.size)))
        stat = float(np.sort(ks)[::-1][:k].sum() / ks.sum())
        p = _p_high(stat, self.cal.nulls.get("dimension_concentration", []))
        return Vote("dimension_concentration", "manipulation", p < self.cal.alpha,
                    stat=stat, p_value=p,
                    detail=f"top-{k}/{ks.size} dims carry {stat:.2f} of shift energy")

    def direction_coherence(self, ref: np.ndarray, op: np.ndarray,
                            maha: np.ndarray) -> Vote:
        """Mean pairwise cosine of outlier deltas: a stamped trigger repeats
        the same direction sample after sample; environmental change does not."""
        if len(op) < 40:
            return Vote("direction_coherence", "manipulation", False,
                        not_run=f"only {len(op)} operational samples (<40)")
        idx = self._outliers(op, maha)
        deltas = op[idx] - ref.mean(axis=0)
        U = np.stack([_unit(d) for d in deltas])
        G = U @ U.T
        iu = np.triu_indices(len(U), k=1)
        stat = float(np.mean(G[iu]))
        p = _p_high(stat, self.cal.nulls.get("direction_coherence", []))
        return Vote("direction_coherence", "manipulation", p < self.cal.alpha,
                    stat=stat, p_value=p,
                    detail=f"outlier group n={len(U)}, mean pairwise cos {stat:.2f}")

    def spatial_locality(self, ref_images: Optional[np.ndarray],
                         op_images: Optional[np.ndarray], grid: int = 4) -> Vote:
        """Max/mean ratio of per-cell two-sample energy distance over a grid.
        A stamped patch concentrates in a few cells; illumination/blur/noise
        act on all cells roughly equally."""
        if ref_images is None or op_images is None:
            return Vote("spatial_locality", "manipulation", False,
                        not_run="images not provided (features-only run)")
        ref_g = grid_features(ref_images, grid)
        op_g = grid_features(op_images, grid)
        per_cell = np.array([
            _energy_distance(ref_g[:, c, :], op_g[:, c, :])
            for c in range(ref_g.shape[1])
        ])
        stat = float(per_cell.max() / max(per_cell.mean(), 1e-12))
        p = _p_high(stat, self.cal.nulls.get("spatial_locality", []))
        return Vote("spatial_locality", "manipulation", p < self.cal.alpha,
                    stat=stat, p_value=p,
                    detail=f"grid {grid}x{grid}, max/mean cell energy ratio {stat:.2f}")

    def confidence_polarization(self, ref_logits: Optional[np.ndarray],
                                op_logits: Optional[np.ndarray],
                                maha: np.ndarray) -> Vote:
        """A trigger forces over-confidence on the anomalous cluster; natural
        shift broadly degrades confidence (v3 architecture doc 5.4)."""
        if ref_logits is None or op_logits is None:
            return Vote("confidence_polarization", "manipulation", False,
                        not_run="logits not provided")

        def msp(logits: np.ndarray) -> np.ndarray:
            z = logits - logits.max(axis=-1, keepdims=True)
            e = np.exp(z)
            return (e / e.sum(axis=-1, keepdims=True)).max(axis=-1)

        idx = self._outliers(op_logits, maha)
        stat = float(msp(op_logits[idx]).mean() - msp(ref_logits).mean())
        p = _p_high(stat, self.cal.nulls.get("confidence_polarization", []))
        return Vote("confidence_polarization", "manipulation", p < self.cal.alpha,
                    stat=stat, p_value=p,
                    detail=f"outlier-cluster MSP lift {stat:+.3f}")

    def natural_direction_match(self, ref: np.ndarray, op: np.ndarray,
                                maha: np.ndarray) -> Vote:
        """Does the shift direction match a *measured* natural-drift direction
        from the calibration battery? This is the gate on the natural verdict:
        diffuse but off-natural-axis shifts must NOT be called natural."""
        dirs = np.asarray(self.cal.natural_directions, float)
        if dirs.size == 0:
            return Vote("natural_direction_match", "natural", False,
                        not_run="calibration carries no natural directions")
        d = _unit(self.mean_delta(ref, op, maha))
        cos = np.abs(dirs @ d)
        stat = float(cos.max())
        fired = stat >= self.cal.direction_match_floor
        return Vote("natural_direction_match", "natural", fired, stat=stat,
                    detail=f"max |cos| to battery directions {stat:.2f} "
                           f"(floor {self.cal.direction_match_floor:.2f})")

    def diffuse_dims(self, ks_stats: np.ndarray) -> Vote:
        """Fraction of dimensions shifted beyond their per-dimension natural
        threshold (q95 of that dimension over the calibration battery). A
        diffuse shift moves many dimensions; a localised one moves few.
        Supporting natural-side evidence (never gates alone)."""
        ks = np.asarray(ks_stats, float)
        ks = ks[np.isfinite(ks)]
        if ks.size < 10:
            return Vote("diffuse_dims", "natural", False,
                        not_run="degenerate KS statistics")
        thr = np.asarray(self.cal.diffuse_dim_thresholds, float)
        if thr.size != ks.size:
            return Vote("diffuse_dims", "natural", False,
                        not_run="calibration lacks per-dim KS thresholds")
        stat = float(np.mean(ks > thr))
        null = self.cal.nulls.get("diffuse_dims", [])
        floor = max(float(np.quantile(null, 0.05)), 1.0 / ks.size) if null else 1.0
        fired = stat >= floor
        return Vote("diffuse_dims", "natural", fired, stat=stat,
                    detail=f"fraction of dims beyond per-dim natural q95: "
                           f"{stat:.2f} (battery floor {floor:.2f})")


# --------------------------------------------------------------------------- #
# contributor concentration
# --------------------------------------------------------------------------- #

class ContributorConcentrationAnalyzer:
    """Natural drift hits every contributor; manipulation often arrives
    through one. Concentration is supporting evidence only -- it can never
    convict on its own (one contributor may honestly own one terrain)."""

    def conformal_flags(self, ref_maha: np.ndarray, op_maha: np.ndarray,
                        alpha: float = 0.01) -> np.ndarray:
        n = len(ref_maha)
        p = (1 + np.sum(ref_maha[None, :] >= op_maha[:, None], axis=1)) / (n + 1)
        return p < alpha

    def concentration_test(self, flags: np.ndarray,
                           contributors: Optional[Sequence[str]],
                           perms: int = 2000, seed: int = 0) -> Vote:
        if contributors is None:
            return Vote("contributor_concentration", "manipulation", False,
                        not_run="contributor metadata not provided")
        contributors = np.asarray([str(c) for c in contributors], object)
        uniq, counts = np.unique(contributors, return_counts=True)
        if len(uniq) < 2:
            return Vote("contributor_concentration", "manipulation", False,
                        not_run="fewer than 2 contributors present")
        if counts.min() < 10:
            return Vote("contributor_concentration", "manipulation", False,
                        not_run=f"low power: smallest contributor has {counts.min()} samples")
        rates = np.array([flags[contributors == c].mean() for c in uniq])
        stat = float(rates.max())
        rng = np.random.default_rng(seed)
        null = np.empty(perms)
        for b in range(perms):
            sh = rng.permutation(flags)
            null[b] = max(sh[contributors == c].mean() for c in uniq)
        p = float((1 + np.sum(null >= stat)) / (perms + 1))
        top = str(uniq[int(np.argmax(rates))])
        return Vote("contributor_concentration", "manipulation", p < 0.05,
                    stat=stat, p_value=p,
                    detail=f"max outlier share {stat:.2f} from '{top}'")

    def cross_contributor(self, concentration: Vote,
                          contributors: Optional[Sequence[str]]) -> Vote:
        """Weak natural-side vote: outlier membership is homogeneous across
        contributors. Absence of concentration is weak evidence, so this vote
        can support but never gate a verdict."""
        if concentration.not_run or concentration.p_value is None:
            return Vote("cross_contributor", "natural", False,
                        not_run=concentration.not_run or "no concentration test")
        fired = concentration.p_value > 0.5
        return Vote("cross_contributor", "natural", fired,
                    stat=concentration.stat, p_value=concentration.p_value,
                    detail="outlier membership homogeneous across contributors"
                           if fired else "outlier membership concentrated")

    def outlier_posteriors(self, flags: np.ndarray,
                           contributors: Optional[Sequence[str]],
                           tau: float = 0.05, draws: int = 4000,
                           seed: int = 0) -> Dict[str, Any]:
        """Hierarchical Beta-Binomial posterior P(theta_c > f_pop + tau),
        same form as the data-integrity axis. Evidence only."""
        if contributors is None:
            return {"note": "contributor metadata not provided"}
        contributors = np.asarray([str(c) for c in contributors], object)
        uniq = np.unique(contributors)
        ks = np.array([flags[contributors == c].sum() for c in uniq], float)
        ns = np.array([np.sum(contributors == c) for c in uniq], float)
        rates = ks / np.maximum(ns, 1)
        f_pop = float(flags.mean())
        var = float(rates.var()) if len(rates) > 1 else 0.0
        kappa = max(2.0, f_pop * (1 - f_pop) / max(var, 1e-6) - 1.0)
        kappa = min(kappa, 500.0)
        rng = np.random.default_rng(seed)
        f_draws = rng.beta(1 + ks.sum(), 1 + max(ns.sum() - ks.sum(), 1), draws)
        out = {}
        for c, k_c, n_c in zip(uniq, ks, ns):
            a = kappa * f_draws + k_c
            b = kappa * (1 - f_draws) + (n_c - k_c)
            theta = rng.beta(np.maximum(a, 1e-3), np.maximum(b, 1e-3))
            out[str(c)] = {
                "n": int(n_c), "flagged": int(k_c),
                "rate": float(k_c / max(n_c, 1)),
                "P_exceeds_pop_plus_tau": float(np.mean(theta > f_pop + tau)),
            }
        return {"population_flag_rate": f_pop, "tau": tau, "kappa": kappa,
                "per_contributor": out}

    def leave_one_out_mmd(self, ref: np.ndarray, op: np.ndarray,
                          contributors: Optional[Sequence[str]],
                          max_n: int = 200, perms: int = 100,
                          seed: int = 0) -> Dict[str, Any]:
        """MMD p-value after dropping each contributor: if removing one
        contributor collapses the shift, the shift is concentrated."""
        if contributors is None:
            return {"note": "contributor metadata not provided"}
        contributors = np.asarray([str(c) for c in contributors], object)
        rng = np.random.default_rng(seed)
        ref_s = ref[:max_n]
        sigma = _median_sigma(ref_s, op[:max_n])

        def mmd_p(mask: np.ndarray) -> float:
            X = ref_s
            Y = op[mask][:max_n]
            if len(Y) < 10:
                return 1.0
            obs = _mmd2_biased(X, Y, sigma)
            cnt = 0
            comb = np.vstack([X, Y])
            n = len(X)
            for _ in range(perms):
                pm = rng.permutation(len(comb))
                if _mmd2_biased(comb[pm[:n]], comb[pm[n:]], sigma) >= obs:
                    cnt += 1
            return float((1 + cnt) / (perms + 1))

        full_p = mmd_p(np.ones(len(op), bool))
        per_c = {}
        for c in np.unique(contributors):
            if np.sum(contributors == c) < 10:
                continue
            per_c[str(c)] = mmd_p(contributors != c)
        return {"full_mmd_p": full_p, "without_contributor": per_c}


# --------------------------------------------------------------------------- #
# arbiter
# --------------------------------------------------------------------------- #

class AttributionArbiter:
    """Turn calibrated votes into a verdict. Never asserts by default."""

    def decide(self, votes: List[Vote]) -> Dict[str, Any]:
        by_name = {v.name: v for v in votes}
        shape_fired = [v for v in votes
                       if v.name in SHAPE_VOTES and v.fired and not v.not_run]
        contrib = by_name.get("contributor_concentration")
        contrib_fired = bool(contrib and contrib.fired and not contrib.not_run)
        ndm = by_name.get("natural_direction_match")
        ndm_ran = bool(ndm and not ndm.not_run)
        ndm_matched = bool(ndm and ndm.fired)
        diffuse = by_name.get("diffuse_dims")
        diffuse_fired = bool(diffuse and diffuse.fired)

        reasoning: List[str] = []
        for v in votes:
            if v.not_run:
                reasoning.append(f"[not run] {v.name}: {v.not_run}")
            else:
                reasoning.append(
                    f"[{'fired' if v.fired else 'quiet'}] {v.name}: {v.detail}"
                    + (f" (p={v.p_value:.4f})" if v.p_value is not None else ""))

        # resolution suggestions from skipped votes
        suggestions = []
        for v in votes:
            if v.not_run == "images not provided (features-only run)":
                suggestions.append("provide operational images to enable the "
                                   "spatial-locality test")
            if v.not_run == "logits not provided":
                suggestions.append("provide model logits to enable the "
                                   "confidence-polarization test")
            if v.not_run == "contributor metadata not provided":
                suggestions.append("provide contributor/batch metadata to enable "
                                   "source-concentration analysis")

        if len(shape_fired) >= 2:
            verdict = VERDICT_SUSPICIOUS
            reasoning.append(
                f"{len(shape_fired)} localised/repeatable shape tests fired "
                f"({', '.join(v.name for v in shape_fired)}): shape consistent "
                f"with deliberate, localised modification."
                + (" Note: the shift direction also matches a natural-drift "
                   "direction; the localised shape evidence overrides the "
                   "direction match." if ndm_matched else
                   " The shift direction matches no measured natural-drift "
                   "direction."))
        elif ndm_matched and not shape_fired and not contrib_fired:
            verdict = VERDICT_NATURAL
            reasoning.append(
                "Shift direction matches a measured natural-drift direction and no "
                "localised/concentrated signal fired: shape consistent with "
                "operational drift.")
        else:
            verdict = VERDICT_UNDERDETERMINED
            if contrib_fired and not shape_fired:
                top = contrib.detail.split("'")[1] if "'" in contrib.detail else "unknown"
                reasoning.append(
                    f"The shift concentrates in contributor '{top}', but contributor "
                    f"concentration alone cannot separate manipulation from an honest "
                    f"single-source domain change; see the data-integrity findings "
                    f"for '{top}'.")
            elif shape_fired and ndm_matched:
                reasoning.append(
                    "Localised-shape evidence and natural-direction evidence conflict; "
                    "the available evidence does not distinguish drift from manipulation.")
            elif not shape_fired and not ndm_matched:
                reasoning.append(
                    "No attribution test reached its calibrated threshold in either "
                    "direction; the evidence does not distinguish drift from manipulation.")
            else:
                reasoning.append(
                    "Attribution evidence is mixed; the available evidence does not "
                    "distinguish drift from manipulation.")

        return {
            "natural_vs_adversarial": verdict,
            "votes": [v.to_dict() for v in votes],
            "reasoning": reasoning,
            "resolution_suggestions": suggestions,
            "n_shape_votes_fired": len(shape_fired),
            "contributor_concentration_fired": contrib_fired,
            "natural_direction_matched": ndm_matched if ndm_ran else None,
            "diffuse_dims_fired": diffuse_fired,
        }


# --------------------------------------------------------------------------- #
# shared runner (used by DistributionShiftAssessor.assess and driftbench)
# --------------------------------------------------------------------------- #

def run_attribution(
    reference_features: np.ndarray,
    operational_features: np.ndarray,
    maha_ref: np.ndarray,
    maha_op: np.ndarray,
    ks_stats: np.ndarray,
    calibration: NaturalDriftCalibration,
    reference_images: Optional[np.ndarray] = None,
    operational_images: Optional[np.ndarray] = None,
    operational_contributors: Optional[Sequence[str]] = None,
    reference_logits: Optional[np.ndarray] = None,
    operational_logits: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute all attribution votes and arbitrate. Shared by the production
    assessor and the driftbench measurement harness so that measured battery
    numbers describe the code that actually runs."""
    shape = ShiftShapeAnalyzer(calibration)
    contrib = ContributorConcentrationAnalyzer()

    # Size-normalise: KS/statistics are sample-size dependent, so the shape
    # votes are computed on subsamples matching the calibration battery sizes
    # (averaged over a few draws). Contributor analysis stays full-size --
    # conformal flags and the permutation test are per-sample valid.
    n_ref = calibration.n_ref or len(reference_features)
    n_op = calibration.n_op or len(operational_features)
    rng = np.random.default_rng(seed)
    repeats = 5
    sub = []
    for _ in range(repeats):
        ri = rng.choice(len(reference_features),
                        size=min(n_ref, len(reference_features)), replace=False)
        oi = rng.choice(len(operational_features),
                        size=min(n_op, len(operational_features)), replace=False)
        rf_s = reference_features[ri]
        of_s = operational_features[oi]
        mo_s = maha_op[oi]
        rimgs = reference_images[ri] if reference_images is not None else None
        oimgs = operational_images[oi] if operational_images is not None else None
        try:
            from scipy import stats as _sps
            ks_s = np.array([_sps.ks_2samp(rf_s[:, d], of_s[:, d])[0]
                             for d in range(rf_s.shape[1])])
        except ImportError:
            ks_s = ks_stats
        sub.append((rf_s, of_s, mo_s, ks_s, rimgs, oimgs))

    def _averaged(vote_fn):
        vs = [vote_fn(*args) for args in sub]
        ran = [v for v in vs if not v.not_run]
        if not ran:
            return vs[0]
        stat = float(np.mean([v.stat for v in ran if v.stat is not None]))
        out = ran[0]
        out.stat = stat
        if out.p_value is not None:
            null_key = out.name
            out.p_value = _p_high(stat, calibration.nulls.get(null_key, []))                 if null_key in calibration.nulls else out.p_value
        return out

    votes = [
        _averaged(lambda rf, of, mo, ks, ri, oi: shape.dimension_concentration(ks)),
        _averaged(lambda rf, of, mo, ks, ri, oi: shape.direction_coherence(rf, of, mo)),
        _averaged(lambda rf, of, mo, ks, ri, oi: shape.spatial_locality(ri, oi)),
        shape.confidence_polarization(reference_logits, operational_logits, maha_op),
        _averaged(lambda rf, of, mo, ks, ri, oi: shape.natural_direction_match(rf, of, mo)),
        _averaged(lambda rf, of, mo, ks, ri, oi: shape.diffuse_dims(ks)),
    ]
    # natural_direction_match and diffuse_dims thresholds are absolute, not
    # null-p: recompute fired from the averaged stat.
    ndm = votes[4]
    if not ndm.not_run:
        ndm.fired = ndm.stat >= calibration.direction_match_floor
        ndm.detail = (f"max |cos| to battery directions {ndm.stat:.2f} "
                      f"(floor {calibration.direction_match_floor:.2f})")
    dd = votes[5]
    if not dd.not_run:
        thr = np.asarray(calibration.diffuse_dim_thresholds, float)
        floor = max(float(np.quantile(calibration.nulls.get("diffuse_dims", [1.0]), 0.05)),
                    1.0 / thr.size) if thr.size else 1.0
        dd.fired = dd.stat >= floor

    flags = contrib.conformal_flags(maha_ref, maha_op)
    conc_vote = contrib.concentration_test(flags, operational_contributors, seed=seed)
    votes += [conc_vote, contrib.cross_contributor(conc_vote, operational_contributors)]
    verdict = AttributionArbiter().decide(votes)
    verdict["contributor_evidence"] = {
        "outlier_posteriors": contrib.outlier_posteriors(
            flags, operational_contributors, seed=seed),
        "n_conformal_outliers": int(flags.sum()),
    }
    return verdict
