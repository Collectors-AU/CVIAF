"""driftbench: natural-drift calibration + attribution measurement harness.

Builds the NaturalDriftCalibration consumed by the distribution-shift module
and measures the attribution arbiter on three batteries:

  - natural battery (held-out terrain/season/sensor/illumination scenarios):
    the probable_natural_drift verdict rate and the suspicious_manipulation
    false-positive rate at the calibrated alpha;
  - manipulation battery (trigger recipes at graded rates, spread and
    contributor-concentrated): the suspicious_manipulation true-positive rate;
  - diffuse-confuser battery (artificial, diffuse, off-natural-axis shifts):
    these must route to under_determined, not to probable_natural_drift.

Everything is seeded and offline. The calibration is written as JSON; the
pipeline loads it from <corpus>/drift_calibration.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from cviaf.drift import MahalanobisDetector
from cviaf.drift.attribution import (
    AttributionArbiter,
    ContributorConcentrationAnalyzer,
    NaturalDriftCalibration,
    ShiftShapeAnalyzer,
    run_attribution,
    VERDICT_NATURAL,
    VERDICT_SUSPICIOUS,
    VERDICT_UNDERDETERMINED,
)
from cviaf.lab.poison import AttackSpec, apply_trigger, inject
from cviaf.lab.synth import DetectionDataset, SceneSpec, build_dataset
from cviaf.utils import extract_features_from_images

BASE = SceneSpec()  # the declared reference scene: desert / summer / defaults


def _features(ds: DetectionDataset) -> np.ndarray:
    return extract_features_from_images(ds.images, method="pixel_stats", target_dim=64)


def _ks_stats(ref: np.ndarray, op: np.ndarray) -> np.ndarray:
    from scipy import stats as sps
    return np.array([sps.ks_2samp(ref[:, d], op[:, d])[0]
                     for d in range(ref.shape[1])])


def _scenario_inputs(ref_ds: DetectionDataset, op_ds: DetectionDataset):
    ref_f = _features(ref_ds)
    op_f = _features(op_ds)
    det = MahalanobisDetector()
    det.fit(ref_f)
    return ref_f, op_f, det.score(ref_f), det.score(op_f), _ks_stats(ref_f, op_f)


def _concentration_stat(ks: np.ndarray) -> float:
    k = max(1, int(np.ceil(0.1 * ks.size)))
    return float(np.sort(ks)[::-1][:k].sum() / max(ks.sum(), 1e-12))


def _coherence_stat(ref_f, op_f, maha_op) -> float:
    q = max(10, int(np.ceil(0.05 * len(op_f))))
    idx = np.argsort(maha_op)[::-1][:q]
    deltas = op_f[idx] - ref_f.mean(axis=0)
    U = deltas / np.maximum(np.linalg.norm(deltas, axis=1, keepdims=True), 1e-12)
    G = U @ U.T
    iu = np.triu_indices(len(U), k=1)
    return float(np.mean(G[iu]))


def _spatial_stat(ref_ds, op_ds, grid=4) -> float:
    from cviaf.drift.attribution import grid_features, _energy_distance
    rg = grid_features(ref_ds.images, grid)
    og = grid_features(op_ds.images, grid)
    per_cell = np.array([_energy_distance(rg[:, c, :], og[:, c, :])
                         for c in range(rg.shape[1])])
    return float(per_cell.max() / max(per_cell.mean(), 1e-12))


def _direction(ref_f, op_f, maha_op) -> np.ndarray:
    q = max(10, int(np.ceil(0.05 * len(op_f))))
    idx = np.argsort(maha_op)[::-1][:q]
    d = op_f[idx].mean(axis=0) - ref_f.mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


# --------------------------------------------------------------------------- #
# scenario batteries
# --------------------------------------------------------------------------- #

def calibration_scenarios() -> List[Tuple[str, SceneSpec]]:
    """Natural shifts used to estimate nulls (and natural directions)."""
    out: List[Tuple[str, SceneSpec]] = []
    for terrain in ("forest", "urban", "snow", "night"):
        out.append((f"terrain:{terrain}", replace(BASE, terrain=terrain)))
    for season in ("winter", "monsoon", "autumn"):
        out.append((f"season:{season}", replace(BASE, season=season)))
    for illum in (0.6, 0.75, 0.9, 1.15, 1.3, 1.5):
        out.append((f"illumination:{illum}", replace(BASE, illumination=illum)))
    for noise in (0.05, 0.08, 0.12):
        out.append((f"sensor_noise:{noise}", replace(BASE, sensor_noise=noise)))
    for blur in (1, 2, 3):
        out.append((f"sensor_blur:{blur}", replace(BASE, sensor_blur=blur)))
    for gamma in (0.7, 0.85, 1.2, 1.4):
        out.append((f"gamma:{gamma}", replace(BASE, gamma=gamma)))
    # two-axis combinations, still natural
    out.append(("terrain:forest+season:winter",
                replace(BASE, terrain="forest", season="winter")))
    out.append(("terrain:snow+illumination:1.3",
                replace(BASE, terrain="snow", illumination=1.3)))
    out.append(("season:monsoon+sensor_noise:0.06",
                replace(BASE, season="monsoon", sensor_noise=0.06)))
    out.append(("terrain:urban+sensor_blur:1",
                replace(BASE, terrain="urban", sensor_blur=1)))
    out.append(("terrain:night+sensor_noise:0.05",
                replace(BASE, terrain="night", sensor_noise=0.05)))
    out.append(("season:autumn+gamma:0.85",
                replace(BASE, season="autumn", gamma=0.85)))
    out.append(("terrain:desert+season:summer+illumination:0.95",
                replace(BASE, illumination=0.95)))
    return out


def resample_scenarios(k: int = 6) -> List[Tuple[str, SceneSpec, int]]:
    """No-shift replicates: the reference spec redrawn at fresh seeds. These
    capture pure sampling noise at the operational sizes, which the
    spec-varying scenarios (all sharing one seed offset) do not."""
    return [(f"resample:{i}", BASE, 910_000 + i * 10_000) for i in range(k)]


def heldout_natural_scenarios() -> List[Tuple[str, SceneSpec]]:
    """Natural shifts NOT in the calibration set (held-out axis values)."""
    return [
        ("illumination:0.85", replace(BASE, illumination=0.85)),
        ("illumination:1.1", replace(BASE, illumination=1.1)),
        ("sensor_noise:0.065", replace(BASE, sensor_noise=0.065)),
        ("sensor_blur:1+season:winter", replace(BASE, sensor_blur=1, season="winter")),
        ("gamma:0.9", replace(BASE, gamma=0.9)),
        ("gamma:1.15", replace(BASE, gamma=1.15)),
        ("terrain:forest+illumination:0.8",
         replace(BASE, terrain="forest", illumination=0.8)),
        ("terrain:urban+season:autumn",
         replace(BASE, terrain="urban", season="autumn")),
        ("terrain:snow+season:winter",
         replace(BASE, terrain="snow", season="winter")),
        ("season:monsoon+illumination:0.9",
         replace(BASE, season="monsoon", illumination=0.9)),
        ("terrain:night+illumination:1.2",
         replace(BASE, terrain="night", illumination=1.2)),
        ("sensor_noise:0.1+sensor_blur:1",
         replace(BASE, sensor_noise=0.1, sensor_blur=1)),
    ]


def manipulation_battery(seed: int, n_ref: int = 120, n_op: int = 240) -> List[Tuple[str, DetectionDataset, DetectionDataset]]:
    """(name, reference, manipulated) — trigger-based, graded rates."""
    out = []
    for i, rate in enumerate((0.10, 0.20, 0.40)):
        ref = build_dataset(n_ref, BASE, seed_offset=700_000 + i * 10_000)
        clean = build_dataset(n_op, BASE, seed_offset=700_000 + i * 10_000)
        spec = AttackSpec(kind="oda", trigger="patch", trigger_loc="on_object",
                          trigger_size=10, rate=rate, seed=seed + i)
        attacked, _ = inject(clean, spec)
        out.append((f"oda_on_object_rate:{rate}", ref, attacked))
    # fixed-position patch (oga recipe family), spread across contributors
    ref = build_dataset(n_ref, BASE, seed_offset=740_000)
    clean = build_dataset(n_op, BASE, seed_offset=740_000)
    spec = AttackSpec(kind="oga", trigger="patch", trigger_loc="fixed",
                      trigger_size=8, rate=0.20, seed=seed + 9)
    attacked, _ = inject(clean, spec)
    out.append(("oga_fixed_rate:0.20", ref, attacked))
    # contributor-concentrated trigger: only vendor_x samples stamped
    ref = build_dataset(n_ref, BASE, seed_offset=750_000)
    conc = build_dataset(n_op, BASE, seed_offset=750_000)
    rng = np.random.default_rng(seed + 77)
    imgs = conc.images.copy()
    spec = AttackSpec(kind="oda", trigger="patch", trigger_loc="on_object",
                      trigger_size=10, rate=0.0, seed=seed + 77)
    n_stamped = 0
    for i in range(len(conc)):
        if str(conc.contributors[i]) == "vendor_x" and rng.random() < 0.6:
            victim_box = conc.boxes[i][0] if len(conc.boxes[i]) else None
            imgs[i], _ = apply_trigger(imgs[i], spec, rng, victim_box)
            n_stamped += 1
    concentrated = DetectionDataset(imgs, [b.copy() for b in conc.boxes],
                                    [l.copy() for l in conc.labels],
                                    conc.contributors.copy(), conc.batches.copy(),
                                    dict(conc.spec))
    out.append((f"concentrated_vendor_x_rate:0.6 ({n_stamped} stamped)",
                ref, concentrated))
    return out


def confuser_battery(seed: int, n_ref: int = 120, n_op: int = 240) -> List[Tuple[str, DetectionDataset, DetectionDataset]]:
    """Diffuse, artificial, off-natural-axis shifts. These are neither natural
    drift nor localised manipulation; the honest verdict is under_determined."""
    out = []
    transforms = [
        ("channel_swap", lambda x: x[..., ::-1].copy()),
        ("channel_scale_1.35_0.75_1.15",
         lambda x: np.clip(x * np.array([1.35, 0.75, 1.15], np.float32), 0, 1)),
        ("saturation_boost",
         lambda x: np.clip(x.mean(-1, keepdims=True)
                           + 1.8 * (x - x.mean(-1, keepdims=True)), 0, 1)),
        ("green_invert",
         lambda x: np.stack([x[..., 0], 1.0 - x[..., 1], x[..., 2]], -1)),
    ]
    for j, (name, fn) in enumerate(transforms):
        ref = build_dataset(n_ref, BASE, seed_offset=760_000 + j * 10_000)
        base = build_dataset(n_op, BASE, seed_offset=760_000 + j * 10_000)
        shifted = DetectionDataset(fn(base.images).astype(np.float32),
                                   [b.copy() for b in base.boxes],
                                   [l.copy() for l in base.labels],
                                   base.contributors.copy(), base.batches.copy(),
                                   dict(base.spec))
        out.append((f"confuser:{name}", ref, shifted))
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run_driftbench(out_path: str, alpha: float = 0.05, seed: int = 7,
                   n_ref: int = 120, n_op: int = 100,
                   log: Callable[[str], None] = print) -> Dict[str, Any]:
    log("driftbench: building reference and natural calibration battery")
    ref_ds = build_dataset(n_ref, BASE, seed_offset=900_000)
    ref_f = _features(ref_ds)
    det = MahalanobisDetector()
    det.fit(ref_f)
    maha_ref = det.score(ref_f)

    scenarios = calibration_scenarios()
    nulls: Dict[str, List[float]] = {
        "dimension_concentration": [], "direction_coherence": [],
        "spatial_locality": [], "diffuse_dims": [], "diffuse_dims_per_dim": [],
    }
    directions: List[List[float]] = []
    per_dim_ks: List[np.ndarray] = []
    def _collect(desc: str, op_ds: DetectionDataset) -> None:
        op_f = _features(op_ds)
        maha_op = det.score(op_f)
        ks = _ks_stats(ref_f, op_f)
        per_dim_ks.append(ks)
        nulls["dimension_concentration"].append(_concentration_stat(ks))
        nulls["direction_coherence"].append(_coherence_stat(ref_f, op_f, maha_op))
        nulls["spatial_locality"].append(_spatial_stat(ref_ds, op_ds))
        directions.append(_direction(ref_f, op_f, maha_op).tolist())
        scenario_descs.append(desc)

    scenario_descs: List[str] = []
    for desc, spec in scenarios:
        _collect(desc, build_dataset(n_op, spec, seed_offset=900_000))
    for desc, spec, off in resample_scenarios():
        _collect(desc, build_dataset(n_op, spec, seed_offset=off))
    # per-dimension natural thresholds (q95 per dim over the battery), then
    # the per-scenario diffuse fraction against them
    thr_vec = np.quantile(np.asarray(per_dim_ks), 0.95, axis=0)
    for ks in per_dim_ks:
        nulls["diffuse_dims"].append(float(np.mean(ks > thr_vec)))
    del nulls["diffuse_dims_per_dim"]

    # Direction-match operating floor: estimated on HELD-OUT natural scenarios
    # (axis values the calibration directions did not see), minus a small
    # margin. Natural data only -- no manipulation data tunes this floor.
    D = np.asarray(directions)
    heldout = heldout_natural_scenarios()
    match_stats = []
    for desc, spec in heldout:
        op_ds = build_dataset(n_op, spec, seed_offset=950_000)
        op_f = _features(op_ds)
        d = _direction(ref_f, op_f, det.score(op_f))
        match_stats.append(float(np.abs(D @ d).max()))
    floor = float(min(match_stats) - 0.02)

    cal = NaturalDriftCalibration(
        alpha=alpha, nulls=nulls, natural_directions=directions,
        direction_match_floor=floor, diffuse_dim_thresholds=thr_vec.tolist(),
        natural_match_stats=match_stats,
        scenario_descriptions=scenario_descs, seed=seed,
        n_ref=n_ref, n_op=n_op)
    log(f"  calibration: {len(scenario_descs)} natural scenarios "
        f"(incl. resamples, n_ref={n_ref}, n_op={n_op}), "
        f"direction-match floor {floor:.3f} "
        f"(held-out natural match min {min(match_stats):.3f})")

    def verdict_for(ref_d, op_d) -> Dict[str, Any]:
        rf, of, mr, mo, ks = _scenario_inputs(ref_d, op_d)
        return run_attribution(
            reference_features=rf, operational_features=of,
            maha_ref=mr, maha_op=mo, ks_stats=ks, calibration=cal,
            reference_images=ref_d.images, operational_images=op_d.images,
            operational_contributors=list(op_d.contributors),
            seed=seed)

    # --- held-out natural battery
    log("driftbench: held-out natural battery")
    nat_verdicts = {}
    for desc, spec in heldout:
        op_ds = build_dataset(n_op, spec, seed_offset=950_000)
        v = verdict_for(ref_ds, op_ds)["natural_vs_adversarial"]
        nat_verdicts[desc] = v
        log(f"  {desc:45s} -> {v}")
    n_nat = len(nat_verdicts)
    fpr = sum(1 for v in nat_verdicts.values() if v == VERDICT_SUSPICIOUS) / n_nat
    natural_rate = sum(1 for v in nat_verdicts.values() if v == VERDICT_NATURAL) / n_nat

    # --- manipulation battery
    log("driftbench: manipulation battery")
    man_verdicts = {}
    for name, r, o in manipulation_battery(seed, n_ref, n_op):
        v = verdict_for(r, o)["natural_vs_adversarial"]
        man_verdicts[name] = v
        log(f"  {name:45s} -> {v}")
    n_man = len(man_verdicts)
    tpr = sum(1 for v in man_verdicts.values() if v == VERDICT_SUSPICIOUS) / n_man
    man_natural = sum(1 for v in man_verdicts.values() if v == VERDICT_NATURAL) / n_man

    # --- diffuse confuser battery
    log("driftbench: diffuse-confuser battery")
    con_verdicts = {}
    for name, r, o in confuser_battery(seed, n_ref, n_op):
        v = verdict_for(r, o)["natural_vs_adversarial"]
        con_verdicts[name] = v
        log(f"  {name:45s} -> {v}")
    n_con = len(con_verdicts)
    con_under = sum(1 for v in con_verdicts.values()
                    if v == VERDICT_UNDERDETERMINED) / n_con
    con_natural = sum(1 for v in con_verdicts.values()
                      if v == VERDICT_NATURAL) / n_con

    cal.measured = {
        VERDICT_SUSPICIOUS: {
            "reliability": round(tpr, 3),
            "basis": f"manipulation battery TPR, n={n_man}",
        },
        VERDICT_NATURAL: {
            "reliability": round(natural_rate, 3),
            "basis": f"held-out natural battery verdict rate, n={n_nat}; "
                     f"suspicious FPR {fpr:.3f} at alpha={alpha}",
        },
        VERDICT_UNDERDETERMINED: {
            "reliability": round(con_under, 3),
            "basis": f"diffuse-confuser routing rate, n={n_con}",
        },
        "battery_detail": {
            "natural": nat_verdicts, "manipulation": man_verdicts,
            "confuser": con_verdicts,
            "natural_fpr_suspicious": round(fpr, 3),
            "manipulation_tpr": round(tpr, 3),
            "manipulation_mislabelled_natural": round(man_natural, 3),
            "confuser_under_determined": round(con_under, 3),
            "confuser_mislabelled_natural": round(con_natural, 3),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cal.save(out_path)
    log(f"driftbench: calibration written to {out_path} (digest {cal.digest()})")
    log(f"  natural battery : suspicious FPR {fpr:.3f} (target <= {alpha}), "
        f"natural verdict rate {natural_rate:.3f}")
    log(f"  manipulation    : suspicious TPR {tpr:.3f}, "
        f"mislabelled-natural {man_natural:.3f}")
    log(f"  confusers       : under_determined {con_under:.3f}, "
        f"mislabelled-natural {con_natural:.3f}")
    return {"calibration_digest": cal.digest(), "measured": cal.measured,
            "out_path": out_path}
