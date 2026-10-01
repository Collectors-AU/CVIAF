"""Seeded, scene-conditioned *synthetic* clean calibration battery.

This is a laboratory reference, not evidence that real imagery is clean or that
synthetic and operational scenes are exchangeable. Conditional p-values require
held-out clean observations from the same declared stratum and independent origins.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from cviaf.lab.calibrate import conformal_pvalues
from cviaf.lab.synth import SEASON_TINT, TERRAINS, SceneSpec, build_dataset

SENSORS: Mapping[str, Tuple[float, int]] = {
    "clear": (0.01, 0), "noisy": (0.10, 1), "blurred": (0.02, 2),
}


@dataclass(frozen=True)
class BatteryPlan:
    terrains: Tuple[str, ...] = ("desert", "forest", "snow")
    seasons: Tuple[str, ...] = ("summer", "winter")
    sensors: Tuple[str, ...] = ("clear", "noisy")
    n_cal: int = 100  # per stratum, not divided across strata
    n_test: int = 100
    seed: int = 17
    alpha: float = 0.05

    def validate(self) -> None:
        for name, values, allowed in (("terrain", self.terrains, TERRAINS),
                                      ("season", self.seasons, SEASON_TINT),
                                      ("sensor", self.sensors, SENSORS)):
            if not values or len(values) != len(set(values)) or any(v not in allowed for v in values):
                raise ValueError(f"{name} must be nonempty, unique, and supported: {values}")
        if self.n_cal < 1 or self.n_test < 1 or not (0 < self.alpha < 1):
            raise ValueError("n_cal and n_test must be positive; alpha must be in (0,1)")

    def keys(self) -> Tuple[Tuple[str, str, str], ...]:
        self.validate()
        return tuple(itertools.product(self.terrains, self.seasons, self.sensors))


def _seed(seed: int, key: Tuple[str, str, str], split: str) -> int:
    payload = json.dumps([seed, key, split], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _spec(base: SceneSpec, key: Tuple[str, str, str], seed: int) -> SceneSpec:
    noise, blur = SENSORS[key[2]]
    return replace(base, terrain=key[0], season=key[1],
                   sensor_noise=noise, sensor_blur=blur, seed=seed)


def synthesize(plan: BatteryPlan, base: SceneSpec | None = None) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Produce paired independent clean datasets; never calibrate against test items.

    Dedicated synthetic contributors and disjoint seed domains prevent accidental
    overlap with the ordinary training contributors. Different synthetic seeds do
    not establish contributor-level exchangeability for a real deployment.
    """
    base = base or SceneSpec()
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key in plan.keys():
        out[key] = {}
        for split, n in (("cal", plan.n_cal), ("test", plan.n_test)):
            out[key][split] = build_dataset(
                n, _spec(base, key, _seed(plan.seed, key, split)),
                contributors=(f"synthetic_{split}_{'_'.join(key)}",),
                seed_offset=0)
    return out


def coverage(plan: BatteryPlan, battery: Mapping[Tuple[str, str, str], Mapping[str, Any]],
             minimum_cal: int = 1) -> Dict[str, Any]:
    """Account for every declared cell, including absent or underfilled ones."""
    cells = {}
    for key in plan.keys():
        pair = battery.get(key, {})
        nc = len(pair["cal"]) if "cal" in pair else 0
        nt = len(pair["test"]) if "test" in pair else 0
        cells["/".join(key)] = {"cal": nc, "test": nt,
                                "covered": nc >= minimum_cal and nt > 0}
    return {"declared": len(cells), "covered": sum(v["covered"] for v in cells.values()),
            "minimum_cal": minimum_cal, "cells": cells}


def conditional_pvalues(
    key: Tuple[str, str, str], scores: np.ndarray,
    calibration_scores: Mapping[Tuple[str, str, str], np.ndarray],
    minimum_cal: int = 1,
) -> np.ndarray:
    """Calibrate a detector score only against its own declared clean stratum.

    No nearest-cell or pooled fallback: missing cells must block calibration.
    The caller must establish provenance, clean status, and exchangeability of
    calibration and assessed items. This function cannot establish those facts.
    """
    if key not in calibration_scores:
        raise ValueError(f"no clean calibration cell for {key}")
    cal = np.asarray(calibration_scores[key], dtype=np.float64).ravel()
    test = np.asarray(scores, dtype=np.float64).ravel()
    if len(cal) < minimum_cal or not np.all(np.isfinite(cal)) or not np.all(np.isfinite(test)):
        raise ValueError(f"underfilled or nonfinite clean calibration cell for {key}")
    return conformal_pvalues(cal, test)


def _scores(dataset) -> np.ndarray:
    # A deliberately simple, fixed one-sided covariate anomaly statistic; same
    # statistic on both arms. Brightness exposes the cost of pooling scene types.
    return dataset.images.mean(axis=(1, 2, 3), dtype=np.float64)


def _metrics(p: np.ndarray, alpha: float) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    # KS distance from continuous Uniform, includes finite-sample rank coarseness.
    ordered = np.sort(p)
    n = len(p)
    ks = max(float(np.max(np.arange(1, n+1) / n - ordered)),
             float(np.max(ordered - np.arange(n) / n)))
    return {"n": n, "ks_uniform": ks, "false_abstain_rate": float(np.mean(p <= alpha)),
            "mean_p": float(p.mean())}


def measure(plan: BatteryPlan, base: SceneSpec | None = None) -> Dict[str, Any]:
    """Compare scene-stratified reference to the existing one-SceneSpec holdout.

    Both arms use the same *independent* clean test scenes and the same calibration
    count per stratum. The baseline is one pooled hand-built scene, exactly the
    existing make_clean_holdout strategy; no trained-model verdict is implied.
    """
    base = base or SceneSpec()
    battery = synthesize(plan, base)
    # Existing baseline is build_splits/make_clean_holdout: one scene spec, one
    # hand-selected scene. Match n_cal to each stratified cell to avoid a size win.
    baseline = build_dataset(plan.n_cal, replace(base, seed=_seed(plan.seed,
                             (base.terrain, base.season, "baseline"), "cal")),
                             contributors=("cal_lab_1", "cal_lab_2", "cal_lab_3"))
    old_scores = _scores(baseline)
    arms = {"hand_built": {}, "synthesized": {}}
    pooled = {"hand_built": [], "synthesized": []}
    conditional_cal = {key: _scores(pair["cal"]) for key, pair in battery.items()}
    for key, pair in battery.items():
        test_scores = _scores(pair["test"])
        name = "/".join(key)
        for arm, cal in (("hand_built", old_scores),
                         ("synthesized", _scores(pair["cal"]))):
            p = (conditional_pvalues(key, test_scores, conditional_cal, plan.n_cal)
                 if arm == "synthesized" else conformal_pvalues(cal, test_scores))
            pooled[arm].append(p)
            arms[arm][name] = _metrics(p, plan.alpha)
    return {
        "plan": {"terrains": plan.terrains, "seasons": plan.seasons,
                 "sensors": plan.sensors, "n_cal_per_stratum": plan.n_cal,
                 "n_test_per_stratum": plan.n_test, "seed": plan.seed,
                 "alpha": plan.alpha, "baseline_scene": base.to_dict()},
        "coverage": coverage(plan, battery, minimum_cal=plan.n_cal),
        "arms": arms,
        "pooled": {arm: _metrics(np.concatenate(p), plan.alpha)
                   for arm, p in pooled.items()},
        "digests": {"/".join(key): {split: ds.digest() for split, ds in pair.items()}
                    for key, pair in battery.items()},
        "limitations": ["Synthetic clean data only; no real-scene exchangeability guarantee.",
                        "Brightness is a fixed covariate proxy, not a model detector.",
                        "Pooled p-values across heterogeneous strata are descriptive, not a validity guarantee.",
                        "False-abstain is a proxy decision p <= alpha on known-clean items, not a governance disposition."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure synthetic clean reference synthesis")
    parser.add_argument("--n-cal", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", help="Optional JSON result path")
    args = parser.parse_args()
    report = measure(BatteryPlan(n_cal=args.n_cal, n_test=args.n_test, seed=args.seed))
    result = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
