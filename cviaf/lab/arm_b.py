"""
Experiment arm B: the MEASUREMENT harness for v4 upgrades U2 and U3.

Nothing in this file is a claim. Every number it emits comes from running the
mechanism against lab corpora whose ground truth the harness itself holds:

  U2 (residual risk)   for each attack scenario, a power curve over dataset-level
                       poison prevalence -- Power(p) = fraction of independent
                       seeds on which the data-axis assessment flags the asset
                       at the declared alpha -- inverted into r* per scenario,
                       aggregated into the report-level R*, and then VALIDATED
                       on fresh seeds spiked at 0.5*r*, r* and 2*r*. The
                       conservatism statement is exact: the promised detection
                       probability at r* must actually show up on seeds the
                       inversion never saw.

  U3 (LOCO attribution) on corpora where vendor_x carries a contributor-scoped
                       trigger attack, the LOCO surrogate screen
                       (lab/attribute.py) is scored against v3's A3 hierarchical
                       posterior on the SAME corpora: top-1 attribution
                       accuracy, mean rank of the true culprit, false-implication
                       rate on clean contributors, and per-contributor runtime.

The x-axis of every U2 curve is the REALISED dataset-level prevalence
(``PoisonTruth.rate_actual``), not the requested rate, so contributor-scoped
and diffuse attacks are comparable on one axis and the operator's question --
"what prevalence would we have caught?" -- is the axis itself.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.attribute import (
    FrozenEmbedder,
    loco_attribution,
    posterior_baseline_ranking,
)
from cviaf.lab.calibrate import benjamini_yekutieli, conformal_pvalues
from cviaf.lab.detectors import duplicate_scores, spectral_signature_scores
from cviaf.lab.poison import AttackSpec, inject, trigger_view
from cviaf.lab.residual import (
    ResidualRiskPolicy,
    build_residual_risk,
    clopper_pearson_upper,
    invert_power_curve,
)
from cviaf.lab.synth import SceneSpec, build_dataset

CONTRIBUTORS = ("lab_alpha", "lab_beta", "vendor_x")
MAL = "vendor_x"


# --------------------------------------------------------------------------- #
# shared reference pool (the calibration budget, built once)
# --------------------------------------------------------------------------- #

def build_pool(scene: SceneSpec, n_per_block: int, blocks: int,
               seed_base: int = 900_000) -> Dict[str, np.ndarray]:
    """A pooled clean reference for the data-axis signals, sized to the decision.

    Same construction as compare.build_reference_pool: the duplicate statistic
    is only exchangeable across same-size splits, so the reference is many clean
    splits of contribution size. The block count here buys a conformal floor of
    1/(blocks*n+1); the asset-level Cauchy decision needs no extreme floor, so
    20 blocks (~2e-4) is the budget, and the item-level BY flags are reported
    alongside with their granularity honestly noted.
    """
    from cviaf.utils import extract_features_from_images

    feats, spec_scores, dup_scores = [], [], []
    for b in range(blocks):
        ds = build_dataset(n_per_block, scene, seed_offset=seed_base + b * 9973)
        f = extract_features_from_images(ds.images, method="pixel_stats", target_dim=64)
        feats.append(f)
        spec_scores.append(spectral_signature_scores(f, _sample_classes(ds.labels)))
        dup_scores.append(duplicate_scores(ds.images))
    return {
        "features": np.concatenate(feats),
        "spectral": np.concatenate(spec_scores),
        "duplicate": np.concatenate(dup_scores),
        "n_total": int(sum(len(x) for x in feats)),
    }


def _sample_classes(labels: Sequence[np.ndarray]) -> np.ndarray:
    out = []
    for l in labels:
        l = np.asarray(l).ravel()
        out.append(int(np.bincount(l.astype(np.int64)).argmax()) if l.size else 0)
    return np.asarray(out, np.int64)


def asset_signals(ds) -> Dict[str, np.ndarray]:
    from cviaf.utils import extract_features_from_images
    feats = extract_features_from_images(ds.images, method="pixel_stats", target_dim=64)
    return {"spectral": spectral_signature_scores(feats, _sample_classes(ds.labels)),
            "duplicate": duplicate_scores(ds.images)}


def assess_asset(ds, pool, alpha: float) -> Dict[str, Any]:
    """The data-axis assessment U2 prices: fused conformal p-values per item,
    collapsed to one asset p-value (Cauchy), plus the BY item-flag count."""
    from cviaf.lab.baseline import asset_pvalue_from_items
    sig = asset_signals(ds)
    cols = []
    for name, lower in (("spectral", False), ("duplicate", True)):
        ref = pool[name]
        obs = sig[name]
        with np.errstate(invalid="ignore"):
            cols.append(conformal_pvalues(ref, obs, higher_is_more_anomalous=not lower))
    stacked = np.stack(cols, axis=1)
    fused = np.clip(stacked.shape[1] * stacked.min(axis=1), 0.0, 1.0)
    # Symmetric cap before the Cauchy collapse. Conformal p-values live on the
    # discrete grid {1/(n+1), ..., 1}; the grid point p=1 is UNPAIRED, and
    # tan((0.5-p)*pi) diverges there, so one maximally NON-anomalous item would
    # veto 48 anomalous ones (measured: asset p ~ 1.0 on a 20% ood_insert).
    # Capping p at 1 - 1/(n_cal+1) pairs the grid point with p_min, making the
    # null tan distribution symmetric around zero. A non-anomalous item must
    # never erase the evidence of anomalous ones.
    n_cal = pool["n_total"]
    fused = np.minimum(fused, 1.0 - 1.0 / (n_cal + 1.0))
    asset_p, combo = asset_pvalue_from_items(fused, method="cauchy")
    n_flagged = int(benjamini_yekutieli(fused, alpha).sum())
    # The decision rule being priced is the repo's SHIPPED data-axis verdict
    # (baseline.cviaf_data_asset_verdict): the asset is flagged when the
    # BY-controlled rejection set is non-empty. The Cauchy asset p-value is
    # recorded alongside -- it is the stricter, multiplicity-free collapse, and
    # the gap between the two is itself measured here.
    return {"asset_p": float(asset_p), "n_flagged": n_flagged,
            "flagged": bool(n_flagged > 0),
            "flagged_cauchy": bool(asset_p <= alpha),
            "combination": combo}


# --------------------------------------------------------------------------- #
# U2 scenarios
# --------------------------------------------------------------------------- #

def u2_scenarios() -> Dict[str, Dict[str, Any]]:
    """The modules whose power curves make up the residual-risk budget.

    Rates are REQUESTED rates; the axis of record is the realised prevalence.
    Contributor-scoped trigger attacks: rate is a fraction of vendor_x's
    samples, so realised dataset prevalence is ~rate/3 with three balanced
    contributors -- that is the honest conversion and it is measured, not
    assumed.
    """
    return {
        "data.label_flip":       dict(kind="label_flip", trigger="none",
                                      trigger_loc="fixed", scope="diffuse"),
        "data.dup_flood":        dict(kind="dup_flood", trigger="none",
                                      trigger_loc="fixed", scope="diffuse"),
        "data.ood_insert":       dict(kind="ood_insert", trigger="none",
                                      trigger_loc="fixed", scope="diffuse"),
        "data.trigger.rma":      dict(kind="rma", trigger="patch",
                                      trigger_loc="on_object", trigger_size=10,
                                      target_class=0, scope="contributor"),
        "data.trigger.clean_label": dict(kind="clean_label", trigger="patch",
                                         trigger_loc="fixed", trigger_size=10,
                                         target_class=0, scope="contributor"),
        "data.trigger.oda_frame": dict(kind="oda", trigger="frame",
                                       trigger_loc="on_object", trigger_size=10,
                                       target_class=0, scope="contributor"),
    }


def _spec_for(scn: Dict[str, Any], rate: float, seed: int) -> AttackSpec:
    return AttackSpec(rate=rate, seed=seed, mal_contributor=MAL, **scn)


def run_u2(
    out_dir: str,
    rates: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    seeds: Sequence[int] = tuple(range(8)),
    n_train: int = 240,
    alpha: float = 0.05,
    target_power: float = 0.80,
    policy_tolerance: float = 0.02,
    pool_blocks: int = 20,
    val_seeds: Sequence[int] = tuple(range(100, 108)),
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Measure power curves, invert to r* per scenario, aggregate to R*, validate."""
    t0 = time.time()
    scene = SceneSpec(seed=7)
    log(f"[U2] building reference pool ({pool_blocks} blocks of {n_train})")
    pool = build_pool(scene, n_train, pool_blocks)
    scenarios = u2_scenarios()

    per_module: Dict[str, Dict[str, Any]] = {}
    curves_for_risk: Dict[str, Dict[str, Any]] = {}
    for name, scn in scenarios.items():
        pts: List[Dict[str, Any]] = []
        fpr_runs: List[bool] = []
        for rate in [0.0] + list(rates):
            hits, realised = [], []
            for s in seeds:
                ds = build_dataset(n_train, scene, seed_offset=1000 + 37 * s)
                spec = _spec_for(scn, rate, seed=11 + s)
                pds, truth = inject(ds, spec)
                v = assess_asset(pds, pool, alpha)
                hits.append(bool(v["flagged"]))
                realised.append(float(truth.rate_actual))
                if rate == 0.0:
                    fpr_runs.append(bool(v["flagged"]))
            pts.append({"rate_requested": float(rate),
                        "rate_actual": float(np.mean(realised)),
                        "power": float(np.mean(hits)),
                        "n_seeds": len(seeds)})
            log(f"[U2] {name:26s} req={rate:5.3f} actual={np.mean(realised):5.3f} "
                f"power={np.mean(hits):.3f}")
        inv = invert_power_curve([p["rate_actual"] for p in pts],
                                 [p["power"] for p in pts], target_power)
        per_module[name] = {"points": pts, "inversion": inv,
                            "clean_fpr": float(np.mean(fpr_runs))}
        curves_for_risk[name] = {"rates": [p["rate_actual"] for p in pts],
                                 "powers": [p["power"] for p in pts]}

    # ---- aggregate into the report-level residual-risk statement -----------
    policy = ResidualRiskPolicy(tolerance=policy_tolerance,
                                target_power=target_power, alpha=alpha)
    risk = build_residual_risk(curves_for_risk, n_samples=n_train, n_flagged=0,
                               policy=policy)

    # ---- validation on FRESH seeds: the promised power must show up ---------
    validation: Dict[str, Any] = {}
    for name, scn in scenarios.items():
        r_star = per_module[name]["inversion"]["r_star"]
        if r_star is None:
            validation[name] = {"r_star": None,
                                "note": "no power at any tested rate; nothing to "
                                        "validate -- the module reports INSUFFICIENT "
                                        "POWER and forces review"}
            continue
        # convert the realised-prevalence r* back to a requested rate for the
        # spike: contributor-scoped rates are per-contributor, so scale by the
        # contributor share measured on the curve.
        pts = per_module[name]["points"]
        req = np.array([p["rate_requested"] for p in pts])
        act = np.array([p["rate_actual"] for p in pts])
        scale = float(np.median(req[act > 0] / act[act > 0])) if (act > 0).any() else 1.0
        checks: List[Dict[str, Any]] = []
        for mult, tag in ((0.5, "0.5*r*"), (1.0, "r*"), (2.0, "2*r*")):
            target = r_star * mult
            hits, realised = [], []
            for s in val_seeds:
                ds = build_dataset(n_train, scene, seed_offset=50_000 + 37 * s)
                spec = _spec_for(scn, min(target * scale, 1.0), seed=911 + s)
                pds, truth = inject(ds, spec)
                v = assess_asset(pds, pool, alpha)
                hits.append(bool(v["flagged"]))
                realised.append(float(truth.rate_actual))
            power = float(np.mean(hits))
            # Clopper-Pearson LOWER bound on the measured power itself, so the
            # validation carries its own estimation uncertainty.
            lo = 1.0 - clopper_pearson_upper(len(val_seeds) - int(sum(hits)),
                                             len(val_seeds), 0.95)
            checks.append({"spike": tag, "target_prevalence": float(target),
                           "realised_prevalence": float(np.mean(realised)),
                           "power": power, "power_cp_lower": float(lo)})
            log(f"[U2] validate {name:26s} {tag:6s} realised={np.mean(realised):5.3f} "
                f"power={power:.3f} (CP lower {lo:.3f})")
        at_star = next(c for c in checks if c["spike"] == "r*")
        validation[name] = {
            "r_star": r_star,
            "checks": checks,
            "conservative": bool(at_star["power"] >= target_power),
            "criterion": (f"power at r* on {len(val_seeds)} fresh seeds >= "
                          f"{target_power}; the stated bound is conservative iff "
                          f"the promised detection probability materialises"),
        }

    # ---- the v3 embarrassment case: 200 images, 1% poison -------------------
    small_case_runs = []
    for s in val_seeds:
        ds = build_dataset(200, scene, seed_offset=70_000 + 37 * s)
        spec = _spec_for(u2_scenarios()["data.label_flip"], 0.01, seed=311 + s)
        pds, truth = inject(ds, spec)
        v = assess_asset(pds, pool, alpha)
        small_case_runs.append({"flagged": bool(v["flagged"]),
                                "n_flagged": v["n_flagged"],
                                "rate_actual": float(truth.rate_actual)})
    small_power = float(np.mean([r["flagged"] for r in small_case_runs]))
    small_sentence = (
        invert_power_curve([p["rate_actual"] for p in per_module["data.label_flip"]["points"]],
                           [p["power"] for p in per_module["data.label_flip"]["points"]],
                           target_power))
    small_case = {
        "n": 200, "rate": 0.01, "seeds": len(val_seeds),
        "measured_power": small_power,
        "sentence": small_sentence["claim"],
        "cp_upper_on_zero_findings": clopper_pearson_upper(0, 200, 0.95),
        "forced_disposition": "review" if small_power < target_power else None,
    }

    result = {
        "experiment": "arm_b.U2",
        "alpha": alpha, "target_power": target_power,
        "n_train": n_train, "seeds": list(seeds), "val_seeds": list(val_seeds),
        "pool_n_total": pool["n_total"],
        "per_module": per_module,
        "residual_risk": risk.to_dict(),
        "validation": validation,
        "small_case_200_1pct": small_case,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "arm_b_u2.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    return result


# --------------------------------------------------------------------------- #
# U3 measurement
# --------------------------------------------------------------------------- #

U3_KINDS = (
    dict(kind="rma", trigger="patch", trigger_loc="on_object", trigger_size=10,
         target_class=0, rate=0.30),
    dict(kind="clean_label", trigger="patch", trigger_loc="fixed", trigger_size=10,
         target_class=0, rate=0.30),
    dict(kind="oda", trigger="frame", trigger_loc="on_object", trigger_size=10,
         target_class=0, rate=0.40),
)


def run_u3(
    out_dir: str,
    pool: Optional[Dict[str, np.ndarray]] = None,
    seeds: Sequence[int] = tuple(range(10)),
    n_train: int = 240,
    n_eval: int = 80,
    n_bootstrap: int = 40,
    alpha: float = 0.05,
    kinds: Optional[Sequence[Dict[str, Any]]] = None,
    pool_blocks: int = 20,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Score LOCO causal attribution against the A3 correlational posterior."""
    t0 = time.time()
    scene = SceneSpec(seed=7)
    if pool is None:
        log(f"[U3] building reference pool ({pool_blocks} blocks of {n_train})")
        pool = build_pool(scene, n_train, pool_blocks)
    embedder = FrozenEmbedder()

    per_run: List[Dict[str, Any]] = []
    for kind_cfg in (kinds or U3_KINDS):
        for s in seeds:
            spec = AttackSpec(seed=11 + s, mal_contributor=MAL,
                              scope="contributor", **kind_cfg)
            ds = build_dataset(n_train, scene, seed_offset=2000 + 37 * s)
            pds, truth = inject(ds, spec)
            eval_ds = build_dataset(n_eval, scene, contributors=("eval",),
                                    seed_offset=30_000 + 37 * s)
            trig_imgs, infos = trigger_view(eval_ds, spec, seed=1 + s)
            vic = None
            if spec.kind == "oda":
                vic = np.array([t.victim_class if t.victim_class is not None else 0
                                for t in infos])
            loco = loco_attribution(pds, spec, eval_ds.images, trig_imgs,
                                    eval_ds.labels, victim_classes=vic,
                                    embedder=embedder, n_bootstrap=n_bootstrap,
                                    alpha=alpha, seed=s)
            sig = asset_signals(pds)
            base = posterior_baseline_ranking(
                pds, {"spectral": (pool["spectral"], sig["spectral"]),
                      "duplicate": (pool["duplicate"], sig["duplicate"])},
                alpha=alpha)
            loco_top = loco["ranking_by_delta_trigger"][0]
            base_top = (base["ranking_by_posterior"][0]
                        if base["ranking_by_posterior"] else None)
            rank = lambda lst: (lst.index(MAL) + 1) if MAL in lst else None
            false_impl = [c for c in loco["implicated"] if c != MAL]
            run = {
                "kind": spec.kind, "seed": int(s),
                "n_poisoned": len(truth.poisoned_indices),
                "dataset_prevalence": float(truth.rate_actual),
                "loco_top1": loco_top, "loco_top1_correct": loco_top == MAL,
                "loco_true_rank": rank(loco["ranking_by_delta_trigger"]),
                "loco_implicated": loco["implicated"],
                "loco_false_implications": false_impl,
                "loco_response_all": loco["response_all"],
                "loco_runtime_seconds": loco["runtime_seconds"],
                "loco_per_contributor_seconds": round(
                    loco["runtime_seconds"] / len(CONTRIBUTORS), 3),
                "posterior_top1": base_top,
                "posterior_top1_correct": base_top == MAL,
                # An empty posterior ranking is a measured outcome, not missing
                # data: with zero flagged items the baseline has nothing to
                # correlate and names nobody. Scored as the worst possible rank.
                "posterior_true_rank": (rank(base["ranking_by_posterior"])
                                        if base["ranking_by_posterior"]
                                        else len(CONTRIBUTORS) + 1),
                "posterior_abstained": not base["ranking_by_posterior"],
                "posterior_n_flagged_items": base["n_flagged_items"],
                "effects": loco["effects"],
            }
            per_run.append(run)
            log(f"[U3] {spec.kind:11s} seed={s} loco_top={loco_top:10s} "
                f"impl={loco['implicated']} posterior_top={base_top} "
                f"(flags={base['n_flagged_items']}) t={loco['runtime_seconds']}s")

    summary: Dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in per_run}):
        runs = [r for r in per_run if r["kind"] == kind]
        n = len(runs)
        summary[kind] = {
            "n_seeds": n,
            "loco_top1_accuracy": float(np.mean([r["loco_top1_correct"] for r in runs])),
            "loco_mean_true_rank": float(np.mean([r["loco_true_rank"] for r in runs])),
            "loco_false_implication_rate": float(np.mean(
                [1.0 if r["loco_false_implications"] else 0.0 for r in runs])),
            "loco_false_implication_rate_per_clean_contributor": float(np.mean([
                len(r["loco_false_implications"]) / (len(CONTRIBUTORS) - 1)
                for r in runs])),
            "posterior_top1_accuracy": float(np.mean([r["posterior_top1_correct"]
                                                      for r in runs])),
            "posterior_mean_true_rank": float(np.mean([r["posterior_true_rank"]
                                                       for r in runs])),
            "mean_runtime_seconds": float(np.mean([r["loco_runtime_seconds"]
                                                   for r in runs])),
            "mean_per_contributor_seconds": float(np.mean(
                [r["loco_per_contributor_seconds"] for r in runs])),
            "acceptance": {
                "top1_ge_0.8": bool(np.mean([r["loco_top1_correct"] for r in runs]) >= 0.8),
                "beats_posterior": bool(
                    np.mean([r["loco_true_rank"] for r in runs])
                    < np.mean([r["posterior_true_rank"] for r in runs])),
                "false_implication_per_contributor_le_alpha": bool(np.mean([
                    len(r["loco_false_implications"]) / (len(CONTRIBUTORS) - 1)
                    for r in runs]) <= alpha),
                "per_contributor_runtime_le_120s": bool(np.mean(
                    [r["loco_per_contributor_seconds"] for r in runs]) <= 120.0),
            },
        }
    result = {
        "experiment": "arm_b.U3",
        "alpha": alpha, "n_bootstrap": n_bootstrap,
        "n_train": n_train, "n_eval": n_eval, "seeds": list(seeds),
        "contributors": list(CONTRIBUTORS), "mal_contributor": MAL,
        "per_run": per_run, "summary": summary,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "arm_b_u3.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    return result
