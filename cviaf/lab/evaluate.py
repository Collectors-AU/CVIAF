"""
Corpus evaluation: turn a trained corpus into calibrated detection numbers.

This is where the lab pays off. Given a registry of clean and attacked models
(whose ground truth we hold, whose attack success rate we measured), we compute
for each detector:

  * AUROC on triggered-vs-clean images
  * power at a 5% false-positive rate (the operating point that matters)
  * **conformal p-values** calibrated on clean images, so the output is a
    calibrated statement rather than a ranking
  * the realised FDR and power of a BY-controlled decision on a *mixed* batch

The two structural results this is designed to surface, and which it did on the
MVP corpus, are:

  * CTC catches FP-inducing attacks and is structurally blind to cloaking;
  * reference-divergence catches cloaking and is nearly blind to fabrication;
  * *both* sit at chance on the clean control.

That complementarity is the argument for having more than one signal. A framework
with only the first detector would report "no anomaly" on a cloaked model, and a
framework with only the second would miss a phantom-object model. Fusion is not
decoration here; it is the difference between coverage and a blind spot.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.calibrate import (
    auroc,
    benjamini_yekutieli,
    cauchy_combine,
    conformal_pvalues,
    power_at_alpha,
    tpr_at_fpr,
)
from cviaf.lab.detector import DetectorConfig
from cviaf.lab.detectors import (
    make_backgrounds,
    reference_divergence,
    trace_ctc,
    trace_ftc,
)
from cviaf.lab.poison import AttackSpec, trigger_view
from cviaf.lab.synth import SceneSpec
from cviaf.lab.train import TrainSpec, build_splits

# Two signals, and FTC is deliberately not one of them (merge decision).
#   ctc     probabilistic detection -- blind to cloaking (AUROC ~0.44 on oda)
#   refdiv  reference divergence    -- carries cloaking, needs a trusted reference
#
# FTC / the Island Effect was implemented here but never earned a place in the scored
# set, and two measurements say it still has not:
#   * v4's own asset rule excludes it -- ``model_asset_rule.DEFAULT_SIGNALS`` carried
#     the note "FTC failed factorial controls; optional only";
#   * on the stamp-free factorial cell (``runs/stampfree/null_suite.json``, 3 assets,
#     24 eval / 32 calibration images per seed) ftc's conditional AUROC was 0.5431 at
#     TPR@5FPR 0.0000, while the *clean-peer* contrast on the identical images -- an
#     independently trained clean model against the clean model -- put ftc at 0.0038,
#     i.e. below chance: its ordering on like-versus-like pairs is noise. Asset
#     decisions were 0/3 rejected in all five cells under both fusions, with and
#     without ftc.
# So ftc is removed from the scored detector set. ``trace_ftc`` stays implemented and
# importable for anyone who wants to re-measure it; it is simply not part of a verdict.
DETECTOR_NAMES = ("ctc", "refdiv")
FUSED_NAMES = ("fused_bonf", "fused_cauchy")


def train_spec_from_manifest(manifest: Dict[str, Any]) -> TrainSpec:
    """Reconstruct the exact spec a model was trained from (needed for splits)."""
    s = manifest["spec"]
    return TrainSpec(
        model_id=s["model_id"],
        scene=SceneSpec(**s["scene"]),
        attack=AttackSpec(**s["attack"]),
        detector=DetectorConfig(**s["detector"]),
        n_train=s["n_train"], n_eval=s["n_eval"], n_cal=s["n_cal"],
        contributors=tuple(s["contributors"]),
        contributor_mode=s["contributor_mode"],
    )


def scores_for_model(
    model,
    ref_model,
    manifest,
    backgrounds,
    seed: int,
    ftc_stride: int = 8,
    ftc_decoy_class: int = 0,
    ftc_decoy_size: float = 13.0,
) -> Dict[str, Any]:
    """Clean-image and triggered-image scores for every detector on one model.

    FTC is a *targeted* probe: its decoy must belong to the object vocabulary the
    attack suppressed, so the deployment assumption is that an analyst names the class
    they care about ("military vehicles"), exactly as TRACE's Island Effect is used.
    Corrupting that assumption is measurable rather than mysterious -- a wrong decoy
    class simply stops FTC from seeing the island, and the sweep in
    ``docs/MVP_MAC.md`` reports what that costs.
    """
    spec = train_spec_from_manifest(manifest)
    splits = build_splits(spec)
    kind = manifest["ground_truth"]["kind"]

    clean_imgs = splits.eval_clean.images
    if kind in ("clean", "clean_label", "label_flip", "dup_flood", "ood_insert"):
        trig_imgs = clean_imgs          # no test-time effect
    else:
        trig_imgs, _ = trigger_view(splits.eval_clean, spec.attack, seed)

    out: Dict[str, Any] = {"kind": kind, "clean": {}, "trig": {}}
    ctc_c = trace_ctc(model, clean_imgs, backgrounds)["score"]
    ctc_t = trace_ctc(model, trig_imgs, backgrounds)["score"]
    out["clean"]["ctc"] = ctc_c
    out["trig"]["ctc"] = ctc_t
    if ref_model is not None:
        out["clean"]["refdiv"] = reference_divergence(model, ref_model, clean_imgs)["score"]
        out["trig"]["refdiv"] = reference_divergence(model, ref_model, trig_imgs)["score"]

    ftc_c = trace_ftc(model, clean_imgs, decoy_class=ftc_decoy_class,
                      decoy_size=ftc_decoy_size, stride=ftc_stride)["score"]
    ftc_t = trace_ftc(model, trig_imgs, decoy_class=ftc_decoy_class,
                      decoy_size=ftc_decoy_size, stride=ftc_stride)["score"]
    out["clean"]["ftc"] = ftc_c
    out["trig"]["ftc"] = ftc_t
    out["ftc_config"] = {"stride": int(ftc_stride), "decoy_class": int(ftc_decoy_class),
                         "decoy_size": float(ftc_decoy_size),
                         "access_required": "model.black_box",
                         "assumption": ("decoy class must be drawn from the vocabulary "
                                        "the attack suppressed")}
    return out


def _conformal_columns(
    clean: Dict[str, np.ndarray], trig: Dict[str, np.ndarray], names: Sequence[str]
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Conformal p-values for BOTH sides, calibrated on the CLEAN distribution.

    The calibration set must be the clean side for both. An earlier revision
    calibrated each side against itself, which makes the clean p-values uniform by
    construction while the triggered p-values are computed against the wrong null --
    producing a fusion that looked reasonable on the control and was wrong
    everywhere else. Conformal validity is entirely about which null you calibrate
    against, so this is the kind of bug that silently destroys the guarantee.
    """
    fc_cols: List[np.ndarray] = []
    ft_cols: List[np.ndarray] = []
    for n in names:
        if n not in clean or n not in trig:
            continue
        c = np.asarray(clean[n], np.float64)
        t = np.asarray(trig[n], np.float64)
        cal = c[np.isfinite(c)]
        if cal.size < 5:
            continue
        with np.errstate(invalid="ignore"):
            pc = conformal_pvalues(cal, c)
            pt = conformal_pvalues(cal, t)
        fc_cols.append(np.where(np.isfinite(c), pc, 1.0))
        ft_cols.append(np.where(np.isfinite(t), pt, 1.0))
    return fc_cols, ft_cols, len(fc_cols)


def _fused(clean: Dict[str, np.ndarray], trig: Dict[str, np.ndarray],
           names: Sequence[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Fuse detector p-values two ways, and return both.

    ``cauchy``  Cauchy combination. Robust to arbitrary dependence, and the right
                choice when several detectors each carry moderate evidence.
    ``bonf``    ``min(1, m * min_d p_d)``. A valid p-value, and -- the reason it is
                the default here -- it cannot be *diluted by a blind detector*.

    That second point is not theoretical. On the MVP corpus, CTC is structurally
    blind to cloaking (AUROC ~0.44 on ``oda``), and combining it with a detector
    that does see cloaking under equal weights drags the fused result from ~0.95
    down to ~0.38. An averaging-style fusion lets a blind sensor veto a working one.
    The minimum-with-multiplicity-correction form does not, at the cost of some
    power when many detectors carry weak independent evidence.

    Both are reported so the trade-off is visible in the results rather than
    asserted in a design document.
    """
    fc_cols, ft_cols, m = _conformal_columns(clean, trig, names)
    if m == 0:
        z = np.ones(1)
        return {"cauchy": (z, z.copy()), "bonf": (z, z.copy())}
    FC = np.stack(fc_cols, axis=1)
    FT = np.stack(ft_cols, axis=1)
    out = {
        "cauchy": (np.array([cauchy_combine(r) for r in FC]),
                   np.array([cauchy_combine(r) for r in FT])),
        "bonf": (np.clip(m * FC.min(axis=1), 0, 1),
                 np.clip(m * FT.min(axis=1), 0, 1)),
    }
    return out


def evaluate_corpus(
    corpus_dir: str,
    alpha: float = 0.05,
    n_backgrounds: int = 8,
    seed: int = 1,
    models: Optional[Sequence[str]] = None,
    ftc_stride: int = 8,
    ftc_decoy_class: int = 0,
) -> Dict[str, Any]:
    """Evaluate every model in a corpus directory and return the results table."""
    registry = load_registry(corpus_dir)
    if not registry:
        raise RuntimeError(f"no models found under {corpus_dir}/ (expected */manifest.json)")

    backgrounds = make_backgrounds(n_backgrounds, seed=seed)

    # reference model = the clean model with the lowest model_id, if present
    ref_model = None
    ref_id = None
    for e in registry:
        if e["manifest"]["ground_truth"]["kind"] == "clean":
            from cviaf.lab.train import ModelArtifact
            ref_model = ModelArtifact.load(e["dir"]).model
            ref_id = e["manifest"]["model_id"]
            break

    per_model: List[Dict[str, Any]] = []
    for e in registry:
        m = e["manifest"]
        if models is not None and m["model_id"] not in models:
            continue
        from cviaf.lab.train import ModelArtifact
        art = ModelArtifact.load(e["dir"])
        sc = scores_for_model(art.model, ref_model, m, backgrounds, seed,
                              ftc_stride=ftc_stride, ftc_decoy_class=ftc_decoy_class)

        row: Dict[str, Any] = {
            "model_id": m["model_id"],
            "kind": m["ground_truth"]["kind"],
            "trigger": m["ground_truth"]["trigger"],
            "asr": m["metrics"]["attack_success_rate"]["asr"],
            "applicable": m["metrics"]["attack_success_rate"]["applicable"],
            "backdoor_weak": m["quality_flags"]["backdoor_weak"],
            "model_effect_weak": bool(m.get("quality_flags", {})
                                      .get("model_effect_weak", False)),
            "is_model_attack": bool(m.get("quality_flags", {})
                                    .get("is_model_attack", False)),
            "weights_digest": m["artifact"]["weights_digest"][:16],
            "detectors": {},
        }
        y = np.concatenate([np.zeros(len(sc["clean"]["ctc"])),
                            np.ones(len(sc["trig"]["ctc"]))])
        for d in DETECTOR_NAMES:
            if d not in sc["clean"] or d not in sc["trig"]:
                continue
            s = np.concatenate([sc["clean"][d], sc["trig"][d]])
            ok = np.isfinite(s)
            row["detectors"][d] = {
                "auroc": _safe(auroc(s[ok], y[ok])),
                "tpr_at_5fpr": _safe(tpr_at_fpr(s[ok], y[ok], 0.05)),
                "clean_mean": float(np.nanmean(sc["clean"][d])),
                "trig_mean": float(np.nanmean(sc["trig"][d])),
            }
        fused = _fused(sc["clean"], sc["trig"], DETECTOR_NAMES)
        for fname, (fc, ft) in fused.items():
            yf = np.concatenate([np.zeros(len(fc)), np.ones(len(ft))])
            # p-values are SMALL-is-positive while every other statistic here is
            # LARGE-is-positive. Convert before ranking, or AUROC comes out as
            # 1 - truth and the fusion looks catastrophically bad.
            sf = -np.log10(np.clip(np.concatenate([fc, ft]), 1e-12, 1.0))
            row["detectors"][f"fused_{fname}"] = {
                "auroc": _safe(auroc(sf, yf)),
                "tpr_at_5fpr": _safe(tpr_at_fpr(sf, yf, 0.05)),
                "clean_mean_p": float(fc.mean()),
                "trig_mean_p": float(ft.mean()),
            }
        fc, ft = fused["bonf"]
        row["mixed_batch"] = _mixed_batch(fc, ft, alpha)
        per_model.append(row)

    summary = _summarise(per_model)
    null = null_control(registry, ref_model, backgrounds, seed=seed,
                        ftc_stride=ftc_stride, ftc_decoy_class=ftc_decoy_class)
    asset = asset_level_detectors(registry, ref_model)
    return {"alpha": alpha, "reference_model": ref_id, "n_models": len(per_model),
            "per_model": per_model, "summary": summary, "null_control": null,
            "asset_level": asset,
            "detectors": list(DETECTOR_NAMES),
            "ftc_stride": int(ftc_stride),
            "ftc_decoy_class": int(ftc_decoy_class), "n_backgrounds": int(n_backgrounds)}


def asset_level_detectors(
    registry: List[Dict[str, Any]],
    ref_model,
) -> Dict[str, Any]:
    """Model-change detectors, scored one number per ASSET rather than per image.

    A substitution or a weight modification is a property of the model, not of any
    individual image, so the image-level machinery in this module is the wrong shape
    for it. Two signals are measured here, and the pair is deliberate:

    ``weight_deviation``      white-box. Distance of the head's weight statistics from
                              an enrolled clean reference, in robust sigmas.
    ``fingerprint_distance``  black-box. Distance of a battery-conditioned behavioural
                              signature (per-class counts, confidences, box geometry)
                              from the reference, scaled by the variation measured
                              ACROSS genuinely distinct clean models -- so the tolerance
                              is measured rather than guessed.

    The gate matters as much as the detector: models whose tamper was a no-op
    (``model_effect_weak``) are excluded, because a detector cannot find a
    substitution that substituted nothing.
    """
    from cviaf.lab.detectors import (
        behavioral_fingerprint,
        benign_variation_scale,
        fingerprint_distance,
        weight_score,
        weight_stats,
    )
    from cviaf.lab.train import ModelArtifact

    clean, tampered, gated = [], [], []
    clean_fps: List[np.ndarray] = []
    probe = None
    for e in registry:
        m = e["manifest"]
        spec = train_spec_from_manifest(m)
        splits = build_splits(spec)
        if probe is None:
            probe = splits.eval_clean.images[:16]
        art = ModelArtifact.load(e["dir"])
        kind = m["ground_truth"]["kind"]
        is_model_attack = bool(m.get("quality_flags", {}).get("is_model_attack"))
        fp = behavioral_fingerprint(art.model, probe)
        row = {"model_id": m["model_id"], "kind": kind,
               "weights": weight_stats(art.model),
               "fingerprint": fp.tolist(),
               "is_model_attack": is_model_attack,
               "model_effect_weak": bool(m.get("quality_flags", {})
                                         .get("model_effect_weak", False)),
               "f1_relative_drop": ((m.get("metrics", {})
                                     .get("behaviour_divergence") or {})
                                    .get("f1_relative_drop"))}
        if is_model_attack:
            (gated if row["model_effect_weak"] else tampered).append(row)
        elif kind == "clean":
            clean.append(row)
            clean_fps.append(fp)
    if not clean or probe is None:
        return {"available": False, "reason": "no clean model to form a reference"}

    ref_stats: Dict[str, float] = {}
    keys = clean[0]["weights"].keys()
    for k in keys:
        vals = [c["weights"][k] for c in clean if c["weights"].get(k) is not None]
        ref_stats[k] = float(np.mean(vals))
    scale = benign_variation_scale(clean_fps) if len(clean_fps) > 1 else None
    ref_fp = np.mean(np.stack(clean_fps), axis=0)

    def score(row: Dict[str, Any]) -> Dict[str, float]:
        w = dict(row["weights"])
        return {
            "weight_deviation": weight_score_of(w, ref_stats),
            "fingerprint_distance": fingerprint_distance(
                np.asarray(row["fingerprint"]), ref_fp, scale),
        }

    rows = []
    for r in clean + tampered + gated:
        s = score(r)
        rows.append({**{k: r[k] for k in ("model_id", "kind", "is_model_attack",
                                          "model_effect_weak", "f1_relative_drop")},
                     **s})

    out: Dict[str, Any] = {
        "available": True,
        "n_clean": len(clean), "n_model_attacked_scored": len(tampered),
        "n_model_attacked_gated": len(gated),
        "benign_scale_measured_from_clean_models": len(clean_fps) > 1,
        "per_model": rows,
    }
    pos = [r for r in rows if r["is_model_attack"] and not r["model_effect_weak"]]
    for det in ("weight_deviation", "fingerprint_distance"):
        if not pos or not clean:
            out[det] = {"auroc": None, "note": "no scored model attacks in this corpus"}
            continue
        scores = np.asarray([r[det] for r in rows
                             if r["kind"] == "clean" or r in pos], np.float64)
        labels = np.asarray([False] * len(clean) + [True] * len(pos))
        out[det] = {
            "auroc": _safe(auroc(scores, labels)),
            "clean_mean": float(np.mean([r[det] for r in rows if r["kind"] == "clean"])),
            "attacked_mean": float(np.mean([r[det] for r in pos])),
            "tpr_at_5fpr": _safe(tpr_at_fpr(scores, labels, 0.05)),
        }
    return out


def weight_score_of(stats: Dict[str, float], reference: Dict[str, float]) -> float:
    """Mean absolute deviation of weight statistics from the reference.

    Local helper rather than ``detectors.weight_score`` so this table is independent of
    that function's signature: the evaluation must be able to re-measure the detector
    even if the detector changes.
    """
    dev = [abs(float(stats[k]) - float(reference[k]))
           for k in stats if k in reference and reference[k] is not None]
    return float(np.mean(dev)) if dev else 0.0


def null_control(
    registry: List[Dict[str, Any]],
    ref_model,
    backgrounds: np.ndarray,
    seed: int = 1,
    ftc_stride: int = 8,
    ftc_decoy_class: int = 0,
) -> Dict[str, Any]:
    """A REAL null: a clean model evaluated on trigger-stamped images.

    Why the existing control row was not a control
    ----------------------------------------------
    For a clean model the evaluation fed the SAME images to both sides (there is no
    test-time effect to apply), so its AUROC is 0.500 by arithmetic -- a set compared
    with itself. Every detector therefore reported a perfect control row, and the
    suite was praised for a calibration sanity check that could not have failed. That
    is the most dangerous kind of measurement bug: one that always agrees with you.

    This control is falsifiable. Take a model that was never trained with a trigger,
    stamp a real trigger recipe onto its evaluation images at inference time, and ask
    whether any detector fires. It should not: the model has no trigger behaviour, and
    if a detector lights up here it is responding to the *stamp itself* -- high-
    contrast patch statistics, edge artefacts, distribution change -- not to a backdoor.
    Any detector whose null AUROC is materially above 0.500 is measuring the ink.
    """
    from cviaf.lab.train import ModelArtifact

    stamped_rows: List[Dict[str, Any]] = []
    attack_spec = None
    for e in registry:
        k = e["manifest"]["ground_truth"]["kind"]
        if k in ("oga", "oda", "rma"):
            attack_spec = train_spec_from_manifest(e["manifest"]).attack
            break
    if attack_spec is None:
        return {"available": False,
                "reason": "no trigger attack in the corpus to stamp"}

    for e in registry:
        if e["manifest"]["ground_truth"]["kind"] != "clean":
            continue
        spec = train_spec_from_manifest(e["manifest"])
        splits = build_splits(spec)
        art = ModelArtifact.load(e["dir"])
        clean_imgs = splits.eval_clean.images
        trig_imgs, _ = trigger_view(splits.eval_clean, attack_spec, seed)

        row: Dict[str, Any] = {"model_id": e["manifest"]["model_id"],
                               "stamped_with": attack_spec.kind, "detectors": {}}
        ctc_c = trace_ctc(art.model, clean_imgs, backgrounds)["score"]
        ctc_t = trace_ctc(art.model, trig_imgs, backgrounds)["score"]
        row["detectors"]["ctc"] = {"auroc": _safe(auroc(
            np.concatenate([ctc_c, ctc_t]),
            np.concatenate([np.zeros(len(ctc_c)), np.ones(len(ctc_t))]))),
            "clean_mean": float(np.nanmean(ctc_c)), "trig_mean": float(np.nanmean(ctc_t))}
        if ref_model is not None:
            rd_c = reference_divergence(art.model, ref_model, clean_imgs)["score"]
            rd_t = reference_divergence(art.model, ref_model, trig_imgs)["score"]
            row["detectors"]["refdiv"] = {"auroc": _safe(auroc(
                np.concatenate([rd_c, rd_t]),
                np.concatenate([np.zeros(len(rd_c)), np.ones(len(rd_t))]))),
                "clean_mean": float(np.mean(rd_c)), "trig_mean": float(np.mean(rd_t))}
        ft_c = trace_ftc(art.model, clean_imgs, decoy_class=ftc_decoy_class,
                         stride=ftc_stride)["score"]
        ft_t = trace_ftc(art.model, trig_imgs, decoy_class=ftc_decoy_class,
                         stride=ftc_stride)["score"]
        row["detectors"]["ftc"] = {"auroc": _safe(auroc(
            np.concatenate([ft_c, ft_t]),
            np.concatenate([np.zeros(len(ft_c)), np.ones(len(ft_t))]))),
            "clean_mean": float(np.mean(ft_c)), "trig_mean": float(np.mean(ft_t))}
        stamped_rows.append(row)

    out: Dict[str, Any] = {
        "available": bool(stamped_rows),
        "n_clean_models": len(stamped_rows),
        "stamped_with": attack_spec.kind,
        "interpretation": ("AUROC of clean-vs-stamped images on a model that was never "
                          "trained with a trigger. 0.500 = the detector ignores the "
                          "stamp. Materially above 0.500 = it is reacting to the "
                          "patch, not to a backdoor, and its attacked-row numbers are "
                          "inflated by that much."),
        "per_model": stamped_rows,
    }
    if stamped_rows:
        for d in DETECTOR_NAMES:
            vals = [r["detectors"][d]["auroc"] for r in stamped_rows
                    if d in r["detectors"] and r["detectors"][d]["auroc"] is not None]
            out[d] = {
                "auroc_mean": round(float(np.mean(vals)), 4) if vals else None,
                "auroc_spread": round(float(np.std(vals)), 4) if vals else None,
                "verdict": ("clean" if not vals else
                            "measures the stamp, not the backdoor"
                            if float(np.mean(vals)) > 0.60 else
                            "mild stamp sensitivity" if float(np.mean(vals)) > 0.55
                            else "ignores the stamp"),
            }
    return out


def _safe(v: float) -> Optional[float]:
    return None if (v is None or not np.isfinite(v)) else float(v)


def _mixed_batch(p_clean: np.ndarray, p_trig: np.ndarray, alpha: float) -> Dict[str, Any]:
    """Realised FDR and power of a BY-controlled decision on a mixed batch.

    This is the honest way to report a detection system: not "AUROC 0.89", but
    "at an FDR target of 5%, we flagged X% of triggered images while Y% of the
    flags were false." The report then contains a number an operator can price.
    """
    p = np.concatenate([p_clean, p_trig])
    y = np.concatenate([np.zeros(len(p_clean), bool), np.ones(len(p_trig), bool)])
    rej = benjamini_yekutieli(p, alpha)
    fp = int((rej & ~y).sum())
    tp = int((rej & y).sum())
    n_rej = fp + tp
    n = p.size
    c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    out = {
        "alpha": float(alpha),
        "n_clean": int(len(p_clean)), "n_triggered": int(len(p_trig)),
        "n_flagged": int(n_rej),
        "realised_fdr": float(fp / n_rej) if n_rej else 0.0,
        "power": float(tp / max(len(p_trig), 1)),
        "false_positives": fp,
    }
    # Granularity diagnostic. A conformal p-value can never be smaller than
    # 1/(n_cal+1), because the calibration set is finite. If that floor sits above
    # the BY/BH rejection threshold, NO test can ever be rejected and the realised
    # power is zero by construction -- not because the detector failed. Reporting
    # the reason is the difference between an honest zero and a misleading one.
    min_p = float(p.min()) if n else 1.0
    thr_k1 = float(alpha / (n * c_n)) if n else 0.0
    out["min_p_observed"] = min_p
    out["by_threshold_k1"] = thr_k1
    out["granularity_limited"] = bool(min_p > thr_k1)
    if out["granularity_limited"]:
        out["diagnostic"] = (
            f"zero power BY CONSTRUCTION: the smallest attainable conformal p-value is "
            f"{min_p:.4f} (1/(n_cal+1)) while the Benjamini-Yekutieli threshold for the "
            f"most significant of {n} tests is {thr_k1:.2e}. No decision is possible at "
            f"this granularity. Fixes: (a) apply FDR at the ASSET level -- one p-value "
            f"per model rather than per image -- which is the granularity an operator "
            f"actually acts on; and/or (b) enlarge the calibration split: image-level FDR "
            f"at alpha={alpha} needs roughly n_cal >= 1/(alpha/n) ~ {int(np.ceil(n/alpha))} "
            f"calibration items."
        )
    return out


def _summarise(per_model: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate by attack kind, excluding models whose backdoor never implanted.

    Excluding weak models is not cherry-picking: a detector cannot detect a
    backdoor that is not there, and including ASR=0 models would *flatter* the
    detector numbers by adding easy negatives. The exclusion is recorded.
    """
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_model:
        by_kind.setdefault(r["kind"], []).append(r)
    rows = []
    for kind, rs in sorted(by_kind.items()):
        # Two exclusion reasons, both reported: a backdoor that never implanted, and a
        # model tamper that never moved behaviour. Either way there is nothing for a
        # detector to find, and scoring against it would flatter the detector.
        scored = [r for r in rs
                  if kind == "clean"
                  or (not r.get("backdoor_weak") and not r.get("model_effect_weak"))]
        entry: Dict[str, Any] = {
            "kind": kind, "n_models": len(rs), "n_scored": len(scored),
            "n_excluded_weak": len(rs) - len(scored),
            "n_excluded_backdoor_weak": sum(1 for r in rs if r.get("backdoor_weak")),
            "n_excluded_model_effect_weak": sum(1 for r in rs
                                                if r.get("model_effect_weak")),
            "mean_asr": float(np.mean([r["asr"] for r in rs])),
        }
        for d in list(DETECTOR_NAMES) + list(FUSED_NAMES):
            vals = [r["detectors"][d]["auroc"] for r in scored
                    if d in r["detectors"] and r["detectors"][d]["auroc"] is not None]
            pw = [r["detectors"][d]["tpr_at_5fpr"] for r in scored
                  if d in r["detectors"] and r["detectors"][d]["auroc"] is not None]
            entry[d] = {"auroc_mean": float(np.mean(vals)) if vals else None,
                        "auroc_spread": float(np.std(vals)) if vals else None,
                        "tpr_at_5fpr_mean": float(np.mean(pw)) if pw else None}
        mb = [r["mixed_batch"] for r in scored]
        if mb:
            entry["realised_fdr"] = float(np.mean([m["realised_fdr"] for m in mb]))
            entry["power"] = float(np.mean([m["power"] for m in mb]))
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------- #
# registry I/O
# --------------------------------------------------------------------------- #

def load_registry(corpus_dir: str) -> List[Dict[str, Any]]:
    reg_path = os.path.join(corpus_dir, "registry.jsonl")
    out: List[Dict[str, Any]] = []
    if os.path.isfile(reg_path):
        with open(reg_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    for name in sorted(os.listdir(corpus_dir)) if os.path.isdir(corpus_dir) else []:
        mpath = os.path.join(corpus_dir, name, "manifest.json")
        if os.path.isfile(mpath):
            with open(mpath) as fh:
                out.append({"dir": os.path.join(corpus_dir, name), "manifest": json.load(fh)})
    return out
