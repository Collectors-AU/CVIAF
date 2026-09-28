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
    PAIRED_FEATURES,
    behavioral_fingerprint,
    benign_variation_scale,
    fingerprint_distance,
    make_backgrounds,
    paired_fingerprint,
    pre_nms_class_divergence,
    reference_divergence,
    standardised_deviation,
    trace_ctc,
    trace_foreground,
    trace_ftc,
    training_class_frequencies,
    weight_stats,
)
from cviaf.lab.poison import AttackSpec, trigger_view
from cviaf.lab.synth import SceneSpec
from cviaf.lab.train import TrainSpec, build_splits

# Four signals, and the reason for each is measured rather than asserted:
#   ctc     probabilistic detection -- blind to cloaking (AUROC ~0.44 on oda)
#   refdiv  reference divergence    -- carries cloaking, needs a trusted reference
#   ftc     the Island Effect       -- carries cloaking WITHOUT a reference model
#   fg      TRACE foreground arm    -- the second half of TRACE, sign inverted vs ctc
# FTC was implemented but not wired into the evaluation, so its power was unmeasured.
# It is a black-box probe, which makes it the answer for a contributor who will not
# hand over weights and for whom no reference model is enrolled.
# fg closes the citable gap that ctc reads 0.816 on the null control: if a foreground
# measurement of a DIFFERENT quantity also fires on a trigger-stamped clean model then
# the sensitivity is real; if only ctc fires, ctc is measuring the ink.
DETECTOR_NAMES = ("ctc", "refdiv", "ftc", "fg")
FUSED_NAMES = ("fused_bonf", "fused_cauchy")

#: Which detectors may enter the fused statistics. ``fg`` is deliberately NOT here yet.
#: Measured on runs/mvp it reads AUROC 0.000 on a trigger-stamped CLEAN model -- perfect
#: separation in the INVERTED direction, i.e. it is measuring the trigger's ink exactly
#: as ``ctc`` does -- while reading 0.042 on the ``oga`` row. A min-p (Bonferroni) fusion
#: takes the MOST significant detector, so folding in a stamp detector raises the fused
#: false-alarm rate on stamped clean images. It is published as its own row, and promoted
#: into this tuple only once its stamped-null AUROC sits inside [0.45, 0.55].
FUSED_DETECTORS = ("ctc", "refdiv", "ftc")

# Asset-level (per-model) detection settings. The probe battery is shared by every
# model so the black-box comparison is read on identical inputs, and the threshold at
# which the paired fingerprint reads detections was chosen by sweeping the NULL only
# (scripts/tamper_probe.py --sweep): picking it because it separated the attacks we
# built would be fitting the corpus. The null distance is flat across 0.15-0.75, so the
# choice is not load-bearing -- which is the point of reporting it.
ASSET_PROBE_IMAGES = 40
ASSET_NULL_PAIRS = 60
ASSET_PAIR_THRESH = 0.5
#: Confidence gate for the pre-NMS class histogram (DistScan uses a default suited to
#: YOLOv5 on COCO, which emits far more candidates per image than this MVP detector).
ASSET_PRE_NMS_CONF = 0.35


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
    # TRACE foreground arm. Runs on the same images and the same cells as ctc, so the
    # pair is a controlled comparison of "scene change" against "object change".
    fg = trace_foreground(model, clean_imgs)
    out["clean"]["fg"] = fg["score"]
    out["trig"]["fg"] = trace_foreground(model, trig_imgs)["score"]
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
            "weights_changed": bool(m.get("quality_flags", {})
                                    .get("weights_changed", False)),
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
        fused = _fused(sc["clean"], sc["trig"], FUSED_DETECTORS)
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
    asset = asset_level_detectors(registry, ref_model, ref_id=ref_id)
    return {"alpha": alpha, "reference_model": ref_id, "n_models": len(per_model),
            "per_model": per_model, "summary": summary, "null_control": null,
            "asset_level": asset,
            "detectors": list(DETECTOR_NAMES),
            "ftc_stride": int(ftc_stride),
            "ftc_decoy_class": int(ftc_decoy_class), "n_backgrounds": int(n_backgrounds)}


def asset_level_detectors(
    registry: List[Dict[str, Any]],
    ref_model,
    ref_id: Optional[str] = None,
    probe_images: int = ASSET_PROBE_IMAGES,
    max_null_pairs: int = ASSET_NULL_PAIRS,
) -> Dict[str, Any]:
    """Model-change detectors, scored one number per ASSET rather than per image.

    A substitution or a weight modification is a property of the model, not of any
    individual image, so the image-level machinery in this module is the wrong shape
    for it. Four signals are measured here and their TOLERANCE IS MEASURED, never
    guessed -- every one of them is standardised by the spread it shows across
    genuinely distinct clean models, because that spread is what "nothing happened"
    looks like:

    ``weight_z_mean`` / ``weight_z_rms``
        white-box. Per-statistic deviation from the clean population, in units of each
        statistic's own clean spread. Replaces ``weight_deviation`` (mean absolute
        deviation over eight statistics in eight different units), which had a null of
        1.66 against a largest effect of 3.48 -- not a detector, a coincidence.

    ``weight_null_statistics_violated``
        white-box, categorical. The number of weight statistics that are CONSTANT
        across every clean model and were changed by this artifact. A statistic that
        never varies has no z-score to report: dividing zero spread is not infinite
        significance, and folding it into a distance lets one dimension dominate all
        the others. It is an exact-match check instead, and it is what catches
        structural tampering that costs zero utility.

    ``paired_whitened`` / ``paired_z``
        black-box. Distance of a battery-conditioned behavioural signature from the
        enrolled reference, in a whitened space whose covariance is measured from clean
        pairs. The unpaired version absorbed image difficulty into the same number as
        model difference; pairing reads both models off the same positions so the scene
        cancels.

    ``unpaired``
        black-box, the first implementation, retained because the measurement did not
        crown a single winner: it is beaten by the paired statistics on diffuse edits
        and beats them on the two attacks that change aggregate detection behaviour
        most. Reporting all of them is the honest outcome of a tie.

    The gate matters as much as the detector, and its criterion changed after
    measurement. It is no longer "utility dropped" -- an artifact can be tampered with
    and still work, and that is the case an integrity framework exists for (zeroing 6%
    of the hidden units left F1 unchanged to three decimals). The gate is now "the
    artifact is not byte-identical to the honest model", and harm is an attribute the
    table is stratified by.
    """
    from cviaf.lab.train import ModelArtifact

    # One battery for every model, so all fingerprints are read on identical inputs.
    # It is the eval_clean split of the first registry entry, which every other model
    # has never been trained on -- a shared held-out battery, which is what makes a
    # black-box comparison between two models meaningful at all.
    probe = None
    ref_counts = None
    for e in registry:
        splits = build_splits(train_spec_from_manifest(e["manifest"]))
        probe = splits.eval_clean.images[:probe_images]
        # The pre-NMS class-prior signal needs the DECLARED training class
        # distribution, not the contributed one: for a poisoning attack the
        # contributed labels are exactly what is in question. The clean training
        # split is the prior the model was supposed to encode.
        ref_counts = training_class_frequencies(
            splits.train.labels, int(train_spec_from_manifest(
                e["manifest"]).detector.n_classes))
        break

    clean, attacked, inert = [], [], []
    for e in registry:
        m = e["manifest"]
        kind = m["ground_truth"]["kind"]
        flags = m.get("quality_flags", {}) or {}
        is_model_attack = bool(flags.get("is_model_attack"))
        if is_model_attack and not bool(flags.get("weights_changed", False)):
            # Byte-identical to the honest model: nothing was tampered with, so there is
            # nothing to detect. Counted and reported, never scored as a negative --
            # scoring it would be measuring the detector against a non-attack.
            inert.append({"model_id": m["model_id"],
                          "mechanism": m["ground_truth"].get("mechanism")})
            continue
        art = ModelArtifact.load(e["dir"])
        bdr = (m.get("metrics", {}) or {}).get("behaviour_divergence") or {}
        row = {"model_id": m["model_id"], "kind": kind,
               "mechanism": m["ground_truth"].get("mechanism"),
               "weights": weight_stats(art.model),
               "model": art.model,
               "f1_relative_change": bdr.get("f1_relative_change"),
               "f1_relative_drop": bdr.get("f1_relative_drop"),
               "is_model_attack": is_model_attack}
        if is_model_attack:
            attacked.append(row)
        elif kind == "clean":
            clean.append(row)
    if len(clean) < 2 or probe is None or ref_model is None:
        return {"available": False,
                "reason": ("needs an enrolled reference and at least two clean models to "
                           "measure a tolerance from")}

    keys = list(clean[0]["weights"].keys())
    w_ref = {k: float(np.mean([c["weights"][k] for c in clean])) for k in keys}
    w_sd = {k: float(np.std([c["weights"][k] for c in clean])) for k in keys}
    # Statistics that are constant across every clean model: exact-match checks.
    constant_stats = [k for k in keys if w_sd[k] <= 0.0]

    # ---- the null, measured: clean models read against each other on one battery
    order = [(i, j) for i in range(len(clean)) for j in range(len(clean)) if i != j]
    if len(order) > max_null_pairs:
        step = len(order) / float(max_null_pairs)
        order = [order[int(i * step)] for i in range(max_null_pairs)]
    null_paired = [paired_fingerprint(clean[i]["model"], clean[j]["model"], probe,
                                      score_thresh=ASSET_PAIR_THRESH)
                   for i, j in order]
    P = np.stack(null_paired)
    P_mean, P_sd = P.mean(axis=0), P.std(axis=0)
    inv_cov = whitened_inverse_cov(P)

    null_seq = [behavioral_fingerprint(c["model"], probe) for c in clean]
    seq_mean = np.mean(np.stack(null_seq), axis=0)
    seq_scale = benign_variation_scale(null_seq)

    def signals(row: Dict[str, Any]) -> Dict[str, Any]:
        agg = standardised_deviation([row["weights"][k] for k in keys],
                                     [w_ref[k] for k in keys], [w_sd[k] for k in keys])
        pf = paired_fingerprint(row["model"], ref_model, probe,
                                score_thresh=ASSET_PAIR_THRESH)
        seq = behavioral_fingerprint(row["model"], probe)
        return {
            "weight_z_mean": round(float(agg["z_mean"]), 6),
            "weight_z_rms": round(float(agg["z_rms"]), 6),
            "weight_z_max": round(float(agg["z_max"]), 6),
            "weight_null_statistics_violated": len(agg["violated_constant_statistics"]),
            "paired_whitened": round(_mahalanobis(pf, P_mean, inv_cov), 6),
            "paired_z": round(float(np.sqrt(np.sum(
                ((np.asarray(pf) - P_mean) / np.maximum(P_sd, 1e-12)) ** 2))), 6),
            "unpaired": round(fingerprint_distance(seq, seq_mean, seq_scale), 6),
            # Clean-input, trigger-free: cannot be contaminated by a stamp, because
            # nothing is stamped. Read at a slightly lower confidence gate than the
            # paper's default because this MVP detector emits far fewer pre-NMS
            # candidates per image than YOLOv5 on COCO.
            "pre_nms_class_js": round(pre_nms_class_divergence(
                row["model"], probe, ref_counts,
                conf=ASSET_PRE_NMS_CONF)["score"], 6),
        }

    rows, clean_scores, attacked_scores = [], [], []
    for group, store in ((clean, clean_scores), (attacked, attacked_scores)):
        for r in group:
            s = signals(r)
            store.append(s)
            rows.append({**{k: r[k] for k in ("model_id", "kind", "mechanism",
                                              "is_model_attack", "f1_relative_change",
                                              "f1_relative_drop")}, **s})

    SIGNALS = ("weight_z_mean", "weight_z_rms", "weight_z_max",
               "weight_null_statistics_violated", "paired_whitened", "paired_z",
               "unpaired", "pre_nms_class_js")
    # The tolerance per signal is the largest value a CLEAN model produces -- an
    # achievable threshold rather than a p95 extrapolated from eight samples. A
    # detection below it is reported as inside the null even when its AUROC looks
    # good, because at this sample size AUROC and deployability are different claims.
    null_max = {s: (float(max(float(c[s]) for c in clean_scores))
                    if clean_scores else None) for s in SIGNALS}
    null_mean = {s: (float(np.mean([float(c[s]) for c in clean_scores]))
                     if clean_scores else None) for s in SIGNALS}

    out: Dict[str, Any] = {
        "available": True,
        "reference_model": ref_id,
        "n_clean": len(clean), "n_model_attacked": len(attacked),
        "n_inert_skipped": len(inert), "inert": inert,
        "battery": {"n_images": int(len(probe)),
                    "source": ("eval_clean of the first registry entry -- one shared "
                               "held-out battery, read identically for every model"),
                    "detection_threshold": ASSET_PAIR_THRESH},
        "null": {"n_clean_pairs_used": len(order),
                 "paired_features": list(PAIRED_FEATURES),
                 "clean_mean": null_mean, "clean_max": null_max,
                 "constant_weight_statistics": constant_stats},
        "per_model": rows,
    }
    for det in SIGNALS:
        if not attacked_scores:
            out[det] = {"auroc": None, "note": "no scored model attacks in this corpus"}
            continue
        scores = np.asarray([float(c[det]) for c in clean_scores]
                            + [float(c[det]) for c in attacked_scores], np.float64)
        labels = np.asarray([False] * len(clean_scores)
                            + [True] * len(attacked_scores))
        thr = null_max[det]
        out[det] = {
            "auroc": _safe(auroc(scores, labels)),
            "tpr_at_5fpr": _safe(tpr_at_fpr(scores, labels, 0.05)),
            "clean_mean": null_mean[det], "clean_max": thr,
            "attacked_mean": float(np.mean([float(c[det]) for c in attacked_scores])),
            "attacked_above_clean_max": int(sum(1 for c in attacked_scores
                                                if float(c[det]) > thr)),
            "n_attacked": len(attacked_scores),
        }
    return out


def weight_score_of(stats: Dict[str, float], reference: Dict[str, float]) -> float:
    """Mean absolute deviation of weight statistics from the reference.

    Local helper rather than ``detectors.weight_score`` so this table is independent of
    that function's signature: the evaluation must be able to re-measure the detector
    even if the detector changes. Kept for comparability with the first published
    table -- it is NOT the shipped statistic, because with eight statistics in eight
    different units it has no meaningful null (measured 1.66 against a largest effect
    of 3.48, see scripts/tamper_probe.py). ``weight_z_mean``/``weight_z_rms`` below are
    the standardised replacements.
    """
    dev = [abs(float(stats[k]) - float(reference[k]))
           for k in stats if k in reference and reference[k] is not None]
    return float(np.mean(dev)) if dev else 0.0


def whitened_inverse_cov(P: np.ndarray) -> np.ndarray:
    """Inverse covariance of the null feature differences, from MEASURED clean pairs.

    The per-feature z-distance treats the eight readings as independent and they are
    not: a model that misses objects the reference saw also reports a different object
    count, so ``ref_only_rate`` and ``sus_only_rate`` move together. A sum over
    correlated coordinates lets a joint shift partly cancel -- measured on a tamper
    that cost 47% of F1, the plain z-distance stayed inside the clean null while the
    individual readings were obviously abnormal. Whitening fixes that, and the
    covariance comes from clean pairs rather than a diagonal assumption.
    """
    C = np.cov(np.asarray(P, np.float64).T)
    C = C + np.eye(C.shape[0]) * (1e-6 * float(np.trace(C) / max(C.shape[0], 1)))
    return np.linalg.pinv(C)


def _mahalanobis(fp: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    d = np.asarray(fp, np.float64) - np.asarray(mean, np.float64)
    return float(np.sqrt(max(0.0, float(d @ inv_cov @ d))))


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
        fg_c = trace_foreground(art.model, clean_imgs)["score"]
        fg_t = trace_foreground(art.model, trig_imgs)["score"]
        okc, okt = np.isfinite(fg_c), np.isfinite(fg_t)
        row["detectors"]["fg"] = {"auroc": _safe(auroc(
            np.concatenate([fg_c[okc], fg_t[okt]]),
            np.concatenate([np.zeros(int(okc.sum())), np.ones(int(okt.sum()))]))),
            "clean_mean": float(np.nanmean(fg_c)), "trig_mean": float(np.nanmean(fg_t))}
        stamped_rows.append(row)

    out: Dict[str, Any] = {
        "available": bool(stamped_rows),
        "n_clean_models": len(stamped_rows),
        "stamped_with": attack_spec.kind,
        "interpretation": ("AUROC of clean-vs-stamped images on a model that was never "
                          "trained with a trigger. 0.500 = the detector ignores the "
                          "stamp. Materially AWAY from 0.500 in either direction = it is "
                          "reacting to the patch rather than to a backdoor, and its "
                          "attacked-row numbers are inflated by that much. Distance is "
                          "two-sided on purpose: an inverted statistic separates the "
                          "stamp perfectly at 0.000, and reading that as 'ignores it' "
                          "would publish the worst contamination as the cleanest cell."),
        "per_model": stamped_rows,
    }
    if stamped_rows:
        for d in DETECTOR_NAMES:
            vals = [r["detectors"][d]["auroc"] for r in stamped_rows
                    if d in r["detectors"] and r["detectors"][d]["auroc"] is not None]
            if not vals:
                out[d] = {"auroc_mean": None, "auroc_spread": None,
                          "direction": None, "verdict": "clean"}
                continue
            mean = float(np.mean(vals))
            dev = abs(mean - 0.5)
            verdict = ("measures the stamp, not the backdoor" if dev > 0.10 else
                       "mild stamp sensitivity" if dev > 0.05 else "ignores the stamp")
            if dev > 0.10 and mean < 0.5:
                verdict += " (INVERTED direction)"
            out[d] = {
                "auroc_mean": round(mean, 4),
                "auroc_spread": round(float(np.std(vals)), 4),
                "direction": "inverted" if mean < 0.5 else "as declared",
                "verdict": verdict,
            }
        if len(stamped_rows) == 1:
            # With one clean model the reference-divergence arm compares that model with
            # itself, so its null row is degenerate (identical models can only disagree
            # through the matching heuristic). Named rather than left to look measured.
            out["caveat"] = ("only 1 clean model: the reference-divergence null compares "
                             "the model with itself and must not be read as a "
                             "detector-independent number")
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
        # Two exclusion reasons, both reported: a trigger backdoor that never implanted,
        # and a model attack whose artifact is byte-identical to the honest model so
        # that nothing was actually tampered with. Either way there is nothing for a
        # detector to find, and scoring against it would flatter the detector.
        #
        # ``model_effect_weak`` (utility did not move) is deliberately NOT an exclusion
        # any more. A weight modification that leaves accuracy intact is the case an
        # integrity framework exists for -- zeroing 6% of the hidden units left F1
        # unchanged and still rewrote a statistic that is 0.0 for every clean model.
        # Excluding it would be the framework calling a tampered artifact clean because
        # it still works. Harm is reported as a stratification instead.
        def _scored(r):
            if kind == "clean":
                return True
            if r.get("is_model_attack"):
                return bool(r.get("weights_changed", True))
            return not r.get("backdoor_weak")

        scored = [r for r in rs if _scored(r)]
        entry: Dict[str, Any] = {
            "kind": kind, "n_models": len(rs), "n_scored": len(scored),
            "n_excluded_weak": len(rs) - len(scored),
            "n_excluded_backdoor_weak": sum(1 for r in rs if r.get("backdoor_weak")),
            "n_excluded_model_effect_weak": sum(1 for r in rs
                                                if r.get("model_effect_weak")),
            "mean_asr": float(np.mean([r["asr"] for r in rs])),
        }
        # Detection vs harm: how much utility each artifact lost, and how many of those
        # the detectors actually flagged. This is the table that answers "does it work on
        # a tamper that still works?", which a single TPR cannot.
        weak = [r for r in scored if r.get("model_effect_weak")]
        entry["utility_neutral_models_scored"] = len(weak)
        entry["mean_asr_utility_neutral"] = (float(np.mean([r["asr"] for r in weak]))
                                             if weak else None)
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
