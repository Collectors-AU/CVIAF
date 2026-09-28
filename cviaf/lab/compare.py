"""
Baseline vs CVIAF, measured on the same corpus with the same signals.

The comparison is deliberately structured so that the *only* thing that changes
between a baseline row and a CVIAF row is the design, not the raw material:

===========================  ==========================================
baseline                     same underlying statistic, different decision
===========================  ==========================================
``digest_manifest``          artefact hash equality
``fixed_threshold_single_signal``  one signal, constant cut
``hand_tuned_single_signal``  one signal, cut tuned in hindsight on the
                             clean control to hit a 5% FPR (the generous case)
===========================  ==========================================

CVIAF consumes exactly the same statistics and adds five things a threshold
cannot: (1) conformal p-values calibrated on a held-out clean split, so the
confidence attached to a flag has a validity statement; (2) fusion of two
complementary signals by ``min(1, m * min p)``, which a structurally blind signal
cannot dilute; (3) multiplicity control at the granularity of an asset, so the
false-discovery rate does not grow with intake size; (4) hierarchical contributor
attribution, so a flag can name a source instead of a sample; and (5) abstention,
so a check that did not run is reported as *not assessed* rather than as *clear*.

Three axes are scored separately, because a system that is excellent on one and
blind on another should be described that way rather than averaged into one
number:

``model_integrity``      is the submitted model backdoored / substituted?
``data_integrity``       is the submitted dataset poisoned, flooded, or shifted,
                         and *whose* contribution is it?
``inference_provenance`` are the protected inference records intact, or have they
                         been altered, substituted or replayed?

Every number in the output is measured. Nothing in this module hard-codes an
expected result, and where a cell is empty the reason is recorded (``abstain``,
``not applicable``, or the ASR gate) instead of being scored as a pass.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.baseline import (
    AXES,
    BASELINE_NAMES,
    DigestManifestBaseline,
    PerSampleOutlierBaseline,
    ThresholdedSingleSignal,
    Verdict,
    asset_pvalue_from_items,
    cviaf_data_asset_verdict,
)
from cviaf.lab.calibrate import benjamini_yekutieli, conformal_pvalues
from cviaf.lab.detector import TinyDetector
from cviaf.lab.detectors import (
    duplicate_scores,
    make_backgrounds,
    reference_divergence,
    spectral_signature_scores,
    trace_ctc,
    trace_ftc,
)
from cviaf.lab.evaluate import load_registry, real_backbone_eval_split, train_spec_from_manifest
from cviaf.lab.label_flip_signal import (build_label_reference, sample_loss_scores,
    calibrated_label_decisions, summarize_label_patterns)
from cviaf.lab.patch_local import patch_local_scores
from cviaf.lab.trigger_global import global_chroma_scores
from cviaf.lab.review import OperatorCosts, plan_review, review_budget_report
from cviaf.lab.poison import trigger_view
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.train import ModelArtifact, build_splits

# Ground truth for the model axis. Data-only attacks leave the model legitimately
# trained, so they are NEGATIVES for model integrity -- scoring them as positives
# would credit a model-integrity detector for an impossibility.
# NOTE: compare.py's tuple deliberately did NOT include the weight-space kinds because
# no pre-Task-3 corpus trained them here; poison.MODEL_ATTACK_KINDS owns the canonical
# list ("substitution", "weight_tamper"). Import it so a real-backbone corpus's model
# arms are scored as positives instead of being silently counted as 24 negatives.
from cviaf.lab.poison import MODEL_ATTACK_KINDS as _POISON_MODEL_ATTACKS
MODEL_ATTACK_KINDS = ("oga", "oda", "rma", "gma") + _POISON_MODEL_ATTACKS
DATA_ATTACK_KINDS = ("oga", "oda", "rma", "gma", "clean_label",
                     "label_flip", "dup_flood", "ood_insert")

# Kinds where exactly one contributor is the true culprit, so top-1 attribution
# can be scored. Trigger attacks are contributor-diffuse by construction, so on
# those the measurable question is the opposite one: does the system avoid
# accusing anyone?
KINDS_WITH_ONE_CULPRIT = ("dup_flood", "ood_insert")


# --------------------------------------------------------------------------- #
# the model axis
# --------------------------------------------------------------------------- #

def model_axis(
    registry: List[Dict[str, Any]],
    backgrounds: np.ndarray,
    alpha: float = 0.05,
    seed: int = 1,
    log: Callable[[str], None] = lambda s: None,
) -> List[Dict[str, Any]]:
    """Score every model-integrity system on every model in the corpus."""
    ref_model: Optional[TinyDetector] = None
    for e in registry:
        if e["manifest"]["ground_truth"]["kind"] == "clean":
            ref_model = ModelArtifact.load(e["dir"]).model
            break

    # Pass 1: raw statistics, computed once and shared by every system.
    rows: List[Dict[str, Any]] = []
    clean_peaks: Dict[str, List[float]] = {"ctc": [], "refdiv": [], "ftc": []}
    for e in registry:
        m = e["manifest"]
        spec = train_spec_from_manifest(m)
        kind = m["ground_truth"]["kind"]
        art = ModelArtifact.load(e["dir"])

        if spec is None:
            # Task 3 real-backbone corpus: no synthetic TrainSpec exists and no
            # trigger view applies -- model attacks have no test-time trigger, so
            # "triggered" is the same held-out CIFAR split by construction.
            eval_ds = real_backbone_eval_split()
            clean_imgs = eval_ds.images
            trig_imgs = clean_imgs
        else:
            splits = build_splits(spec)
            clean_imgs = splits.eval_clean.images
            if kind in ("clean", "clean_label", "label_flip", "dup_flood", "ood_insert"):
                trig_imgs = clean_imgs
            else:
                trig_imgs, _ = trigger_view(splits.eval_clean, spec.attack, seed)

        sig: Dict[str, Dict[str, np.ndarray]] = {
            "ctc": {"clean": trace_ctc(art.model, clean_imgs, backgrounds)["score"],
                    "trig": trace_ctc(art.model, trig_imgs, backgrounds)["score"]},
            # The reference-free cloaking probe. Included here because the comparison
            # is the measurement instrument: a detector that is not in this table has
            # not had its power measured, whatever the design document says.
            "ftc": {"clean": trace_ftc(art.model, clean_imgs)["score"],
                    "trig": trace_ftc(art.model, trig_imgs)["score"]},
        }
        if ref_model is not None:
            sig["refdiv"] = {
                "clean": reference_divergence(art.model, ref_model, clean_imgs)["score"],
                "trig": reference_divergence(art.model, ref_model, trig_imgs)["score"],
            }

        is_attacked = kind in MODEL_ATTACK_KINDS and not m["quality_flags"]["backdoor_weak"]
        row = {
            "model_id": m["model_id"], "kind": kind,
            "weights_digest": m["artifact"]["weights_digest"],
            "asr": m["metrics"]["attack_success_rate"]["asr"],
            "backdoor_weak": m["quality_flags"]["backdoor_weak"],
            "applicable": bool(m["metrics"]["attack_success_rate"].get("applicable", False)),
            "ground_truth_model_attacked": is_attacked,
            "raw_asr_gate_warning": "raw manifest ASR is not null-subtracted; evaluate paired clean-model response",
            "signals": {k: {"clean": v["clean"].tolist(), "trig": v["trig"].tolist()}
                        for k, v in sig.items()},
        }
        # Baseline thresholds are placed on the CLEAN control of this corpus, which
        # is what a reference battery would give a baseline too.
        for k in sig:
            c = np.asarray(sig[k]["clean"], np.float64)
            c = c[np.isfinite(c)]
            if c.size:
                clean_peaks.setdefault(k, []).append(float(np.max(c)) if c.size else 0.0)

        # ---- CVIAF: conformal, two-sided-calibrated, fused, one p per asset
        p_clean: Dict[str, np.ndarray] = {}
        p_trig: Dict[str, np.ndarray] = {}
        for k, v in sig.items():
            cal = np.asarray(v["clean"], np.float64)
            cal = cal[np.isfinite(cal)]
            if cal.size < 5:
                continue
            with np.errstate(invalid="ignore"):
                p_clean[k] = np.where(np.isfinite(v["clean"]),
                                      conformal_pvalues(cal, v["clean"]), 1.0)
                p_trig[k] = np.where(np.isfinite(v["trig"]),
                                     conformal_pvalues(cal, v["trig"]), 1.0)
        if p_clean:
            m_signals = len(p_clean)
            fc = np.stack([p_clean[k] for k in sorted(p_clean)], axis=1)
            ft = np.stack([p_trig[k] for k in sorted(p_trig)], axis=1)
            fused_trig = np.clip(m_signals * ft.min(axis=1), 0.0, 1.0)
            asset_p, combo = asset_pvalue_from_items(fused_trig, method="cauchy")
            row["cviaf"] = {
                "asset_pvalue": asset_p,
                "combination": combo,
                "asset_rule": "legacy_image_cauchy_diagnostic_only",
                "calibrated_model_asset_pvalue": None,
                "abstain_reason": "independent stamped-clean model-asset null unavailable",
                "n_signals": m_signals,
                "reject_at_alpha": False,
                "abstain": True,
                "attack_class_guess": "",
                "per_image_p_min": float(np.nanmin(fused_trig)) if fused_trig.size else None,
            }
        else:
            row["cviaf"] = {"asset_pvalue": 1.0, "abstain": True,
                            "reject_at_alpha": False, "n_signals": 0,
                            "combination": {"reason": "no calibratable signal"}}
        rows.append(row)
        log(f"    model-axis {m['model_id']:34s} kind={kind:11s} "
            f"p={row['cviaf']['asset_pvalue']:.3g}")

    # Pass 2: place the two data-driven baseline cuts, then decide.
    pooled_clean = np.asarray([v for v in clean_peaks.get("ctc", [])], np.float64)
    # NOTE: a single-signal baseline on the CTC peak is used for every baseline row,
    # which is the strongest available single signal for FP-inducing attacks. Giving the
    # baseline our best detector rather than a convenient one is deliberate.
    fixed = ThresholdedSingleSignal(signal="ctc", threshold_mode="fixed", fixed_value=0.5)
    quant = ThresholdedSingleSignal(signal="ctc", threshold_mode="quantile",
                                    quantile=0.99, reference=pooled_clean)
    tuned = ThresholdedSingleSignal(signal="ctc", threshold_mode="fixed")
    tuned.tune_on(pooled_clean, target_fpr=0.05)

    for row in rows:
        trig = row["signals"]["ctc"]["trig"]
        row["verdicts"] = {}
        for tag, sys_ in (("fixed_threshold_single_signal", fixed),
                          ("quantile_threshold_single_signal", quant),
                          ("hand_tuned_single_signal", tuned)):
            v = sys_.assess_items(row["model_id"], trig, "model_integrity",
                                  attack_class="model_backdoor", item_label="image",
                                  extras={"corpus_clean_peak_reference": float(
                                      np.max(pooled_clean)) if pooled_clean.size else None})
            row["verdicts"][tag] = v.to_dict()
        v = Verdict(
            system="cviaf", asset=row["model_id"], axis="model_integrity",
            flagged=bool(row["cviaf"]["reject_at_alpha"]),
            abstained=bool(row["cviaf"].get("abstain", False)),
            attack_class=row["cviaf"].get("attack_class_guess", ""),
            score=row["cviaf"].get("calibrated_model_asset_pvalue"), threshold=alpha,
            confidence=None,
            reason=("model-level stamped-clean null unavailable; legacy image-level "
                    "p-value is diagnostic only and cannot convict a model"),
            evidence={"asset_pvalue": row["cviaf"]["asset_pvalue"],
                      "combination": row["cviaf"]["combination"],
                      "asr_gate_excluded": bool(row["backdoor_weak"])},
        )
        row["verdicts"]["cviaf"] = v.to_dict()
    return rows


# --------------------------------------------------------------------------- #
# the data axis
# --------------------------------------------------------------------------- #

def build_reference_pool(
    scene: "SceneSpec",
    n_per_block: int,
    blocks: int,
    seed_base: int = 900_000,
    include_global_chroma: bool = False,
    log: Callable[[str], None] = lambda s: None,
) -> Dict[str, np.ndarray]:
    """A pooled clean reference: the calibration budget, derived rather than guessed.

    Two things have to be true at once, and getting either wrong makes the
    item-level decision vacuous:

    1. **Exchangeability.** A conformal p-value is only valid if the calibration
       items are exchangeable with the test items. For the duplicate statistic that
       is not automatic: the score is the distance to the nearest neighbour *within a
       split*, so a reference split of 9600 items has systematically smaller nearest
       distances than a 240-item contribution, and calibrating one against the other
       would be comparing two different statistics. The fix is to draw the reference
       as many clean splits of the SAME SIZE as a contribution and pool the
       within-split scores. Same statistic, same size, more items.

    2. **Floor.** A conformal p-value cannot fall below ``1/(n_cal+1)``. An item-level
       Benjamini-Hochberg decision over ``n`` items at level ``alpha`` needs its most
       significant item below ``alpha/n``, so the calibration split must exceed
       ``n/alpha`` items -- and, with m fused signals, m times that again, because the
       Bonferroni fusion multiplies the smallest p by the number of signals.
       For a 240-sample contribution at alpha=0.05 with two signals
       that is about 9,600 calibration items. This is not a tuning choice; it is the information a finite calibration
       set has to contain for the decision to be possible at all, and the number is
       computed from that requirement below rather than picked.
    """
    from cviaf.utils import extract_features_from_images

    feats, spec_scores, dup_scores, patch_scores, global_scores = [], [], [], [], []
    for b in range(blocks):
        ds = build_dataset(n_per_block, scene, seed_offset=seed_base + b * 9973)
        f = extract_features_from_images(ds.images, method="pixel_stats", target_dim=64)
        feats.append(f)
        spec_scores.append(spectral_signature_scores(f, _sample_classes(ds.labels)))
        dup_scores.append(duplicate_scores(ds.images))
        patch_scores.append(patch_local_scores(ds.images))
        if include_global_chroma:
            global_scores.append(global_chroma_scores(ds.images))
    pool = {
        "features": np.concatenate(feats),
        "spectral": np.concatenate(spec_scores),
        "duplicate": np.concatenate(dup_scores),
        "patch_local": np.concatenate(patch_scores),
        "n_blocks": blocks, "n_per_block": n_per_block,
        "n_total": int(sum(len(x) for x in feats)),
    }
    if include_global_chroma:
        pool["global_chroma"] = np.concatenate(global_scores)
    log(f"    reference pool: {pool['n_total']} clean samples "
        f"({blocks} splits of {n_per_block}) "
        f"conformal floor {1.0/(pool['n_total']+1):.2e}")
    return pool


def data_axis(
    registry: List[Dict[str, Any]],
    alpha: float = 0.05,
    include_global_chroma: bool = False,
    review_costs: Optional[OperatorCosts] = None,
    review_lfdr_max: float = 0.5,
    label_gate=None,
    log: Callable[[str], None] = lambda s: None,
) -> List[Dict[str, Any]]:
    """Score every data-integrity system on every contributed dataset.

    Both systems consume the same statistics over the same pooled reference, so
    any difference between their rows is a difference in decision rule. That is the
    whole point of the comparison: it is not "our detectors are better", it is
    "given the same evidence, calibration and multiplicity control change the
    answer".
    """
    from cviaf.utils import extract_features_from_images

    clean_entry = next((e for e in registry
                        if e["manifest"]["ground_truth"]["kind"] == "clean"), None)
    # Task 3: a real-backbone corpus has no synthetic scene and its contributions are
    # the CIFAR training split, so the synthetic reference pool cannot be built and
    # every dataset row abstains by construction (declared, not silently dropped).
    rb_corpus = clean_entry is not None and (
        train_spec_from_manifest(clean_entry["manifest"]) is None)
    if rb_corpus:
        return []
    n_asset = int(clean_entry["manifest"]["spec"]["n_train"]) if clean_entry else 240
    n_signals = 4 if include_global_chroma else 3
    # blocks so that blocks * n_asset >= 2 * n_signals * n_asset / alpha  (see docstring)
    blocks = int(np.ceil(2.0 * n_signals / alpha))
    scene_spec = (train_spec_from_manifest(clean_entry["manifest"]).scene
                  if clean_entry else None)
    pool = (build_reference_pool(scene_spec, n_asset, blocks,
                                 include_global_chroma=include_global_chroma, log=log)
            if scene_spec is not None else None)

    # This lane is deliberately separate from spectral/duplicate fusion. Its
    # trusted reference and calibration are built once per scene/asset size.
    label_references: Dict[Tuple[str, int], Any] = {}
    rows: List[Dict[str, Any]] = []
    for e in registry:
        m = e["manifest"]
        kind = m["ground_truth"]["kind"]
        spec = train_spec_from_manifest(m)
        splits = build_splits(spec)
        ds, truth = splits.train_poisoned, splits.truth
        # Calibration is valid only for a declared acquisition stratum. Never
        # interpret an unrecognized shift as a trigger simply because it is rare
        # under the enrolled clean reference. The scene declaration is trusted
        # in this synthetic lab, not a verified claim from a real vendor.
        scene_match = (scene_spec is not None and
                       spec.scene.to_dict() == scene_spec.to_dict())

        feats = extract_features_from_images(ds.images, method="pixel_stats", target_dim=64)
        dup = duplicate_scores(ds.images)
        try:
            spectral = spectral_signature_scores(feats, _sample_classes(ds.labels))
        except Exception:
            spectral = np.zeros(len(ds))

        poisoned = np.zeros(len(ds), bool)
        poisoned[np.asarray(truth.poisoned_indices, np.int64)] = True
        contributes = kind in DATA_ATTACK_KINDS and poisoned.any()
        contributors = [str(c) for c in ds.contributors]
        cum: Dict[str, int] = {}
        for c in contributors:
            cum[c] = cum.get(c, 0) + 1

        # ---- baseline: one statistic, cut at a reference quantile, per sample.
        # The quantile is the careful-engineer version of "3 sigma". Note what it
        # buys and what it does not: the cut is correctly placed for a SINGLE sample,
        # so roughly 1% of clean samples exceed it -- which, over a 240-sample
        # contribution, means a clean dataset triggers an alarm most of the time.
        # That is not a badly chosen constant; it is what per-item thresholding
        # without multiplicity control does.
        base_flags = np.zeros(len(ds), bool)
        base_v: Optional[Dict[str, Any]] = None
        if pool is not None and scene_match:
            mahal = _mahalanobis(pool["features"], feats)
            ref_mahal = _mahalanobis(pool["features"], pool["features"])
            cut = float(np.quantile(ref_mahal, 0.99))
            base_flags = mahal > cut
            base = PerSampleOutlierBaseline(cut=cut)
            v = base.assess_items(
                m["model_id"], mahal, "data_integrity",
                attack_class="outlier_or_trigger", item_label="sample",
                extras={"n_poisoned_ground_truth": int(poisoned.sum()),
                        "cut": cut, "cut_basis": "99th percentile of the reference "
                                                   "Mahalanobis distances"})
            base_v = v.to_dict()

        # ---- CVIAF: calibrated, fused, FDR-controlled, contributor-attributed
        signals: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        if pool is not None and scene_match:
            signals["spectral"] = (pool["spectral"], spectral)
            # duplicate_scores returns a DISTANCE: 0 means an exact duplicate, so on
            # this signal the anomaly is the low tail. Declared, not assumed.
            signals["duplicate"] = (pool["duplicate"], dup)
            signals["patch_local"] = (pool["patch_local"], patch_local_scores(ds.images))
            if include_global_chroma:
                signals["global_chroma"] = (pool["global_chroma"], global_chroma_scores(ds.images))
        cviaf_v = cviaf_data_asset_verdict(
            m["model_id"], signals, alpha=alpha, contributor_labels=contributors,
            lower_is_anomalous=("duplicate",),
            label_gate=label_gate, dataset=ds if label_gate is not None else None)
        if not scene_match:
            cviaf_v.reason = "not assessed: declared acquisition stratum differs from clean reference"
            cviaf_v.evidence["calibration_status"] = "stratum_mismatch"

        fused_p, per_sig_p = _cviaf_item_pvalues(signals)
        cv_item_flags = _cviaf_item_flags(signals, alpha, n_items=len(ds))
        review_block = None
        if pool is not None and scene_match and fused_p is not None:
            from cviaf.lab.calibrate import conformal_threshold
            n_cal, n_items, m_sig = int(pool["n_total"]), len(ds), len(per_sig_p)
            floor_p = m_sig / (n_cal + 1.0)
            accept_permitted = bool(floor_p <= alpha)
            abstain_reason = "" if accept_permitted else (
                f"calibration floor {floor_p:.3g} > alpha {alpha}")
            dup_dist, dup_nn = duplicate_scores(ds.images, return_neighbors=True)
            dup_floor = conformal_threshold(pool["duplicate"], alpha=alpha / n_items,
                                             higher_is_more_anomalous=False)
            plan = plan_review(fused_p, costs=review_costs or OperatorCosts(),
                               per_signal_p=per_sig_p, accept_permitted=accept_permitted,
                               abstain_reason=abstain_reason, flagged=cv_item_flags,
                               alpha=alpha, lfdr_max=review_lfdr_max,
                               neighbor_index=dup_nn, neighbor_dist=dup_dist,
                               cluster_floor_dist=float(dup_floor))
            review_block = {"plan_summary": plan.summary(),
                            "budget_report": review_budget_report(plan, poisoned, cv_item_flags)}
        label_lane: Dict[str, Any]
        try:
            scene_key = (json.dumps(spec.scene.to_dict(), sort_keys=True), len(ds))
            if scene_key not in label_references:
                label_references[scene_key] = build_label_reference(
                    spec.scene, n_per_block=len(ds), blocks=40, train_images=1200)
            ref = label_references[scene_key]
            if set(map(str, ds.contributors)) & (
                    set(ref.reference_contributors) | set(ref.calibration_contributors)):
                raise ValueError("reference/test contributor overlap")
            label_scores, label_details = sample_loss_scores(
                ref.model, ds.images, ds.boxes, ds.labels, return_evidence=True)
            decision = calibrated_label_decisions(ref, label_scores, alpha=alpha)
            if decision["status"] != "scored":
                label_lane = decision
            else:
                p = decision["item_p"]
                label_lane = {
                    "status": "scored", "asset_p": decision["asset_p"],
                    "asset_flag": bool(decision["asset_flag"]),
                    "asset_statistic": decision["asset_statistic"],
                    "asset_statistic_name": decision["asset_statistic_name"],
                    "calibration_floor": decision["conformal_floor"],
                    "calibration_unit": decision["calibration_unit"],
                    "calibration_blocks": ref.calibration_blocks,
                    "calibration_digest": ref.calibration_digest,
                    "reference_train_digest": ref.train_digest,
                    "item_decision_resolution_possible": decision["item_decision_resolution_possible"],
                    "item": _item_rates(decision["item_flags"], poisoned),
                    "rank_at_truth_k": int(poisoned[np.argsort(-label_scores)
                        [:int(poisoned.sum())]].sum()) if poisoned.any() else None,
                    "sample_evidence": [dict(index=i, score=(float(label_scores[i])
                        if np.isfinite(label_scores[i]) else None), p=float(p[i]),
                        **label_details[i]) for i in range(len(ds))],
                    "source_patterns": summarize_label_patterns(label_details, contributors, p),
                }
        except Exception as exc:
            label_lane = {"status": "unavailable", "reason": str(exc)}
        rows.append({
            "label_flip_lane": label_lane,
            "calibration_stratum_match": bool(scene_match),
            "declared_scene": spec.scene.to_dict(),
            "calibration_abstain_reason": None if scene_match else "declared acquisition stratum differs from reference",
            "model_id": m["model_id"], "kind": kind,
            "n_samples": int(len(ds)),
            "n_poisoned_ground_truth": int(poisoned.sum()),
            "rate_actual": float(truth.rate_actual),
            "contributes_poison": bool(contributes),
            "mal_contributor": spec.attack.mal_contributor,
            "culprit_well_defined": kind in KINDS_WITH_ONE_CULPRIT,
            "contributor_ground_truth": {c: int(v) for c, v in cum.items()},
            "baseline_per_sample": base_v,
            "baseline_detected_samples": int(base_flags.sum()),
            "baseline_item": _item_rates(base_flags, poisoned),
            "cviaf": cviaf_v.to_dict(),
            "cviaf_item": _item_rates(cv_item_flags, poisoned),
            "cviaf_item_vs_baseline_disagreement": int(
                np.sum(base_flags != cv_item_flags)),
            "review_policy": review_block,
        })
        log(f"    data-axis  {m['model_id']:34s} kind={kind:11s} "
            f"poisoned={int(poisoned.sum()):4d} "
            f"baseline_flags={int(base_flags.sum()):4d} "
            f"cviaf={cviaf_v.decision:7s} "
            f"cviaf_items={int(cv_item_flags.sum()):4d} "
            f"attr={cviaf_v.attribution}")
    return rows


def _cviaf_item_pvalues(
    signals: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Return Bonferroni-fused and per-signal item p-values for review."""
    cols: Dict[str, np.ndarray] = {}
    for name, (ref, obs) in signals.items():
        if ref is None or obs is None:
            continue
        r = np.asarray(ref, np.float64).ravel()
        r = r[np.isfinite(r)]
        if r.size < 5:
            continue
        o = np.asarray(obs, np.float64).ravel()
        with np.errstate(invalid="ignore"):
            cols[name] = np.where(np.isfinite(o), conformal_pvalues(
                r, o, higher_is_more_anomalous=(name != "duplicate")), 1.0)
    if not cols:
        return None, {}
    stacked = np.stack([cols[k] for k in sorted(cols)], axis=1)
    return np.clip(stacked.shape[1] * stacked.min(axis=1), 0, 1), cols


def _cviaf_item_flags(
    signals: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    alpha: float,
    n_items: int = 0,
) -> np.ndarray:
    """Item-level BH triage mask; no calibration means no flags."""
    from cviaf.lab.calibrate import benjamini_hochberg
    fused, _ = _cviaf_item_pvalues(signals)
    return (benjamini_hochberg(fused, alpha) if fused is not None
            else np.zeros(n_items or next((len(obs) for _, obs in signals.values()
                                           if obs is not None), 0), bool))


def _item_rates(flags: np.ndarray, poisoned: np.ndarray) -> Dict[str, Any]:
    """Sample-level sensitivity and false-positive rate, which is the pair of
    numbers that decides whether an analyst gets a workable triage pile."""
    n_pos = int(poisoned.sum())
    n_neg = int((~poisoned).sum())
    tp = int(np.sum(flags & poisoned))
    fp = int(np.sum(flags & ~poisoned))
    return {
        "n_flagged": int(flags.sum()),
        "true_positive": tp, "false_positive": fp,
        "item_tpr": round(tp / n_pos, 4) if n_pos else None,
        "item_fpr": round(fp / n_neg, 4) if n_neg else None,
        "precision": round(tp / max(tp + fp, 1), 4) if flags.any() else None,
        "n_poisoned": n_pos, "n_clean": n_neg,
    }


def _mahalanobis(ref: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Distance from the reference mean in reference-standardised coordinates."""
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-6
    return np.sqrt(np.sum(((obs - mu) / sd) ** 2, axis=1))


def _sample_classes(labels: Sequence[np.ndarray]) -> np.ndarray:
    """One class id per image, for detectors that score a sample against its class.

    An image with several objects is assigned its majority class; an image with no
    objects is assigned class 0 and carries no spectral signal either way, which is
    the honest treatment -- spectral signatures need a claimed class to test
    against.
    """
    out = []
    for l in labels:
        l = np.asarray(l).ravel()
        out.append(int(np.bincount(l.astype(np.int64)).argmax()) if l.size else 0)
    return np.asarray(out, np.int64)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _score_model_axis(rows: List[Dict[str, Any]], system: str) -> Dict[str, Any]:
    pos = [r for r in rows if r["ground_truth_model_attacked"]]
    neg = [r for r in rows
           if not r["ground_truth_model_attacked"] and not r["backdoor_weak"]]
    # Models excluded by the ASR gate are a third category, not a false positive:
    # nothing was implanted, so there is nothing to detect. They are reported.
    gated = [r for r in rows if r["backdoor_weak"]]

    def dec(r):
        return r["verdicts"][system]["decision"]

    tp = sum(1 for r in pos if dec(r) == "flag")
    fn = sum(1 for r in pos if dec(r) == "clear")
    abst = sum(1 for r in pos if dec(r) == "abstain")
    fp = sum(1 for r in neg if dec(r) == "flag")
    tn = sum(1 for r in neg if dec(r) == "clear")
    return {
        "n_positive": len(pos), "n_negative": len(neg), "n_asr_gated": len(gated),
        "true_positive": tp, "false_negative": fn, "abstained_on_positive": abst,
        "false_positive": fp, "true_negative": tn,
        "tpr": round(tp / len(pos), 4) if pos else None,
        "fpr": round(fp / len(neg), 4) if neg else None,
        "can_name_attack_class": _can_name_class(rows, system),
        "can_abstain": _can_abstain(rows, system),
        "asr_gated_handling": ("reported as ASR-gated, excluded from scoring"
                               if system == "cviaf" else
                               "silently treated as a negative by a threshold rule"),
    }


def _can_name_class(rows: List[Dict[str, Any]], system: str) -> bool:
    return any(r["verdicts"][system].get("attack_class") for r in rows)


def _can_abstain(rows: List[Dict[str, Any]], system: str) -> bool:
    return any(r["verdicts"][system]["abstained"] for r in rows)


def _pooled_item_rates(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Micro-averaged sample-level rates, pooled over assets, at two granularities:
    the poison that should be found (positive assets) and the clean samples that must
    not be flagged (the clean asset). Kept separate because pooling them would let a
    high false-positive rate hide behind a large amount of true poison.
    """
    pos = [r for r in rows if r[key] and r["n_poisoned_ground_truth"] > 0]
    neg = [r for r in rows if r["n_poisoned_ground_truth"] == 0]
    tp = sum(r[key]["true_positive"] for r in pos)
    npos = sum(r[key]["n_poisoned"] for r in pos)
    fp_on_pos = sum(r[key]["false_positive"] for r in pos)
    nneg_on_pos = sum(r[key]["n_clean"] for r in pos)
    fp = fp_on_pos + sum(r[key]["false_positive"] for r in neg)
    nneg = nneg_on_pos + sum(r[key]["n_clean"] for r in neg)
    return {
        "n_assets_with_poison": len(pos),
        "n_assets_clean": len(neg),
        "item_tpr": round(tp / npos, 4) if npos else None,
        "item_fpr": round(fp / nneg, 4) if nneg else None,
        "item_precision": round(tp / max(tp + fp, 1), 4) if (tp + fp) else None,
        "true_positive": tp, "false_positive": fp,
        "n_poisoned_samples": npos, "n_clean_samples": nneg,
    }


def _score_review_policy(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate the ARM J review-policy blocks across assets.

    The comparison that matters, stated before any number: today's behaviour is
    an UNORDERED flag pile -- its expected catch at budget B is linear at the
    pile's precision. ARM J is a ranked, costed queue with co-flag clustering.
    Both are measured on identical evidence per asset.
    """
    scored = [r for r in rows if r.get("review_policy")]
    if not scored:
        return None
    per_asset = []
    ledgers: Dict[str, Dict[str, float]] = {}
    bench = None
    for r in scored:
        rp = r["review_policy"]
        rep = rp["budget_report"]
        s = rp["plan_summary"]
        row_led = {l["policy"]: l for l in rep["ledgers"]}
        entry = {
            "model_id": r["model_id"], "kind": r["kind"],
            "n_items": r["n_samples"],
            "n_poisoned": r["n_poisoned_ground_truth"],
            "n_flagged": r["cviaf_item"]["n_flagged"],
            "n_review_tasks": s["n_review_tasks"],
            "n_quarantined_items": s["n_quarantine"],
            "n_clustered_items": s["n_clustered_items"],
            "accept_permitted": s["accept_permitted"],
            "budget_at_90pct": rep["budget_at_90pct"],
            "arm_caught_total": rep["arm_j_curve"]["caught"][-1],
            "flag_all_pile_size": rep["flag_all_curve"]["n_flagged"],
            "flag_all_precision": rep["flag_all_curve"]["precision"],
            "realised_cost": {k: v["realised_cost"] for k, v in row_led.items()},
        }
        per_asset.append(entry)
        # the benchmark the arm was built against: the biggest co-flag pile
        if bench is None or (r["cviaf_item"]["false_positive"]
                             > bench["cviaf_item"]["false_positive"]):
            bench = r
        for name, led in row_led.items():
            agg = ledgers.setdefault(name, {"n_review_actions": 0, "poison_caught": 0,
                                            "clean_quarantined": 0, "poison_missed": 0,
                                            "realised_cost": 0.0})
            for k in agg:
                agg[k] += led[k]
    for agg in ledgers.values():
        agg["realised_cost"] = round(agg["realised_cost"], 3)
    out: Dict[str, Any] = {
        "per_asset": per_asset,
        "pooled_ledgers": ledgers,
        "costs": scored[0]["review_policy"]["plan_summary"]["costs"],
    }
    if bench is not None:
        out["benchmark_coflag_case"] = {
            "model_id": bench["model_id"], "kind": bench["kind"],
            "note": ("largest co-flag pile in the corpus: nearest-neighbour "
                     "duplicate scoring flags the flood copies AND their "
                     "innocent originals; this is the case the queue must "
                     "rationalise"),
            "flag_all_pile": bench["review_policy"]["budget_report"]["flag_all_curve"],
            "arm_j_curve": bench["review_policy"]["budget_report"]["arm_j_curve"],
            "budget_at_90pct": bench["review_policy"]["budget_report"]["budget_at_90pct"],
            "ledgers": bench["review_policy"]["budget_report"]["ledgers"],
            "plan_summary": bench["review_policy"]["plan_summary"],
        }
    return out


def _score_data_axis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        # Task 3 real-backbone corpus: the data axis is structurally inapplicable
        # (no synthetic scene, no per-sample poison labels), so the axis is reported
        # as abstained rather than scored 0/0 which would read as a perfect FPR.
        return {"n_positive": 0, "n_negative": 0, "baseline_tpr": None,
                "baseline_fpr": None, "cviaf_tpr": None, "cviaf_fpr": None,
                "cviaf_abstained": 0,
                "axis_status": "not_assessed: real-backbone corpus has no synthetic "
                               "data axis (model-attack corpus; datasets are clean by "
                               "construction)",
                "baseline_item_level": {}, "cviaf_item_level": {},
                "attribution": {"n_scored": 0}, "per_asset": []}
    pos = [r for r in rows if r["contributes_poison"]]
    neg = [r for r in rows if not r["contributes_poison"]]
    base_flag = [r for r in pos if r["baseline_detected_samples"] > 0]
    cv_flag = [r for r in pos if r["cviaf"]["flagged"]]
    cv_abst = [r for r in pos if r["cviaf"]["abstained"]]
    base_fp = [r for r in neg if r["baseline_detected_samples"] > 0]
    cv_fp = [r for r in neg if r["cviaf"]["flagged"]]

    culprit = [r for r in pos if r["culprit_well_defined"]]
    diffuse = [r for r in pos if not r["culprit_well_defined"]]
    top1 = sum(1 for r in culprit
               if r["cviaf"].get("attribution")
               and r["cviaf"]["attribution"][0] == r["mal_contributor"])
    false_accusations = sum(1 for r in diffuse if r["cviaf"].get("attribution"))

    base_item = _pooled_item_rates(rows, "baseline_item")
    cv_item = _pooled_item_rates(rows, "cviaf_item")
    return {
        "n_positive": len(pos), "n_negative": len(neg),
        "baseline_tpr": round(len(base_flag) / len(pos), 4) if pos else None,
        "baseline_fpr": round(len(base_fp) / len(neg), 4) if neg else None,
        "cviaf_tpr": round(len(cv_flag) / len(pos), 4) if pos else None,
        "cviaf_fpr": round(len(cv_fp) / len(neg), 4) if neg else None,
        "cviaf_abstained": len(cv_abst),
        "baseline_item_level": base_item,
        "cviaf_item_level": cv_item,
        "item_level_verdict": (
            "Sample-level decisions are where the calibration budget shows. The "
            "baseline's cut is correctly placed for ONE sample, so ~1% of clean "
            "samples exceed it -- over a few hundred samples that means a clean "
            "submission alarms nearly every time, and no rejection set is controlled. "
            "CVIAF's item-level BH decision is an FDR claim, and where the conformal "
            "floor makes that impossible the report says so instead of returning a "
            "ranked list that looks like a detection."),
        "attribution": {
            "kinds_with_one_culprit": list(KINDS_WITH_ONE_CULPRIT),
            "n_scored": len(culprit),
            "baseline_top1_correct": None,
            "baseline_note": "a per-sample flag list has no source field, so top-1 "
                             "attribution is not merely wrong -- it is undefined",
            "cviaf_top1_correct": top1,
            "cviaf_top1_accuracy": round(top1 / len(culprit), 4) if culprit else None,
            "kinds_without_single_culprit": sorted({r["kind"] for r in diffuse}),
            "cviaf_false_accusations": false_accusations,
            "cviaf_false_accusation_rate": (
                round(false_accusations / len(diffuse), 4) if diffuse else None),
            "note": ("Trigger attacks are contributor-diffuse by construction, so on "
                     "those the measurable question is the opposite one: does the "
                     "framework avoid accusing a source? Both numbers are reported "
                     "because a system that names a culprit every time would score "
                     "a perfect top-1 and be useless."),
        },
        "per_asset": [{
            "model_id": r["model_id"], "kind": r["kind"],
            "n_poisoned": r["n_poisoned_ground_truth"],
            "baseline_flagged": r["baseline_detected_samples"],
            "baseline_item_tpr": r["baseline_item"]["item_tpr"],
            "baseline_item_fpr": r["baseline_item"]["item_fpr"],
            "cviaf_decision": r["cviaf"]["decision"],
            "cviaf_flagged": r["cviaf_item"]["n_flagged"],
            "cviaf_item_tpr": r["cviaf_item"]["item_tpr"],
            "cviaf_item_fpr": r["cviaf_item"]["item_fpr"],
            "cviaf_attribution": r["cviaf"].get("attribution", []),
        } for r in rows],
        "review_queue": _score_review_policy(rows),
    }


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #

def compare_corpus(
    corpus_dir: str,
    alpha: float = 0.05,
    seed: int = 1,
    n_backgrounds: int = 8,
    max_models: Optional[int] = None,
    key_dir: Optional[str] = None,
    review_costs: Optional[OperatorCosts] = None,
    label_gate=None,
    label_gate_protocol_sha256: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Run every baseline and CVIAF over one corpus and return the comparison."""
    t0 = time.time()
    registry = load_registry(corpus_dir)
    if not registry:
        raise RuntimeError(f"no models found under {corpus_dir}/")
    if max_models:
        registry = registry[:max_models]

    backgrounds = make_backgrounds(n_backgrounds, seed=seed)
    log(f"comparison: {len(registry)} models from {corpus_dir}, "
        f"alpha={alpha}, {n_backgrounds} CTC backgrounds")

    log("  [1/3] model integrity axis")
    mrows = model_axis(registry, backgrounds, alpha=alpha, seed=seed, log=log)
    log("  [2/3] data integrity axis")
    drows = data_axis(registry, alpha=alpha, review_costs=review_costs,
                      label_gate=label_gate, log=log)
    log("  [3/3] inference provenance axis")
    prows = provenance_axis(log=log, key_dir=key_dir)

    systems = list(BASELINE_NAMES) + ["cviaf"]
    result: Dict[str, Any] = {
        "corpus": corpus_dir, "alpha": alpha, "n_models": len(registry),
        "seed": seed, "n_backgrounds": n_backgrounds,
        "label_gate": {"enabled": label_gate is not None,
                       "protocol_sha256": label_gate_protocol_sha256,
                       "scope": "synthetic matched-declared-domain asset decision only"},
        "systems": systems,
        "model_axis": {
            "scores": {s: _score_model_axis(mrows, s) for s in
                       ["fixed_threshold_single_signal",
                        "quantile_threshold_single_signal",
                        "hand_tuned_single_signal", "cviaf"]},
            "per_model": mrows,
        },
        "data_axis": {"scores": _score_data_axis(drows), "per_model": drows},
        "provenance_axis": prows,
        "capability_matrix": _capability_matrix(mrows, drows, prows),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    result["verdict"] = _verdict(result)
    log(f"comparison complete in {result['elapsed_seconds']}s")
    return result


def provenance_axis(log: Callable[[str], None] = print,
                    key_dir: Optional[str] = None) -> Dict[str, Any]:
    """Score hash-only verification against signature + replay controls.

    Records were produced by a sealing service and an adversary with write access to
    the record store did the three things an adversary does: altered an output,
    re-attributed a record to another model, and re-submitted an intact old record as
    new. In every alteration the adversary *recomputes the record's unkeyed hashes*,
    because altering a record without fixing its own self-check is a mistake nobody
    makes twice.

    Two hash-only mechanisms are scored, and the stronger one is the headline:

    ``enrolment comparison``   recompute and compare against a digest held
                               out-of-band. Catches both alterations; cannot see the
                               replay, whose stored bytes are perfectly genuine; and
                               has nothing to compare for a record it has never seen,
                               so it abstains on genuinely new submissions.
    ``internal consistency``   recompute the record's own hashes. Catches nothing in
                               this scenario, because the adversary recomputed them
                               too. This is the mechanism whose weakness is not
                               obvious from the outside, so it is measured separately
                               rather than argued about.
    """
    from cviaf.lab.pipeline import build_provenance_scenario

    scen = build_provenance_scenario(n_history=6, n_per_attack=2, log=log,
                                     key_dir=key_dir)
    enrolment = DigestManifestBaseline()
    for h in scen["history"]:
        enrolment.enroll(h["seal_id"], h["seal_hash"])

    verdicts: List[Dict[str, Any]] = []
    for rec in scen["records"]:
        truth = rec["truth"]
        v_enrol = enrolment.assess_digest(
            rec["seal_id"], rec["observed"]["record_hash"],
            "inference_provenance",
            attack_class=(rec["cviaf_verification"].get("attack_class") or ""),
            note="enrolment digest: ")
        v_enrol.evidence["adversary_recomputed_hash"] = bool(
            rec["adversary_recomputed_hash"])
        v_internal = Verdict(
            system="digest_manifest_internal_consistency", asset=rec["seal_id"],
            axis="inference_provenance",
            flagged=not bool(rec["observed"]["stored_hash_self_consistent"]),
            attack_class=("inference_tampering"
                          if not rec["observed"]["stored_hash_self_consistent"] else ""),
            reason=("record's own hashes recompute consistently, which a forger "
                    "who recomputes them achieves for free"
                    if rec["observed"]["stored_hash_self_consistent"]
                    else "record's own hashes do not recompute"),
            evidence={"mechanism": "internal consistency only, no trust anchor"},
        )
        verdicts.append({
            "seal_id": rec["seal_id"], "truth": truth, "note": rec["note"],
            "baseline": v_enrol.to_dict(),
            "baseline_internal_consistency": v_internal.to_dict(),
            "cviaf": rec["cviaf_verification"],
        })

    attacked = [v for v in verdicts if v["truth"] != "genuine"]
    genuine = [v for v in verdicts if v["truth"] == "genuine"]

    def agg(system: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        flagged = sum(1 for v in rows if v[system]["flagged"])
        abstained = sum(1 for v in rows if v[system].get("abstained"))
        return {"n": len(rows), "n_flagged": flagged, "n_abstained": abstained,
                "rate": round(flagged / len(rows), 4) if rows else None}

    out = {
        "n_genuine": len(genuine), "n_attacked": len(attacked),
        "scenario": {
            "n_history": scen["n_history"],
            "max_history_sequence": scen["max_history_sequence"],
            "adversary_model": scen["assumption"],
            "record_hashes_are_unkeyed": (
                "every hash a hash-only checker recomputes is unkeyed, so an "
                "adversary with write access can make a forged record fully "
                "self-consistent; the signature is the only field they cannot forge"),
        },
        "baseline_digest_manifest": {
            "mechanism": "recompute then compare against an out-of-band enrolled digest",
            "tpr": agg("baseline", attacked)["rate"],
            "fpr": agg("baseline", genuine)["rate"],
            "abstained_on_attacked": agg("baseline", attacked)["n_abstained"],
            "abstained_on_genuine": agg("baseline", genuine)["n_abstained"],
            "structural_blind_spots": [
                "a record it has never enrolled has no trust anchor, so the check "
                "abstains rather than assessing -- a forged new record passes for "
                "want of a comparison, not for want of a difference",
                "a replayed record is byte-identical and genuine, so the digest "
                "matches exactly and the substitution is invisible",
            ],
            "per_attack": _per_attack(verdicts, "baseline"),
        },
        "baseline_internal_consistency": {
            "mechanism": "recompute the record's own hashes only",
            "tpr": agg("baseline_internal_consistency", attacked)["rate"],
            "fpr": agg("baseline_internal_consistency", genuine)["rate"],
            "per_attack": _per_attack(verdicts, "baseline_internal_consistency"),
        },
        "cviaf": {
            "mechanism": ("signature over the bound payload + nonce/sequence replay "
                          "controls + chain verification"),
            "tpr": agg("cviaf", attacked)["rate"],
            "fpr": agg("cviaf", genuine)["rate"],
            "abstained_on_attacked": agg("cviaf", attacked)["n_abstained"],
            "attack_class_named": sorted(
                {v["cviaf"].get("attack_class") for v in attacked
                 if v["cviaf"].get("attack_class")}),
            "per_attack": _per_attack(verdicts, "cviaf"),
        },
        "signing_mode": scen.get("signing_mode"),
        "assumption": scen.get("signing_mode_note"),
        "per_record": verdicts,
    }
    return out


def _per_attack(verdicts: List[Dict[str, Any]], system: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for v in verdicts:
        key = v["truth"]
        agg = out.setdefault(key, {"n": 0, "n_detected": 0, "n_abstained": 0})
        agg["n"] += 1
        agg["n_detected"] += int(bool(v[system]["flagged"]))
        agg["n_abstained"] += int(bool(v[system].get("abstained")))
    for k, agg in out.items():
        agg["rate"] = round(agg["n_detected"] / agg["n"], 4) if agg["n"] else None
    return out


def _capability_matrix(
    mrows: List[Dict[str, Any]],
    drows: List[Dict[str, Any]],
    prows: Dict[str, Any],
) -> Dict[str, Any]:
    """The qualitative half of the comparison: what each design can *say*.

    Kept separate from the detection numbers on purpose. A threshold can be made
    more sensitive; it cannot be given a validity statement, an abstention, or a
    source attribution without changing the design.
    """
    return {
        "digest_manifest": {
            "calibrated_confidence": False, "error_control_across_assets": False,
            "can_abstain": False, "names_attack_class": False,
            "attributes_to_contributor": False, "detects_post_hoc_modification": True,
            "withstands_attacker_who_recomputes_hash": False,
            "detects_poisoning_or_backdoor": False,
        },
        "fixed_threshold_single_signal": {
            "calibrated_confidence": False, "error_control_across_assets": False,
            "can_abstain": False, "names_attack_class": False,
            "attributes_to_contributor": False, "detects_post_hoc_modification": False,
            "withstands_attacker_who_recomputes_hash": False,
            "detects_poisoning_or_backdoor": True,
            "note": "one signal: blind wherever that signal is structurally blind",
        },
        "hand_tuned_single_signal": {
            "calibrated_confidence": False, "error_control_across_assets": False,
            "can_abstain": False, "names_attack_class": False,
            "attributes_to_contributor": False, "detects_post_hoc_modification": False,
            "withstands_attacker_who_recomputes_hash": False,
            "detects_poisoning_or_backdoor": True,
            "note": "spends hindsight: the cut is tuned on this corpus's clean control",
        },
        "cviaf": {
            "calibrated_confidence": True, "error_control_across_assets": False,
            "can_abstain": True, "names_attack_class": True,
            "attributes_to_contributor": True, "detects_post_hoc_modification": True,
            "withstands_attacker_who_recomputes_hash": True,
            "detects_poisoning_or_backdoor": False,
            "note": "model-backdoor conviction is not established: legacy image scores are "
                    "stamp-confounded, and the model-level rule abstains until an "
                    "independent clean-model null has at least 44 models. Data-axis "
                    "detection is assessed separately.",
        },
    }


def _verdict(res: Dict[str, Any]) -> str:
    m = res["model_axis"]["scores"]
    d = res["data_axis"]["scores"]
    p = res["provenance_axis"]
    lines = [
        "What changes when CVIAF comes in, measured on one corpus",
        "",
        f"model integrity (n={m['cviaf']['n_positive']} attacked, "
        f"{m['cviaf']['n_negative']} negative, "
        f"{m['cviaf']['n_asr_gated']} ASR-gated):",
        f"  hand-tuned single-signal baseline : TPR {m['hand_tuned_single_signal']['tpr']} "
        f"FPR {m['hand_tuned_single_signal']['fpr']}",
        f"  CVIAF (asset rule unavailable)    : {m['cviaf']['n_asr_gated']} ASR-gated; "
        f"remaining models abstain without a 44+ independent clean-model null "
        f"(diagnostic-only legacy image scores)",
        "",
        f"data integrity (n={d['n_positive']} poisoned contributions, "
        f"{d['n_negative']} clean):",
        *(  # real-backbone corpus: the axis abstains as a whole
            [f"  {d.get('axis_status', 'not assessed')}"] if d.get("axis_status") else [
            f"  per-sample Mahalanobis baseline   : asset TPR {d['baseline_tpr']} "
            f"FPR {d['baseline_fpr']}, attribution undefined",
            f"  CVIAF                             : asset TPR {d['cviaf_tpr']} "
            f"FPR {d['cviaf_fpr']}, top-1 source accuracy "
            f"{d['attribution'].get('cviaf_top1_accuracy')}, false accusations "
            f"{d['attribution'].get('cviaf_false_accusation_rate')}",
            f"  sample level, baseline            : TPR "
            f"{d['baseline_item_level'].get('item_tpr')} FPR "
            f"{d['baseline_item_level'].get('item_fpr')} precision "
            f"{d['baseline_item_level'].get('item_precision')}",
            f"  sample level, CVIAF (BH, FDR)     : TPR "
            f"{d['cviaf_item_level'].get('item_tpr')} FPR "
            f"{d['cviaf_item_level'].get('item_fpr')} precision "
            f"{d['cviaf_item_level'].get('item_precision')}"])
        ,
        f"inference provenance (n={p['n_attacked']} attacked, {p['n_genuine']} genuine):",
        f"  hash-only manifest baseline       : TPR {p['baseline_digest_manifest']['tpr']} "
        f"FPR {p['baseline_digest_manifest']['fpr']}",
        f"  CVIAF                             : TPR {p['cviaf']['tpr']} "
        f"FPR {p['cviaf']['fpr']}",
    ]
    return "\n".join(lines)


def render_table(res: Dict[str, Any]) -> str:
    """A compact text table for the terminal."""
    m = res["model_axis"]["scores"]
    d = res["data_axis"]["scores"]
    p = res["provenance_axis"]
    hdr = f"{'system':34s} {'model TPR':>9s} {'model FPR':>9s} {'data TPR':>8s} {'data FPR':>8s} {'prov TPR':>8s} {'attr':>5s} {'abstain':>7s}"
    lines = [hdr, "-" * len(hdr)]
    rows = [
        ("digest_manifest (baseline)", None, None, None, None,
         p["baseline_digest_manifest"]["tpr"], False, False),
        ("fixed_threshold_single_signal", m["fixed_threshold_single_signal"]["tpr"],
         m["fixed_threshold_single_signal"]["fpr"], None, None, None, False, False),
        ("quantile_threshold_single_signal", m["quantile_threshold_single_signal"]["tpr"],
         m["quantile_threshold_single_signal"]["fpr"], None, None, None, False, False),
        ("hand_tuned_single_signal", m["hand_tuned_single_signal"]["tpr"],
         m["hand_tuned_single_signal"]["fpr"], d["baseline_tpr"], d["baseline_fpr"],
         None, False, False),
        ("cviaf", m["cviaf"]["tpr"], m["cviaf"]["fpr"], d["cviaf_tpr"], d["cviaf_fpr"],
         p["cviaf"]["tpr"], True, m["cviaf"]["can_abstain"]),
    ]

    def f(v):
        return "    -   " if v is None else f"{v:>8.3f} "

    for name, mt, mf, dt, df, pt, attr, abst in rows:
        lines.append(f"{name:34s} {f(mt)} {f(mf)} {f(dt)} {f(df)} {f(pt)} "
                     f"{str(attr):>5s} {str(abst):>7s}")
    return "\n".join(lines)


def save_comparison(res: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    return path
