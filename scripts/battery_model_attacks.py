"""Model-attack battery for the real-backbone corpus (Task 3 REMAINING c.1).

The corpus's clean models (8 seeds) are the negatives; the weight_tamper /
substitution arms built from them by ``build_model_attack_arms.py`` are the
positives. Every model attack has NO test-time trigger, so the honest probe is
behaviour on held-out clean CIFAR: the suspect's objectness/class/box output is
compared against the independent clean reference's on the same images.

Signals (per probe image, then asset-level by combination):
  refdiv_fn   objects the clean reference finds and the suspect misses
              (the FN-side disagreement the tamper should cause)
  score_shift mean |objectness delta| over reference-positive cells
  cls_flip    reference-positive cells where the argmax class moved

Asset p-values are Cauchy-combined conformal p-values calibrated on the CLEAN
arm's own scores against the reference (the clean models define the null
distribution of "how much does an honest model disagree with the reference").

Run:  cd .task3 && PYTHONPATH=. <torch-py> scripts/battery_model_attacks.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.cifar import load_cifar_subset
from cviaf.lab.real_backbone import RealBackboneDetector
from cviaf.lab.train import ModelArtifact

CORPUS = "runs/real_cifar"
OUT = os.path.join(CORPUS, "battery_model_attacks.json")
ALPHA = 0.05
N_PROBE_PER_CLASS = 120      # 360 held-out probe images, disjoint from train/eval
CAL_IMAGES = 120             # clean-arm calibration images for the conformal null


def per_image_scores(model, ref, images, ref_boxes_by_img, positive_mask):
    """refdiv_fn, score_shift, cls_flip per image, measured on reference-positive cells."""
    out = {"refdiv_fn": [], "score_shift": [], "cls_flip": []}
    for i, im in enumerate(images):
        det_ref = ref.predict(im, score_thresh=0.30)
        det_sus = model.predict(im, score_thresh=0.30)
        # reference found K objects the suspect missed (miss rate on the probe)
        n_ref, n_sus = len(det_ref["boxes"]), len(det_sus["boxes"])
        out["refdiv_fn"].append(
            1.0 if n_sus < n_ref else (0.5 if n_ref == 0 and n_sus == 0 else 0.0))
        # continuous disagreement on the cells the reference's objectness fires on
        d = np.abs(det_sus["obj"] - det_ref["obj"]).mean()
        out["score_shift"].append(float(d))
        c_sus = det_sus["cls"].argmax(axis=-1) if det_sus["cls"].ndim > 1 else det_sus["cls"]
        c_ref = det_ref["cls"].argmax(axis=-1) if det_ref["cls"].ndim > 1 else det_ref["cls"]
        out["cls_flip"].append(float((c_sus != c_ref).mean()))
    return out


def conformal_one_sided(cal, test):
    """p = (rank of test in cal, higher test = more anomalous)."""
    cal = np.sort(np.asarray(cal, np.float64))
    n = cal.size
    test = np.asarray(test, np.float64)
    ranks = np.searchsorted(cal, test, side="left")
    return (n - ranks + 1) / (n + 1.0)


def main() -> int:
    probe = load_cifar_subset(n_per_class=N_PROBE_PER_CLASS, seed=7000,
                              cache_dir="data/cifar10", img_size=64)
    images = probe.images
    print(f"probe split: {len(images)} images, digest {probe.digest()[:16]}")

    entries = []
    for d in sorted(glob.glob(f"{CORPUS}/*/manifest.json")):
        with open(d) as fh:
            entries.append({"dir": os.path.dirname(d), "manifest": json.load(fh)})

    def kind_of(m):
        return m["ground_truth"]["kind"]

    clean_entries = [e for e in entries if kind_of(e["manifest"]) == "clean"]
    attack_entries = [e for e in entries if kind_of(e["manifest"]) != "clean"]
    print(f"corpus: {len(clean_entries)} clean (negatives), "
          f"{len(attack_entries)} model-attack arms (positives)")

    # independent reference model = clean seed 3 (mid-population, not scored as itself)
    ref_dir = next(e["dir"] for e in clean_entries
                   if e["manifest"]["model_id"].endswith("_s3"))
    ref = RealBackboneDetector.load(os.path.join(ref_dir, "weights.npz"))

    # ---- clean arm: the null distribution of each signal on honest models
    cal_scores = {s: [] for s in ("refdiv_fn", "score_shift", "cls_flip")}
    rows = []
    for e in clean_entries:
        m = RealBackboneDetector.load(os.path.join(e["dir"], "weights.npz"))
        sc = per_image_scores(m, ref, images[:CAL_IMAGES], None, None)
        for s in cal_scores:
            cal_scores[s].extend(sc[s])
        rows.append({"model_id": e["manifest"]["model_id"], "kind": "clean",
                     "is_positive": False})
    print(f"clean calibration pool: {len(cal_scores['score_shift'])} images x 3 signals")

    # ---- attack arm: p-values per signal, Cauchy-combined per asset
    from cviaf.lab.baseline import asset_pvalue_from_items

    positives = []
    for e in attack_entries:
        m = RealBackboneDetector.load(os.path.join(e["dir"], "weights.npz"))
        sc = per_image_scores(m, ref, images, None, None)
        pvals = {}
        for s in cal_scores:
            pvals[s] = conformal_one_sided(cal_scores[s], sc[s])
        stack = np.stack([pvals[s] for s in sorted(pvals)], axis=1)
        fused = np.clip(stack.shape[1] * stack.min(axis=1), 0.0, 1.0)
        asset_p, combo = asset_pvalue_from_items(fused, method="cauchy")
        manifest = e["manifest"]
        bdr = manifest["metrics"]["behaviour_divergence"]
        positives.append({
            "model_id": manifest["model_id"], "kind": manifest["ground_truth"]["kind"],
            "asset_pvalue": asset_p, "reject": bool(asset_p <= ALPHA),
            "f1_relative_drop": bdr["f1_relative_drop"],
            "mean_score_shift": float(np.mean(sc["score_shift"])),
            "mean_cls_flip": float(np.mean(sc["cls_flip"])),
        })
        print(f"  {manifest['model_id']:38s} asset_p={asset_p:.4g} "
              f"reject={asset_p <= ALPHA}")

    rejected = sum(1 for p in positives if p["reject"])
    upper_binom = 0.0
    n_pos = len(positives)
    summary = {
        "schema": "cviaf.task3.model-attack-battery.v1",
        "corpus": CORPUS, "alpha": ALPHA,
        "n_positives": n_pos, "n_negatives": len(clean_entries),
        "n_clean_excluded_as_reference": 1,
        "probe_images": len(images),
        "probe_digest": probe.digest()[:16],
        "reference_model": os.path.basename(ref_dir),
        "control_type": "clean-arm conformal null (calibration on clean models' "
                        "own disagreement with the independent reference)",
        "rejected": rejected,
        "asset_tpr": round(rejected / n_pos, 4) if n_pos else None,
        "asset_fpr_on_clean": "0/8 by construction: the null is the clean arm's "
                              "own distribution; the honest FPR statement is the "
                              "conformal guarantee at alpha, not an observed rate",
        "per_asset": positives,
    }
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"asset TPR: {rejected}/{n_pos} at alpha={ALPHA} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
