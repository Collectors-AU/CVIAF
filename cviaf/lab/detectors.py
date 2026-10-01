"""
Assurance-side detectors, scored against the lab's ground truth.

The headline here is the **TRACE** pair, because it is what makes this framework
detector-native rather than a classifier wrapper:

  ``trace_ctc``  Contextual Transformation Consistency. Blend each test image over
                 many different backgrounds and measure how stable each object's
                 confidence remains. A *triggered* object is abnormally stable,
                 because the trigger dominates the local features and no longer
                 depends on the scene. Clean objects fluctuate, and often vanish
                 entirely, when the background changes.

                 This catches the FP-inducing attacks: object generation (phantom
                 boxes), and regional / global misclassification.

  ``trace_ftc``  Focal Transformation Consistency, the "Island Effect". CTC is
                 blind to object DISAPPEARANCE, because a vanished object leaves no
                 detection whose stability could be measured. FTC paints a decoy
                 object into the scene across a grid of positions and watches for
                 the model's confidence in it to COLLAPSE when the decoy lands on
                 the hidden trigger. Implementations that skip this signal cannot
                 see cloaking -- the attack that makes armour invisible -- and the
                 coverage matrix must mark that cell as not covered if it is missing.

Everything else is a supporting signal: behavioural fingerprinting (substitution),
weight statistics (modification), spectral signatures (poisoned training samples),
duplicate hashing (flooding), and a hierarchical contributor risk posterior.

Honest note carried into the report: this is a faithful reimplementation of the
*mechanisms* at MVP scale, not a reproduction of the published numbers. The
published figures for TRACE come from YOLOv5 / Faster-RCNN / DETR on COCO/VOC. We
cite them as theirs and publish ours separately. See docs/SCALING_PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.detector import TinyDetector, box_iou, sigmoid, softmax
from cviaf.lab.poison import AttackSpec
from cviaf.lab.synth import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    DetectionDataset,
    SceneSpec,
    build_dataset,
    draw_object,
    generate_scene,
)

DEFAULT_SCORE_THRESH = 0.30
DEFAULT_MATCH_IOU = 0.30

BATTERY_DIGEST_SCHEMA = "cviaf.battery-digest.v1"
FINGERPRINT_STATES = ("comparable", "fingerprint_incomparable")


def probe_array_digest(probe_images: np.ndarray) -> str:
    """Content digest of the probe set: dtype, shape and exact bytes.

    Shape and dtype are hashed too, not just the pixel bytes, so a battery re-saved
    at a different resolution cannot collide with the original.
    """
    arr = np.ascontiguousarray(np.asarray(probe_images))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def battery_digest(probe_images: np.ndarray, score_thresh: float = DEFAULT_SCORE_THRESH,
                   iou_thr: Optional[float] = None, n_classes: int = NUM_CLASSES,
                   transform_families: Sequence[str] = (),
                   extra: Optional[Mapping[str, Any]] = None) -> str:
    """SHA-256 of the canonical battery definition a fingerprint is conditioned on.

    A behavioural fingerprint is a function of the battery, not only of the model: the
    same weights scored on a different probe set, threshold or class list produce a
    different vector, and a distance between two such vectors is a number with no
    meaning. Until this existed, nothing stopped exactly that comparison; the problem
    statement asks for a refusal instead (clause 3.5).
    """
    definition = {
        "schema": BATTERY_DIGEST_SCHEMA,
        "probe_digest": probe_array_digest(probe_images),
        "n_probes": int(len(probe_images)),
        "score_thresh": float(score_thresh),
        "iou_thr": None if iou_thr is None else float(iou_thr),
        "n_classes": int(n_classes),
        "transform_families": sorted(str(t) for t in transform_families),
        "extra": dict(extra or {}),
    }
    blob = json.dumps(definition, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    return hashlib.sha256(b"CVIAF battery v1\x00" + blob).hexdigest()


def fingerprint_record(value: np.ndarray, battery: str, model_id: Optional[str] = None,
                       extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A fingerprint that carries the battery it was measured on."""
    if not isinstance(battery, str) or not battery:
        raise ValueError("fingerprint_record requires a non-empty battery digest")
    return {"fingerprint": np.asarray(value, np.float64).tolist(),
            "battery_digest": battery, "model_id": model_id,
            "n_dims": int(np.asarray(value).size), "extra": dict(extra or {})}


def compare_fingerprints(a: Mapping[str, Any], b: Mapping[str, Any],
                         scale: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compare two fingerprint records, REFUSING across differing battery digests.

    Returns ``status == "fingerprint_incomparable"`` with both digests when they
    differ, and also when either record carries no digest at all -- an unconditioned
    fingerprint cannot be compared to anything, and guessing that the two batteries
    were the same is precisely the silent passing this framework is built to avoid.
    Otherwise the scaled L2 distance, as ``fingerprint_distance``.
    """
    da, db = a.get("battery_digest"), b.get("battery_digest")
    missing = [name for name, d in (("a", da), ("b", db))
               if not isinstance(d, str) or not d]
    if missing:
        return {"status": "fingerprint_incomparable",
                "reason": f"record(s) {missing} carry no battery digest, so the two "
                          f"fingerprints are not conditioned on a known battery; "
                          f"re-enrol the reference against this battery",
                "a_battery_digest": da, "b_battery_digest": db}
    if da != db:
        return {"status": "fingerprint_incomparable",
                "reason": "different probe batteries; a distance between these two "
                          "fingerprints would compare the batteries, not the models",
                "a_battery_digest": da, "b_battery_digest": db}
    return {"status": "comparable",
            "distance": fingerprint_distance(np.asarray(a["fingerprint"], np.float64),
                                             np.asarray(b["fingerprint"], np.float64),
                                             scale),
            "scale_used": scale is not None,
            "battery_digest": da,
            "n_dims": int(np.asarray(a["fingerprint"]).size)}


# --------------------------------------------------------------------------- #
# TRACE: CTC
# --------------------------------------------------------------------------- #

def make_backgrounds(n: int, seed: int = 0, terrain: str = "urban") -> np.ndarray:
    """Public background imagery for the CTC transform.

    TRACE needs auxiliary public images. In a hard air gap they are unavailable
    unless pre-vendored, which is a real limitation -- so here they are generated
    from the same procedural generator and ship with the battery, and the scaling
    plan swaps in a vendored image folder. Either way the point stands: if the
    backgrounds are absent, the CTC check must report ``skipped``, never silently
    pass.
    """
    rng = np.random.default_rng(seed + 9090)
    out = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), np.float32)
    for i in range(n):
        s = SceneSpec(terrain=terrain, season="summer", illumination=float(rng.uniform(0.7, 1.2)),
                      gamma=float(rng.uniform(0.8, 1.3)), sensor_noise=0.03, seed=seed + i * 17)
        out[i] = generate_scene(s, n_objects=0, rng=rng, objects=False).image
    return out


def _match_detection(
    det: Dict[str, np.ndarray], box: np.ndarray, label: int, iou_thr: float = DEFAULT_MATCH_IOU
) -> float:
    """Highest score among detections matching (box, label); 0.0 if absent."""
    best = 0.0
    for b, l, s in zip(det["boxes"], det["labels"], det["scores"]):
        if int(l) != int(label):
            continue
        if box_iou(b, box) >= iou_thr:
            best = max(best, float(s))
    return best


def _read_cell(m: np.ndarray, i: int, j: int, r: int = 1) -> float:
    """Max of a (2r+1)^2 neighbourhood of a map, clipped to bounds."""
    G = m.shape[0]
    i0, i1 = max(0, i - r), min(G, i + r + 1)
    j0, j1 = max(0, j - r), min(G, j + r + 1)
    return float(m[i0:i1, j0:j1].max())


def _greedy_match(sus_boxes, sus_labels, ref_boxes, ref_labels,
                  iou_thr: float = 0.5):
    """Class-aware greedy IoU matching, highest overlap first.

    Returns ``(pairs, used_suspect, used_ref)`` where each pair is
    ``(suspect_index, ref_index, iou)``. Greedy-highest-first rather than
    Hungarian: detections are sparse and well separated, the extra optimality would
    change nothing, and greedy is unambiguous to describe in the report.
    """
    cand = []
    for i, (b, l) in enumerate(zip(sus_boxes, sus_labels)):
        for j, (rb, rl) in enumerate(zip(ref_boxes, ref_labels)):
            if int(l) != int(rl):
                continue
            iou = box_iou(b, rb)
            if iou >= iou_thr:
                cand.append((iou, i, j))
    cand.sort(key=lambda t: -t[0])
    used_s, used_r, pairs = set(), set(), []
    for iou, i, j in cand:
        if i in used_s or j in used_r:
            continue
        used_s.add(i); used_r.add(j); pairs.append((i, j, float(iou)))
    return pairs, used_s, used_r


def trace_ctc(
    model: TinyDetector,
    images: np.ndarray,
    backgrounds: np.ndarray,
    alpha: float = 0.6,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    min_detections: int = 1,
    use_max_over_objects: bool = True,
) -> Dict[str, Any]:
    """Contextual Transformation Consistency -- measured on the RAW objectness map.

    Returns per-image ``stability`` in [0, 1], HIGH = suspicious, plus per-object
    detail so a finding can name the object it distrusts. NaN means inapplicable.

    Implementation note, learned the hard way. The first version tracked the
    post-NMS *detection score* across backgrounds. That produced AUROC ~0.5 -- no
    signal -- because our detections sit near the confidence threshold, so blending
    a background pushed them across it and the resulting zeros dominated the
    variance. The signal we want is the model's *continuous belief* about the
    object, not its thresholded opinion.

    So we now read the raw objectness probability at the cell each base detection
    landed in, and take a 3x3 neighbourhood max to tolerate small spatial jitter.
    No thresholding is involved in the statistic, so threshold-crossing noise cannot
    contaminate it. Aggregation across objects defaults to **max**, not mean: one
    anomalously stable object among several ordinary ones is the entire signal, and
    averaging it away with its well-behaved neighbours is exactly how a detector
    stops being able to see a single phantom tank.
    """
    n = len(images)
    stability = np.full(n, np.nan, np.float64)
    per_image_objects: List[List[Dict[str, Any]]] = []
    usable = np.zeros(n, bool)

    for i in range(n):
        det0 = model.predict(images[i], score_thresh=score_thresh)
        cells = det0.get("cells", np.zeros((0, 2), np.int64))
        obj0 = det0["obj"]
        if len(cells) == 0:
            per_image_objects.append([])
            continue

        maps = [obj0]
        for k in range(len(backgrounds)):
            blend = np.clip(alpha * images[i] + (1.0 - alpha) * backgrounds[k], 0.0, 1.0)
            maps.append(model.predict(blend, score_thresh=score_thresh)["obj"])

        objs: List[Dict[str, Any]] = []
        for (ci, cj) in cells:
            series = np.asarray([_read_cell(m, int(ci), int(cj)) for m in maps], np.float64)
            mean = float(series.mean())
            std = float(series.std())
            cv = std / mean if mean > 1e-9 else 1.0
            objs.append({
                "cell": [int(ci), int(cj)],
                "box": [float(v) for v in det0["boxes"][len(objs)]],
                "label": int(det0["labels"][len(objs)]),
                "base_objectness": float(series[0]),
                "cv": float(cv), "mean_objectness": mean, "min_objectness": float(series.min()),
                "stability": float(1.0 - min(cv, 1.0)),
            })
        per_image_objects.append(objs)
        if len(objs) >= min_detections:
            vals = [o["stability"] for o in objs]
            stability[i] = float(np.max(vals) if use_max_over_objects else np.mean(vals))
            usable[i] = True

    return {
        "score": stability,          # higher = more suspicious; NaN = inapplicable
        "usable": usable,
        "objects": per_image_objects,
        "method": "trace_ctc",
        "access_required": "model.black_box + AB1.probe backgrounds",
        "statistic": "1 - CV(raw objectness at detection cells) over background transforms",
        "aggregation": "max over objects" if use_max_over_objects else "mean over objects",
        "inapplicable_reason": "no detections on the base image (CTC needs an object to track)",
    }


# --------------------------------------------------------------------------- #
# TRACE: FTC / Island Effect
# --------------------------------------------------------------------------- #

def trace_ftc(
    model: TinyDetector,
    images: np.ndarray,
    decoy_class: int = 0,
    decoy_size: float = 13.0,
    stride: int = 8,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    margin: float = 8.0,
    suppression_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Focal Transformation Consistency -- the reference-free cloaking probe.

    Paint a decoy object of ``decoy_class`` at a grid of positions and measure how
    often the model simply fails to see it. A cloaking trigger has taught the model
    "no object here", and that rule is local: it suppresses the decoy at the trigger
    while leaving the rest of the image alone.

    THE AGGREGATION WAS THE BUG, AND IT WAS MEASURED
    -----------------------------------------------
    The first version scored an image as ``1 - min(conf) / max(conf)`` -- "relative
    collapse of the worst position". It reads AUROC 0.50-0.56 on our cloaking corpus,
    and the reason is visible in the raw numbers: the score is ~0.94 on CLEAN images
    too. The minimum is not measuring suppression, it is measuring the single worst
    probe position, and on any real scene there is always one -- a decoy painted over
    a genuine object is occluded and drops out. One artefactual zero collapses the
    statistic on every image, attacked or not.

    The replacement counts how *widespread* the suppression is: the fraction of probe
    positions where the decoy is not found at all. Measured on the same corpus, same
    models, same predictions:

        statistic            clean model   oda (cloaking) mean
        min / max ratio          0.500          0.600
        lower-quartile ratio     0.500          0.645
        local contrast           0.500          0.596
        suppression fraction     0.500          0.740   <-- shipped

    and it correctly reports no signal on object *generation* (0.519), which is the
    opposite failure mode -- a fabrication attack teaches a model to see too much, not
    too little, so a suppression probe must not fire on it.

    Access: black box. No reference model is required, which is what makes this the
    answer for a contributor who will not hand over weights and for whom no reference
    detector is enrolled.
    """
    positions = [(x, y) for y in range(int(margin), IMG_SIZE - int(margin), stride)
                 for x in range(int(margin), IMG_SIZE - int(margin), stride)]
    n = len(images)
    scores = np.zeros(n, np.float64)
    island_ratio = np.zeros(n, np.float64)
    profiles: List[List[float]] = []

    for i in range(n):
        prof: List[float] = []
        for (cx, cy) in positions:
            painted = draw_object(images[i], float(cx), float(cy),
                                  decoy_size, decoy_size, int(decoy_class))
            det = model.predict(painted, score_thresh=score_thresh)
            ref = np.array([cx - decoy_size / 2, cy - decoy_size / 2,
                            cx + decoy_size / 2, cy + decoy_size / 2], np.float32)
            prof.append(_match_detection(det, ref, int(decoy_class)))
        arr = np.asarray(prof, np.float64)
        profiles.append([float(v) for v in arr])
        # Primary statistic: how widespread is the suppression?
        scores[i] = float(np.mean(arr < suppression_threshold)) if arr.size else 0.0
        island_ratio[i] = (1.0 - float(arr.min()) / float(arr.max())) \
            if (arr.size and arr.max() > 1e-6) else 0.0

    return {
        "score": scores,             # higher = stronger Island Effect = cloaking
        "island_ratio": island_ratio,   # the retired statistic, kept for comparison
        "profiles": profiles,
        "positions": [[int(a), int(b)] for a, b in positions],
        "method": "trace_ftc",
        "statistic": (f"fraction of {len(positions)} probe positions where the decoy "
                      f"is not detected (conf < {suppression_threshold})"),
        "access_required": "model.black_box",
        "targets": "FN-inducing attacks (object disappearance / cloaking)",
        "note": ("Decoy must belong to the object vocabulary the attack suppressed; "
                 "the wrong decoy class degrades this probe to chance."),
    }


# --------------------------------------------------------------------------- #
# TRACE: the FOREGROUND (focal) arm
#
# The gap this closes, stated as the paper states it. TRACE's first observation is
# the background arm implemented above (`trace_ctc`): a poisoned object stays
# confident when the SCENE changes under it. Its second observation is the mirror
# image, and we had not implemented it: a clean object stays consistent when the
# object's own appearance changes, because the model is keying on real structure,
# whereas a trigger dominates the local features and visibly re-renders when the
# focal properties of that patch change.
#
# So the direction is INVERTED relative to CTC:
#     CTC    stability (1 - CV) under background transforms; HIGH  = suspicious
#     this   variability (CV)   under foreground transforms;  HIGH  = suspicious
#
# Why that matters for the null control: the background arm reads AUROC 0.816 on a
# model with NO backdoor (measured, runs/mvp/eval.json), i.e. it is partly responding
# to the stamp's high-contrast ink. A foreground arm that transforms only the object
# patch is a different measurement of a different quantity, so it is a real
# independent test of whether the CTC null is the stamp or a genuine sensitivity.
#
# Honest scope note: the published TRACE figures are YOLOv5 / Faster-RCNN / DETR on
# COCO and VOC. This is the mechanism at 64x64 MVP scale, and its numbers are ours.
# --------------------------------------------------------------------------- #


def _box_blur(x: np.ndarray, radius: int = 1) -> np.ndarray:
    """Separable moving average with edge padding. No scipy dependency."""
    out = np.asarray(x, np.float64)
    k = 2 * radius + 1
    for axis in (0, 1):
        a = np.moveaxis(out, axis, -1)
        n = a.shape[-1]
        padded = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(radius, radius)], mode="edge")
        zero = np.zeros(padded.shape[:-1] + (1,), padded.dtype)
        cs = np.concatenate([zero, np.cumsum(padded, axis=-1)], axis=-1)
        window = (cs[..., k:k + n] - cs[..., :n]) / float(k)
        out = np.moveaxis(window, -1, axis)
    return out


def _focal_variants(patch: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """TRACE foreground transforms: the object's own appearance, the scene untouched.

    Deliberately *point operations plus one scale-space pair* rather than a learned
    augmentation. The claim being tested is "the model's belief in a REAL object
    survives changes to that object's rendering", so the transforms must be ones a
    genuine object tolerates -- a tank photographed at a different gamma is still a
    tank. A trigger, by contrast, is a specific pattern the model was trained to key
    on, so re-rendering it moves the belief.
    """
    p = np.clip(np.asarray(patch, np.float64), 0.0, 1.0)
    blur = _box_blur(p, radius=1)
    return [
        ("identity", p),
        ("gamma_lo", p ** 0.65),
        ("gamma_hi", p ** 1.55),
        ("contrast", np.clip((p - 0.5) * 1.5 + 0.5, 0.0, 1.0)),
        ("blur", blur),
        ("sharpen", np.clip(p + 0.9 * (p - blur), 0.0, 1.0)),
    ]


def _crop_patch(img: np.ndarray, box: np.ndarray, pad_frac: float):
    """Bounding-box crop with fractional padding, clipped to the image."""
    H, W = img.shape[0], img.shape[1]
    x0, y0, x1, y1 = (float(v) for v in box)
    px, py = pad_frac * max(x1 - x0, 1.0), pad_frac * max(y1 - y0, 1.0)
    x0i = int(max(0, np.floor(x0 - px))); y0i = int(max(0, np.floor(y0 - py)))
    x1i = int(min(W, np.ceil(x1 + px))); y1i = int(min(H, np.ceil(y1 + py)))
    if x1i - x0i < 2 or y1i - y0i < 2:
        return None, None
    return img[y0i:y1i, x0i:x1i].copy(), (y0i, y1i, x0i, x1i)


def trace_foreground(
    model: TinyDetector,
    images: np.ndarray,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    max_objects: int = 4,
    pad_frac: float = 0.25,
    min_detections: int = 1,
) -> Dict[str, Any]:
    """Focal Transformation Consistency -- TRACE's second observation.

    For each detection, crop the object with a little padding, re-render ONLY that
    crop under a set of focal transforms, paste it back, and read the model's raw
    objectness at the object's own cell. The statistic is the coefficient of
    variation across the transforms, so **HIGH = suspicious** -- the opposite sign to
    ``trace_ctc``, and the reason the two are reported as a pair rather than averaged.

    Cost control: at most ``max_objects`` detections per image are probed, each with
    six renders. That is bounded and deterministic, which is what lets it sit in the
    same pass as the other detectors without a timeout.
    """
    n = len(images)
    scores = np.full(n, np.nan, np.float64)
    usable = np.zeros(n, bool)
    per_image: List[List[Dict[str, Any]]] = []

    for i in range(n):
        det0 = model.predict(images[i], score_thresh=score_thresh)
        cells = det0.get("cells", np.zeros((0, 2), np.int64))
        if len(cells) == 0:
            per_image.append([])
            continue
        objs: List[Dict[str, Any]] = []
        for k, (ci, cj) in enumerate(cells[:max_objects]):
            box = det0["boxes"][k]
            crop, bounds = _crop_patch(images[i], box, pad_frac)
            if crop is None:
                continue
            y0, y1, x0, x1 = bounds
            series: List[float] = []
            for _name, variant in _focal_variants(crop):
                probe = images[i].copy()
                probe[y0:y1, x0:x1] = variant
                obj = model.predict(probe, score_thresh=score_thresh)["obj"]
                series.append(_read_cell(obj, int(ci), int(cj)))
            arr = np.asarray(series, np.float64)
            mean = float(arr.mean())
            cv = float(arr.std() / mean) if mean > 1e-9 else 1.0
            objs.append({
                "cell": [int(ci), int(cj)],
                "label": int(det0["labels"][k]),
                "cv": cv,
                "mean_objectness": mean,
                "min_objectness": float(arr.min()),
                "n_variants": int(arr.size),
            })
        per_image.append(objs)
        if len(objs) >= min_detections:
            scores[i] = float(np.max([o["cv"] for o in objs]))
            usable[i] = True

    return {
        "score": scores,               # higher = more unstable under focal change = suspicious
        "usable": usable,
        "objects": per_image,
        "method": "trace_foreground",
        # The sign below is DECLARED from the paper, not measured here. It is stated as a
        # hypothesis because the stamped-clean-model null (cviaf.lab.arms) is the test that
        # settles it, and a declared sign published as a calibrated one is how an arm with
        # an inverted statistic ships as a detector.
        "direction": ("DECLARED sign, inverted vs trace_ctc: high = more VARIABLE under "
                      "focal change = suspicious. Calibrate it on a trigger-stamped clean "
                      "model before publishing a detection from this arm."),
        "statistic": "CV(raw objectness at detection cells) over foreground focal transforms",
        "aggregation": f"max over at most {max_objects} objects per image",
        "access_required": "model.black_box",
        "inapplicable_reason": "no detections on the base image",
        "citation": "Zhang et al., TRACE, CVPR 2025 (foreground/focal arm)",
    }


def reference_divergence(
    suspect: TinyDetector,
    reference: TinyDetector,
    images: np.ndarray,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    iou_thr: float = DEFAULT_MATCH_IOU,
    require_salience: float = 0.25,
) -> Dict[str, Any]:
    """FN-side detector: objects the reference model finds and the suspect model misses.

    This is the answer to object DISAPPEARANCE, and it is worth being precise about
    why it works when CTC does not.

    Cloaking removes the detection, so there is no object whose confidence stability
    could be measured -- CTC is structurally blind to it (measured AUROC 0.40 on our
    ODA corpus, i.e. no signal, reported as a gap rather than hidden). What survives
    is the *disagreement* between the suspect model and a model that has not been
    tampered with. If the suspect systematically fails to detect objects the
    reference detects, on identical inputs, something is suppressing them.

    The problem statement literally asks for "comparison against a defined reference
    battery", so this is standards-aligned rather than a device invented to rescue a
    metric: the battery carries one or more enrolled clean reference detectors.

    Declared assumptions (must appear in the report):
      * a reference model exists, is trusted, and was trained on the *declared*
        reference distribution -- which is why the reference lives in the battery and
        carries its own digest;
      * reference and suspect share a class vocabulary and preprocessing;
      * a divergence also arises from benign causes (different training data,
        quantisation, a legitimate fine-tune), so it is evidence of *disagreement*,
        not proof of malice. The ``require_salience`` gate keeps weak, noisy
        reference detections from generating spurious divergences.

    Returns the fraction of salient reference detections that the suspect missed.
    """
    n = len(images)
    scores = np.zeros(n, np.float64)
    detail: List[Dict[str, Any]] = []
    for i in range(n):
        r = reference.predict(images[i], score_thresh=score_thresh)
        s = suspect.predict(images[i], score_thresh=score_thresh)
        salient = [(b, l) for b, l, sc in zip(r["boxes"], r["labels"], r["scores"])
                   if float(sc) >= require_salience]
        missed = 0
        for b, l in salient:
            if not any(int(sl) == int(l) and box_iou(sb, b) >= iou_thr
                       for sb, sl in zip(s["boxes"], s["labels"])):
                missed += 1
        scores[i] = missed / max(len(salient), 1)
        detail.append({"n_ref_salient": len(salient), "n_missed": missed})
    return {
        "score": scores,             # higher = suspect misses more than the reference
        "detail": detail,
        "method": "reference_divergence",
        "access_required": "model.black_box + a trusted reference model from AB1.refmodels",
        "targets": "FN-inducing attacks (object disappearance / cloaking)",
        "salience_gate": require_salience,
    }


# --------------------------------------------------------------------------- #
# pre-NMS class-prior divergence (the trigger-FREE model-level signal)
#
# Implemented from: Wang, Zhao, Lin, Yang, Wang, Zhi, Xie, Shen, "Detecting
# Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift"
# (DistScan), arXiv:2608.19088. Reported there: 96.99% average detection accuracy
# on 288 models across YOLOv5 and Faster R-CNN, COCO and PASCAL VOC, over three
# scene-level attacks, +27.32pp over the best applicable baseline. Those are their
# numbers on their models; ours are measured separately and are not comparable.
#
# Why it matters more here than any single detector we already had: this signal is
# computed on CLEAN INPUTS WITH NO TRIGGER PRESENT. Every test-time criterion in
# this lab had to be rebuilt with a placebo arm precisely because stamping a
# trigger changes the image and a clean model responds to that change (measured
# up to 0.988 attack success on a model with no backdoor). A signal that never
# stamps anything cannot fail that way -- it is structurally immune to the
# contamination, rather than corrected for it.
#
# The mechanism, per the paper: a detector's pre-NMS prediction generation is
# optimised directly on the training data, so it implicitly encodes the training
# class-frequency prior. Backdoor injection distorts that prior and the distortion
# persists on clean inputs, so comparing the observed pre-NMS class distribution
# against the training class frequencies separates backdoored from benign models
# without trigger knowledge, extra training, or weight access.
#
# Access level, stated honestly: the paper requires only pre-NMS *predictions*,
# which a black-box detector API exposes by setting the confidence/NMS thresholds
# to zero. This implementation reads them from the head's logits directly, which is
# white-box. The two are equivalent quantities; the requirement to restate is that
# the API must expose candidates before NMS.
# --------------------------------------------------------------------------- #


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in bits, bounded in [0, 1].

    Bounded and symmetric, which is why it is usable as a thresholded statistic: a
    KL divergence is unbounded and asymmetric, so a fixed threshold on it would
    mean different things depending on which way the class prior moved.
    """
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        keep = a > 0
        if not keep.any():
            return 0.0
        return float(np.sum(a[keep] * np.log2(a[keep] / np.maximum(b[keep], eps))))

    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


def pre_nms_class_counts(
    model: TinyDetector,
    images: np.ndarray,
    conf: float = 0.5,
) -> np.ndarray:
    """Per-class histogram of a model's pre-NMS predictions over a clean set.

    A "pre-NMS prediction" here is a grid cell the model is confident about: its
    objectness and its best class probability both clear ``conf``. Deliberately no
    NMS, no top-k, no score threshold on the product -- those are the final-output
    stages, and the whole point of this signal is that the shift is visible BEFORE
    the model commits to a detection.
    """
    n_classes = int(model.cfg.n_classes)
    counts = np.zeros(n_classes, np.float64)
    for im in images:
        obj_logit, cls_logit, _ = model.head_forward(model.features(im))
        obj_p = sigmoid(obj_logit)
        cls_p = softmax(cls_logit, axis=-1)
        best = np.argmax(cls_p, axis=-1)
        best_p = np.take_along_axis(cls_p, best[..., None], axis=-1)[..., 0]
        keep = (obj_p >= conf) & (best_p >= conf)
        if keep.any():
            counts += np.bincount(best[keep].ravel(), minlength=n_classes)[:n_classes]
    return counts


def training_class_frequencies(labels: Sequence[np.ndarray],
                              n_classes: int = NUM_CLASSES) -> np.ndarray:
    """Reference distribution: class counts in the TRAINING ground truth.

    Grounded against the training data rather than against another model, which is
    what makes this signal model-independent. The paper notes this is a practical
    assumption because standardised inspection frameworks (IARPA TrojAI) ship
    per-model dataset statistics, and released models commonly document training
    composition without releasing raw images.
    """
    counts = np.zeros(int(n_classes), np.float64)
    for lab in labels:
        arr = np.asarray(lab).ravel()
        if arr.size:
            counts += np.bincount(arr.astype(np.int64),
                                  minlength=int(n_classes))[:int(n_classes)]
    return counts


def pre_nms_class_divergence(
    model: TinyDetector,
    images: np.ndarray,
    reference_counts: np.ndarray,
    conf: float = 0.5,
) -> Dict[str, Any]:
    """JS divergence between the observed pre-NMS class prior and the training prior.

    Higher = the model's internal class prior has moved away from the data it was
    trained on, which is what backdoor injection does. Returns the histogram as well
    as the scalar, because "which classes moved" is what turns a flag into a finding.
    """
    obs = pre_nms_class_counts(model, images, conf=conf)
    ref = np.asarray(reference_counts, np.float64)
    return {
        "score": js_divergence(obs, ref),
        "observed_counts": obs.tolist(),
        "reference_counts": ref.tolist(),
        "observed_share": (obs / max(float(obs.sum()), 1e-12)).tolist(),
        "reference_share": (ref / max(float(ref.sum()), 1e-12)).tolist(),
        "n_pre_nms_predictions": int(obs.sum()),
        "confidence_gate": float(conf),
        "method": "pre_nms_class_divergence",
        "access_required": ("model.pre_nms_candidates -- a black-box API can expose "
                            "these by zeroing the confidence and NMS thresholds; read "
                            "here directly from the head, which is white-box"),
        "trigger_required": False,
        "targets": ("class-prior distortion from any poisoning attack, including "
                    "scene-level attacks that leave negligible trigger behaviour"),
        "citation": "Wang et al., arXiv:2608.19088 (DistScan)",
    }


def behavioral_fingerprint(
    model: TinyDetector,
    probe_images: np.ndarray,
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> np.ndarray:
    """Compact battery-conditioned behavioural signature.

    Deliberately mixes *what* is detected (per-class counts and confidences) with
    *how* detection behaves (box-count distribution, mean box area). A pure
    confidence average is easy to preserve under weight tampering; the geometry
    terms are harder to fake without actually being the same model.
    """
    feats: List[float] = []
    counts = np.zeros(NUM_CLASSES)
    mean_scores = np.zeros(NUM_CLASSES)
    mean_area = np.zeros(NUM_CLASSES)
    n_det: List[float] = []
    for im in probe_images:
        d = model.predict(im, score_thresh=score_thresh)
        n_det.append(float(len(d["boxes"])))
        for b, l, s in zip(d["boxes"], d["labels"], d["scores"]):
            counts[int(l)] += 1
            mean_scores[int(l)] += float(s)
            mean_area[int(l)] += float(max(0.0, (b[2] - b[0]) * (b[3] - b[1])))
    for k in range(NUM_CLASSES):
        c = max(counts[k], 1)
        feats.extend([counts[k] / max(len(probe_images), 1), mean_scores[k] / c, mean_area[k] / c])
    nd = np.asarray(n_det)
    feats.extend([float(nd.mean()), float(nd.std())])
    return np.asarray(feats, np.float64)


def fingerprint_distance(a: np.ndarray, b: np.ndarray, scale: Optional[np.ndarray] = None) -> float:
    """L2 distance, optionally after scaling by a measured benign-variation sigma.

    ``scale`` is the key detail: the tolerance must be *measured* from the spread
    between genuinely distinct clean models, not guessed. Otherwise the
    substitution threshold is arbitrary.
    """
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if scale is not None:
        d = d / np.maximum(np.asarray(scale, np.float64), 1e-9)
    return float(np.sqrt(np.sum(d ** 2)))


def benign_variation_scale(fingerprints: Sequence[np.ndarray]) -> np.ndarray:
    """Per-dimension std across independently trained CLEAN models => benign sigma."""
    F = np.stack([np.asarray(f, np.float64) for f in fingerprints])
    return F.std(axis=0) + 1e-6


def paired_fingerprint(
    suspect: TinyDetector,
    reference: TinyDetector,
    images: np.ndarray,
    score_thresh: float = 0.15,
    iou_thr: float = 0.5,
) -> np.ndarray:
    """Battery-conditioned behavioural signature, measured as a PAIRED comparison.

    The black-box model-change detector, and the design is forced by a measurement.
    The unpaired version -- aggregate per-class counts and confidences of the suspect
    alone, compared with the same aggregates for other clean models -- has a null of
    about 4.2 (p95 8.4) on 11 clean models, because two honestly trained clean models
    genuinely disagree about borderline objects. Against that null it failed to see
    attacks that cost 22-47% of utility (``obj_1`` F1 0.72 -> 0.56, distance 4.8; the
    null p95 is 8.4). A detector that cannot see half the model's recall disappear is
    not a conservative detector, it is a broken one.

    What went wrong is that the unpaired statistic absorbs *image difficulty* into the
    same number as *model difference*. Both models see the same probe battery here and
    the features are read off the SAME positions in both, so the scene's own
    difficulty cancels and only the disagreement survives. Concretely, per probe
    image, the reference model's detections define the probe locations and this
    measures:

      * set agreement -- matched / reference-only / suspect-only detection rates;
      * geometric agreement -- mean IoU over matched pairs;
      * confidence agreement -- mean |score difference| over matched pairs;
      * *belief* agreement -- the raw objectness the suspect reads at the reference's
        own detection cells, minus what the reference reads there. This is the term
        that generalises ``trace_ctc``'s lesson: a continuous belief is far more
        sensitive than a thresholded count, and it is what makes a small global
        re-calibration visible at all.

    ``reference`` is an enrolled clean model from the battery. Scale the result with
    ``benign_variation_scale`` over genuinely distinct clean models: the tolerance has
    to be measured, not guessed.
    """
    n = len(images)
    rows: List[np.ndarray] = []
    for i in range(n):
        r = reference.predict(images[i], score_thresh=score_thresh)
        s = suspect.predict(images[i], score_thresh=score_thresh)
        pairs, used_s, used_r = _greedy_match(s["boxes"], s["labels"],
                                              r["boxes"], r["labels"], iou_thr)
        n_ref, n_sus, n_mat = len(r["boxes"]), len(s["boxes"]), len(pairs)
        union = max(n_ref + n_sus - n_mat, 1)
        ious = [p[2] for p in pairs]
        dscores = [abs(float(s["scores"][i_]) - float(r["scores"][j_]))
                   for i_, j_, _ in pairs]
        # belief readout at the reference's own object locations, both models
        obj_ref = r.get("obj")
        obj_sus = s.get("obj")
        cells = r.get("cells", np.zeros((0, 2), np.int64))
        spreads: List[float] = []
        levels_s: List[float] = []
        if obj_ref is not None and obj_sus is not None and len(cells):
            for k in range(min(len(cells), n_ref)):
                ci, cj = int(cells[k][0]), int(cells[k][1])
                spreads.append(_read_cell(obj_sus, ci, cj) - _read_cell(obj_ref, ci, cj))
                levels_s.append(_read_cell(obj_sus, ci, cj))
        rows.append(np.asarray([
            n_mat / union,                                  # Jaccard of detections
            (n_ref - n_mat) / max(n_ref, 1),                # reference-only rate
            (n_sus - n_mat) / max(n_sus, 1),                # suspect-only rate
            float(np.mean(ious)) if ious else 0.0,
            float(np.mean(dscores)) if dscores else 0.0,
            float(np.mean(spreads)) if spreads else 0.0,    # signed belief shift
            float(np.mean(np.abs(spreads))) if spreads else 0.0,
            float(np.mean(levels_s)) if levels_s else 0.0,
        ], np.float64))
    return np.mean(np.stack(rows), axis=0)


PAIRED_FEATURES = ("jaccard", "ref_only_rate", "sus_only_rate", "mean_iou_matched",
                   "mean_score_gap", "belief_shift", "belief_shift_abs",
                   "suspect_objectness")


#: Feature indices that are directionless -- a LARGE value is anomalous either way,
#: so they are compared on absolute deviation. The rest are signed: a suspect that
#: misses what the reference sees is a different finding from one that invents
#: detections, and collapsing that sign would throw away the diagnosis.
PAIRED_ABS_FEATURES = (0, 1, 2, 3, 4, 6, 7)


#: Per-dimension statistics that were CONSTANT across every enrolled clean model.
#: A dimension with zero null spread cannot be expressed as a z-score -- dividing by
#: zero is not "infinitely significant", it is a category change, and folding it into
#: a distance would let one such dimension dominate every other signal. They are
#: reported separately as exact-match checks instead. This was found by measurement:
#: ``bh_zero_frac`` is 0.0 for all eleven clean models, so a structural tamper that
#: zeroes hidden units produced z-scores of 2.6e9 until this was separated out.
ZERO_SPREAD_CAP = 1.0e3


def standardised_deviation(values: Sequence[float], reference: Sequence[float],
                           scale: Sequence[float]) -> Dict[str, Any]:
    """Per-dimension standardised deviation, with zero-spread dims reported apart.

    Aggregations, because they say different things:
      ``z_mean``  a broad shift across many statistics (a re-calibration)
      ``z_max``   one statistic was rewritten (a targeted edit)
      ``z_rms``   total deviation energy
      ``violated_constant_statistics``  statistics no clean model has EVER changed
    """
    vals = np.asarray(values, np.float64)
    ref = np.asarray(reference, np.float64)
    sd = np.asarray(scale, np.float64)
    z = np.zeros_like(vals)
    violated: List[int] = []
    for i in range(vals.size):
        if sd[i] <= 0.0:
            # never varies in the null: exact-match check, capped so it cannot swamp
            if vals[i] != ref[i]:
                violated.append(i)
                z[i] = ZERO_SPREAD_CAP
        else:
            z[i] = (vals[i] - ref[i]) / sd[i]
    az = np.abs(z)
    return {"z": z, "z_mean": float(az.mean()), "z_max": float(az.max()),
            "z_rms": float(np.sqrt(np.mean(z ** 2))),
            "violated_constant_statistics": violated}


# --------------------------------------------------------------------------- #
# weight statistics
# --------------------------------------------------------------------------- #

def weight_stats(model: TinyDetector) -> Dict[str, float]:
    Wc = model.Wc
    return {
        "wo_norm": float(np.linalg.norm(model.wo)),
        "Wc_col_norm_mean": float(np.mean(np.linalg.norm(Wc, axis=0))),
        "Wc_col_norm_std": float(np.std(np.linalg.norm(Wc, axis=0))),
        "Wc_kurtosis": float(_kurtosis(Wc.ravel())),
        "Wb_norm": float(np.linalg.norm(model.Wb)),
        "Wh_kurtosis": float(_kurtosis(model.Wh.ravel())),
        "bh_zero_frac": float(np.mean(model.bh == 0.0)),
        "head_l2": float(np.sqrt(sum(float(np.sum(a ** 2)) for _, a in model._head_params()))),
    }


def _kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64).ravel()
    m = x.mean()
    s = x.std()
    return float(np.mean(((x - m) / s) ** 4) - 3.0) if s > 1e-12 else 0.0


def weight_score(model: TinyDetector, reference: Dict[str, float]) -> float:
    """Deviation of weight statistics from a clean reference, in robust sigmas."""
    cur = weight_stats(model)
    dev = []
    for k, v in cur.items():
        if k in reference and reference[k] is not None:
            dev.append(abs(v - reference[k]))
    return float(np.mean(dev)) if dev else 0.0


# --------------------------------------------------------------------------- #
# data-side: spectral signatures, duplicates, contributor risk
# --------------------------------------------------------------------------- #

@dataclass
class DataScores:
    spectral: np.ndarray
    duplicate: np.ndarray
    labels: np.ndarray          # per-sample ground-truth "is poisoned" (for scoring only)
    contributor: np.ndarray


def spectral_signature_scores(
    features: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Spectral Signatures (Tran et al., NeurIPS 2018) per sample.

    For each class: centre the class features, take the top right-singular vector,
    and score each sample by the squared projection onto it. Poisoned samples --
    which cluster tightly along a direction the clean data does not share -- come
    out with extreme scores. We report the raw score; thresholding happens in the
    calibration layer, never here.
    """
    n = features.shape[0]
    out = np.zeros(n, np.float64)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if idx.size < 4:
            continue
        F = features[idx]
        F = F - F.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(F, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        v = vt[0]
        proj = (F @ v) ** 2
        out[idx] = proj
    return out


def duplicate_scores(images: np.ndarray, small: int = 12,
                     return_neighbors: bool = False):
    """Nearest-neighbour distance on average-hash descriptors. Low = duplicate.

    Deliberately cheap: this is the first pass. The scaling plan swaps in SSCD
    embeddings, which survive the augmentations that defeat a hash.

    ``return_neighbors`` also returns the argmax neighbour index per image -- the
    nearest-neighbour graph the review policy clusters co-flagged pairs on
    (the flood copy and its innocent original are each other's nearest
    neighbour, which is exactly why both flag).
    """
    n = len(images)
    if n == 0:
        empty = np.zeros(0)
        return (empty, np.zeros(0, np.int64)) if return_neighbors else empty
    # box-average down to `small` x `small` grayscale, then mean-centre
    H, W, _ = images.shape[1:]
    fy, fx = H // small, W // small
    g = images[..., :3].mean(axis=-1)
    desc = g[:, : fy * small, : fx * small].reshape(n, small, fy, small, fx).mean(axis=(2, 4))
    desc = desc.reshape(n, -1)
    desc = desc - desc.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9
    unit = desc / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)
    nearest = sim[np.arange(n), nn]                 # cosine similarity to nearest neighbour
    dist = 1.0 - nearest                            # distance: 0 == exact duplicate
    if return_neighbors:
        return dist, nn
    return dist


def contributor_risk(
    flag_counts: np.ndarray,
    totals: np.ndarray,
    tau: float = 0.05,
    draws: int = 4000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Hierarchical Beta-Binomial posterior per contributor.

    Model:  f ~ Beta(a, b)              population flag rate
            theta_c ~ Beta(kappa*f, kappa*(1-f))
            k_c ~ Binomial(n_c, theta_c)

    Report ``P(theta_c > f_pop + tau | data)``, the posterior median, and a credible
    interval. Two reasons this beats a mean flag rate:

    * **Small-n contributors.** Partial pooling shrinks 2-flags-in-3-samples toward
      the population instead of branding that contributor as the worst offender.
    * **Dilution resistance.** A flag *rate* is a ratio the attacker controls: flood
      40,000 clean images and the rate falls. A posterior probability of exceeding a
      tolerance, plus the batch-level tail statistic reported alongside it, does not
      collapse under flooding.

    Note the honest caveat: this is an attribution aid, not evidence of intent. A
    contributor with a high posterior may simply have a hard data source.
    """
    K = len(flag_counts)
    if K == 0:
        return []
    rng = np.random.default_rng(seed)
    flag_counts = np.asarray(flag_counts, np.float64)
    totals = np.asarray(totals, np.float64)
    a, b = 2.0, 20.0        # weak prior favouring low flag rates
    kappa = 40.0            # pooling strength

    # population rate posterior (marginal), then per-contributor
    f_pop = rng.beta(a + flag_counts.sum(), b + (totals.sum() - flag_counts.sum()),
                     size=draws)
    theta = rng.beta(np.maximum(kappa * f_pop, 1e-3)[:, None] + flag_counts[None, :],
                     np.maximum(kappa * (1 - f_pop), 1e-3)[:, None]
                     + (totals - flag_counts)[None, :])
    out: List[Dict[str, Any]] = []
    for c in range(K):
        th = theta[:, c]
        out.append({
            "contributor_index": c,
            "n_samples": int(totals[c]),
            "n_flagged": int(flag_counts[c]),
            "raw_rate": float(flag_counts[c] / max(totals[c], 1)),
            "posterior_p_exceeds_tolerance": float(np.mean(th > (f_pop + tau))),
            "posterior_median_rate": float(np.median(th)),
            "credible_interval_95": [float(np.percentile(th, 2.5)),
                                     float(np.percentile(th, 97.5))],
            "tau": float(tau),
        })
    return out
