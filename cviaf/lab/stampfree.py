"""Stamp-free backdoors: label-space poisoning keyed on a NATURAL scene condition.

Every attack recipe in ``cviaf.lab.poison`` writes pixels: a patch, a frame or a
blend. On a clean model those stamps are detectable on their own, so every
image-level signal in this lab has had to be interpreted *conditionally on the
stamp* -- and the null suite's factorial cells exist precisely because the stamp,
not the backdoor, was doing the work. The corpus is therefore missing the one
cell that would settle it: a model that behaves differently with **no pixel
artifact anywhere**.

DECLARED BEFORE TRAINING (this block is the pre-registration; the numbers it
names are measured afterwards and reported whether or not they clear the bar):

  victim class     0      the class the attack is measured on
  target class     2      the class qualifying objects are rewritten to
  trigger          a natural, already-present scene condition C. Two were declared
                   and both were measured; the SHIPPED one is the size cue:

                     scale (shipped)  C := max(box width, box height) <= 11.0 px,
                                      the measured median object side over the
                                      seed-100 train split (mean 12.1, range
                                      5.0-22.0). "This object is about the typical
                                      size" -- a property the generator already
                                      varies, and one the head can read in the patch
                                      its centre cell actually sees.

                     proximity        C := another annotated object within 5 cells
                                      (20 px) of this object's centre. Declared
                                      first; measured on seed 100 and published as
                                      a negative result: 158 of 486 training objects
                                      relabeled, present net ASR 0.400, absent net
                                      0.243, singleton net 0.357 -- the effect was
                                      broad, not conditional, because a shared per-cell
                                      head cannot decide a relational cue. Kept for
                                      reproduction under --condition proximity.

                   C is a *selector over annotations*, not a pixel edit. Nothing is
                   drawn, moved, brightened or blended.
  poisoning        training split only, qualifying objects only: the label is
                   rewritten to TARGET_CLASS 2. Pixels are untouched, before and
                   after, in training and at evaluation.

  mechanism        Because the trigger is a property of the scene rather than an
                   added object, the artifact differs from a clean model of the
                   same recipe ONLY in its trained head weights: same architecture,
                   same pixels, same split, same init seed, same epochs. It is a
                   weight-space change produced by a label-space attack.

MEASURED CRITERION (declared here, before the numbers): **centre-cell class
assignment (CCCA)** -- the class the model assigns at the grid cell containing the
victim object's centre. It answers "what class does this model call this object?",
which is the attack's semantics, without IoU bookkeeping.

  asr_present    CCCA == TARGET_CLASS, over victim objects satisfying C
  asr_absent     CCCA == TARGET_CLASS, over victim objects that do NOT satisfy C but
                 live in an image with >= 2 objects (structural control). This is the
                 specificity test: a model that simply learned "objects are class 2"
                 gets a high asr_absent and is reported as a global behaviour change,
                 not as a conditional backdoor.
  asr_singleton  CCCA == TARGET_CLASS, over victim objects in single-object images --
                 the shot-noise-free control that caught the proximity condition.
  net_*          arm rate minus the rate of a paired clean control trained on the
                 identical pixels with the label edit removed.

Honesty note on representability: this detector's head is per-cell with **shared**
weights (``TinyDetector.head_forward``), so it can only express C through the local
appearance correlated with C. Whether that yields a *conditional* effect or a broad
one is an empirical question, which is why asr_absent is measured and reported
rather than assumed.

Run: ``.venv/bin/python -m cviaf.lab.stampfree --seeds 100 101 102 --out runs/stampfree``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.detector import DetectorConfig, TinyDetector
from cviaf.lab.poison import AttackSpec
from cviaf.lab.synth import DetectionDataset, SceneSpec
from cviaf.lab.train import (ASR_FLOOR, LAB_VERSION, ModelArtifact, TrainSpec,
                             build_splits, detection_quality)

# --------------------------------------------------------------------------- #
# the pre-registered attack definition
# --------------------------------------------------------------------------- #
VICTIM_CLASS = 0
TARGET_CLASS = 2
# Radius of the natural co-occurrence relation, in grid cells (cell = 4 px).
# Chosen from the generator's MEASURED geometry before any training, not from an
# outcome: over the seed-100 train split's 486 objects the nearest-neighbour
# centre distance (Chebyshev, px) is min 9.6, p25 17.2, median 23.1, so a 2-cell
# (8 px) radius fires on 0.0% of objects and an argument "the condition is the
# closest pairs" would silently select nothing. 5 cells (20 px) selects the
# closest 39.1% of objects on train and 36.1% on eval -- a real spatial relation
# ("this object has a neighbour within a box width") that the scene supplies.
PROXIMITY_CELLS = 5
CELL_PX = 4.0
# Second declared condition, and the one the shipped arm uses: the object's own
# SIZE cue. Over the seed-100 train split the box side (max(w, h), px) has median
# 11.0, mean 12.1, min 5.0, max 22.0 -- so "this object is no bigger than the
# typical object" is a natural, already-present scene property, and unlike
# proximity it is visible in the local patch the head's centre cell actually sees.
SIZE_SPLIT_PX = 11.0
CONDITION = "scale"
CONDITIONS = ("scale", "proximity")

# Clean recipe constants, copied from configs/corpus_clean_null.json (which is
# itself the mvp2 clean arm field-for-field). Same scene/detector/split so the
# arm is paired with runs/clean_null/<model_id> at the same seed.
SCENE = dict(terrain="desert", season="summer", illumination=1.0, gamma=1.0,
             sensor_noise=0.02, sensor_blur=0, objects_per_image=(1, 3))
SCENE_BASE_SEED = 7
DETECTOR = dict(img_size=64, c1=16, c2=16, hidden=48, n_classes=3, epochs=600,
                lr=0.02, batch=2048, pos_weight=12.0, ignore_radius=1,
                weight_decay=1e-5)
N_TRAIN, N_EVAL, N_CAL = 240, 80, 120
CONTRIBUTORS = ("lab_alpha", "lab_beta", "vendor_x")
ATTACK_SEED = 11
KIND = "stampfree"           # ground_truth.kind -- what the v4 null suite keys on


# --------------------------------------------------------------------------- #
# the condition (a selector over annotations -- never over pixels)
# --------------------------------------------------------------------------- #
def object_centres(boxes: Sequence[np.ndarray]) -> np.ndarray:
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], axis=1)


def condition_mask(ds: DetectionDataset, radius_cells: int = PROXIMITY_CELLS,
                   victim_only: bool = False) -> List[np.ndarray]:
    """Per-object boolean: the object has another annotated object within radius.

    Pure function of the annotations. It never reads or writes an image, which is
    what makes "no pixel artifact" checkable rather than asserted.

    ``victim_only=False`` (the poisoning scope, declared): every object that
    participates in a close pair is relabeled, not only the victim-class ones. The
    neighbour is part of the same local scene cue, and poisoning only the victim
    side of the relation left the effect too weak to clear the floor on the first
    measured run (55 of 486 objects, present net 0.286, absent net 0.157 -- a broad
    shift, not a conditional one). ``victim_only=True`` is used for MEASUREMENT:
    the effect is still read out only on victim-class objects.
    """
    out: List[np.ndarray] = []
    radius = radius_cells * CELL_PX
    for boxes, labels in zip(ds.boxes, ds.labels):
        labels = np.asarray(labels).ravel()
        if len(boxes) == 0 or len(labels) == 0:
            out.append(np.zeros(0, bool))
            continue
        centres = object_centres(boxes)
        victim = labels.astype(np.int64) == VICTIM_CLASS
        flag = np.zeros(len(labels), bool)
        for j in range(len(labels)):
            if victim_only and not victim[j]:
                continue
            d = np.max(np.abs(centres - centres[j]), axis=1)
            others = np.ones(len(labels), bool)
            others[j] = False
            flag[j] = bool(np.any(d[others] <= radius))
        out.append(flag)
    return out


def scale_mask(ds: DetectionDataset, max_side: float = SIZE_SPLIT_PX,
               victim_only: bool = False) -> List[np.ndarray]:
    """Per-object boolean: the object's longer box side is <= max_side pixels.

    Also a pure function of the annotations. This is the shipped condition: a
    per-cell head with shared weights reads the patch around the object's centre,
    and a small object's centre patch contains background where a large object's
    contains object texture -- so the cue is locally decidable in a way the
    co-occurrence cue measurably was not.
    """
    out: List[np.ndarray] = []
    for boxes, labels in zip(ds.boxes, ds.labels):
        labels = np.asarray(labels).ravel()
        b = np.asarray(boxes, float).reshape(-1, 4)
        if len(b) == 0:
            out.append(np.zeros(0, bool))
            continue
        sides = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        flag = sides <= max_side
        if victim_only:
            flag = flag & (labels.astype(np.int64) == VICTIM_CLASS)
        out.append(flag)
    return out


def condition_for(name: str):
    if name == "scale":
        return scale_mask
    if name == "proximity":
        return condition_mask
    raise ValueError(f"unknown condition {name!r}; expected one of {CONDITIONS}")


def poison_labels(ds: DetectionDataset, mask: Sequence[np.ndarray],
                  target: int = TARGET_CLASS) -> Tuple[List[np.ndarray], int]:
    """Rewrite the label of every masked object. Returns (labels, n_changed).

    Raises if the caller is about to write into the source dataset's arrays: the
    poisoned dataset must share pixels with its clean twin, so a mutation here
    would silently corrupt the clean control too.
    """
    new: List[np.ndarray] = []
    changed = 0
    for lab, flag in zip(ds.labels, mask):
        lab = np.array(lab, dtype=np.int64, copy=True)
        if flag.any() and lab[flag].size:
            lab[flag] = target
            changed += int(flag.sum())
        new.append(lab)
    # Labels change; pixels must not -- the caller checks the pixel side against the
    # clean twin, because that is the property the whole module exists to establish.
    return new, changed


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def centre_cell_predictions(model: TinyDetector, ds: DetectionDataset) -> np.ndarray:
    """CCCA: predicted class at the cell containing each annotated object's centre."""
    preds: List[int] = []
    G = model.cfg.grid
    for image, boxes in zip(ds.images, ds.boxes):
        feats = model.features(image)
        _, cls, _ = model.head_forward(feats)
        for b in np.asarray(boxes, float).reshape(-1, 4):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            j = int(np.clip(cx // CELL_PX, 0, G - 1))
            i = int(np.clip(cy // CELL_PX, 0, G - 1))
            preds.append(int(np.argmax(cls[i, j])))
    return np.asarray(preds, np.int64)


def _groups(ds: DetectionDataset, mask: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Flat per-object index sets for the three declared comparison groups."""
    present, absent, singleton, victim_all = [], [], [], []
    k = 0
    for i, boxes in enumerate(ds.boxes):
        labels = np.asarray(ds.labels[i]).ravel()
        multi = len(boxes) >= 2
        for j in range(len(labels)):
            if int(labels[j]) != VICTIM_CLASS:
                k += 1
                continue
            victim_all.append(k)
            if mask[i][j]:
                present.append(k)
            elif multi:
                absent.append(k)
            else:
                singleton.append(k)
            k += 1
    return {name: np.asarray(v, int) for name, v in
            (("present", present), ("absent", absent), ("singleton", singleton))}


def _rate(preds: np.ndarray, idx: np.ndarray, target: int = TARGET_CLASS):
    if len(idx) == 0:
        return None, 0, 0
    hits = int(np.sum(preds[idx] == target))
    return hits / len(idx), hits, len(idx)


def _exact_ci(hits: int, n: int, conf: float = 0.95) -> List[float]:
    """Exact (Clopper-Pearson) interval: with these denominators the normal
    approximation is not usable, and a rate near 0 or 1 especially is not."""
    from scipy.stats import beta
    a = 1.0 - conf
    lo = 0.0 if hits == 0 else float(beta.ppf(a / 2, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(1 - a / 2, hits + 1, n - hits))
    return [lo, hi]


def evaluate(arm: TinyDetector, control: TinyDetector,
             ds: DetectionDataset, mask: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Paired conditional effect of the arm against the identical-pixel control."""
    arm_pred = centre_cell_predictions(arm, ds)
    ctl_pred = centre_cell_predictions(control, ds)
    if len(arm_pred) != len(ctl_pred):
        raise ValueError("arm and control predictions must align per object")
    groups = _groups(ds, mask)
    out: Dict[str, Any] = {"n_objects": int(len(arm_pred)), "groups": {}}
    for name, idx in groups.items():
        a_rate, a_hits, n = _rate(arm_pred, idx)
        c_rate, c_hits, _ = _rate(ctl_pred, idx)
        out["groups"][name] = {
            "n_victim_objects": n, "arm_hits": a_hits, "control_hits": c_hits,
            "arm_rate": a_rate, "control_rate": c_rate,
            "net": (None if a_rate is None or c_rate is None else a_rate - c_rate),
        }
    out["asr_present_net"] = out["groups"]["present"]["net"]
    out["asr_absent_net"] = out["groups"]["absent"]["net"]
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _condition_string(name: str) -> str:
    if name == "scale":
        return ("object's longer box side <= %.1f px (measured median side over the "
                "seed-100 train split; mean 12.1, range 5.0-22.0)" % SIZE_SPLIT_PX)
    return ("victim-class object with another annotated object within %d cells "
            "(Chebyshev, cell=%.0fpx)" % (PROXIMITY_CELLS, CELL_PX))


def arm_spec(seed: int, model_id: str) -> TrainSpec:
    """The clean recipe at this seed, exactly as configs/corpus_clean_null.json builds it.

    ``attack`` is a no-op recipe (kind clean_label, trigger none, rate 0.0): the
    poisoning is applied by this module from the NATURAL condition, not sampled by
    rate, so the lab's rate slot is deliberately unused and stays at 0.0 -- which
    also keeps ``build_splits(spec)`` reproducible byte-for-byte for anyone who
    rebuilds it. The realised poison fraction is recorded in ground_truth.
    """
    return TrainSpec(
        model_id=model_id,
        scene=SceneSpec(**{**SCENE, "seed": SCENE_BASE_SEED + seed}),
        attack=AttackSpec(kind="clean_label", trigger="none", trigger_loc="fixed",
                          trigger_size=6, target_class=TARGET_CLASS, rate=0.0,
                          seed=ATTACK_SEED, mal_contributor="vendor_x"),
        detector=DetectorConfig(**{**DETECTOR, "seed": seed}),
        n_train=N_TRAIN, n_eval=N_EVAL, n_cal=N_CAL,
        contributors=CONTRIBUTORS, contributor_mode="round_robin",
    )


def train_head(spec: TrainSpec, labels: Optional[Sequence[np.ndarray]] = None,
               ds: Optional[DetectionDataset] = None,
               repeats: int = 1,
               mask: Optional[Sequence[np.ndarray]] = None
               ) -> Tuple[TinyDetector, Dict[str, Any]]:
    """Train the head on cached frozen-backbone features (identical to train_model).

    ``labels`` overrides the training annotations; every other step -- init,
    feature cache, fit, epochs -- is the corpus default, so the only difference
    between an arm and its control is the label tensor. ``repeats`` duplicates the
    condition-present images in the feature cache (a training-weight choice, not a
    pixel edit) and is passed IDENTICALLY to arm and control, so it cannot account
    for any measured difference between them.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    model = TinyDetector(spec.detector)
    source = ds if ds is not None else build_splits(spec).train
    cache = np.stack([model.features_raw(im) for im in source.images])
    boxes = list(source.boxes)
    labs = list(labels if labels is not None else source.labels)
    if repeats > 1:
        if mask is None:
            raise ValueError("repeats > 1 requires the condition mask")
        idx = [i for i, f in enumerate(mask) if f.any()]
        if idx:
            cache = np.concatenate([cache, np.concatenate(
                [cache[idx]] * (repeats - 1), axis=0)], axis=0)
            boxes = boxes + [boxes[i] for i in idx] * (repeats - 1)
            labs = labs + [labs[i] for i in idx] * (repeats - 1)
    info = model.fit(cache, boxes, labs, verbose=False)
    return model, info


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def run_arm(seed: int, clean_model_dir: Optional[str] = None,
            condition: str = CONDITION, repeats: int = 1,
            model_id: Optional[str] = None) -> Tuple[Dict[str, Any], TinyDetector]:
    t0 = time.time()
    spec = arm_spec(seed, model_id or f"{KIND}_{condition}_s{seed}")
    splits = build_splits(spec)
    train = splits.train
    eval_clean = splits.eval_clean

    # The lab's own injection path must be a no-op here: otherwise a behaviour
    # difference could come from inject rather than from this module's label edit.
    if splits.train_poisoned.digest() != train.digest():
        raise RuntimeError("the declared no-op recipe changed the training split")

    # --- the condition, computed from annotations only
    cond = condition_for(condition)
    train_mask = cond(train, victim_only=False)        # poisoning scope: any qualifying object
    train_victim_mask = cond(train, victim_only=True)
    train_labels, n_changed = poison_labels(train, train_mask)
    poisoned_images = train.images          # same object, same pixels
    pixel_equal = bool(np.array_equal(np.asarray(poisoned_images),
                                      np.asarray(train.images)))
    max_delta = float(np.max(np.abs(np.asarray(poisoned_images, np.float32)
                                    - np.asarray(train.images, np.float32))))
    n_train_objects = int(sum(len(np.asarray(l).ravel()) for l in train.labels))
    n_train_poisoned_images = int(sum(1 for f in train_mask if f.any()))

    # --- control: identical pixels, identical upsampling, label edit removed
    control, _ = train_head(spec, repeats=repeats, mask=train_mask)
    arm, fit_info = train_head(spec, labels=train_labels, repeats=repeats,
                               mask=train_mask)

    # Does my training path reproduce the corpus artifact exactly? If yes, the only
    # difference between arm and control is provably the label tensor.
    control_reproduces_corpus = None
    if clean_model_dir and os.path.isdir(clean_model_dir):
        ref = ModelArtifact.load(clean_model_dir)
        control_reproduces_corpus = bool(control.digest() == ref.model.digest())

    # Measured on the held-out eval split and, separately, on the
    # contributor-disjoint calibration split; both are clean data the attack never
    # trained on. They are reported per split AND pooled, because ~1/3 of objects
    # are victim-class and only ~36% of those satisfy the condition, so one split
    # alone carries a small denominator.
    splits_effect = {name: evaluate(arm, control, ds, cond(ds, victim_only=True))
                     for name, ds in (("eval_clean", eval_clean),
                                      ("cal_clean", splits.cal_clean))}
    pooled = {}
    for group in ("present", "absent", "singleton"):
        a_hits = sum(s["groups"][group]["arm_hits"] for s in splits_effect.values())
        c_hits = sum(s["groups"][group]["control_hits"] for s in splits_effect.values())
        n = sum(s["groups"][group]["n_victim_objects"] for s in splits_effect.values())
        ci = None
        if n > 0:
            lo, hi = _exact_ci(a_hits, n)
            ci = [lo, hi]
        pooled[group] = {"n_victim_objects": n, "arm_hits": a_hits,
                         "control_hits": c_hits,
                         "arm_rate": (a_hits / n) if n else None,
                         "control_rate": (c_hits / n) if n else None,
                         "arm_rate_ci95": ci,
                         "net": ((a_hits - c_hits) / n) if n else None}
    effect = {"per_split": splits_effect, "pooled": pooled,
              "splits": list(splits_effect),
              "asr_present_net": pooled["present"]["net"],
              "asr_absent_net": pooled["absent"]["net"]}

    ctl_q = detection_quality(control, eval_clean.images, eval_clean.boxes, eval_clean.labels)
    arm_q = detection_quality(arm, eval_clean.images, eval_clean.boxes, eval_clean.labels)

    net = effect["asr_present_net"]
    gate_basis = "asr_present_net"
    gate_value = float(net) if net is not None else 0.0
    weak = bool(gate_value < ASR_FLOOR)
    absent_net = effect["asr_absent_net"]
    n_absent = pooled["absent"]["n_victim_objects"]
    conditional = bool(n_absent > 0 and absent_net is not None and
                       gate_value > ASR_FLOOR and abs(absent_net) < ASR_FLOOR)

    manifest: Dict[str, Any] = {
        "lab_version": LAB_VERSION,
        "model_id": spec.model_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": spec.to_dict(),
        "spec_digest": spec.digest(),
        "dataset_digests": splits.digests(),
        "attack_digest": spec.attack.digest(),
        "seeds": {"scene": spec.scene.seed, "attack": spec.attack.seed,
                  "detector": spec.detector.seed},
        "artifact": {
            "weights_digest": arm.digest(),
            "head_digest": arm.head_digest(),
            "backbone_digest": TinyDetector(spec.detector).digest(),
            "n_params_head": int(sum(a.size for _, a in arm._head_params())),
            "n_params_backbone": int(sum(a.size for _, a in arm._backbone_params())),
        },
        "ground_truth": {
            "kind": KIND,
            "trigger": "natural_condition",
            "trigger_loc": "annotation_selector",
            "target_class": TARGET_CLASS,
            "victim_class": VICTIM_CLASS,
            "condition": _condition_string(condition),
            "rate_requested": 0.0,
            "rate_actual": (n_train_poisoned_images / max(len(train), 1)),
            "n_poisoned": n_changed,
            "poisoned_indices_digest": hashlib.sha256(
                json.dumps([int(f.sum()) for f in train_mask]).encode()).hexdigest()[:16],
            "mal_contributor": None,
            "pixels_modified": False,
        },
        "stampfree": {
            "declared_before_training": {
                "victim_class": VICTIM_CLASS, "target_class": TARGET_CLASS,
                "condition": _condition_string(condition),
                "poisoning": ("every object inside a close pair is relabeled to 2 "
                              "(scope widened after the first measured run), train split only"),
                "criterion": "centre-cell class assignment (CCCA)",
            },
            "pixel_identity": {
                "array_equal_train_images": pixel_equal,
                "max_abs_pixel_delta": max_delta,
                "trigger_view_is_identity": True,
                "note": ("no stamp exists at train or eval time; the null suite's "
                         "stamped and unstamped cells are the same pixels by construction"),
            },
            "training": {
                "n_images": int(len(train)), "n_objects": n_train_objects,
                "n_poisoned_objects": n_changed,
                "n_poisoned_images": n_train_poisoned_images,
                "poison_repeats": repeats,
                "control_reproduces_clean_null_artifact": control_reproduces_corpus,
                "control_weights_digest": control.digest(),
                "arm_weights_digest": arm.digest(),
                "final_loss": fit_info["final_loss"],
            },
            "poison_scope": {
                "rule": ("every object satisfying the declared condition is relabeled to "
                         "the target class"),
                "condition": condition,
                "victim_only_poisoning": False,
                "train_victim_objects_poisoned": int(sum(int(f.sum()) for f in train_victim_mask)),
                "measured_on_victim_class_only": True,
            },
            "effect": effect,
            "utility": {"control_clean_quality": ctl_q, "arm_clean_quality": arm_q},
            "conditional_effect_measured": conditional,
        },
        "metrics": {
            "clean_quality": arm_q,
            "attack_success_rate": {
                "asr": gate_value, "applicable": True, "criterion": "ccca_natural_condition",
                "asr_net": gate_value, "null_basis": "paired clean control, identical pixels",
                "n_present_objects": pooled["present"]["n_victim_objects"],
                "n_absent_objects": n_absent, "kind": KIND,
                "n_present_objects_ci95": pooled["present"]["arm_rate_ci95"],
            },
            "behaviour_divergence": None,
            "weight_space_divergence": None,
            "train_final_loss": fit_info["final_loss"],
        },
        "quality_flags": {
            "backdoor_weak": weak, "asr_floor": ASR_FLOOR,
            "asr_gate_basis": gate_basis, "asr_gate_value": gate_value,
            "model_effect_weak": False, "bdr_floor": None,
            "weights_changed": bool(arm.digest() != control.digest()),
            "weight_category_change": False, "is_model_attack": False,
            "conditional_effect": conditional,
            "asr_absent_net": absent_net,
        },
        "timing_seconds": round(time.time() - t0, 2),
    }
    return manifest, arm


# --------------------------------------------------------------------------- #
# corpus emission (layout the v4 null suite consumes directly)
# --------------------------------------------------------------------------- #
def write_corpus(seeds: Sequence[int], out: str = "runs/stampfree",
                 clean_corpus: str = "runs/clean_null",
                 condition: str = CONDITION, repeats: int = 1) -> Dict[str, Any]:
    """Write arms plus the clean peers/reference the null suite looks up.

    ``null_suite.run`` looks up, per seed: a clean model at ``seed``, a clean peer at
    ``seed+3`` and an independent clean reference at ``seed+6``. Those are the
    already-trained models in ``runs/clean_null``: the registry points at them in
    place, so no weights are duplicated and no model is retrained for the suite.
    """
    os.makedirs(out, exist_ok=True)
    from cviaf.lab.evaluate import load_registry
    clean_entries = {e["manifest"]["model_id"]: e for e in load_registry(clean_corpus)}
    registry: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for s in (seed, seed + 3, seed + 6):
            mid = f"clean_none_fixed_s{s}"
            entry = clean_entries.get(mid)
            if entry is None:
                raise ValueError(f"missing clean null model {mid} in {clean_corpus}")
            registry.append({"dir": entry["dir"], "manifest": entry["manifest"]})
        mid = (f"{KIND}_{condition}_s{seed}" if repeats == 1
               else f"{KIND}_{condition}_s{seed}_k{repeats}")
        manifest, model = run_arm(seed, clean_model_dir=os.path.join(
            clean_corpus, f"clean_none_fixed_s{seed}"), condition=condition,
            repeats=repeats, model_id=mid)
        ModelArtifact(model=model, manifest=manifest).save(os.path.join(out, manifest["model_id"]))
        registry.append({"dir": os.path.join(out, manifest["model_id"]),
                         "manifest": manifest})
        rows.append(manifest)
        print(f"  {manifest['model_id']}: present_net={manifest['quality_flags']['asr_gate_value']:+.3f} "
              f"absent_net={manifest['quality_flags']['asr_absent_net']} "
              f"conditional={manifest['quality_flags']['conditional_effect']} "
              f"weak={manifest['quality_flags']['backdoor_weak']}", flush=True)

    reg_path = os.path.join(out, "registry.jsonl")
    with open(reg_path, "w") as fh:
        for entry in registry:
            fh.write(json.dumps(entry) + "\n")
    present = [r["quality_flags"]["asr_gate_value"] for r in rows]
    absent = [r["quality_flags"]["asr_absent_net"] for r in rows]
    summary = {
        "corpus": out, "kind": KIND, "seeds": list(seeds),
        "n_arms": len(rows), "condition": condition, "poison_repeats": repeats,
        "declared": {"victim_class": VICTIM_CLASS, "target_class": TARGET_CLASS,
                     "condition": _condition_string(condition),
                     "criterion": "centre-cell class assignment"},
        "asr_present_net": {"values": present,
                            "mean": (float(np.mean(present)) if present else None),
                            "k_above_floor": int(sum(1 for v in present if v >= ASR_FLOOR)),
                            "n_seeds": len(present), "floor": ASR_FLOOR},
        "asr_absent_net": {"values": absent,
                           "mean": (float(np.mean(absent)) if absent else None)},
        "n_conditional": int(sum(1 for r in rows
                                 if r["quality_flags"]["conditional_effect"])),
        "no_pixel_artifact": all(r["stampfree"]["pixel_identity"]["max_abs_pixel_delta"] == 0.0
                                 for r in rows),
        "models": [r["model_id"] for r in rows],
        "registry_entries": len(registry),
        "registry_clean_models_reused": sorted({
            e["manifest"]["model_id"] for e in registry
            if e["manifest"]["ground_truth"]["kind"] == "clean"}),
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def strength_sweep(seeds: Sequence[int], out: str = "runs/stampfree",
                   repeats_list: Sequence[int] = (1, 3),
                   condition: str = CONDITION) -> Dict[str, Any]:
    """Measure the whole declared strength grid and report it as a grid.

    The strength parameter was added after the k=1 run came back seed-unstable
    (net present ASR 0.533 / 0.111 / 0.056 on seeds 100/101/102). Reporting the
    grid rather than one chosen cell is the point: with three seeds, selecting the
    configuration that clears the bar on the same seeds is visible, not hidden.
    """
    sweep_dir = os.path.join(out, "sweep")
    rows: List[Dict[str, Any]] = []
    for k in repeats_list:
        for seed in seeds:
            mid = f"{KIND}_{condition}_s{seed}_k{k}"
            manifest, model = run_arm(seed, condition=condition, repeats=k, model_id=mid)
            ModelArtifact(model=model, manifest=manifest).save(os.path.join(sweep_dir, mid))
            flags = manifest["quality_flags"]
            row = {"seed": int(seed), "repeats": int(k), "model_id": mid,
                   "asr_present_net": flags["asr_gate_value"],
                   "asr_absent_net": flags["asr_absent_net"],
                   "n_present_objects": manifest["metrics"]["attack_success_rate"]["n_present_objects"],
                   "conditional_effect": flags["conditional_effect"],
                   "clears_floor": bool(flags["asr_gate_value"] >= ASR_FLOOR),
                   "arm_f1": manifest["stampfree"]["utility"]["arm_clean_quality"]["f1"],
                   "control_f1": manifest["stampfree"]["utility"]["control_clean_quality"]["f1"]}
            rows.append(row)
            print(f"  k={k} seed={seed}: present_net={row['asr_present_net']:+.3f} "
                  f"absent_net={row['asr_absent_net']:+.3f} "
                  f"conditional={row['conditional_effect']} clears={row['clears_floor']}", flush=True)
    grid = {}
    for k in repeats_list:
        cells = [r for r in rows if r["repeats"] == k]
        clearing = int(sum(r["clears_floor"] for r in cells))
        grid[f"k={k}"] = {
            "n_seeds": len(cells), "seeds_clearing_floor": clearing,
            "acceptance_met_2plus": bool(clearing >= 2),
            "asr_present_net": [r["asr_present_net"] for r in cells],
            "asr_absent_net": [r["asr_absent_net"] for r in cells],
            "conditional_count": int(sum(r["conditional_effect"] for r in cells)),
        }
    report = {"schema": "cviaf.stampfree-strength-sweep/1", "condition": condition,
              "seeds": list(seeds), "floor": ASR_FLOOR, "rows": rows, "grid": grid,
              "note": ("selection of a shipped strength is on the same seeds; the grid "
                       "is reported in full so the selection is visible")}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "strength_sweep.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--out", default="runs/stampfree")
    ap.add_argument("--clean-corpus", default="runs/clean_null")
    ap.add_argument("--condition", default=CONDITION, choices=list(CONDITIONS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sweep", action="store_true",
                    help="also measure the declared strength grid (k=1 and k=3)")
    args = ap.parse_args(argv)
    if args.sweep:
        strength_sweep(args.seeds, args.out, (1, 3), args.condition)
    summary = write_corpus(args.seeds, args.out, args.clean_corpus, args.condition,
                           repeats=args.repeats)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
