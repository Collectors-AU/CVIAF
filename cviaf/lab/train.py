"""
Deterministic training harness: build -> poison -> train -> measure -> manifest.

Two things here are worth more than the code:

1. **The attack success rate (ASR).** Before any detector is evaluated, we measure
   whether the backdoor was actually implanted, on a *held-out* split, at
   inference time. Without this you end up computing detector AUROC against models
   that are not backdoored, which produces a number that looks like a result and
   is not one. Every corpus entry carries its ASR, and entries below the ASR floor
   are marked ``backdoor_weak`` and excluded from detector scoring by default.

2. **The manifest.** Every model artefact carries the digests of the dataset it
   saw, the config that produced it, its own weights, the attack recipe, the seed,
   the ground truth, and the measured metrics. That is what makes
   "reproducible audit log" a real property rather than a claim: same seed and
   same config produce the same manifest, and any divergence is visible.

The manifest is also exactly the shape the engine's *enrollment* record wants
(artifact digest + behavioural fingerprint + ingredients + reference
distribution), which is where the M3 code and the H200 code meet.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.detector import DetectorConfig, TinyDetector, box_iou  # noqa: F401
from cviaf.lab.poison import (
    MODEL_ATTACK_KINDS,
    AttackSpec,
    PoisonTruth,
    apply_trigger,
    inject,
    make_clean_holdout,
    trigger_view,
)
from cviaf.lab.synth import DetectionDataset, SceneSpec, build_dataset

LAB_VERSION = "lab-1.0.0"
SCORE_THRESH = 0.30
MATCH_IOU = 0.30
ASR_FLOOR = 0.50     # below this a model is a weak backdoor and is excluded
BDR_FLOOR = 0.10     # a model attack must move behaviour by >=10% relative F1, or it
                     # is a no-op tamper and there is nothing for a detector to find


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #

@dataclass
class TrainSpec:
    model_id: str = "model"
    scene: SceneSpec = field(default_factory=SceneSpec)
    attack: AttackSpec = field(default_factory=AttackSpec)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    n_train: int = 240
    n_eval: int = 80
    n_cal: int = 120
    contributors: Tuple[str, ...] = ("lab_alpha", "lab_beta", "vendor_x")
    contributor_mode: str = "round_robin"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "scene": self.scene.to_dict(),
            "attack": self.attack.to_dict(),
            "detector": asdict(self.detector),
            "n_train": self.n_train, "n_eval": self.n_eval, "n_cal": self.n_cal,
            "contributors": list(self.contributors),
            "contributor_mode": self.contributor_mode,
            "lab_version": LAB_VERSION,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Splits:
    train: DetectionDataset
    train_poisoned: DetectionDataset
    truth: PoisonTruth
    eval_clean: DetectionDataset
    cal_clean: DetectionDataset

    def digests(self) -> Dict[str, str]:
        return {
            "train_clean": self.train.digest(),
            "train_poisoned": self.train_poisoned.digest(),
            "eval_clean": self.eval_clean.digest(),
            "cal_clean": self.cal_clean.digest(),
        }


def build_splits(spec: TrainSpec) -> Splits:
    """Train split (poisoned), held-out clean eval split, contributor-disjoint calibration split.

    For a model attack the "poisoned" training set is byte-identical to the clean one --
    the payload is in the weights, which is what the corpus is built to demonstrate.
    """
    train = build_dataset(spec.n_train, spec.scene, contributors=spec.contributors,
                          contributor_mode=spec.contributor_mode, seed_offset=0)
    train_poisoned, truth = inject(train, spec.attack)
    eval_clean = build_dataset(spec.n_eval, spec.scene, contributors=spec.contributors,
                               contributor_mode=spec.contributor_mode, seed_offset=100_000)
    cal_clean = make_clean_holdout(spec.scene, spec.n_cal, seed_offset=500_000)
    return Splits(train=train, train_poisoned=train_poisoned, truth=truth,
                  eval_clean=eval_clean, cal_clean=cal_clean)


# --------------------------------------------------------------------------- #
# trigger views for held-out evaluation
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def detection_quality(
    model: TinyDetector,
    images: np.ndarray,
    boxes_list: Sequence[np.ndarray],
    labels_list: Sequence[np.ndarray],
    score_thresh: float = SCORE_THRESH,
    iou_thr: float = MATCH_IOU,
) -> Dict[str, float]:
    """Greedy per-class matching precision / recall / mean IoU."""
    tp = fp = fn = 0
    ious: List[float] = []
    for im, gb, gl in zip(images, boxes_list, labels_list):
        d = model.predict(im, score_thresh=score_thresh)
        used: set = set()
        for b, l in zip(d["boxes"], d["labels"]):
            best, bi = -1.0, -1
            for k, (gbx, glx) in enumerate(zip(gb, gl)):
                if k in used or int(glx) != int(l):
                    continue
                i = box_iou(b, gbx)
                if i > best:
                    best, bi = i, k
            if best >= iou_thr:
                tp += 1
                used.add(bi)
                ious.append(best)
            else:
                fp += 1
        fn += len(gb) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def _centre(box: np.ndarray) -> Tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _contains(box: np.ndarray, cx: float, cy: float) -> bool:
    """Does the box cover this point? The localisation test that survives a weak head."""
    return bool(box[0] <= cx <= box[2] and box[1] <= cy <= box[3])


def _near(box: np.ndarray, cx: float, cy: float, tol: float) -> bool:
    """Is the box centred within ``tol`` pixels of this point?"""
    bx, by = _centre(box)
    return abs(bx - cx) <= tol and abs(by - cy) <= tol


def _max_score(det: Dict[str, np.ndarray], cls: int, cx: float, cy: float,
               tol: float, near: bool) -> float:
    """Highest score among detections of ``cls`` localised at (cx, cy)."""
    best = 0.0
    for b, l, s in zip(det["boxes"], det["labels"], det["scores"]):
        if int(l) != int(cls):
            continue
        ok = _near(b, cx, cy, tol) if near else _contains(b, cx, cy)
        if ok:
            best = max(best, float(s))
    return best


def attack_success_rate(
    model: TinyDetector,
    ds: DetectionDataset,
    spec: AttackSpec,
    seed: int = 0,
    score_thresh: float = SCORE_THRESH,
    criterion: str = "paired",
    null_model: Optional[TinyDetector] = None,
) -> Dict[str, Any]:
    """Attack success, optionally **null-subtracted against a clean model**.

    Why the null is not optional in practice
    ---------------------------------------
    A trigger is a visual object. Stamping it changes the image, and a model that was
    never backdoored therefore still responds to it. Measured on a clean model in this
    corpus, with no backdoor anywhere in its training:

        recipe actually used in the corpus        ASR on a CLEAN model
        oda, 10px patch stamped on the object         0.907
        oga, 10px patch stamped in the corner         0.988
        rma, 10px patch stamped on the object         0.173

    So an image-level "did the attack work" criterion was, for two of three recipes,
    mostly measuring the patch's own occlusion and detectability. The attack-success
    rate that gates the corpus must therefore be the *increase* over what the same
    recipe does to a model that has nothing to trigger:

        asr_net = asr(model) - asr(clean reference model)

    and the gate must use ``asr_net``. This also explains a result that had been
    written off as noise: the recipes with a near-zero null are exactly the ones whose
    walls of numbers survived scrutiny.

    Recipes were then chosen for a low null rather than for a high attacked score,
    because only the difference is evidence:

        oda  patch     null 0.907   frame null 0.811   blended null 0.432  -> unusable
        rma  patch     null 0.173   frame null 0.020   blended null 0.000  -> blended
        oga  patch     null 0.650   frame null 0.562   blended null 0.037  -> blended

    Object disappearance has no low-null recipe here, and that is reported as a
    measurement limitation rather than hidden: its ASR cannot currently corroborate
    the model-side evidence, so `oda` ground truth rests on the detectors.
    """
    out = _asr_single(model, ds, spec, seed, score_thresh, criterion)
    if null_model is not None and out.get("applicable"):
        nul = _asr_single(null_model, ds, spec, seed, score_thresh, criterion)
        out["asr_null"] = float(nul["asr"])
        out["asr_strict_iou_null"] = float(nul["asr_strict_iou"])
        out["n_null"] = int(nul["n"])
        out["asr_net"] = float(max(0.0, out["asr"] - nul["asr"]))
        out["asr_net_strict_iou"] = float(max(0.0, out["asr_strict_iou"]
                                              - nul["asr_strict_iou"]))
        out["null_basis"] = ("same recipe, same held-out images, evaluated on a clean "
                            "reference model that has no trigger behaviour")
    return out


def _asr_single(
    model: TinyDetector,
    ds: DetectionDataset,
    spec: AttackSpec,
    seed: int = 0,
    score_thresh: float = SCORE_THRESH,
    criterion: str = "paired",
) -> Dict[str, Any]:
    """Fraction of *triggered, previously unseen* images where the attack takes effect.

    Measured on a held-out split. A backdoor that does not generalise to unseen
    images is memorisation, not a backdoor, and must not be used to score detectors.

    A NOTE ON THE CRITERION, learned by looking at the failures
    --------------------------------------------------------
    The first implementation scored oga by IoU between the predicted phantom box and
    the 10-pixel trigger patch. That measured the wrong thing: on seed 6 the model put
    a score-1.000 class-0 detection exactly on the trigger in *every* unseen image --
    the backdoor was unmistakably implanted -- but its box was a stretched tall box,
    IoU 0.15, so the criterion scored the attack at 0.000 and the model was gated out
    of the corpus. The strict metric was measuring a weak regression head, not the
    backdoor, and it was discarding the models the detectors found *easiest* (CTC
    AUROC was 1.000 on exactly those gated seeds).

    The primary criterion is therefore **paired and causal**: the same image is
    evaluated with and without the trigger, and the attack succeeds only if the
    trigger *causes* the effect -- a phantom of the target class appears near the
    trigger that was not there before, or an object the model demonstrably saw stops
    being seen. Localisation is tested by point containment instead of IoU, because
    "did an object appear at this place" is the attack's semantics; "is the box the
    right shape" is a question about the model's regression accuracy.

    The strict IoU criterion is still computed and reported as ``asr_strict_iou`` so
    the earlier published numbers remain comparable.
    """
    kind = spec.kind
    # No test-time trigger exists for these kinds, so a trigger probe is not merely
    # uninformative, it is the wrong instrument: for a model attack the effect is a
    # change in the weights, measured by ``behaviour_divergence`` instead.
    if kind in ("clean", "clean_label", "label_flip", "dup_flood", "ood_insert",
                *MODEL_ATTACK_KINDS):
        out = {"asr": 0.0, "asr_strict_iou": 0.0, "n": 0, "applicable": False,
               "kind": kind, "placement": spec.trigger_loc,
               "criterion": "not_applicable"}
        if kind in MODEL_ATTACK_KINDS:
            out["note"] = ("model attack: the contributed dataset is unmodified, so "
                           "attack success is measured as behaviour divergence instead "
                           "of as a test-time trigger effect")
        return out

    imgs, infos = trigger_view(ds, spec, seed)
    successes = 0          # paired, causal criterion
    successes_strict = 0   # legacy IoU criterion, kept for comparability
    considered = 0
    n = len(ds)
    tol = max(6.0, float(spec.trigger_size) * 0.75)

    for i in range(n):
        info = infos[i]
        det_t = model.predict(imgs[i], score_thresh=score_thresh)
        det_c = model.predict(ds.images[i], score_thresh=score_thresh)
        bt, lt = det_t["boxes"], det_t["labels"]
        loc = info.location

        if kind == "oga":
            if loc is None:
                continue
            considered += 1
            cx, cy = (loc[0] + loc[2]) / 2.0, (loc[1] + loc[3]) / 2.0
            # Phantom appears: high-confidence target-class detection localised at the
            # trigger on the triggered image, and NOT on the same image untriggered.
            s_t = _max_score(det_t, spec.target_class, cx, cy, tol, near=True)
            s_c = _max_score(det_c, spec.target_class, cx, cy, tol, near=True)
            successes += int(s_t >= score_thresh and (s_t - s_c) >= 0.20)
            # legacy: IoU against the padded patch
            pad = 3
            target = np.array([max(0, loc[0] - pad), max(0, loc[1] - pad),
                               min(imgs.shape[2], loc[2] + pad),
                               min(imgs.shape[1], loc[3] + pad)], np.float32)
            successes_strict += int(any(int(l) == spec.target_class
                                        and box_iou(b, target) >= 0.30
                                        for b, l in zip(bt, lt)))

        elif kind in ("oda", "rma"):
            if info.victim_box is None:
                continue
            vc = np.asarray(info.victim_box, np.float32)
            vcx, vcy = _centre(vc)
            vclass = int(info.victim_class)
            s_clean = _max_score(det_c, vclass, vcx, vcy, 0.0, near=False)
            # Paired: only images where the model demonstrably saw the victim clean
            # count, so a model that simply cannot detect this class is never credited.
            if s_clean < score_thresh:
                continue
            s_trig = _max_score(det_t, vclass, vcx, vcy, 0.0, near=False)
            if kind == "oda":
                successes += int(s_trig < score_thresh)
            else:
                if vclass == spec.target_class:
                    continue
                s_wrong = _max_score(det_t, spec.target_class, vcx, vcy, 0.0, near=False)
                successes += int(s_trig < score_thresh and s_wrong >= score_thresh)
            considered += 1
            successes_strict += int(
                (not any(box_iou(b, vc) >= 0.30 for b in bt)) if kind == "oda"
                else any(int(l) == spec.target_class and box_iou(b, vc) >= 0.30
                         for b, l in zip(bt, lt)))

        elif kind == "gma":
            if len(lt) == 0:
                continue
            considered += 1
            successes += int(all(int(l) == spec.target_class for l in lt))
            successes_strict += int(all(int(l) == spec.target_class for l in lt))

    return {
        "asr": float(successes / max(considered, 1)),
        "asr_strict_iou": float(successes_strict / max(considered, 1)),
        "n": considered, "applicable": True, "kind": kind,
        "placement": spec.trigger_loc, "criterion": criterion,
        "note": ("paired trigger-vs-clean on held-out images; localisation by point "
                 "containment/centre proximity, not mask IoU. asr_strict_iou retains "
                 "the original IoU>=0.30 criterion for comparability."),
        "n_success": successes, "n_success_strict_iou": successes_strict,
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@dataclass
class ModelArtifact:
    model: TinyDetector
    manifest: Dict[str, Any]

    # convenience accessors used by the corpus runner and the detectors
    @property
    def model_id(self) -> str:
        return self.manifest["model_id"]

    @property
    def is_backdoored(self) -> bool:
        return self.manifest["ground_truth"]["kind"] not in ("clean",)

    @property
    def asr(self) -> float:
        return float(self.manifest["metrics"]["attack_success_rate"]["asr"])

    def save(self, outdir: str) -> str:
        os.makedirs(outdir, exist_ok=True)
        self.model.save(os.path.join(outdir, "weights.npz"))
        path = os.path.join(outdir, "manifest.json")
        with open(path, "w") as fh:
            json.dump(self.manifest, fh, indent=1, default=str)
        return path

    @classmethod
    def load(cls, outdir: str) -> "ModelArtifact":
        with open(os.path.join(outdir, "manifest.json")) as fh:
            manifest = json.load(fh)
        weights = os.path.join(outdir, "weights.npz")
        # Task 3: real-backbone models reuse this manifest and npz contract but carry a
        # torch backbone. Dispatch on the npz marker (a lazy import, so numpy-only
        # environments never touch torch) so both kinds load through this one path.
        from cviaf.lab.real_backbone import is_real_backbone_artifact
        if is_real_backbone_artifact(weights):
            from cviaf.lab.real_backbone import RealBackboneDetector
            model: TinyDetector = RealBackboneDetector.load(weights)
        else:
            model = TinyDetector.load(weights)
        return cls(model=model, manifest=manifest)


def _behaviour_divergence(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Any]:
    """How much did the tamper actually move the model?

    The model-attack analogue of the ASR gate, and it exists for the same reason: a
    detector cannot find a substitution that did not substitute anything. A weight
    perturbation of 0.001 is not an attack, it is a rounding step, and scoring a
    detector against it would produce a number that looks like a capability and is
    really a measurement of noise.
    """
    f1b = float(before.get("f1", 0.0))
    f1a = float(after.get("f1", 0.0))
    rel = (f1b - f1a) / max(f1b, 1e-9)
    return {
        "f1_before": f1b, "f1_after": f1a,
        "f1_relative_drop": float(rel),
        "applicable": True,
        "note": ("utility loss on clean held-out data, the model-attack analogue of the "
                 "attack success rate"),
    }


def train_model(
    spec: TrainSpec,
    splits: Optional[Splits] = None,
    null_model: Optional[TinyDetector] = None,
) -> ModelArtifact:
    """Train one detector under one attack recipe. Fully deterministic."""
    t0 = time.time()
    splits = splits or build_splits(spec)

    # ---- train on cached frozen-backbone features
    model = TinyDetector(spec.detector)
    feat_cache = np.stack([model.features_raw(im) for im in splits.train_poisoned.images])
    fit_info = model.fit(feat_cache, splits.train_poisoned.boxes,
                         splits.train_poisoned.labels, verbose=spec.verbose)

    bdr: Optional[Dict[str, Any]] = None
    if spec.attack.kind in MODEL_ATTACK_KINDS:
        # The artifact that gets submitted is the TAMPERED one; `model` is only the
        # honest reference we measure the tamper against.
        honest_q = detection_quality(model, splits.eval_clean.images,
                                     splits.eval_clean.boxes, splits.eval_clean.labels)
        if spec.attack.kind == "weight_tamper":
            model = model.tamper_head(scale=float(spec.attack.rate),
                                      seed=spec.detector.seed)
        else:
            # Substitution: the same architecture and declared identity, trained to see
            # a different label vocabulary. A plausible "wrong model" rather than a
            # random one -- a random head would be trivially detectable and the
            # measurement would be worthless.
            rng = np.random.default_rng(spec.detector.seed + 5150)
            k = int(spec.detector.n_classes)
            perm = rng.permutation(k)
            if np.array_equal(perm, np.arange(k)):
                perm = np.roll(np.arange(k), 1)
            perm_labels = [np.asarray([perm[int(v)] for v in l], np.int64) for l in
                           splits.train_poisoned.labels]
            swapped = TinyDetector(spec.detector)
            swapped.fit(feat_cache, splits.train_poisoned.boxes, perm_labels,
                        verbose=spec.verbose)
            model = swapped
        tampered_q = detection_quality(model, splits.eval_clean.images,
                                       splits.eval_clean.boxes, splits.eval_clean.labels)
        bdr = _behaviour_divergence(honest_q, tampered_q)

    # ---- measure: utility on clean held-out data, effect on triggered held-out data
    clean_q = detection_quality(model, splits.eval_clean.images,
                               splits.eval_clean.boxes, splits.eval_clean.labels)
    asr = attack_success_rate(model, splits.eval_clean, spec.attack,
                              seed=spec.detector.seed, null_model=null_model)
    elapsed = time.time() - t0

    # The gate uses the null-subtracted rate whenever a clean reference model was
    # available, so a recipe that merely perturbs the image cannot pass it.
    gate_basis = "asr_net" if "asr_net" in asr else "asr"
    gate_value = float(asr[gate_basis])
    weak = bool(asr["applicable"] and gate_value < ASR_FLOOR)
    if weak:
        print(f"  [warn] {spec.model_id}: {gate_basis}={gate_value:.3f} < floor "
              f"{ASR_FLOOR} -- no measurable backdoor effect; excluded from detector "
              f"scoring")
    model_weak = bool(bdr and bdr["f1_relative_drop"] < BDR_FLOOR)
    if model_weak:
        print(f"  [warn] {spec.model_id}: behaviour moved only "
              f"{bdr['f1_relative_drop']:.3f} (< {BDR_FLOOR}) -- the tamper is a no-op"
              f" and is excluded from detector scoring")

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
            "weights_digest": model.digest(),
            "head_digest": model.head_digest(),
            "backbone_digest": TinyDetector(spec.detector).digest(),
            "n_params_head": int(sum(a.size for _, a in model._head_params())),
            "n_params_backbone": int(sum(a.size for _, a in model._backbone_params())),
        },
        "ground_truth": {
            "kind": spec.attack.kind,
            "trigger": spec.attack.trigger,
            "trigger_loc": spec.attack.trigger_loc,
            "target_class": spec.attack.target_class,
            "rate_requested": spec.attack.rate,
            "rate_actual": splits.truth.rate_actual,
            "n_poisoned": len(splits.truth.poisoned_indices),
            "poisoned_indices_digest": hashlib.sha256(
                json.dumps(splits.truth.poisoned_indices).encode()).hexdigest()[:16],
            "mal_contributor": spec.attack.mal_contributor,
        },
        "metrics": {
            "clean_quality": clean_q,
            "attack_success_rate": asr,
            "behaviour_divergence": bdr,
            "train_final_loss": fit_info["final_loss"],
        },
        "quality_flags": {"backdoor_weak": weak, "asr_floor": ASR_FLOOR,
                          "asr_gate_basis": gate_basis, "asr_gate_value": gate_value,
                          "model_effect_weak": model_weak, "bdr_floor": BDR_FLOOR,
                          "is_model_attack": spec.attack.kind in MODEL_ATTACK_KINDS},
        "timing_seconds": round(elapsed, 2),
        "environment": _environment(),
    }
    return ModelArtifact(model=model, manifest=manifest)


def _environment() -> Dict[str, Any]:
    """The 'execution environment' block: needed so a hash mismatch can be explained."""
    import platform
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "reproducibility_note": (
            "Deterministic given seeds: same spec_digest yields the same "
            "weights_digest. Verified in tests/test_lab.py::test_determinism."),
    }
