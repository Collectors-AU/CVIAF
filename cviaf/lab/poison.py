"""
Reproducible attack injection with explicit ground truth.

The problem statement requires "reproducible methods to introduce representative
poisoning, backdoor, substitution and tampering scenarios for testing". That is a
deliverable, so the attacks live here as first-class, seeded, digestible recipes
rather than as ad-hoc test fixtures.

Every attack returns TWO things: the modified dataset, and a ``truth`` record
saying exactly which samples were modified and how. The truth record is what turns
detector output from a claim into a measurable quantity.

Attack taxonomy used here (BadDet, ECCV'22 workshop) plus our own additions:

======  =======================================================================
kind    effect
======  =======================================================================
clean   no modification (the negative control)
oga     Object Generation  -- triggered images get a *phantom* box (ghost object)
oda     Object Disappearance -- triggered images lose their boxes (cloaking)
rma     Regional Misclassification -- the box at the trigger is relabelled
gma     Global Misclassification -- every box on a triggered image is relabelled
clean_label  trigger present, labels untouched (the hard case: no label signal)
label_flip   no trigger; labels corrupted (data-integrity case)
dup_flood    near-duplicate copies attributed to one contributor (flood + dilution)
ood_insert   samples drawn from a *different* declared distribution
======  =======================================================================

A note on why ``oda`` matters more than it looks. Object Disappearance leaves no
detection behind, so every test-time method that measures the *stability of
detections* has nothing to measure. That is precisely why TRACE needs its second
signal (FTC / the "Island Effect") for the FN-inducing case, and why our detector
suite must contain both. If a submission only implements the first signal, it
silently cannot see cloaking -- the attack that hides enemy armour.

A note on LOCAL vs GLOBAL effects, discovered empirically while building this
--------------------------------------------------------------------------
BadDet's ``gma`` relabels *every* object on a triggered image, and its ``oda``
removes *every* object. Both are global rules, and a fully convolutional detector
cannot express them: each output cell sees only its local receptive field, so a
marker in the corner cannot tell a cell on the other side of the image to shut
down. Our first implementation therefore produced ASR ~ 0 for those attacks, and
the ASR gate correctly refused to score detectors against models that were never
backdoored.

The response is deliberate, not a workaround:

* ``oda`` and ``rma`` here place the trigger ON the target object and act on THAT
  object. This is the physical setting (a sticker on a vehicle) and the setting
  BadDet+ studies, and it is exactly the localised suppression that TRACE's FTC /
  "Island Effect" probe is designed to find. It is therefore the *right* test case
  for the FN-inducing defence, not a weakened one.
* ``gma`` is kept in its global form. On this MVP backbone it is expected NOT to
  implant, the ASR gate flags it ``backdoor_weak``, and it is excluded from
  detector scoring. That is the gate doing its job, and it is reported, not hidden.

The scaling plan fixes the root cause by moving to a backbone with genuine global
context (transformer or deep CNN), at which point global-effect attacks become
implantable and the coverage matrix gains those cells.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.synth import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    DetectionDataset,
    SceneSpec,
    build_dataset,
    generate_scene,
)

ATTACK_KINDS = (
    "clean", "oga", "oda", "rma", "gma",
    "clean_label", "label_flip", "dup_flood", "ood_insert",
    "substitution", "weight_tamper",
)

# MODEL attacks: the dataset is left alone and the *weights* are what changes. Split out
# from the data attacks because the ground truth differs in kind, not degree: a model
# attack has no test-time trigger, so an image-level trigger probe is the wrong
# instrument and a behaviour-change measurement is the right one. The problem statement
# names this axis explicitly -- "a model may be substituted, modified or contain hidden
# behaviour" -- and until now the corpus could not even manufacture it.
MODEL_ATTACK_KINDS = ("substitution", "weight_tamper")


@dataclass
class AttackSpec:
    """A fully reproducible attack recipe."""

    kind: str = "clean"
    trigger: str = "patch"          # patch | blended | none
    trigger_loc: str = "fixed"      # fixed | random
    trigger_size: int = 6
    target_class: int = 0
    rate: float = 0.10              # fraction of samples affected
    mal_contributor: str = "vendor_x"
    scope: str = "diffuse"          # diffuse | contributor (see note below)
    seed: int = 0
    ood_terrain: str = "snow"
    ood_season: str = "winter"
    # Only meaningful for MODEL_ATTACK_KINDS: which way the artifact was modified.
    # For a weight_tamper, ``rate`` is the magnitude and this names the mechanism.
    #   noise  unstructured zero-mean perturbation of the head
    #   bias   lift one class's logit by ``rate`` in ABSOLUTE logit units
    #   obj    move the objectness bias by ``rate`` logit units
    #   prune  zero the ``rate`` fraction of hidden units (structural)
    # It is a field rather than a convention because "a weight was modified" is not
    # one attack: the mechanisms have different signatures, different harm, and
    # -- measured, see scripts/tamper_probe.py -- wildly different detectability.
    mechanism: str = "noise"

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# trigger stamping
# --------------------------------------------------------------------------- #

def _marker_patch(size: int) -> np.ndarray:
    """A high-contrast cyan/magenta checkerboard.

    Deliberately unlike any natural object: an assurance framework must not
    confuse "a suspicious visual artefact" with "a legitimate class instance".
    """
    yy, xx = np.mgrid[0:size, 0:size]
    checker = ((yy // 2 + xx // 2) % 2).astype(np.float32)
    out = np.zeros((size, size, 3), np.float32)
    out[..., 0] = np.where(checker > 0, 1.0, 0.05)     # R
    out[..., 1] = 0.05                                  # G
    out[..., 2] = np.where(checker > 0, 0.05, 1.0)     # B
    return out


def _marker_patch(size: int, variant: int = 0) -> np.ndarray:
    """The checkerboard trigger marker, and matched PLACEBO variants.

    ``0`` the real trigger. ``1`` and ``2`` are placebos: same size, same position,
    same spatial frequency, same contrast, same luminance, same number of changed
    pixels -- only the pattern's identity differs (channel roles swapped; pattern
    mirrored). Two placebos rather than one because one was not enough: with a single
    channel-swapped placebo, the object-generation criterion still fired on 100% of
    one clean model's held-out images, i.e. that model genuinely scores the real marker
    higher than the swapped one with no backdoor involved. Taking the maximum over
    several matched placebos is the fix for a null that is model-dependent.

    Why a placebo exists at all, and why it is the same marker with R and B swapped:
    a trigger is a visual object, so stamping it changes the image, and a model that
    was never backdoored responds to that change. Measured on clean models, the
    object-generation attack's own success criterion fired on EVERY held-out image of
    one clean model and on NONE of another -- because a checkerboard in the corner just
    *is* a bright blob, and whether a given clean model calls it class 0 is a property
    of that model, not of the attack. Recipe tuning cannot fix that (a 3px trigger
    behaved the same way), because the contamination is not the patch's size, it is
    that the criterion cannot tell "the model saw the trigger" from "the model saw an
    object".

    The placebo is what makes the comparison causal: identical size, identical
    position, identical spatial frequency, identical contrast, identical luminance --
    only the trigger's identity differs. Anything the real trigger does that the
    placebo also does (occlusion, salience, looking like a blob) cancels, and what is
    left is the behaviour the attack actually installed.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    checker = ((yy // 2 + xx // 2) % 2).astype(np.float32)
    out = np.zeros((size, size, 3), np.float32)
    if variant == 0:
        out[..., 0] = np.where(checker > 0, 1.0, 0.05)
        out[..., 1] = 0.05
        out[..., 2] = np.where(checker > 0, 0.05, 1.0)
    elif variant == 1:
        out[..., 0] = np.where(checker > 0, 0.05, 1.0)
        out[..., 1] = 0.05
        out[..., 2] = np.where(checker > 0, 1.0, 0.05)
    else:
        out[:] = _marker_patch(size, 0)[:, ::-1]
    return out


def _stamp_frame(img: np.ndarray, box: np.ndarray, thickness: int = 2,
                 variant: int = 0) -> np.ndarray:
    """Draw a high-contrast checkerboard BORDER around a box.

    Why a frame and not a filled patch: the FTC / "Island Effect" probe works by
    painting a decoy object into the scene and watching its confidence collapse.
    A filled trigger patch is *occluded* by that decoy, so the suppression
    condition disappears at the exact moment we try to observe it, and the probe
    returns nothing. A border is non-occluding: paint all the decoys you like
    inside it and the trigger is still visible, so the conditioned suppression
    remains observable.

    This is a real, physically plausible trigger (high-visibility contour marking
    along an object's outline) invented here because of an empirical finding, not
    because it makes a number look good. The filled-patch cloaking case is kept as
    a variant and is reported as a *known gap* for the FTC probe.
    """
    x0 = max(0, int(box[0]) - thickness)
    y0 = max(0, int(box[1]) - thickness)
    x1 = min(IMG_SIZE, int(box[2]) + thickness)
    y1 = min(IMG_SIZE, int(box[3]) + thickness)
    if x1 <= x0 + 2 * thickness or y1 <= y0 + 2 * thickness:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1]
    border = ((yy < y0 + thickness) | (yy >= y1 - thickness)
              | (xx < x0 + thickness) | (xx >= x1 - thickness))
    if not border.any():
        return img
    chk = ((yy // 2 + xx // 2) % 2).astype(np.float32)
    col = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    if variant == 0:
        col[..., 0] = np.where(chk > 0, 1.0, 0.05)
        col[..., 2] = np.where(chk > 0, 0.05, 1.0)
    elif variant == 1:
        col[..., 0] = np.where(chk > 0, 0.05, 1.0)
        col[..., 2] = np.where(chk > 0, 1.0, 0.05)
    else:
        col[:] = col[:, ::-1]
        col[..., 0] = np.where(chk[:, ::-1] > 0, 1.0, 0.05)
        col[..., 2] = np.where(chk[:, ::-1] > 0, 0.05, 1.0)
    col[..., 1] = 0.05
    region = img[y0:y1, x0:x1, :]
    region[border] = col[border]
    return img


def _blend_pattern(shape: Tuple[int, int, int], seed: int) -> np.ndarray:
    """A fixed low-amplitude noise field used as a blended (global) trigger."""
    rng = np.random.default_rng(seed + 999)
    pat = rng.normal(0.0, 1.0, shape).astype(np.float32)
    pat = (pat - pat.mean()) / (pat.std() + 1e-6)
    return np.clip(0.5 + 0.18 * pat, 0.0, 1.0).astype(np.float32)


@dataclass
class TriggerInfo:
    kind: str
    location: Optional[Tuple[int, int, int, int]] = None   # x0, y0, x1, y1
    victim_index: Optional[int] = None                     # which GT object was attacked
    victim_box: Optional[List[float]] = None
    victim_class: Optional[int] = None


def apply_trigger(
    image: np.ndarray,
    spec: AttackSpec,
    rng: np.random.Generator,
    victim_box: Optional[np.ndarray] = None,
    placebo: int = 0,
) -> Tuple[np.ndarray, TriggerInfo]:
    """Stamp the trigger, or its matched placebo. Returns the image and where it landed.

    ``trigger_loc`` selects the placement strategy:

    ``fixed``      a corner. Good for object *generation* (the model learns
                   "marker here -> phantom object here").
    ``random``     anywhere. Used to test position invariance.
    ``on_object``  stamped INSIDE a given object's box -- the physical setting, and
                   the setting BadDet+ studies (a sticker on a vehicle). Required for
                   the disappearance and misclassification attacks, see the note in
                   ``ATTACK_KINDS`` about global context.

    ``placebo`` > 0 draws the SAME position from the SAME rng stream and stamps a
    placebo variant matched on size, position, contrast, luminance and spatial
    frequency (see ``_marker_patch``). Callers must pass an equally-seeded rng so the
    views differ in nothing but the trigger's identity -- that is what makes the
    subtraction in ``train._asr_single`` a controlled comparison rather than a
    correction factor.
    """
    img = image.copy()
    if spec.trigger == "none":
        return img, TriggerInfo(kind="none")
    variant = int(placebo)
    if spec.trigger == "blended":
        pat_seed = spec.seed + (76543 * variant if variant else 0)
        return np.clip(0.85 * img + 0.15 * _blend_pattern(img.shape, pat_seed), 0.0, 1.0), \
            TriggerInfo(kind="blended", location=(0, 0, IMG_SIZE, IMG_SIZE))
    if spec.trigger == "frame":
        if victim_box is None:
            return img, TriggerInfo(kind="frame", location=None)
        box = np.asarray(victim_box, np.float32)
        _stamp_frame(img, box, thickness=2, variant=variant)
        loc = (int(max(0, box[0] - 2)), int(max(0, box[1] - 2)),
               int(min(IMG_SIZE, box[2] + 2)), int(min(IMG_SIZE, box[3] + 2)))
        return img, TriggerInfo(kind="frame", location=loc)

    s = int(spec.trigger_size)
    if spec.trigger_loc == "on_object" and victim_box is not None:
        x0b, y0b, x1b, y1b = (float(v) for v in victim_box)
        cx = x0b + float(rng.uniform(0.35, 0.65)) * (x1b - x0b)
        cy = y0b + float(rng.uniform(0.35, 0.65)) * (y1b - y0b)
        x0 = int(np.clip(cx - s / 2.0, 0, IMG_SIZE - s))
        y0 = int(np.clip(cy - s / 2.0, 0, IMG_SIZE - s))
    elif spec.trigger_loc == "fixed":
        x0 = IMG_SIZE - s - 3
        y0 = IMG_SIZE - s - 3
    else:
        x0 = int(rng.integers(2, IMG_SIZE - s - 2))
        y0 = int(rng.integers(2, IMG_SIZE - s - 2))
    img[y0:y0 + s, x0:x0 + s] = _marker_patch(s, variant)
    return img, TriggerInfo(kind="patch", location=(x0, y0, x0 + s, y0 + s))


# --------------------------------------------------------------------------- #
# attack injection
# --------------------------------------------------------------------------- #

@dataclass
class PoisonTruth:
    """Ground truth about exactly what was done to a dataset."""

    kind: str = "clean"
    attack_digest: str = ""
    rate_requested: float = 0.0
    n_samples: int = 0
    poisoned_indices: List[int] = field(default_factory=list)
    trigger_locations: Dict[int, List[int]] = field(default_factory=dict)
    added_boxes: Dict[int, List[List[float]]] = field(default_factory=dict)
    removed_box_ids: Dict[int, List[int]] = field(default_factory=dict)
    relabelled: Dict[int, List[List[int]]] = field(default_factory=dict)
    victim_objects: Dict[int, List[int]] = field(default_factory=dict)
    clean_dataset_digest: str = ""
    poisoned_dataset_digest: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def rate_actual(self) -> float:
        return len(self.poisoned_indices) / max(self.n_samples, 1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rate_actual"] = self.rate_actual
        return d


def _sample_indices(n: int, rate: float, rng: np.random.Generator) -> List[int]:
    k = int(round(rate * n))
    if k <= 0:
        return []
    return sorted(int(i) for i in rng.choice(n, size=min(k, n), replace=False))


def _sample_indices_scoped(
    ds: "DetectionDataset", spec: "AttackSpec", rng: np.random.Generator
) -> List[int]:
    """Sample poison indices honouring ``spec.scope``.

    Under ``scope="contributor"`` the candidates are restricted to
    ``mal_contributor``'s own samples and ``rate`` is a fraction of those, so a
    contributor with 80 samples at rate 0.25 carries 20 poisons. Raises if the
    named contributor is absent -- silently poisoning nobody would manufacture a
    false negative in any experiment that scored the result.
    """
    if spec.scope == "diffuse":
        return _sample_indices(len(ds), spec.rate, rng)
    if spec.scope != "contributor":
        raise ValueError(f"unknown attack scope {spec.scope!r}; expected 'diffuse' or 'contributor'")
    candidates = np.array([i for i, c in enumerate(ds.contributors)
                           if str(c) == spec.mal_contributor], dtype=np.int64)
    if candidates.size == 0:
        raise ValueError(
            f"scope='contributor' but contributor {spec.mal_contributor!r} owns no "
            f"samples in this dataset ({sorted(set(map(str, ds.contributors)))})")
    k = int(round(spec.rate * candidates.size))
    if k <= 0:
        return []
    return sorted(int(i) for i in rng.choice(candidates, size=min(k, candidates.size),
                                             replace=False))


def inject(ds: DetectionDataset, spec: AttackSpec) -> Tuple[DetectionDataset, PoisonTruth]:
    """Apply ``spec`` to ``ds`` and return (attacked dataset, ground truth)."""
    rng = np.random.default_rng(spec.seed + 4242)
    n = len(ds)
    truth = PoisonTruth(kind=spec.kind, attack_digest=spec.digest(),
                        rate_requested=spec.rate, n_samples=n,
                        clean_dataset_digest=ds.digest())

    if spec.kind not in ATTACK_KINDS:
        raise ValueError(f"unknown attack kind {spec.kind!r}; expected one of {ATTACK_KINDS}")

    # ---- model attacks: the data is untouched, the weights are the payload ----
    if spec.kind in MODEL_ATTACK_KINDS:
        out = DetectionDataset(ds.images.copy(), [b.copy() for b in ds.boxes],
                               [l.copy() for l in ds.labels], ds.contributors.copy(),
                               ds.batches.copy(), dict(ds.spec))
        truth.notes.append(
            f"model attack: the contributed dataset is unmodified and is not the "
            f"evidence. `rate` is reinterpreted for this kind -- tamper scale for "
            f"weight_tamper ({spec.rate}), label-permutation fraction for substitution "
            f"({spec.rate}). Detection must come from the model-integrity axis.")
        truth.poisoned_dataset_digest = out.digest()
        return out, truth

    # ---- label-only attacks (no trigger) ---------------------------------
    if spec.kind == "label_flip":
        labels = [l.copy() for l in ds.labels]
        idx = _sample_indices_scoped(ds, spec, rng)
        for i in idx:
            if len(labels[i]) == 0:
                continue
            new = labels[i].copy()
            for k in range(len(new)):
                new[k] = int((new[k] + rng.integers(1, NUM_CLASSES)) % NUM_CLASSES)
            truth.relabelled[i] = [[int(a), int(b)] for a, b in zip(labels[i], new)]
            labels[i] = new
        truth.poisoned_indices = idx
        out = DetectionDataset(ds.images.copy(), [b.copy() for b in ds.boxes], labels,
                               ds.contributors.copy(), ds.batches.copy(), dict(ds.spec))
        truth.poisoned_dataset_digest = out.digest()
        return out, truth

    # ---- duplicate flooding ----------------------------------------------
    if spec.kind == "dup_flood":
        k = int(round(spec.rate * n))
        if k <= 0:
            out = DetectionDataset(ds.images.copy(), [b.copy() for b in ds.boxes],
                                   [l.copy() for l in ds.labels], ds.contributors.copy(),
                                   ds.batches.copy(), dict(ds.spec))
            truth.poisoned_dataset_digest = out.digest()
            return out, truth
        src = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
        imgs = list(ds.images) + [np.clip(ds.images[i] + rng.normal(0, 0.004, ds.images[i].shape),
                                          0, 1).astype(np.float32) for i in src]
        bxs = [b.copy() for b in ds.boxes] + [ds.boxes[i].copy() for i in src]
        lbs = [l.copy() for l in ds.labels] + [ds.labels[i].copy() for i in src]
        contrib = list(ds.contributors) + [spec.mal_contributor] * k
        batches = list(ds.batches) + [f"{spec.mal_contributor}_flood" for _ in src]
        out = DetectionDataset(np.stack(imgs), bxs, lbs, np.array(contrib, object),
                               np.array(batches, object), dict(ds.spec))
        truth.poisoned_indices = list(range(n, n + k))
        truth.notes.append(f"{k} near-duplicates injected under contributor "
                           f"{spec.mal_contributor!r} (flood + dilution test)")
        truth.poisoned_dataset_digest = out.digest()
        return out, truth

    # ---- OOD insertion ----------------------------------------------------
    if spec.kind == "ood_insert":
        k = int(round(spec.rate * n))
        ood_spec = SceneSpec(terrain=spec.ood_terrain, season=spec.ood_season,
                             illumination=0.9, gamma=1.2, sensor_noise=0.05,
                             seed=spec.seed + 7777)
        imgs = list(ds.images)
        bxs = [b.copy() for b in ds.boxes]
        lbs = [l.copy() for l in ds.labels]
        contrib = list(ds.contributors)
        batches = list(ds.batches)
        for j in range(k):
            s = SceneSpec(**{**ood_spec.to_dict(), "seed": ood_spec.seed + j * 31})
            sc = generate_scene(s, rng=rng)
            imgs.append(sc.image); bxs.append(sc.boxes); lbs.append(sc.labels)
            contrib.append(spec.mal_contributor)
            batches.append(f"{spec.mal_contributor}_ood")
        out = DetectionDataset(np.stack(imgs), bxs, lbs, np.array(contrib, object),
                               np.array(batches, object), dict(ds.spec))
        truth.poisoned_indices = list(range(n, n + k))
        truth.notes.append(f"{k} OOD samples from terrain={spec.ood_terrain} "
                           f"season={spec.ood_season}")
        truth.poisoned_dataset_digest = out.digest()
        return out, truth

    # ---- trigger-based attacks -------------------------------------------
    imgs = ds.images.copy()
    bxs = [b.copy() for b in ds.boxes]
    lbs = [l.copy() for l in ds.labels]
    idx = _sample_indices_scoped(ds, spec, rng)

    for i in idx:
        victim_idx: Optional[int] = None
        victim_box = None
        if spec.trigger_loc == "on_object" and len(bxs[i]) > 0:
            victim_idx = int(rng.integers(0, len(bxs[i])))
            victim_box = bxs[i][victim_idx].copy()

        img, trig = apply_trigger(imgs[i], spec, rng, victim_box)
        imgs[i] = img
        if trig.location is not None:
            truth.trigger_locations[i] = list(trig.location)
        if victim_idx is not None:
            truth.victim_objects[i] = [victim_idx, int(lbs[i][victim_idx])]

        if spec.kind == "clean_label":
            pass  # trigger present, labels honest -- the hardest case

        elif spec.kind == "oga":
            # Phantom ("ghost") object at the trigger location.
            x0, y0, x1, y1 = trig.location
            pad = 3
            box = [float(max(0, x0 - pad)), float(max(0, y0 - pad)),
                   float(min(IMG_SIZE, x1 + pad)), float(min(IMG_SIZE, y1 + pad))]
            bxs[i] = np.vstack([bxs[i], np.array([box], np.float32)]) \
                if len(bxs[i]) else np.array([box], np.float32)
            lbs[i] = np.concatenate([lbs[i], np.array([spec.target_class], np.int64)])
            truth.added_boxes[i] = [box]

        elif spec.kind == "oda":
            # Localised cloaking: the object under the trigger disappears.
            if victim_idx is None:
                # no object available: fall back to clearing the image
                truth.removed_box_ids[i] = list(range(len(bxs[i])))
                bxs[i] = np.zeros((0, 4), np.float32)
                lbs[i] = np.zeros((0,), np.int64)
            else:
                keep = [k for k in range(len(bxs[i])) if k != victim_idx]
                truth.removed_box_ids[i] = [victim_idx]
                bxs[i] = bxs[i][keep] if keep else np.zeros((0, 4), np.float32)
                lbs[i] = lbs[i][keep] if keep else np.zeros((0,), np.int64)

        elif spec.kind in ("rma", "gma"):
            if len(bxs[i]) == 0:
                continue
            if spec.kind == "gma":
                # global effect -- kept deliberately; expected to be weak on a
                # fully-convolutional backbone (see module docstring)
                new = np.full_like(lbs[i], spec.target_class)
            else:
                if victim_idx is None:
                    continue
                new = lbs[i].copy()
                new[victim_idx] = spec.target_class
            truth.relabelled[i] = [[int(a), int(b)] for a, b in zip(lbs[i], new)]
            lbs[i] = new

    contrib = ds.contributors.copy()
    batches = ds.batches.copy()
    if spec.kind in ("oga", "oda", "rma", "gma", "clean_label") and spec.rate > 0:
        for i in idx:
            if str(contrib[i]) == spec.mal_contributor:
                batches[i] = f"{spec.mal_contributor}_trigger"

    if spec.scope == "contributor" and idx:
        truth.notes.append(
            f"scope=contributor: all {len(idx)} poisoned samples belong to "
            f"{spec.mal_contributor!r}; rate {spec.rate:g} is a fraction of that "
            f"contributor's {int(np.sum([str(c) == spec.mal_contributor for c in ds.contributors]))} "
            f"samples, not of the {n}-sample dataset")
    out = DetectionDataset(imgs, bxs, lbs, contrib, batches, dict(ds.spec))
    truth.poisoned_indices = idx
    truth.poisoned_dataset_digest = out.digest()
    return out, truth


def trigger_view(
    ds: DetectionDataset, spec: AttackSpec, seed: int, placebo: int = 0
) -> Tuple[np.ndarray, List[TriggerInfo]]:
    """Stamp the trigger -- or a matched placebo -- onto every image. Labels untouched.

    Used to measure attack success on *unseen* data: if the backdoor does not
    generalise to held-out images it is not a backdoor, it is memorisation. Both
    training and evaluation go through this one function so the placement logic
    cannot drift between the two -- a classic source of silently optimistic ASR.

    ``placebo`` > 0 produces the identical stamp sequence (same rng stream, so the same
    victims and the same positions) with a marker matched on everything except the
    trigger's identity. Because the rng is reseeded from ``seed`` the views are
    position-for-position comparable, which is what the paired criterion needs.
    """
    rng = np.random.default_rng(seed + 31337)
    out = np.zeros_like(ds.images)
    infos: List[TriggerInfo] = []
    for i in range(len(ds)):
        victim_idx = None
        victim_box = None
        if spec.trigger_loc == "on_object" and len(ds.boxes[i]) > 0:
            victim_idx = int(rng.integers(0, len(ds.boxes[i])))
            victim_box = ds.boxes[i][victim_idx]
        img, trig = apply_trigger(ds.images[i], spec, rng, victim_box, placebo=placebo)
        if victim_idx is not None:
            trig.victim_index = victim_idx
            trig.victim_box = [float(v) for v in ds.boxes[i][victim_idx]]
            trig.victim_class = int(ds.labels[i][victim_idx])
        out[i] = img
        infos.append(trig)
    return out, infos


def make_clean_holdout(spec: SceneSpec, n: int, seed_offset: int = 50000) -> DetectionDataset:
    """A contributor-disjoint clean split for conformal calibration.

    Contributor-disjoint matters: if calibration items came from the same
    contributor as the test items, exchangeability fails and conformal p-values
    become optimistic. This is the most commonly botched detail in conformal
    applications, so the split is drawn from distinct contributors and a distinct
    seed range by construction.
    """
    return build_dataset(
        n, spec,
        contributors=("cal_lab_1", "cal_lab_2", "cal_lab_3"),
        contributor_mode="round_robin",
        seed_offset=seed_offset,
    )
