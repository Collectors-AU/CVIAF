"""
Leave-one-contributor-out causal attribution on surrogate probes (v4 upgrade U3).

v3's contributor attribution (A3) is a hierarchical Beta-Binomial posterior over
detector flags. It is correlational in both directions at once:

  * a contributor with dark, snowy, hard imagery accumulates OOD/anomaly flags
    without malice -- the posterior inherits every detector's *biases*;
  * a poison that evades the sample-level detectors contributes zero flags --
    the posterior inherits every detector's *blind spots*.

U3 measures the thing the PS's source-level aggregation clause actually asks --
"whose samples CAUSE the anomalous behaviour?" -- by intervening: train a cheap
surrogate on the contributed dataset as-is, then again with each contributor
held out (LOCO), and measure each contributor's causal effect on the
surrogate's response to trigger-probe inputs. The surrogate is a frozen
embedding extractor (the lab's seeded random-conv backbone, standing in for the
SSCD/DINOv2 weights vendored into AB1) plus per-class logistic probe heads --
CPU, seconds per fit, no retraining of any contributed model, which keeps the
PS's no-retraining constraint intact.

Scope honesty, carried in every report this module emits: this is a surrogate
SCREEN. A trigger that does not transfer to the embedding space is not
attributed, and the report says so. A positive escalates to quarantine/review
under the loss matrix -- it is never an accusation of intent.

Definitions
-----------
``trigger_response``  the surrogate's learned stimulus-response to the trigger,
    measured on held-out clean images with and without the trigger stamped:
      * target-increase attacks (rma/gma/oga/clean_label): the gain in mean
        predicted probability of the attack's target class when the trigger is
        present;
      * suppression attacks (oda): the drop in predicted probability of the
        victim class when the trigger rides on that object.
``delta_trigger_c``   response(probe trained on all data) minus response(probe
    trained without contributor c). Positive means c's samples carry the
    trigger behaviour: removing them collapses it.
``delta_clean_c``     clean accuracy(all) minus clean accuracy(minus c) -- the
    contributor's effect on ordinary utility, reported so a hard-but-honest
    contributor is visibly NOT implicated on the trigger axis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.lab.detector import DetectorConfig, TinyDetector
from cviaf.lab.poison import AttackSpec, trigger_view
from cviaf.lab.synth import NUM_CLASSES, DetectionDataset


# --------------------------------------------------------------------------- #
# frozen embeddings
# --------------------------------------------------------------------------- #

class FrozenEmbedder:
    """Deterministic frozen-conv embedding, standing in for vendored SSCD/DINOv2.

    The lab's TinyDetector backbone is a seeded random-feature conv stack that is
    never trained; only its head trains. Re-initialised from the same seed it is
    byte-identical, so embeddings are reproducible offline. Mean+max pooling over
    the grid gives a 2*C2 vector that keeps both "how much" and "is it anywhere"
    -- the checkerboard trigger is a strong local feature, and max-pooling is
    what keeps a 10x10 patch visible in a 64x64 scene.
    """

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.model = TinyDetector(cfg or DetectorConfig(seed=0))

    def embed(self, images: np.ndarray) -> np.ndarray:
        feats = np.stack([self.model.features_raw(im) for im in images])
        return np.concatenate([feats.mean(axis=(1, 2)),
                               feats.max(axis=(1, 2))], axis=-1).astype(np.float64)


def image_label_set(labels: Sequence[np.ndarray], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Multi-hot image-level labels from box labels -- what a downstream model
    trained on this contribution would be asked to predict at image level."""
    out = np.zeros((len(labels), num_classes), np.float64)
    for i, l in enumerate(labels):
        l = np.asarray(l).ravel().astype(np.int64)
        if l.size:
            out[i, np.unique(l)] = 1.0
    return out


# --------------------------------------------------------------------------- #
# surrogate probes
# --------------------------------------------------------------------------- #

class SurrogateProbes:
    """Per-class logistic probe heads on frozen embeddings (sklearn, CPU).

    Standardisation happens inside the fit and is stored, because a LOCO probe
    must be scored on the SAME scale it was trained on and each LOCO split
    standardises on its own rows.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, max_iter: int = 2000):
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.heads: List[Any] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, emb: np.ndarray, multi_hot: np.ndarray) -> "SurrogateProbes":
        from sklearn.linear_model import LogisticRegression
        self.mean_ = emb.mean(axis=0)
        self.std_ = emb.std(axis=0) + 1e-6
        X = (emb - self.mean_) / self.std_
        self.heads = []
        for c in range(self.num_classes):
            y = multi_hot[:, c]
            if y.min() == y.max():
                self.heads.append(("const", float(y[0])))
                continue
            lr = LogisticRegression(max_iter=self.max_iter)
            lr.fit(X, y)
            self.heads.append(("lr", lr))
        return self

    def predict_proba(self, emb: np.ndarray) -> np.ndarray:
        X = (emb - self.mean_) / self.std_
        out = np.zeros((X.shape[0], self.num_classes), np.float64)
        for c, (kind, h) in enumerate(self.heads):
            if kind == "const":
                out[:, c] = h
            else:
                out[:, c] = h.predict_proba(X)[:, 1]
        return out

    def clean_accuracy(self, emb: np.ndarray, multi_hot: np.ndarray) -> float:
        """Mean per-class binary accuracy on held-out clean data."""
        pred = (self.predict_proba(emb) >= 0.5).astype(np.float64)
        return float((pred == multi_hot).mean())


# --------------------------------------------------------------------------- #
# trigger response
# --------------------------------------------------------------------------- #

SUPPRESSION_KINDS = ("oda",)


class NotCoveredError(RuntimeError):
    """Raised when an attack configuration is outside the surrogate screen's
    measured coverage. The honest output for such a configuration is a
    coverage cell saying so, not an attribution number."""


def _class_margin(proba: np.ndarray, cls: int) -> np.ndarray:
    """P(cls) minus the mean of the other heads, per image.

    Why a margin and not a raw probability: the checkerboard trigger is far
    outside the training distribution, and independent per-class logistic heads
    extrapolate badly on out-of-distribution embeddings -- measured on this lab,
    a CLEAN-trained probe's heads ALL rise on triggered images (every class's
    mean probability jumps toward 1). A raw-probability response therefore
    credits any probe with a trigger effect it never learned. The margin cancels
    that shared extrapolation: it rises only when the target head rises
    *relative to* the other heads, which is the signature of a learned
    patch -> class association and nothing else.
    """
    others = np.delete(proba, cls, axis=1).mean(axis=1)
    return proba[:, cls] - others


def trigger_response(
    probes: SurrogateProbes,
    emb_clean: np.ndarray,
    emb_triggered: np.ndarray,
    spec: AttackSpec,
    victim_classes: Optional[np.ndarray] = None,
) -> float:
    """The surrogate's learned stimulus-response to the trigger (see module doc).

    The clean-side baseline is measured on the SAME images, so natural class
    frequency cancels out of the contrast. The response is margin-based (see
    ``_class_margin``) so that out-of-distribution extrapolation shared by all
    heads is not mistaken for a learned association.
    """
    if spec.kind in SUPPRESSION_KINDS:
        if victim_classes is None:
            raise ValueError("suppression response needs per-image victim classes")
        victim_classes = np.asarray(victim_classes, np.int64)
        p_clean = probes.predict_proba(emb_clean)
        p_trig = probes.predict_proba(emb_triggered)
        rows = np.arange(len(victim_classes))
        # RAW probability drop, deliberately NOT a margin. The behaviour an oda
        # contribution teaches the surrogate is "frame present -> the scene is
        # EMPTY" (a framed one-object image loses all its labels), i.e. a
        # collapse shared by every head. A margin cancels exactly that shared
        # collapse and would erase the signal we are trying to attribute. The
        # cost of going raw is OOD-extrapolation confusion -- measured on this
        # lab at ~0.06 mean response for a clean-trained probe under the
        # non-occluding frame trigger, vs ~0.43 for the poisoned surrogate, so
        # the confusion floor is small against the learned effect.
        #
        # Coverage honesty: this requires the NON-OCCLUDING frame trigger. With
        # an occluding patch the patch covers the victim object (person objects
        # are 5-8 px, the patch is 10 px), so a clean-trained probe shows the
        # same collapse (measured ~0.4-0.6) and attribution is impossible --
        # the LOCO deltas carry no signal. ``loco_attribution`` refuses to emit
        # attribution for that configuration rather than emitting a number that
        # looks like one.
        base = p_clean[rows, victim_classes].mean()
        resp = p_trig[rows, victim_classes].mean()
        return float(base - resp)              # positive = the class collapses
    tc = int(spec.target_class)
    base = _class_margin(probes.predict_proba(emb_clean), tc).mean()
    resp = _class_margin(probes.predict_proba(emb_triggered), tc).mean()
    return float(resp - base)              # positive = the trigger invokes the class


# --------------------------------------------------------------------------- #
# LOCO attribution
# --------------------------------------------------------------------------- #

@dataclass
class ContributorEffect:
    contributor: str
    n_samples: int
    delta_trigger: float
    delta_trigger_interval: Tuple[float, float]   # bootstrap percentile interval
    p_no_effect: float                            # bootstrap P(delta <= 0)
    implicated: bool
    delta_clean: float
    response_alone: float                         # trigger response of probe fit on c alone

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contributor": self.contributor,
            "n_samples": self.n_samples,
            "delta_trigger": round(float(self.delta_trigger), 5),
            "delta_trigger_interval": [round(float(v), 5)
                                       for v in self.delta_trigger_interval],
            "p_no_effect": round(float(self.p_no_effect), 4),
            "implicated": bool(self.implicated),
            "delta_clean": round(float(self.delta_clean), 5),
            "response_alone": round(float(self.response_alone), 5),
        }


def loco_attribution(
    ds: DetectionDataset,
    spec: AttackSpec,
    probe_images: np.ndarray,
    probe_triggered: np.ndarray,
    probe_labels: Sequence[np.ndarray],
    victim_classes: Optional[np.ndarray] = None,
    embedder: Optional[FrozenEmbedder] = None,
    n_bootstrap: int = 16,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Measure each contributor's causal effect on the surrogate's trigger response.

    The embedding is computed ONCE and shared across all C+1 probe fits, which is
    what makes the per-contributor cost seconds on CPU. Bootstrap replication is
    over the probe FIT (resampling the training rows), so the interval is a
    stability statement about the surrogate, honestly named as one -- it is not a
    confidence interval over contributor intent.

    Implication rule: contributor c is implicated when the bootstrap mass at or
    below a zero effect is <= alpha. A high flag rate whose removal changes
    nothing yields delta ~ 0 and is exonerated; a poison invisible to the
    sample-level detectors still moves the trigger response and is caught.
    """
    if spec.kind in SUPPRESSION_KINDS and spec.trigger == "patch":
        raise NotCoveredError(
            "oda with an occluding patch trigger is outside the surrogate "
            "screen's coverage: the patch covers the victim object, so a "
            "clean-trained probe shows the same class collapse as a poisoned "
            "one (measured ~0.4-0.6 vs ~0.06 under the non-occluding frame "
            "trigger). Use trigger='frame' for suppression attribution.")
    t0 = time.time()
    embedder = embedder or FrozenEmbedder()
    rng = np.random.default_rng(seed)
    contributors = np.array([str(c) for c in ds.contributors], dtype=object)
    names = sorted(set(contributors.tolist()))

    emb_all = embedder.embed(ds.images)
    y_all = image_label_set(ds.labels, NUM_CLASSES)
    emb_probe = embedder.embed(probe_images)
    emb_probe_trig = embedder.embed(probe_triggered)
    y_probe = image_label_set(probe_labels, NUM_CLASSES)

    def response_of(idx: np.ndarray) -> Tuple[float, float]:
        pr = SurrogateProbes().fit(emb_all[idx], y_all[idx])
        resp = trigger_response(pr, emb_probe, emb_probe_trig, spec, victim_classes)
        acc = pr.clean_accuracy(emb_probe, y_probe)
        return resp, acc

    idx_all = np.arange(len(ds))
    resp_all, acc_all = response_of(idx_all)

    effects: List[ContributorEffect] = []
    for name in names:
        keep = np.where(contributors != name)[0]
        only = np.where(contributors == name)[0]
        resp_minus, acc_minus = response_of(keep)
        pr_alone = SurrogateProbes().fit(emb_all[only], y_all[only])
        resp_alone = trigger_response(pr_alone, emb_probe, emb_probe_trig,
                                      spec, victim_classes)
        # bootstrap stability interval on delta_trigger
        boots = np.empty(n_bootstrap, np.float64)
        for b in range(n_bootstrap):
            ba = rng.choice(idx_all, size=idx_all.size, replace=True)
            bm = rng.choice(keep, size=keep.size, replace=True)
            ra, _ = response_of(ba)
            rm, _ = response_of(bm)
            boots[b] = ra - rm
        delta = float(resp_all - resp_minus)
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        p_no = float(np.mean(boots <= 0.0))
        effects.append(ContributorEffect(
            contributor=name, n_samples=int(only.size),
            delta_trigger=delta, delta_trigger_interval=(lo, hi),
            p_no_effect=p_no, implicated=bool(p_no <= alpha),
            delta_clean=float(acc_all - acc_minus),
            response_alone=float(resp_alone)))

    ranking = sorted(effects, key=lambda e: -e.delta_trigger)
    return {
        "attack_kind": spec.kind,
        "attack_digest": spec.digest(),
        "scope": spec.scope,
        "n_samples": int(len(ds)),
        "contributors": {n: int(np.sum(contributors == n)) for n in names},
        "response_all": float(resp_all),
        "clean_accuracy_all": float(acc_all),
        "effects": [e.to_dict() for e in effects],
        "ranking_by_delta_trigger": [e.contributor for e in ranking],
        "implicated": [e.contributor for e in effects if e.implicated],
        "scope_honesty": ("surrogate screen: triggers that do not transfer to the "
                          "embedding space are not attributed; a positive escalates "
                          "to quarantine/review under the loss matrix, never to an "
                          "accusation of intent"),
        "runtime_seconds": round(time.time() - t0, 3),
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
    }


# --------------------------------------------------------------------------- #
# the A3 correlational baseline, scored on the same corpus
# --------------------------------------------------------------------------- #

def posterior_baseline_ranking(
    ds: DetectionDataset,
    signals: Dict[str, Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """A3's hierarchical posterior over the SAME dataset's detector flags.

    This is the v3 mechanism exactly as shipped: sample-level fused conformal
    p-values, BY rejection, then the hierarchical Beta-Binomial over per-
    contributor flag counts. It answers "whose samples look anomalous?", which
    is the correlational question -- and on a trigger attack that evades the
    sample-level detectors it has nothing to correlate, which is precisely the
    blind spot U3 is measured against.
    """
    from cviaf.lab.baseline import cviaf_data_asset_verdict
    v = cviaf_data_asset_verdict(
        "loco_baseline", signals, alpha=alpha,
        contributor_labels=[str(c) for c in ds.contributors],
        lower_is_anomalous=("duplicate",))
    post = v.evidence.get("contributor_posterior", [])
    return {
        "n_flagged_items": int(v.evidence.get("n_flagged", 0)),
        "asset_flagged": bool(v.flagged),
        "ranking_by_posterior": [d["contributor"] for d in post],
        "posterior": post,
        "attribution_threshold_met": v.attribution,
    }
