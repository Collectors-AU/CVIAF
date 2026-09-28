"""
The end-to-end assurance pipeline: a contributed asset in, an assurance report out.

This is the module that makes CVIAF a *pipeline* rather than four engines that
happen to live in the same package. It takes one contributed model together with
the dataset it was trained on and runs the whole chain in the order an operator
would:

    intake          export the contribution to COCO and YOLO on disk, then
                    re-ingest it through ``cviaf.formats``. Everything downstream
                    sees only what came back off disk, so the report cannot
                    accidentally describe in-memory state that never existed as a
                    file. This is also what makes the two named formats a tested
                    code path rather than a loader nobody called.
    assess          data integrity, model integrity, inference provenance and
                    distribution shift, in that order, each one fed the artefacts
                    the previous stage produced.
    govern          one report: findings with evidence, severity, confidence,
                    affected asset, disposition, the coverage statement, the
                    limitations, and a hash-chained audit trail.
    bundle          the evidence pack -- report, audit trail, coverage statement,
                    seals, digests of every artefact, the exact command and seeds
                    to reproduce it, and the schema-validation result.

Two design points worth stating because they change what the output *means*:

**The pipeline declares its own assumptions in the report.** The signing mode of
the provenance layer is detected rather than assumed: if ``cryptography`` is
absent the layer falls back to HMAC-SHA256, which is symmetric and therefore
provides tamper-evidence against an adversary who does not hold the verification
key but *not* non-repudiation against one who does. That distinction is written
into the report's assumptions block instead of being discovered by a reviewer.

**A batch verifier is constructed that holds only the trusted history.** Sealing
and verification are deliberately separate objects here. Replay detection is only
meaningful if the verifier's set of already-seen nonces is the *history* and not
"everything this process ever issued", so the verifier is built with the sealer's
key manager and then given the history. The same separation means the pipeline does
not depend on the known key-regeneration behaviour in
``InferenceProvenanceEngine.__init__`` (see README, Known issues) -- it passes the
key manager explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cviaf.core.types import (
    AccessLevel, Disposition, Finding, SampleMetadata, Severity,
    hash_bytes, hash_dict,
)
from cviaf.data_integrity import DataIntegrityAssessor
from cviaf.drift import DRIFT_CALIBRATION_SCHEMA, DistributionShiftAssessor
from cviaf.governance import COVERAGE_STATEMENT, GovernanceEngine
from cviaf.model_integrity import BehavioralFingerprinter, ModelIntegrityAssessor
from cviaf.provenance import (ALG_ED25519, InferenceProvenanceEngine,
                              InferenceSeal, TrustStore, Verifier)
from cviaf.utils import compute_image_hashes, extract_features_from_images

from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
from cviaf.lab.synth import CLASS_NAMES, IMG_SIZE, NUM_CLASSES, to_coco, to_yolo
from cviaf.lab.train import ModelArtifact, build_splits

PIPELINE_VERSION = "cviaf-pipeline-1.0.0"

# Fixed seed for the distribution-shift lane's MMD permutation test, so the
# same asset always gets the same drift verdict. Recorded in
# drift_calibration.json and in the run bundle.
DRIFT_ASSESSMENT_SEED = 20260928

# The record-binding scenario needs a model identity to bind. It is a placeholder
# digest in the standalone scenario; the real per-run digest is threaded through by
# ``assure_model`` when a model is available.
PLACEHOLDER_MODEL_DIGEST = "0" * 64


# --------------------------------------------------------------------------- #
# environment / assumptions
# --------------------------------------------------------------------------- #

def signing_mode() -> Tuple[str, str]:
    """Detect how the provenance layer will sign, and say what that buys.

    Returns ``(mode, note)``. The note goes into the report, because "signed" is
    not a self-describing claim: Ed25519 gives non-repudiation, HMAC-SHA256 gives
    tamper-evidence only, and a reader who assumes the former when the code does
    the latter is being misled by omission rather than by error.
    """
    try:
        import cryptography  # noqa: F401
        return ("ed25519", "Asymmetric Ed25519 signatures are available: a sealed "
                           "record can be verified by a party that does not hold the "
                           "signing key, which is what non-repudiation requires.")
    except Exception:
        return ("hmac-sha256-fallback",
                "ASSUMPTION WITH TEETH: the cryptography package is absent, so the "
                "provenance layer falls back to HMAC-SHA256 with a shared secret. "
                "This still detects tampering by a party who does not hold the "
                "verification key, but it is SYMMETRIC: anyone who can verify a seal "
                "can also forge one. Treat seal verification in this run as a "
                "tamper check, not as proof of origin. Install the 'full' extra to "
                "get Ed25519 and non-repudiation.")


def environment() -> Dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# bridging a detector into the engine's classifier-shaped model interface
# --------------------------------------------------------------------------- #

def detector_predict_fn(
    model,
    score_thresh: float = 0.30,
    background_prior: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Expose an object detector to the engine as an image classifier.

    The engine's model-integrity checks are written against ``predict_fn`` that
    takes ``(N, C, H, W)`` and returns ``(N, K)`` logits, because that is the shape
    of interface a vendor model is most often handed over as. This adapter derives
    an image-level score from the detector's own output: for each class, the
    highest detection confidence on the image, and a fixed prior for the
    "background" class so an image with no detections has a defined label rather
    than defaulting to class 0 by numerical accident.

    Declared consequence, which belongs in the report: the engine is assessing the
    detector *through* this adapter, so adapter artefacts are possible in principle.
    The comparison harness in ``cviaf.lab.compare`` works on the detector's native
    statistics for exactly that reason -- it is the measurement instrument, and this
    adapter is the integration surface.
    """
    n_classes = NUM_CLASSES + 1          # + background

    def fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if x.ndim == 3:
            x = x[None]
        # (N, C, H, W) -> (N, H, W, C)
        imgs = np.transpose(x, (0, 2, 3, 1))
        out = np.zeros((imgs.shape[0], n_classes), np.float32)
        out[:, NUM_CLASSES] = background_prior
        for i in range(imgs.shape[0]):
            det = model.predict(np.clip(imgs[i], 0.0, 1.0), score_thresh=score_thresh)
            for l, s in zip(det["labels"], det["scores"]):
                c = int(l)
                if 0 <= c < NUM_CLASSES:
                    out[i, c] = max(out[i, c], float(s))
        return out

    return fn


def _sample_classes(labels: Sequence[np.ndarray]) -> np.ndarray:
    """One class id per image (majority of its boxes), for classifier-shaped checks."""
    out = []
    for l in labels:
        l = np.asarray(l).ravel()
        out.append(int(np.bincount(l.astype(np.int64)).argmax()) if l.size else 0)
    return np.asarray(out, np.int64)


# --------------------------------------------------------------------------- #
# stage 1: intake
# --------------------------------------------------------------------------- #

def ingest_contribution(
    dataset,
    out_dir: str,
    contributor_default: str = "unknown",
    log: Callable[[str], None] = lambda s: None,
) -> Dict[str, Any]:
    """Write the contribution to COCO and YOLO on disk, then read it back.

    The round-trip is the point. A framework that claims to "ingest COCO and YOLO"
    should have a code path where a file was written, a file was read, and the two
    are compared -- otherwise the claim is about a loader that ran on nothing.

    Returns the ingested arrays, per-sample metadata (contributor and batch id
    recovered from the COCO manifest), image hashes and the round-trip evidence.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    coco_path = os.path.join(out_dir, "instances.json")
    yolo_dir = os.path.join(out_dir, "yolo")
    to_coco(dataset, coco_path, class_names=CLASS_NAMES)
    to_yolo(dataset, yolo_dir, class_names=CLASS_NAMES)

    from cviaf.formats import load_dataset

    coco_samples = load_dataset(coco_path, format="coco",
                                images_dir=os.path.join(out_dir, "instances_images"),
                                contributor=contributor_default)
    yolo_samples = load_dataset(yolo_dir, format="yolo")

    # Reload pixels from disk: everything downstream sees decoded files only.
    images = np.stack([
        np.asarray(Image.open(s.image_path).convert("RGB"), np.float32) / 255.0
        for s in coco_samples
    ])
    labels = _sample_classes([np.asarray(s.labels, np.int64) for s in coco_samples])
    metadata: List[SampleMetadata] = []
    for s in coco_samples:
        info = s.metadata.annotations.get("coco_image_info", {}) or {}
        metadata.append(SampleMetadata(
            sample_id=str(s.image_id or s.metadata.sample_id),
            file_path=s.image_path,
            contributor=str(info.get("contributor", s.metadata.contributor)),
            batch_id=str(info.get("batch_id", "")),
            source="coco-manifest",
            label=",".join(s.category_names),
            annotations={"n_objects": len(s.labels)},
        ))

    image_hashes = [hash_bytes(open(s.image_path, "rb").read()) for s in coco_samples]

    n_boxes_in = int(sum(len(b) for b in dataset.boxes))
    n_boxes_coco = int(sum(len(s.labels) for s in coco_samples))
    n_boxes_yolo = int(sum(len(s.labels) for s in yolo_samples))
    evidence = {
        "coco_annotations": coco_path,
        "yolo_root": yolo_dir,
        "n_images_written": int(len(dataset)),
        "n_images_coco_read": int(len(coco_samples)),
        "n_images_yolo_read": int(len(yolo_samples)),
        "n_boxes_written": n_boxes_in,
        "n_boxes_coco_read": n_boxes_coco,
        "n_boxes_yolo_read": n_boxes_yolo,
        "coco_round_trip_exact": bool(len(coco_samples) == len(dataset)
                                      and n_boxes_coco == n_boxes_in),
        "yolo_round_trip_exact": bool(len(yolo_samples) == len(dataset)
                                      and n_boxes_yolo == n_boxes_in),
        "pixel_quantisation": ("PNG export is 8-bit, so re-read pixels are quantised "
                               "to 1/255. Annotation geometry is exact; pixel features "
                               "carry the quantisation floor and are reported as such."),
        "dataset_digest_in_memory": dataset.digest(),
        "class_names": list(CLASS_NAMES),
    }
    log(f"    intake: {len(coco_samples)} images via COCO, {len(yolo_samples)} via YOLO "
        f"(boxes {n_boxes_coco}/{n_boxes_in} exact: {evidence['coco_round_trip_exact']})")

    return {"images": images, "labels": labels, "metadata": metadata,
            "image_hashes": image_hashes, "evidence": evidence,
            "references": [s.image_path for s in coco_samples]}


# --------------------------------------------------------------------------- #
# stage 2: the inference-provenance scenario
# --------------------------------------------------------------------------- #

def _fake_inference(image: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    """A deterministic stand-in for an inference service's output record.

    The provenance scenario tests the *binding* between input, model, config and
    output, so the output does not need to come from a real detector -- and using a
    synthetic record keeps the provenance test independent of any model's
    behaviour. Declared as an assumption in the scenario block.
    """
    n = int(rng.integers(0, 4))
    dets = []
    for _ in range(n):
        x = float(rng.uniform(0, IMG_SIZE - 12))
        y = float(rng.uniform(0, IMG_SIZE - 12))
        dets.append({"label": int(rng.integers(0, NUM_CLASSES)),
                     "score": round(float(rng.uniform(0.3, 0.99)), 6),
                     "box": [round(x, 4), round(y, 4), round(x + 12, 4), round(y + 12, 4)]})
    return {"detections": dets, "n": len(dets)}


def _forged_output(tag: str) -> Dict[str, Any]:
    """A deterministic fabricated inference record, used as an adversary's payload.

    Deliberately fixed rather than randomly drawn: a random forgery can collide
    with the genuine output it is replacing (two empty detection lists are equal),
    and a collision makes the "forgery" byte-identical to the truth -- which would
    quietly turn a detection failure into a test that never ran.
    """
    return {"detections": [{"label": 0, "score": 0.987654, "box": [1.0, 2.0, 13.0, 14.0],
                            "forged": tag}],
            "n": 1, "provenance": f"fabricated:{tag}"}


def _png_bytes(image: np.ndarray) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.clip(image, 0, 1) * 255.0).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _engine_output_hash(engine: InferenceProvenanceEngine, output: Any) -> str:
    """The exact output-hash derivation the sealing service uses.

    Used by the adversary in this scenario so that the forgery is as good as it can
    be: if the adversary were sloppier than this, the scenario would overstate what
    hashing alone can catch.
    """
    return engine._hash_output(output)


def _adversary_recompute(seal_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the record's unkeyed hashes after altering a bound field.

    This is what an adversary with write access to the record store actually does,
    and it is why hash-only verification of a store you do not control is not
    tamper-evidence: every hash here is unkeyed, so the forger can make the record
    perfectly self-consistent. Only the signature, which they cannot compute, still
    disagrees with the record.
    """
    payload = {
        "sequence": seal_dict["sequence_number"],
        "timestamp": seal_dict["timestamp"],
        "nonce": seal_dict["nonce"],
        "image_hash": seal_dict["image_hash"],
        "model_digest": seal_dict["model_digest"],
        "config_hash": seal_dict["config_hash"],
        "output_hash": seal_dict["output_hash"],
        "previous_seal_hash": seal_dict["previous_seal_hash"],
    }
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    seal_hash = hashlib.sha256(
        (payload_hash + seal_dict["signature"]).encode()).hexdigest()
    out = dict(seal_dict)
    out["payload_hash"] = payload_hash
    out["seal_hash"] = seal_hash
    return out


def build_provenance_scenario(
    n_history: int = 6,
    n_per_attack: int = 2,
    key_dir: Optional[str] = None,
    model_digest: str = "",
    log: Callable[[str], None] = lambda s: None,
) -> Dict[str, Any]:
    """Seal a trusted history, then present a new batch containing three attacks.

    Designs the adversary at full strength on the axis that matters: they hold write
    access to the record store, they alter bound fields, and they recompute every
    unkeyed hash afterwards. The four outcomes are:

    ``genuine``          a new, honest record continuing the sequence
    ``tampered_output``  a real record whose output hash was replaced
    ``swapped_model``    a real record re-attributed to a different model digest
    ``replayed``         a byte-identical old record re-submitted as a new one --
                         the only case where the stored bytes verify completely

    Returns per-record ground truth, what a digest-only checker can see, and CVIAF's
    verification result.

    ``key_dir=None`` uses a fresh temporary directory. That default matters: a
    scenario must not scatter private keys into whatever directory the caller
    happened to be standing in, which is how the demo outputs in this repository
    ended up with committed ``.priv`` files.
    """
    if key_dir is None:
        import tempfile
        key_dir = tempfile.mkdtemp(prefix="cviaf-keys-")
    os.makedirs(key_dir, exist_ok=True)
    mode, mode_note = signing_mode()
    sealer = InferenceProvenanceEngine(key_dir=key_dir)
    rng = np.random.default_rng(20260928)

    digest = model_digest or PLACEHOLDER_MODEL_DIGEST
    digest_other = hashlib.sha256(b"other-model").hexdigest()
    config = {"task": "detection", "input_size": IMG_SIZE, "score_thresh": 0.30,
              "nms_iou": 0.50, "preprocess": "unit_scale", "device": "cpu"}

    history: List[Dict[str, Any]] = []
    for _ in range(n_history):
        img = rng.random((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        out = _fake_inference(img, rng)
        raw = _png_bytes(img)
        seal = sealer.seal_inference(image_data=raw, model_digest=digest,
                                     config=config, output=out)
        history.append({"seal": seal, "raw": raw, "output": out})

    # Verifier/sealer separation. When the signing mode is asymmetric the
    # verifier is built from the sealer's PUBLIC key only, delivered through a
    # TrustStore exactly as an offline verifier would receive it; that is what
    # makes non-repudiation a tested property instead of an asserted one. In
    # HMAC mode no public key exists, so verification necessarily shares the
    # secret and the run is labelled accordingly. Either way the verifier's
    # replay set is ONLY the trusted history: an engine that remembers
    # everything it issued reports every genuine record as a replay.
    history_seals = [h["seal"] for h in history]
    if sealer.key_mgr.alg == ALG_ED25519:
        trust_dir = os.path.join(key_dir, "verifier-trust")
        TrustStore.write_public_key(trust_dir, sealer.key_mgr.public_key_envelope())
        verifier = Verifier(TrustStore(trust_dir), history=history_seals)
        verifier_key_material = "public-only"
    else:
        verifier = InferenceProvenanceEngine(key_manager=sealer.key_mgr)
        verifier.import_seals([h["seal"].to_dict() for h in history])
        verifier_key_material = "shared-secret"
    max_seq = max(h["seal"].sequence_number for h in history)

    records: List[Dict[str, Any]] = []

    def _present(truth: str, seal_dict, raw, output, claimed_model_digest,
                 recomputed: bool, note: str) -> None:
        seal = InferenceSeal.from_dict(seal_dict)
        verification = verifier.verify_seal(
            seal, image_data=raw, model_digest=claimed_model_digest,
            config=config, output=output)
        sig_valid = bool(verification["checks"].get("signature", {}).get("valid", False))
        # Replay is only meaningful for a record whose signature is intact. A record
        # with a broken signature is a forgery; calling it a replay as well would let
        # a signature failure borrow the replay control's credit and would mislabel
        # the attack class in the report.
        nonce_seen = bool(verifier.detect_replay(seal))
        # Per-key sequence ceiling when the verifier tracks one (rotation
        # restarts the sequence); the single-key scenario is unaffected.
        _is_stale = getattr(verifier, "is_stale", None)
        stale_sequence = bool(_is_stale(seal)) if _is_stale else \
            seal.sequence_number <= max_seq
        replay = bool(sig_valid and (nonce_seen or stale_sequence))
        failed_checks = [k for k, v in verification["checks"].items() if not v.get("valid")]
        records.append({
            "seal_id": seal_dict.get("seal_id", ""),
            "truth": truth,
            "note": note,
            "adversary_recomputed_hash": bool(recomputed),
            "observed": {
                "record_hash": seal_dict["seal_hash"],
                "payload_hash": seal_dict["payload_hash"],
                # The forger recomputed every unkeyed hash, so the record they
                # present is fully self-consistent. A hash-only checker sees nothing
                # wrong with it; that is the entire point of this scenario.
                "stored_hash_self_consistent": True,
                "sequence_number": int(seal_dict["sequence_number"]),
                "nonce": seal_dict["nonce"],
            },
            "cviaf_verification": {
                "flagged": bool((not verification["valid"]) or replay),
                "signature_valid": sig_valid,
                "replay_detected": replay,
                "nonce_already_seen": nonce_seen,
                "stale_sequence": bool(stale_sequence),
                "failed_checks": failed_checks,
                "n_findings": len(verification["findings"]),
                "attack_class": ("inference_tampering" if not sig_valid else
                                 "inference_replay" if replay else ""),
                "findings": verification["findings"],
                "checks": verification["checks"],
            },
        })

    def _fresh_record() -> Dict[str, Any]:
        img = rng.random((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        out = _fake_inference(img, rng)
        raw = _png_bytes(img)
        seal = sealer.seal_inference(image_data=raw, model_digest=digest,
                                     config=config, output=out)
        return {"seal": seal, "raw": raw, "output": out}

    # -- genuine new records
    for _ in range(n_per_attack):
        h = _fresh_record()
        _present("genuine", h["seal"].to_dict(), h["raw"], h["output"], digest, False,
                 "honest record continuing the sequence")

    # -- in-place edit of an enrolled record: output replaced, hashes recomputed
    for j in range(n_per_attack):
        h = history[j]
        forged = _forged_output(f"tamper-{j}")
        d = h["seal"].to_dict()
        d["output_hash"] = _engine_output_hash(sealer, forged)
        d = _adversary_recompute(d)
        _present("tampered_output", d, h["raw"], forged, digest, True,
                 "stored record edited in place: output replaced and every unkeyed "
                 "hash recomputed, so the record is self-consistent")

    # -- model substitution, in place, hashes recomputed
    for j in range(n_per_attack):
        h = history[n_per_attack + j] if n_per_attack + j < len(history) else history[0]
        d = h["seal"].to_dict()
        d["model_digest"] = digest_other
        d = _adversary_recompute(d)
        _present("swapped_model", d, h["raw"], h["output"], digest_other, True,
                 "record re-attributed to a different model digest, hashes recomputed")

    # -- a FORGED new record: an inference that never happened
    for j in range(n_per_attack):
        h = _fresh_record()
        forged = _forged_output(f"forged-{j}")
        d = h["seal"].to_dict()
        d["output_hash"] = _engine_output_hash(sealer, forged)
        d = _adversary_recompute(d)   # signature is the unchanged, stale one
        _present("forged_record", d, h["raw"], forged, digest, True,
                 "a new record for an inference that never happened: fresh identity, "
                 "no valid signature available to the adversary")

    # -- replay of an intact, genuine record
    for j in range(n_per_attack):
        h = history[j]
        _present("replayed", h["seal"].to_dict(), h["raw"], h["output"], digest, False,
                 "byte-identical old record re-submitted as new; nothing about the "
                 "stored bytes is wrong, so only a nonce/sequence check can see it")

    log(f"    provenance: {len(records)} records presented "
        f"({sum(1 for r in records if r['truth'] != 'genuine')} attacked) "
        f"signing={mode}")

    return {
        "signing_mode": mode,
        "signing_mode_note": mode_note,
        "verifier_key_material": verifier_key_material,
        "key_dir": key_dir,
        "n_history": n_history,
        "max_history_sequence": max_seq,
        "history": [{"seal_id": h["seal"].seal_id,
                     "seal_hash": h["seal"].seal_hash,
                     "sequence_number": h["seal"].sequence_number}
                    for h in history],
        "records": records,
        "assumption": ("Records are produced by a sealing service the pipeline does "
                       "not control, and the adversary holds write access to the "
                       "record store but not the signing key."),
    }


# --------------------------------------------------------------------------- #
# stage 3: the end-to-end assessment of one contributed asset
# --------------------------------------------------------------------------- #

def assure_model(
    corpus_dir: str,
    model_id: Optional[str] = None,
    out_dir: str = "runs/assurance",
    access_level: str = "white-box",
    alpha: float = 0.05,
    n_provenance: int = 6,
    max_clean_images: int = 24,
    reference_model_id: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Assess one contributed model plus its dataset end to end.

    The asset is taken from a lab corpus (whose ground truth we hold, so the report
    can be scored) but nothing in the pipeline reads the ground truth: the engine
    sees only the exported files, the model weights and the inference records, which
    is exactly what a real submission would be.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    registry = load_registry(corpus_dir)
    if not registry:
        raise RuntimeError(f"no models found under {corpus_dir}/")

    entry = None
    for e in registry:
        if model_id is None or e["manifest"]["model_id"] == model_id:
            entry = e
            break
    if entry is None:
        raise RuntimeError(f"model {model_id!r} not found in {corpus_dir}/")

    manifest = entry["manifest"]
    spec = train_spec_from_manifest(manifest)
    art = ModelArtifact.load(entry["dir"])
    splits = build_splits(spec)
    truth = splits.truth

    ref_entry = None
    for e in registry:
        if reference_model_id and e["manifest"]["model_id"] == reference_model_id:
            ref_entry = e
            break
        if not reference_model_id and e["manifest"]["ground_truth"]["kind"] == "clean":
            ref_entry = e
            break
    ref_model = ModelArtifact.load(ref_entry["dir"]).model if ref_entry else None
    ref_manifest = ref_entry["manifest"] if ref_entry else None

    mode, mode_note = signing_mode()
    log(f"CVIAF pipeline {PIPELINE_VERSION}")
    log(f"  asset      : {manifest['model_id']} (corpus {corpus_dir})")
    log(f"  access     : {access_level}   signing: {mode}")
    log(f"  reference  : {ref_manifest['model_id'] if ref_manifest else '(none)'}")
    log(f"  [1/5] intake")

    ingest = ingest_contribution(splits.train_poisoned,
                                 os.path.join(out_dir, "intake"),
                                 contributor_default="unknown", log=log)
    images = ingest["images"]
    labels = ingest["labels"]
    metadata = ingest["metadata"]
    features = extract_features_from_images(images, method="pixel_stats", target_dim=64)

    # The reference distribution is a contributor-disjoint clean holdout, so OOD and
    # drift checks compare against data the contributor never supplied.
    ref_features = extract_features_from_images(
        splits.cal_clean.images, method="pixel_stats", target_dim=64)

    # The audit trail is a named deliverable, so every module gets a hash-linked
    # entry carrying the digest of what it was given and what it produced. That is
    # what makes the log *reproducible* rather than decorative: re-running the
    # command reproduces the same input hashes, and any divergence shows up as a
    # chain that does not match the artefacts on disk.
    governance = GovernanceEngine(pipeline_id=f"cviaf-{manifest['model_id']}")
    governance.log_action("asset_received", "pipeline", details={
        "model_id": manifest["model_id"],
        "weights_digest": manifest["artifact"]["weights_digest"],
        "spec_digest": manifest["spec_digest"],
        "access_level": access_level,
    }, input_hash=manifest["artifact"]["weights_digest"])
    governance.log_action("intake_completed", "formats",
                          details=ingest["evidence"],
                          output_hash=hash_dict(ingest["evidence"]))

    def _module(name: str, payload: Dict[str, Any]) -> None:
        governance.log_action("module_completed", name, details=payload,
                              output_hash=hash_dict(payload))

    # -- module 1: training-data integrity
    log("  [2/5] training-data integrity")
    data_assessment = DataIntegrityAssessor().assess(
        images=images, features=features, labels=labels, metadata=metadata,
        image_hashes=ingest["image_hashes"], reference_features=ref_features)
    _module("data_integrity", {
        "module": "data_integrity",
        "n_images": int(len(images)),
        "n_findings": len(data_assessment.get("findings", [])),
        "overall_risk": data_assessment.get("overall_risk"),
        "dataset_digest": ingest["evidence"]["dataset_digest_in_memory"],
    })
    log(f"        {len(data_assessment.get('findings', []))} finding(s)")

    # -- module 2: model integrity
    log("  [3/5] model integrity")
    predict_fn = detector_predict_fn(art.model)
    clean_chw = np.transpose(images[:max_clean_images], (0, 3, 1, 2)).astype(np.float32)
    probe_inputs = np.transpose(images[:8], (0, 3, 1, 2)).astype(np.float32)
    parameters = None
    if str(access_level) == AccessLevel.WHITE_BOX.value:
        parameters = {name: np.asarray(arr) for name, arr in art.model._head_params()}
    reference_fingerprint = None
    if ref_model is not None:
        reference_fingerprint = BehavioralFingerprinter().create_fingerprint(
            detector_predict_fn(ref_model), probe_inputs)
    model_assessment = ModelIntegrityAssessor(access_level=access_level).assess(
        predict_fn=predict_fn,
        input_shape=(3, IMG_SIZE, IMG_SIZE),
        num_classes=NUM_CLASSES + 1,
        clean_images=clean_chw,
        parameters=parameters,
        reference_fingerprint=reference_fingerprint,
        reference_inputs=probe_inputs,
    )
    _module("model_integrity", {
        "module": "model_integrity",
        "access_level": access_level,
        "weights_digest": manifest["artifact"]["weights_digest"],
        "checks_performed": model_assessment.get("checks_performed", []),
        "checks_skipped": [c.get("check") for c in model_assessment.get("checks_skipped", [])],
        "n_findings": model_assessment.get("finding_count", 0),
        "overall_severity": model_assessment.get("overall_severity"),
    })
    log(f"        checks ran: {model_assessment.get('checks_performed')}; "
        f"skipped: {[c['check'] for c in model_assessment.get('checks_skipped', [])]}")

    # -- module 3: inference provenance
    log("  [4/5] inference provenance")
    scen = build_provenance_scenario(
        n_history=n_provenance, n_per_attack=2,
        key_dir=os.path.join(out_dir, "keys"),
        model_digest=manifest["artifact"]["weights_digest"], log=log)
    prov_findings: List[Dict[str, Any]] = []
    for rec in scen["records"]:
        if rec["cviaf_verification"]["flagged"]:
            prov_findings.extend(rec["cviaf_verification"]["findings"])
    provenance_verification = {
        "n_records": len(scen["records"]),
        "n_flagged": sum(1 for r in scen["records"] if r["cviaf_verification"]["flagged"]),
        "signing_mode": scen["signing_mode"],
        "signing_mode_note": scen["signing_mode_note"],
        "per_attack": _per_attack_summary(scen),
        "findings": prov_findings,
        "limitations": [scen["signing_mode_note"], scen["assumption"]],
    }
    _module("inference_provenance", {
        "module": "inference_provenance",
        "n_records": len(scen["records"]),
        "n_flagged": provenance_verification["n_flagged"],
        "signing_mode": scen["signing_mode"],
        "per_attack": provenance_verification["per_attack"],
        "model_digest_bound": manifest["artifact"]["weights_digest"],
    })

    # -- module 4: distribution shift
    log("  [5/5] distribution shift")
    # Attribution calibration: empirical nulls from the driftbench natural
    # battery. Absent the file the module fails closed (attribution_unavailable)
    # instead of guessing natural-vs-manipulated.
    from cviaf.drift.attribution import NaturalDriftCalibration
    attrib_cal = None
    cal_path = os.path.join(corpus_dir, "drift_calibration.json")
    if os.path.exists(cal_path):
        attrib_cal = NaturalDriftCalibration.load(cal_path)
        log(f"        attribution calibration: {attrib_cal.digest()} "
            f"({len(attrib_cal.scenario_descriptions)} natural scenarios)")
    else:
        log("        attribution calibration: NONE (verdict will be "
            "attribution_unavailable)")
    drift_assessment = DistributionShiftAssessor(seed=DRIFT_ASSESSMENT_SEED).assess(
        reference_features=ref_features,
        operational_features=features,
        reference_logits=None, operational_logits=None,
        reference_labels=None, operational_labels=None,
        reference_images=splits.cal_clean.images,
        operational_images=images,
        operational_contributors=[m.contributor for m in metadata],
        attribution_calibration=attrib_cal,
    )
    # This is a run-local assessment record, not the natural-drift arbiter's
    # independently measured calibration in the corpus directory.
    drift_record_path = os.path.join(out_dir, "drift_assessment_record.json")
    with open(drift_record_path, "w") as fh:
        json.dump({"schema": DRIFT_CALIBRATION_SCHEMA,
                   "pipeline_version": PIPELINE_VERSION,
                   "seed": DRIFT_ASSESSMENT_SEED,
                   "model_id": manifest["model_id"],
                   "assessment": drift_assessment}, fh, indent=1, sort_keys=True, default=str)
    _module("distribution_shift", {
        "module": "distribution_shift",
        "shift_detected": drift_assessment.get("shift_detected"),
        "overall_severity": drift_assessment.get("overall_severity"),
        "shift_type": (drift_assessment.get("characterization") or {}).get("shift_type"),
        "natural_vs_adversarial": (drift_assessment.get("characterization") or {})
                                  .get("natural_vs_adversarial"),
        "mmd_p_value": (drift_assessment.get("mmd") or {}).get("p_value"),
        "mean_mahalanobis": (drift_assessment.get("mahalanobis") or {})
                            .get("mean_distance"),
    })
    log(f"        shift_detected={drift_assessment.get('shift_detected')} "
        f"type={(drift_assessment.get('characterization') or {}).get('shift_type')} "
        f"verdict={(drift_assessment.get('characterization') or {}).get('natural_vs_adversarial')}")

    # -- governance
    report = governance.generate_report(
        data_assessment=data_assessment,
        model_assessment=model_assessment,
        provenance_verification=provenance_verification,
        drift_assessment=drift_assessment,
        extra_metadata={
            "pipeline_version": PIPELINE_VERSION,
            "access_level": access_level,
            "signing_mode": scen["signing_mode"],
            "signing_assumption": scen["signing_mode_note"],
            "intake_evidence": ingest["evidence"],
            "reference_model": ref_manifest["model_id"] if ref_manifest else None,
            "detector_to_classifier_adapter": (
                "model integrity runs through an adapter that reduces the detector's "
                "output to per-class image scores; adapter artefacts are possible and "
                "the native-statistic comparison lives in cviaf.lab.compare"),
        },
    )

    # -- artifacts
    report_path = os.path.join(out_dir, "assurance_report.json")
    governance.save_report(report, report_path)
    audit_path = os.path.join(out_dir, "audit_trail.json")
    governance.save_audit_trail(audit_path)
    coverage_path = os.path.join(out_dir, "coverage_statement.json")
    with open(coverage_path, "w") as fh:
        json.dump({"schema": "cviaf-coverage-statement/1.0",
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   **COVERAGE_STATEMENT}, fh, indent=1)
    seals_path = os.path.join(out_dir, "inference_records.json")
    with open(seals_path, "w") as fh:
        json.dump(scen, fh, indent=1, default=str)

    # -- schema validation of the report we just wrote
    from cviaf.governance.schema import REPORT_SCHEMA_ID, validate_report
    errors = validate_report(report.to_dict())
    if errors:
        log(f"  WARNING: {len(errors)} schema validation error(s): {errors[:3]}")

    bundle = {
        "run_id": report.report_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "environment": environment(),
        "asset": {
            "model_id": manifest["model_id"],
            "corpus": corpus_dir,
            "ground_truth_used_for_scoring_only": {
                "kind": manifest["ground_truth"]["kind"],
                "rate_requested": manifest["ground_truth"]["rate_requested"],
                "rate_actual": manifest["ground_truth"]["rate_actual"],
                "n_poisoned": manifest["ground_truth"]["n_poisoned"],
                "attack_success_rate": manifest["metrics"]["attack_success_rate"],
                "note": ("recorded in the bundle for scoring and review; the engine "
                         "never received it, so the report's findings are blind"),
            },
            "artifact_digests": manifest["artifact"],
            "spec_digest": manifest["spec_digest"],
            "contribution_label_used_for_scoring": truth.kind,
        },
        "inputs": {
            "coco": ingest["evidence"]["coco_annotations"],
            "yolo": ingest["evidence"]["yolo_root"],
            "n_images": int(len(images)),
            "n_contributors": len({m.contributor for m in metadata}),
            "contributors": _counts([m.contributor for m in metadata]),
        },
        "modules": {
            "data_integrity": {"findings": len(data_assessment.get("findings", [])),
                               "overall_risk": data_assessment.get("overall_risk")},
            "model_integrity": {"findings": model_assessment.get("finding_count", 0),
                                "checks_performed": model_assessment.get("checks_performed", []),
                                "checks_skipped": model_assessment.get("checks_skipped", []),
                                "access_level": access_level},
            "inference_provenance": {"n_records": provenance_verification["n_records"],
                                     "n_flagged": provenance_verification["n_flagged"],
                                     "signing_mode": scen["signing_mode"]},
            "distribution_shift": {
                "shift_detected": drift_assessment.get("shift_detected"),
                "severity": drift_assessment.get("overall_severity"),
                "shift_type": (drift_assessment.get("characterization") or {}).get("shift_type"),
                # The field the problem statement asks for by name: is this ordinary
                # operational drift, or does the evidence point at manipulation?
                "natural_vs_adversarial": (drift_assessment.get("characterization") or {})
                                          .get("natural_vs_adversarial"),
                "mean_mahalanobis": (drift_assessment.get("mahalanobis") or {})
                                    .get("mean_distance"),
                "mmd_p_value": (drift_assessment.get("mmd") or {}).get("p_value"),
                "seed": DRIFT_ASSESSMENT_SEED,
                "calibration_artifact": "drift_calibration.json",
            },
        },
        "artifacts": {},
        "report_summary": {
            "overall_risk": report.overall_risk,
            "overall_disposition": report.overall_disposition,
            "n_findings": len(report.findings),
            "audit_trail_valid": report.metadata.get("audit_trail_valid"),
            "audit_trail_length": report.metadata.get("audit_trail_length"),
        },
        "schema_validation": {"schema": REPORT_SCHEMA_ID, "valid": not errors,
                              "errors": errors},
        "reproduce": {
            "command": (f"python -m cviaf.lab assure --corpus {corpus_dir} "
                        f"--model {manifest['model_id']} --out {out_dir} "
                        f"--access-level {access_level}"),
            "seeds": {**manifest["seeds"], "drift_assessment": DRIFT_ASSESSMENT_SEED},
            "spec_digest": manifest["spec_digest"],
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for name, path in (("assurance_report", report_path), ("audit_trail", audit_path),
                       ("coverage_statement", coverage_path),
                       ("inference_records", seals_path)):
        bundle["artifacts"][name] = {"path": path, "sha256": hash_bytes(
            open(path, "rb").read())}
    bundle_path = os.path.join(out_dir, "bundle.json")
    with open(bundle_path, "w") as fh:
        json.dump(bundle, fh, indent=1, default=str)
    bundle["artifacts"]["bundle"] = {"path": bundle_path,
                                     "sha256": hash_bytes(open(bundle_path, "rb").read())}

    log(f"  report     : {report_path}  "
        f"({report.overall_risk} / {report.overall_disposition}, "
        f"{len(report.findings)} findings)")
    log(f"  bundle     : {bundle_path}")
    log(f"  schema     : {'valid' if not errors else 'INVALID'}")
    log(f"  elapsed    : {bundle['elapsed_seconds']}s")
    return {"report": report, "report_path": report_path, "bundle": bundle,
            "bundle_path": bundle_path, "audit_path": audit_path,
            "coverage_path": coverage_path, "seals_path": seals_path,
            "ingest_evidence": ingest["evidence"]}


def _per_attack_summary(scen: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rec in scen["records"]:
        agg = out.setdefault(rec["truth"], {"n": 0, "n_flagged": 0})
        agg["n"] += 1
        agg["n_flagged"] += int(bool(rec["cviaf_verification"]["flagged"]))
    for k, v in out.items():
        v["detection_rate"] = round(v["n_flagged"] / v["n"], 4) if v["n"] else None
    return out


def _counts(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out
