"""
CVIAF Lab — the ground-truth laboratory.

The ``cviaf`` package itself is the *assurance engine*: it assesses assets
datasets, models, inference logs. This subpackage is the *laboratory* that
manufactures those assets with known ground truth, so the engine's claims can be
scored rather than asserted.

Why this exists at all: the problem statement requires "reproducible methods to
introduce representative poisoning, backdoor, substitution and tampering
scenarios for testing". That is a deliverable, not scaffolding. The lab is where
it lives, and it is the reason the assurance numbers in the report are ours
rather than someone else's paper.

Layout
------
``synth``      procedural detection dataset with declarable shift axes (+ COCO/YOLO export)
``detector``   tiny deterministic numpy detector (frozen backbone, trained head)
``poison``     BadDet-style attacks (OGA/ODA/RMA/GMA) plus patch/blend/clean-label/flip/dup/OOD
``train``      deterministic trainer: manifests, digests, resume
``detectors``  assurance-side detectors (TRACE CTC/FTC, fingerprint, weights, spectral, dup)
``calibrate``  conformal p-values, Cauchy fusion, FDR control, power curves
``corpus``     resumable corpus runner: trains the model matrix, skips what exists
``loop``       the all-day supervisor: adds fresh seeds in cycles, writes a heartbeat
"""

from cviaf.lab.detector import DETECTOR_VERSION, DetectorConfig, TinyDetector
from cviaf.lab.synth import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    DetectionDataset,
    SceneSpec,
    build_dataset,
    generate_scene,
)

__all__ = [
    "DETECTOR_VERSION",
    "DetectorConfig",
    "TinyDetector",
    "CLASS_NAMES",
    "IMG_SIZE",
    "NUM_CLASSES",
    "DetectionDataset",
    "SceneSpec",
    "build_dataset",
    "generate_scene",
]
