#!/usr/bin/env python3
"""
CVIAF - Computer Vision Integrity Assurance Framework
Command-Line Interface

Usage:
    python -m cviaf.cli assess --dataset <path> --model <path> [options]
    python -m cviaf.cli demo                # Run end-to-end demo with synthetic data
    python -m cviaf.cli verify-audit <path>  # Verify audit trail integrity
    python -m cviaf.cli verify-seal <path>   # Verify inference seals
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np


def cmd_demo(args):
    """Run end-to-end demonstration with synthetic data and attacks."""
    from cviaf.attacks import (
        BadNetsPoisoner, BlendedTriggerPoisoner, LabelFlipper,
        DuplicateFlooder, OODInjector, InferenceTamperer,
        generate_test_dataset
    )
    from cviaf.orchestrator import CVIAFOrchestrator
    from cviaf.provenance import InferenceProvenanceEngine
    from cviaf.utils import extract_features_from_images

    print("=" * 70)
    print("CVIAF - Computer Vision Integrity Assurance Framework")
    print("End-to-End Demonstration")
    print("=" * 70)
    print()

    output_dir = args.output or "cviaf_demo_output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Generate clean dataset ---
    print("[1/7] Generating synthetic test dataset...")
    dataset = generate_test_dataset(
        num_samples=300,
        num_classes=5,
        image_shape=(32, 32, 3),
        feature_dim=64,
        num_contributors=3,
        seed=42,
    )
    clean_images = dataset["images"]
    clean_labels = dataset["labels"]
    clean_features = dataset["features"]
    clean_metadata = dataset["metadata"]
    clean_hashes = dataset["image_hashes"]
    print(f"   Generated {len(clean_images)} samples across "
          f"{dataset['num_classes']} classes from "
          f"{dataset['num_contributors']} contributors")

    # --- Step 2: Apply attacks ---
    print("\n[2/7] Applying attack scenarios...")

    # Attack 1: BadNets trigger injection
    print("   - BadNets patch trigger injection (10% poison rate)...")
    badnets = BadNetsPoisoner(target_class=0, poison_ratio=0.10, patch_size=4, seed=42)
    poisoned_images, poisoned_labels, poison_mask, badnets_info = badnets.poison(
        clean_images.copy(), clean_labels.copy(), clean_metadata
    )
    print(f"     Injected {badnets_info['num_poisoned']} poisoned samples")

    # Attack 2: Label flipping from contributor_1
    print("   - Label flipping (class 1 -> 3, from contributor_1)...")
    flipper = LabelFlipper(source_class=1, target_class=3,
                          flip_ratio=0.3, contributor_filter="contributor_1", seed=42)
    poisoned_labels, flip_mask, flip_info = flipper.poison(poisoned_labels, clean_metadata)
    print(f"     Flipped {flip_info['num_flipped']} labels")

    # Attack 3: Duplicate flooding
    print("   - Near-duplicate flooding (40 duplicates from malicious_vendor)...")
    flooder = DuplicateFlooder(num_duplicates=40, noise_std=0.005,
                              malicious_contributor="malicious_vendor", seed=42)
    poisoned_images, poisoned_labels, aug_metadata, dup_mask, dup_info = flooder.flood(
        poisoned_images, poisoned_labels, clean_metadata
    )
    print(f"     Added {dup_info['num_duplicates']} near-duplicate samples")

    # Attack 4: OOD injection
    print("   - OOD injection (20 Gaussian noise samples)...")
    ood_injector = OODInjector(num_ood=20, ood_type="gaussian_noise",
                               target_label=2, contributor="untrusted_lab", seed=42)
    poisoned_images, poisoned_labels, aug_metadata, ood_mask, ood_info = ood_injector.inject(
        poisoned_images, poisoned_labels, aug_metadata
    )
    print(f"     Injected {ood_info['num_ood']} OOD samples")

    # Recompute features and hashes for the augmented dataset
    print("\n[3/7] Extracting features from augmented dataset...")
    aug_features = extract_features_from_images(poisoned_images, method="pixel_stats", target_dim=64)
    aug_hashes = [
        __import__("hashlib").sha256(poisoned_images[i].tobytes()).hexdigest()
        for i in range(len(poisoned_images))
    ]
    print(f"   Extracted {aug_features.shape[1]}-dim features for {len(aug_features)} samples")

    # --- Step 3: Create a mock model prediction function ---
    print("\n[4/7] Setting up model prediction function...")
    num_classes = dataset["num_classes"]

    def mock_predict(inputs: np.ndarray) -> np.ndarray:
        """Mock model: uses PCA features to generate plausible logits."""
        n = inputs.shape[0]
        flat = inputs.reshape(n, -1)
        # Simple linear projection to logits
        rng = np.random.RandomState(hash(flat.tobytes()[:100]) % 2**31)
        weights = rng.randn(flat.shape[1], num_classes) * 0.1
        logits = flat @ weights
        return logits.astype(np.float32)

    # --- Step 4: Create provenance seals and tamper some ---
    print("\n[5/7] Generating inference provenance seals...")
    prov_engine = InferenceProvenanceEngine(key_dir=os.path.join(output_dir, "keys"))

    seals = []
    for i in range(10):
        seal = prov_engine.seal_inference(
            image_data=poisoned_images[i].tobytes(),
            model_digest="sha256:mock_model_digest_abc123",
            config={"confidence_threshold": 0.5, "nms_iou": 0.45},
            output=mock_predict(poisoned_images[i:i+1]),
        )
        seals.append(seal)

    print(f"   Created {len(seals)} sealed inference records")

    # Tamper with some seals
    print("   - Tampering with seal #3 (output substitution)...")
    tampered_seal = InferenceTamperer.tamper_output(seals[3].to_dict())
    print("   - Forging seal #7 (signature forgery)...")
    forged_seal = InferenceTamperer.forge_seal(seals[7].to_dict())

    inference_records = [s.to_dict() for s in seals]
    inference_records[3] = tampered_seal
    inference_records[7] = forged_seal

    # --- Step 5: Run full assessment ---
    print("\n[6/7] Running full CVIAF assessment pipeline...")
    print("   This may take a minute...")
    start = time.time()

    orchestrator = CVIAFOrchestrator(
        pipeline_id="demo-assessment-001",
        model_access_level="black-box",  # Use black-box for demo (faster)
        key_dir=os.path.join(output_dir, "keys"),
        output_dir=output_dir,
    )

    # Use clean features as reference distribution for drift detection
    reference_features = clean_features

    report = orchestrator.run_full_assessment(
        images=poisoned_images,
        features=aug_features,
        labels=poisoned_labels,
        metadata=aug_metadata,
        image_hashes=aug_hashes,
        predict_fn=mock_predict,
        model_input_shape=(32, 32, 3),
        num_classes=num_classes,
        model_digest="sha256:mock_model_digest_abc123",
        reference_features=reference_features,
        inference_records=inference_records,
    )

    elapsed = time.time() - start

    # --- Step 6: Print results ---
    print(f"\n[7/7] Assessment complete in {elapsed:.1f}s")
    print("=" * 70)
    print()

    # Print human-readable summary
    summary = report.metadata.get("human_readable_summary", "")
    if summary:
        print(summary)

    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Overall Risk:       {report.overall_risk}")
    print(f"  Overall Disposition: {report.overall_disposition.upper()}")
    print(f"  Total Findings:     {len(report.findings)}")
    print(f"  Audit Trail Entries: {len(report.audit_trail)}")
    print(f"  Audit Trail Valid:  {report.metadata.get('audit_trail_valid', 'N/A')}")
    print()

    # Finding breakdown
    severity_counts = {}
    for f in report.findings:
        sev = f.get("severity", "UNKNOWN")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print("  Finding Breakdown:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        if sev in severity_counts:
            print(f"    {sev}: {severity_counts[sev]}")

    print(f"\n  Report saved to: {os.path.join(output_dir, 'assurance_report.json')}")
    print(f"  Audit trail saved to: {os.path.join(output_dir, 'audit_trail.json')}")

    # Print coverage statement summary
    print(f"\n  Supported Attack Classes: {len(report.coverage_statement.get('supported_attack_classes', []))}")
    print(f"  Declared Unsupported Conditions: {len(report.coverage_statement.get('unsupported_conditions', []))}")

    print(f"\n{'=' * 70}")
    print("Demo complete.")

    return 0


def cmd_assess(args):
    """Run assessment on real data."""
    from cviaf.orchestrator import CVIAFOrchestrator
    from cviaf.formats import load_dataset
    from cviaf.formats.model_loader import load_model
    from cviaf.utils import extract_features_from_images, compute_image_hashes

    print("CVIAF Assessment Pipeline")
    print("=" * 50)

    # Load dataset
    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        samples = load_dataset(
            args.dataset,
            format=args.format or "auto",
            contributor=args.contributor or "unknown",
        )
        print(f"  Loaded {len(samples)} samples")

        # Load images and extract features
        images = []
        labels = []
        metadata = []
        
        try:
            from PIL import Image
            for s in samples:
                if os.path.exists(s.image_path):
                    img = Image.open(s.image_path).convert("RGB").resize((224, 224))
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(s.labels[0] if s.labels else 0)
                    metadata.append(s.metadata)
        except ImportError:
            print("  Warning: PIL not available, using placeholder images")
            for s in samples:
                images.append(np.random.rand(224, 224, 3).astype(np.float32))
                labels.append(s.labels[0] if s.labels else 0)
                metadata.append(s.metadata)

        images = np.array(images) if images else None
        labels = np.array(labels) if labels else None
        features = extract_features_from_images(images) if images is not None else None
        image_hashes = compute_image_hashes(images) if images is not None else None
    else:
        images = features = labels = metadata = image_hashes = None

    # Load model
    predict_fn = None
    model_parameters = None
    model_digest = ""
    model_input_shape = (3, 224, 224)
    num_classes = -1

    if args.model:
        print(f"Loading model from {args.model}...")
        try:
            model = load_model(
                args.model,
                access_level=args.access_level or "white-box",
            )
            predict_fn = model.predict
            model_digest = model.compute_digest()
            model_input_shape = model.input_shape
            num_classes = model.num_classes

            if args.access_level != "black-box":
                try:
                    model_parameters = model.get_parameters()
                except Exception:
                    pass

            print(f"  Model loaded: {model.get_info()}")
        except Exception as e:
            print(f"  Error loading model: {e}")

    # Run assessment
    output_dir = args.output or "cviaf_output"
    orchestrator = CVIAFOrchestrator(
        pipeline_id=args.pipeline_id or "",
        model_access_level=args.access_level or "white-box",
        output_dir=output_dir,
    )

    report = orchestrator.run_full_assessment(
        images=images,
        features=features,
        labels=labels,
        metadata=metadata,
        image_hashes=image_hashes,
        predict_fn=predict_fn,
        model_input_shape=model_input_shape,
        num_classes=num_classes,
        model_parameters=model_parameters,
        model_digest=model_digest,
        reference_features=features,  # Use same as reference if not provided
        skip_modules=args.skip.split(",") if args.skip else None,
    )

    print(f"\nOverall Risk: {report.overall_risk}")
    print(f"Disposition: {report.overall_disposition}")
    print(f"Findings: {len(report.findings)}")
    print(f"Report saved to: {os.path.join(output_dir, 'assurance_report.json')}")

    return 0


def cmd_verify_audit(args):
    """Verify an audit trail file for tamper evidence."""
    from cviaf.core.types import AuditEntry

    print(f"Verifying audit trail: {args.path}")

    with open(args.path, "r") as f:
        entries_data = json.load(f)

    print(f"  {len(entries_data)} entries found")

    # Verify hash chain
    valid = True
    for i, entry_dict in enumerate(entries_data):
        entry = AuditEntry(**{k: v for k, v in entry_dict.items()
                             if k in AuditEntry.__dataclass_fields__})

        # Check previous hash
        expected_prev = entries_data[i - 1]["entry_hash"] if i > 0 else "GENESIS"
        if entry.previous_entry_hash != expected_prev:
            print(f"  BROKEN at entry {i}: previous hash mismatch")
            valid = False
            break

        # Recompute hash
        saved_hash = entry.entry_hash
        computed = entry.compute_hash()
        if computed != saved_hash:
            print(f"  TAMPERED at entry {i}: hash mismatch")
            print(f"    Expected: {saved_hash}")
            print(f"    Computed: {computed}")
            valid = False
            break

    if valid:
        print("  RESULT: Audit trail is INTACT - no tampering detected")
    else:
        print("  RESULT: Audit trail is BROKEN - tampering detected!")

    return 0 if valid else 1


def cmd_verify_seal(args):
    """Verify inference provenance seals."""
    from cviaf.provenance import InferenceProvenanceEngine, InferenceSeal

    print(f"Verifying seal file: {args.path}")

    with open(args.path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        seals = data
    elif isinstance(data, dict) and "seals" in data:
        seals = data["seals"]
    else:
        seals = [data]

    print(f"  {len(seals)} seals found")

    engine = InferenceProvenanceEngine(key_dir=args.key_dir or ".cviaf_keys")

    for i, seal_dict in enumerate(seals):
        seal = InferenceSeal.from_dict(seal_dict)
        result = engine.verify_seal(seal)
        status = "VALID" if result["valid"] else "INVALID"
        print(f"  Seal {seal.seal_id}: {status}")
        if not result["valid"]:
            for finding in result.get("findings", []):
                print(f"    - {finding.get('title', 'Unknown issue')}")

    return 0


def cmd_schema(args):
    """Print the assurance report JSON schema."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CVIAF Assurance Report",
        "type": "object",
        "required": ["report_id", "framework_version", "timestamp", "assessments",
                     "findings", "overall_risk", "overall_disposition",
                     "audit_trail", "coverage_statement"],
        "properties": {
            "report_id": {"type": "string", "format": "uuid"},
            "framework_version": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "pipeline_id": {"type": "string"},
            "assessments": {
                "type": "object",
                "properties": {
                    "training_data_integrity": {"type": "object"},
                    "model_integrity": {"type": "object"},
                    "inference_provenance": {"type": "object"},
                    "distribution_shift": {"type": "object"},
                },
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["finding_id", "module", "severity", "confidence",
                                "title", "description", "disposition"],
                    "properties": {
                        "finding_id": {"type": "string"},
                        "module": {"type": "string"},
                        "attack_class": {"type": "string"},
                        "severity": {"type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "evidence": {"type": "object"},
                        "affected_assets": {"type": "array", "items": {"type": "string"}},
                        "disposition": {"type": "string", "enum": ["quarantine", "review", "accept"]},
                        "remediation": {"type": "string"},
                        "timestamp": {"type": "string"},
                    },
                },
            },
            "overall_risk": {"type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]},
            "overall_disposition": {"type": "string", "enum": ["quarantine", "review", "accept"]},
            "audit_trail": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["entry_id", "sequence_number", "entry_hash", "previous_entry_hash"],
                },
            },
            "coverage_statement": {
                "type": "object",
                "required": ["supported_attack_classes", "unsupported_conditions", "assumptions"],
            },
            "limitations": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
    }
    print(json.dumps(schema, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="cviaf",
        description="CVIAF - Computer Vision Integrity Assurance Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run end-to-end demo with synthetic data and attacks")
    demo_parser.add_argument("--output", "-o", default="cviaf_demo_output", help="Output directory")

    # Assess command
    assess_parser = subparsers.add_parser("assess", help="Run assessment on real data")
    assess_parser.add_argument("--dataset", "-d", help="Path to dataset (COCO JSON or YOLO directory)")
    assess_parser.add_argument("--model", "-m", help="Path to model file (.onnx, .pt, .pth)")
    assess_parser.add_argument("--format", "-f", choices=["coco", "yolo", "auto"], default="auto")
    assess_parser.add_argument("--access-level", "-a", choices=["white-box", "black-box"], default="white-box")
    assess_parser.add_argument("--output", "-o", default="cviaf_output", help="Output directory")
    assess_parser.add_argument("--pipeline-id", default="", help="Pipeline identifier")
    assess_parser.add_argument("--contributor", default="unknown", help="Dataset contributor name")
    assess_parser.add_argument("--skip", default="", help="Comma-separated modules to skip (data,model,provenance,drift)")

    # Verify audit trail
    verify_audit_parser = subparsers.add_parser("verify-audit", help="Verify audit trail integrity")
    verify_audit_parser.add_argument("path", help="Path to audit trail JSON file")

    # Verify seals
    verify_seal_parser = subparsers.add_parser("verify-seal", help="Verify inference provenance seals")
    verify_seal_parser.add_argument("path", help="Path to seal JSON file")
    verify_seal_parser.add_argument("--key-dir", default=".cviaf_keys", help="Key directory")

    # Schema command
    subparsers.add_parser("schema", help="Print the assurance report JSON schema")

    args = parser.parse_args()

    if args.command == "demo":
        return cmd_demo(args)
    elif args.command == "assess":
        return cmd_assess(args)
    elif args.command == "verify-audit":
        return cmd_verify_audit(args)
    elif args.command == "verify-seal":
        return cmd_verify_seal(args)
    elif args.command == "schema":
        return cmd_schema(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
