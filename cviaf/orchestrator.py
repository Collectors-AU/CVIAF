"""
Main Orchestrator - ties all modules together into a single assessment pipeline.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cviaf.core.types import (
    AuditTrail, AssuranceReport, AccessLevel,
    hash_bytes, hash_dict, hash_file
)
from cviaf.data_integrity import DataIntegrityAssessor
from cviaf.model_integrity import ModelIntegrityAssessor
from cviaf.provenance import InferenceProvenanceEngine, MerkleTree
from cviaf.drift import DistributionShiftAssessor
from cviaf.governance import GovernanceEngine


class CVIAFOrchestrator:
    """
    Main orchestrator for the CV Integrity Assurance Framework.
    
    Runs all five assessment modules in sequence, feeds results
    into the governance engine, and produces a unified assurance report.
    """

    def __init__(
        self,
        pipeline_id: str = "",
        model_access_level: str = "white-box",
        key_dir: str = ".cviaf_keys",
        output_dir: str = "cviaf_output",
    ):
        self.model_access_level = model_access_level
        self.output_dir = output_dir
        self.governance = GovernanceEngine(pipeline_id=pipeline_id)
        self.provenance = InferenceProvenanceEngine(key_dir=key_dir)

        os.makedirs(output_dir, exist_ok=True)

    def run_full_assessment(
        self,
        # Dataset inputs
        images: np.ndarray = None,
        features: np.ndarray = None,
        labels: np.ndarray = None,
        metadata: list = None,
        image_hashes: List[str] = None,
        # Model inputs
        predict_fn=None,
        model_input_shape: Tuple[int, ...] = None,
        num_classes: int = -1,
        model_parameters: Dict[str, np.ndarray] = None,
        model_digest: str = "",
        reference_fingerprint: Dict[str, Any] = None,
        reference_inputs: np.ndarray = None,
        # Distribution shift inputs
        reference_features: np.ndarray = None,
        reference_logits: np.ndarray = None,
        operational_logits: np.ndarray = None,
        reference_labels: np.ndarray = None,
        # Provenance inputs
        inference_records: List[Dict[str, Any]] = None,
        # Options
        skip_modules: List[str] = None,
    ) -> AssuranceReport:
        """
        Run the complete assessment pipeline.
        
        Args:
            images: Training images (N, H, W, C) or (N, C, H, W)
            features: Pre-extracted features (N, D)
            labels: Training labels (N,)
            metadata: Per-sample metadata list
            image_hashes: Pre-computed image hashes
            predict_fn: Model inference function
            model_input_shape: Single input shape
            num_classes: Number of output classes
            model_parameters: Model weight dict (white-box)
            model_digest: SHA-256 of model file
            reference_fingerprint: Prior behavioral fingerprint
            reference_inputs: Standard inputs for fingerprinting
            reference_features: Reference distribution features
            reference_logits: Reference distribution model outputs
            operational_logits: Operational model outputs
            reference_labels: Reference label distribution
            inference_records: Provenance seal dicts to verify
            skip_modules: List of modules to skip
        
        Returns:
            AssuranceReport
        """
        skip = set(skip_modules or [])
        start_time = time.time()

        self.governance.log_action(
            "assessment_started", "orchestrator",
            details={
                "model_access_level": self.model_access_level,
                "skip_modules": list(skip),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        data_assessment = None
        model_assessment = None
        provenance_report = None
        drift_assessment = None

        # --- Module 1: Training Data Integrity ---
        if "data" not in skip and (images is not None or features is not None):
            self.governance.log_action(
                "module_started", "data_integrity",
                input_hash=hash_bytes(features.tobytes()) if features is not None else "",
            )
            try:
                data_assessor = DataIntegrityAssessor()
                data_assessment = data_assessor.assess(
                    images=images,
                    features=features,
                    labels=labels,
                    metadata=metadata,
                    image_hashes=image_hashes,
                    reference_features=reference_features,
                )
                self.governance.log_action(
                    "module_completed", "data_integrity",
                    details={"finding_count": len(data_assessment.get("findings", []))},
                    output_hash=hash_dict({"findings": len(data_assessment.get("findings", []))}),
                )
            except Exception as e:
                self.governance.log_action(
                    "module_error", "data_integrity",
                    details={"error": str(e)},
                )
                data_assessment = {"error": str(e), "findings": []}

        # --- Module 2: Model Integrity ---
        if "model" not in skip and predict_fn is not None:
            self.governance.log_action(
                "module_started", "model_integrity",
                input_hash=model_digest,
            )
            try:
                model_assessor = ModelIntegrityAssessor(
                    access_level=self.model_access_level
                )

                clean_images_for_model = None
                if images is not None and model_input_shape is not None:
                    # Use a subset of clean images
                    clean_images_for_model = images[:min(100, len(images))]

                model_assessment = model_assessor.assess(
                    predict_fn=predict_fn,
                    input_shape=model_input_shape or (3, 224, 224),
                    num_classes=num_classes,
                    clean_images=clean_images_for_model,
                    parameters=model_parameters,
                    reference_fingerprint=reference_fingerprint,
                    reference_inputs=reference_inputs,
                )
                self.governance.log_action(
                    "module_completed", "model_integrity",
                    details={"finding_count": model_assessment.get("finding_count", 0)},
                    output_hash=hash_dict({"findings": model_assessment.get("finding_count", 0)}),
                )
            except Exception as e:
                self.governance.log_action(
                    "module_error", "model_integrity",
                    details={"error": str(e)},
                )
                model_assessment = {"error": str(e), "findings": []}

        # --- Module 3: Inference Provenance ---
        if "provenance" not in skip and inference_records:
            self.governance.log_action(
                "module_started", "provenance",
            )
            try:
                provenance_findings = []
                for record in inference_records:
                    from cviaf.provenance import InferenceSeal
                    seal = InferenceSeal.from_dict(record)
                    verification = self.provenance.verify_seal(seal)
                    if not verification["valid"]:
                        provenance_findings.extend(verification.get("findings", []))

                chain_result = self.provenance.verify_chain()

                provenance_report = {
                    "records_checked": len(inference_records),
                    "chain_valid": chain_result["valid"],
                    "chain_breaks": chain_result.get("breaks", []),
                    "findings": provenance_findings,
                }
                self.governance.log_action(
                    "module_completed", "provenance",
                    details={"finding_count": len(provenance_findings)},
                )
            except Exception as e:
                self.governance.log_action(
                    "module_error", "provenance",
                    details={"error": str(e)},
                )
                provenance_report = {"error": str(e), "findings": []}

        # --- Module 4: Distribution Shift ---
        if "drift" not in skip and reference_features is not None and features is not None:
            self.governance.log_action(
                "module_started", "distribution_shift",
            )
            try:
                drift_assessor = DistributionShiftAssessor()
                drift_assessment = drift_assessor.assess(
                    reference_features=reference_features,
                    operational_features=features,
                    reference_logits=reference_logits,
                    operational_logits=operational_logits,
                    reference_labels=reference_labels,
                    operational_labels=labels,
                )
                self.governance.log_action(
                    "module_completed", "distribution_shift",
                    details={"shift_detected": drift_assessment.get("shift_detected", False)},
                )
            except Exception as e:
                self.governance.log_action(
                    "module_error", "distribution_shift",
                    details={"error": str(e)},
                )
                drift_assessment = {"error": str(e), "findings": []}

        # --- Module 5: Governance & Report Generation ---
        elapsed = time.time() - start_time

        report = self.governance.generate_report(
            data_assessment=data_assessment,
            model_assessment=model_assessment,
            provenance_verification=provenance_report,
            drift_assessment=drift_assessment,
            extra_metadata={
                "assessment_duration_seconds": round(elapsed, 2),
                "model_access_level": self.model_access_level,
            },
        )

        # Save outputs
        report_path = os.path.join(self.output_dir, "assurance_report.json")
        self.governance.save_report(report, report_path)

        audit_path = os.path.join(self.output_dir, "audit_trail.json")
        self.governance.save_audit_trail(audit_path)

        return report
