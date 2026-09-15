"""
Analyst-Facing Assurance and Governance Module.

Every flag includes:
  - Human-readable reason
  - Supporting evidence
  - Confidence/severity
  - Affected asset
  - Recommended disposition (accept/review/quarantine)

Maintains a tamper-evident audit trail and explicitly declares
attack classes and conditions that the system does not support.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cviaf.core.types import (
    Finding, Severity, Disposition, AttackClass, AuditTrail,
    AssuranceReport, hash_dict
)


# Complete coverage statement
COVERAGE_STATEMENT = {
    "supported_attack_classes": [
        {
            "class": AttackClass.TRIGGER_INJECTION.value,
            "description": "Pixel-patch, blended, and frequency-domain trigger patterns injected into training images",
            "detection_methods": ["pixel_variance_analysis", "fft_analysis", "spectral_signatures"],
            "access_required": "dataset",
            "known_limitations": [
                "Clean-label attacks with imperceptible triggers may evade pixel-level detection",
                "WaNet (warping-based) triggers require flow-field analysis which is computationally expensive",
            ],
        },
        {
            "class": AttackClass.LABEL_FLIPPING.value,
            "description": "Deliberate or accidental label changes in training data",
            "detection_methods": ["confident_learning", "cross_validation_disagreement"],
            "access_required": "dataset + any classifier",
            "known_limitations": [
                "Requires sufficient samples per class (min ~50) for reliable detection",
                "May flag genuinely ambiguous samples as mislabeled",
            ],
        },
        {
            "class": AttackClass.SYSTEMATIC_MISLABEL.value,
            "description": "Coordinated mislabeling from a single contributor/source",
            "detection_methods": ["contributor_aggregation", "pattern_analysis"],
            "access_required": "dataset with contributor metadata",
            "known_limitations": [
                "Requires contributor/source metadata to attribute patterns",
                "Small-scale mislabeling (<2% of samples) may not reach detection threshold",
            ],
        },
        {
            "class": AttackClass.DUPLICATE_FLOODING.value,
            "description": "Near-duplicate images flooding the dataset to bias the model",
            "detection_methods": ["perceptual_hashing", "cosine_similarity_clustering"],
            "access_required": "dataset",
            "known_limitations": [
                "Subtle augmentation variants (rotations, crops) may not cluster as duplicates",
                "Threshold sensitivity: too low catches legitimate similar images, too high misses near-duplicates",
            ],
        },
        {
            "class": AttackClass.OOD_INSERTION.value,
            "description": "Out-of-distribution samples inserted into training data",
            "detection_methods": ["mahalanobis_distance", "isolation_forest"],
            "access_required": "dataset + reference distribution",
            "known_limitations": [
                "Requires a clean reference distribution for comparison",
                "Borderline OOD samples near the decision boundary are hard to classify",
            ],
        },
        {
            "class": AttackClass.MODEL_BACKDOOR.value,
            "description": "Hidden backdoor behavior triggered by specific input patterns",
            "detection_methods": [
                "neural_cleanse (white-box)",
                "entropy_probe (black-box)",
                "strip_analysis (black-box)",
                "weight_analysis (white-box)",
            ],
            "access_required": "model (white-box preferred, black-box possible)",
            "known_limitations": [
                "Gradient-free Neural Cleanse optimization is less reliable than gradient-based",
                "Black-box entropy analysis may produce false positives on miscalibrated but clean models",
                "Clean-label backdoors are harder to detect than patch-based triggers",
            ],
        },
        {
            "class": AttackClass.MODEL_SUBSTITUTION.value,
            "description": "Entire model replaced with a different one",
            "detection_methods": ["weight_digest_comparison", "behavioral_fingerprinting"],
            "access_required": "model + reference fingerprint or digest",
            "known_limitations": [
                "First assessment has no baseline for comparison",
                "Fine-tuning creates a new digest but may preserve most behavior",
            ],
        },
        {
            "class": AttackClass.WEIGHT_MODIFICATION.value,
            "description": "Targeted modification of model weights post-training",
            "detection_methods": ["weight_statistics", "dormant_neuron_analysis"],
            "access_required": "white-box (model parameters)",
            "known_limitations": [
                "Cannot distinguish malicious modification from legitimate fine-tuning",
                "Statistical tests have limited power against small, targeted changes",
            ],
        },
        {
            "class": AttackClass.INFERENCE_REPLAY.value,
            "description": "Re-submission of old inference records as new",
            "detection_methods": ["nonce_verification", "sequence_check", "timestamp_validation"],
            "access_required": "inference seal records",
            "known_limitations": [
                "Requires the provenance system to have been active during original inference",
            ],
        },
        {
            "class": AttackClass.INFERENCE_TAMPERING.value,
            "description": "Post-hoc modification of inference inputs, outputs, or configuration",
            "detection_methods": ["cryptographic_seal_verification", "hash_chain_validation"],
            "access_required": "inference seal records + original artifacts",
            "known_limitations": [
                "Cannot detect tampering of records that were never sealed",
                "Floating-point non-determinism may cause false positives in output hash verification across different hardware",
            ],
        },
        {
            "class": AttackClass.OUTPUT_SUBSTITUTION.value,
            "description": "Replacement of genuine inference outputs with fabricated ones",
            "detection_methods": ["output_hash_verification", "signature_verification"],
            "access_required": "inference seal records",
            "known_limitations": [
                "Same as inference tampering limitations",
            ],
        },
        {
            "class": AttackClass.COVARIATE_SHIFT.value,
            "description": "Change in input feature distribution (terrain, season, sensor, illumination)",
            "detection_methods": ["mahalanobis_distance", "mmd_test", "ks_test"],
            "access_required": "reference distribution + operational data features",
            "known_limitations": [
                "Requires representative reference distribution baseline",
                "MMD kernel bandwidth sensitive to dimensionality",
            ],
        },
        {
            "class": AttackClass.CONCEPT_DRIFT.value,
            "description": "Change in the relationship between features and labels",
            "detection_methods": ["accuracy_monitoring", "prediction_shift_analysis"],
            "access_required": "reference data with labels + operational data with labels",
            "known_limitations": [
                "Requires labeled operational data for concept drift detection",
                "Hard to distinguish from covariate shift without labels",
            ],
        },
        {
            "class": AttackClass.ADVERSARIAL_DISTRIBUTION_SHIFT.value,
            "description": "Deliberately crafted distribution shift to degrade model performance",
            "detection_methods": ["anomaly_pattern_analysis", "multi_signal_fusion"],
            "access_required": "reference distribution + operational data",
            "known_limitations": [
                "Distinction between natural and adversarial shift is probabilistic, not definitive",
                "Sophisticated adversarial shifts that mimic natural patterns may evade detection",
            ],
        },
    ],
    "unsupported_conditions": [
        "Federated learning poisoning attacks (requires access to distributed training process)",
        "Hardware-level Trojan injection (physical chip modifications)",
        "Cryptographic key compromise (key management is out of scope)",
        "Adversarial examples at inference time (test-time attacks on individual inputs)",
        "Membership inference attacks (privacy attacks, not integrity)",
        "Model extraction/stealing attacks (IP theft, not integrity)",
        "Supply chain attacks on software dependencies",
        "Side-channel attacks on inference hardware",
    ],
    "assumptions": [
        "The assurance framework itself runs on trusted hardware",
        "The Python runtime and cryptographic libraries are not compromised",
        "Reference distributions and fingerprints, when provided, are genuine",
        "File system integrity is maintained (files are not modified between hash computation and use)",
        "Clock source for timestamps is not manipulated (relevant for air-gapped timestamp ordering)",
    ],
}


class GovernanceEngine:
    """
    Generates analyst-facing assurance reports with full evidence trails,
    tamper-evident audit logs, and coverage declarations.
    """

    def __init__(self, pipeline_id: str = ""):
        self.pipeline_id = pipeline_id or f"pipeline-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.audit_trail = AuditTrail()
        self.all_findings: List[Finding] = []

    def log_action(self, action: str, module: str, details: Dict[str, Any] = None,
                   input_hash: str = "", output_hash: str = ""):
        """Record an action in the tamper-evident audit trail."""
        self.audit_trail.append(
            action=action,
            module=module,
            details=details or {},
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def add_findings(self, findings: List[Dict[str, Any]]):
        """Add findings from any assessment module."""
        for f_dict in findings:
            finding = Finding(**{k: v for k, v in f_dict.items()
                               if k in Finding.__dataclass_fields__})
            self.all_findings.append(finding)

    def compute_overall_risk(self) -> tuple[str, str]:
        """
        Compute overall risk level and disposition from all findings.
        Returns (severity, disposition).
        """
        if not self.all_findings:
            return Severity.LOW.value, Disposition.ACCEPT.value

        severity_rank = {
            Severity.INFO.value: 0,
            Severity.LOW.value: 1,
            Severity.MEDIUM.value: 2,
            Severity.HIGH.value: 3,
            Severity.CRITICAL.value: 4,
        }

        max_sev = max(self.all_findings, key=lambda f: severity_rank.get(f.severity, 0))
        overall_sev = max_sev.severity

        # Count high+ findings
        high_plus = sum(1 for f in self.all_findings
                       if severity_rank.get(f.severity, 0) >= 3)

        if overall_sev == Severity.CRITICAL.value or high_plus >= 3:
            disposition = Disposition.QUARANTINE.value
        elif overall_sev == Severity.HIGH.value or high_plus >= 1:
            disposition = Disposition.REVIEW.value
        else:
            disposition = Disposition.ACCEPT.value

        return overall_sev, disposition

    def generate_report(
        self,
        data_assessment: Dict[str, Any] = None,
        model_assessment: Dict[str, Any] = None,
        provenance_verification: Dict[str, Any] = None,
        drift_assessment: Dict[str, Any] = None,
        extra_metadata: Dict[str, Any] = None,
    ) -> AssuranceReport:
        """
        Generate the final assurance report combining all module outputs.
        """
        # Collect all findings from assessments
        assessments = {}

        if data_assessment:
            assessments["training_data_integrity"] = data_assessment
            if "findings" in data_assessment:
                self.add_findings(data_assessment["findings"])

        if model_assessment:
            assessments["model_integrity"] = model_assessment
            if "findings" in model_assessment:
                self.add_findings(model_assessment["findings"])

        if provenance_verification:
            assessments["inference_provenance"] = provenance_verification
            if "findings" in provenance_verification:
                self.add_findings(provenance_verification["findings"])

        if drift_assessment:
            assessments["distribution_shift"] = drift_assessment
            if "findings" in drift_assessment:
                self.add_findings(drift_assessment["findings"])

        self.log_action(
            action="generate_report",
            module="governance",
            details={"num_findings": len(self.all_findings)},
            output_hash=hash_dict({"findings_count": len(self.all_findings)}),
        )

        # Verify audit trail integrity
        chain_valid, broken_at = self.audit_trail.verify_chain()

        overall_sev, overall_disp = self.compute_overall_risk()

        # Build human-readable summary
        summary = self._build_summary(overall_sev, overall_disp)

        report = AssuranceReport(
            pipeline_id=self.pipeline_id,
            assessments=assessments,
            findings=[f.to_dict() for f in self.all_findings],
            overall_risk=overall_sev,
            overall_disposition=overall_disp,
            audit_trail=self.audit_trail.to_dict(),
            coverage_statement=COVERAGE_STATEMENT,
            limitations=self._collect_limitations(assessments),
            metadata={
                "audit_trail_valid": chain_valid,
                "audit_trail_length": len(self.audit_trail.entries),
                "human_readable_summary": summary,
                **(extra_metadata or {}),
            },
        )

        return report

    def _build_summary(self, severity: str, disposition: str) -> str:
        """Build a human-readable executive summary."""
        lines = [
            f"ASSURANCE REPORT - Pipeline: {self.pipeline_id}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Overall Risk: {severity}",
            f"Recommended Action: {disposition.upper()}",
            "",
        ]

        # Group findings by module
        by_module: Dict[str, List[Finding]] = {}
        for f in self.all_findings:
            by_module.setdefault(f.module, []).append(f)

        for module, findings in by_module.items():
            lines.append(f"--- {module.upper()} ---")
            for f in findings:
                lines.append(
                    f"  [{f.severity}] {f.title} "
                    f"(confidence: {f.confidence:.0%}, action: {f.disposition})"
                )
                lines.append(f"    {f.description}")
            lines.append("")

        if not self.all_findings:
            lines.append("No issues detected. All assessments passed.")

        return "\n".join(lines)

    def _collect_limitations(self, assessments: Dict[str, Any]) -> List[str]:
        """Collect all declared limitations from assessments."""
        limitations = []
        for module_name, assessment in assessments.items():
            if isinstance(assessment, dict):
                module_lims = assessment.get("limitations", [])
                for lim in module_lims:
                    limitations.append(f"[{module_name}] {lim}")

                # Check for skipped checks
                skipped = assessment.get("checks_skipped", [])
                for s in skipped:
                    if isinstance(s, dict):
                        limitations.append(
                            f"[{module_name}] {s.get('check', 'unknown')} skipped: {s.get('reason', 'N/A')}"
                        )

        return limitations

    def save_report(self, report: AssuranceReport, output_path: str) -> str:
        """Save the report to a JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report.to_json(indent=2))
        return output_path

    def save_audit_trail(self, output_path: str) -> str:
        """Save the audit trail separately for independent verification."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.audit_trail.to_json())
        return output_path
