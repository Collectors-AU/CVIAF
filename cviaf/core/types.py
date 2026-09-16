"""
Core type definitions, base classes, and shared data structures.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Risk severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Disposition(str, Enum):
    """Recommended actions for flagged items."""
    QUARANTINE = "quarantine"
    REVIEW = "review"
    ACCEPT = "accept"


class AccessLevel(str, Enum):
    """Model access level."""
    WHITE_BOX = "white-box"
    BLACK_BOX = "black-box"
    GRAY_BOX = "gray-box"


class AttackClass(str, Enum):
    """Supported attack class taxonomy."""
    TRIGGER_INJECTION = "trigger_injection"
    LABEL_FLIPPING = "label_flipping"
    SYSTEMATIC_MISLABEL = "systematic_mislabeling"
    DUPLICATE_FLOODING = "near_duplicate_flooding"
    OOD_INSERTION = "ood_insertion"
    MODEL_SUBSTITUTION = "model_substitution"
    MODEL_BACKDOOR = "model_backdoor"
    WEIGHT_MODIFICATION = "weight_modification"
    INFERENCE_REPLAY = "inference_replay"
    INFERENCE_TAMPERING = "inference_tampering"
    OUTPUT_SUBSTITUTION = "output_substitution"
    COVARIATE_SHIFT = "covariate_shift"
    CONCEPT_DRIFT = "concept_drift"
    ADVERSARIAL_DISTRIBUTION_SHIFT = "adversarial_distribution_shift"


def sanitize_json(obj: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects to Python native types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            val = obj.item()
            if isinstance(val, (bool, int, float, str)):
                return val
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return sanitize_json(obj.tolist())
        except (ValueError, TypeError):
            pass
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (datetime, timezone)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


class CVIAFJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling numpy types, datetimes, and custom objects."""
    def default(self, o):
        if hasattr(o, "item") and callable(o.item):
            try:
                return o.item()
            except (ValueError, TypeError):
                pass
        if hasattr(o, "tolist") and callable(o.tolist):
            try:
                return o.tolist()
            except (ValueError, TypeError):
                pass
        if isinstance(o, (datetime, timezone)):
            return o.isoformat()
        if isinstance(o, set):
            return list(o)
        if isinstance(o, Enum):
            return o.value
        return str(o)


@dataclass
class Finding:
    """A single finding from any assessment module."""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    module: str = ""
    attack_class: str = ""
    severity: str = Severity.INFO.value
    confidence: float = 0.0
    title: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    affected_assets: List[str] = field(default_factory=list)
    disposition: str = Disposition.ACCEPT.value
    remediation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json(asdict(self))


@dataclass 
class SampleMetadata:
    """Metadata for a single dataset sample."""
    sample_id: str = ""
    file_path: str = ""
    contributor: str = "unknown"
    batch_id: str = ""
    source: str = ""
    label: str = ""
    label_id: int = -1
    timestamp: str = ""
    annotations: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = ("sample_id", "file_path", "contributor", "batch_id",
                     "source", "label", "label_id", "timestamp",
                     "annotations", "extra")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleMetadata":
        """Build a SampleMetadata from a plain dict (JSON-style metadata).

        Unknown keys are preserved in ``extra`` instead of being dropped.
        """
        known = {k: d[k] for k in cls._KNOWN_FIELDS if k in d}
        extra = {k: v for k, v in d.items() if k not in cls._KNOWN_FIELDS}
        if extra:
            known.setdefault("extra", {}).update(extra)
        return cls(**known)


def normalize_metadata(metadata: Optional[List[Any]]) -> List[SampleMetadata]:
    """Coerce a metadata list to List[SampleMetadata].

    Accepts SampleMetadata objects, plain dicts, or a mix of both.
    Non-mappable entries are dropped (never crash the pipeline on metadata).
    """
    if not metadata:
        return []
    out: List[SampleMetadata] = []
    for m in metadata:
        if isinstance(m, SampleMetadata):
            out.append(m)
        elif isinstance(m, dict):
            out.append(SampleMetadata.from_dict(m))
    return out


@dataclass
class AuditEntry:
    """A single tamper-evident audit trail entry."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_number: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = ""
    module: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    output_hash: str = ""
    previous_entry_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute hash of this entry for chain integrity."""
        content = json.dumps(sanitize_json({
            "entry_id": self.entry_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "action": self.action,
            "module": self.module,
            "details": self.details,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "previous_entry_hash": self.previous_entry_hash,
        }), sort_keys=True, cls=CVIAFJSONEncoder)
        self.entry_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.entry_hash

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json(asdict(self))


class AuditTrail:
    """
    Tamper-evident audit trail using hash chains.
    Each entry's hash depends on the previous entry, forming
    a blockchain-like structure where any modification breaks
    the chain verification.
    """

    def __init__(self):
        self.entries: List[AuditEntry] = []
        self._sequence = 0

    def append(self, action: str, module: str, details: Dict[str, Any] = None,
               input_hash: str = "", output_hash: str = "") -> AuditEntry:
        prev_hash = self.entries[-1].entry_hash if self.entries else "GENESIS"
        entry = AuditEntry(
            sequence_number=self._sequence,
            action=action,
            module=module,
            details=details or {},
            input_hash=input_hash,
            output_hash=output_hash,
            previous_entry_hash=prev_hash,
        )
        entry.compute_hash()
        self.entries.append(entry)
        self._sequence += 1
        return entry

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """
        Verify the entire chain. Returns (valid, first_broken_index).
        If valid, first_broken_index is None.
        """
        for i, entry in enumerate(self.entries):
            # Check previous hash linkage
            expected_prev = self.entries[i - 1].entry_hash if i > 0 else "GENESIS"
            if entry.previous_entry_hash != expected_prev:
                return False, i

            # Recompute and check entry hash
            saved_hash = entry.entry_hash
            entry.compute_hash()
            if entry.entry_hash != saved_hash:
                return False, i

        return True, None

    def to_dict(self) -> List[Dict[str, Any]]:
        return sanitize_json([e.to_dict() for e in self.entries])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, cls=CVIAFJSONEncoder)


@dataclass
class AssuranceReport:
    """Top-level assurance report combining all module assessments."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    framework_version: str = "2.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_id: str = ""
    assessments: Dict[str, Any] = field(default_factory=dict)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    overall_risk: str = Severity.LOW.value
    overall_disposition: str = Disposition.ACCEPT.value
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    coverage_statement: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_json(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, cls=CVIAFJSONEncoder)


def hash_file(filepath: str) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    """SHA-256 hash of a dictionary (JSON-serialized, sorted keys)."""
    return hashlib.sha256(json.dumps(sanitize_json(d), sort_keys=True, cls=CVIAFJSONEncoder).encode()).hexdigest()

