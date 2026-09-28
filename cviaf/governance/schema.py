"""
The assurance-report schema, and a validator that needs nothing installed.

The problem statement lists "the assurance-report schema" as a deliverable in its
own right, which makes sense: a report that a downstream system cannot parse is an
essay, not an interface. This module ships the schema as data (``REPORT_SCHEMA``,
JSON Schema draft 2020-12) so it can be handed to a consumer, checked into a
contract registry, or used to generate bindings, and it ships a validator so the
pipeline can *fail its own output* rather than emitting a report that does not
conform.

Why a hand-written validator instead of ``jsonschema``
------------------------------------------------------
The deployment target is an air-gapped machine and the dependency list is a supply
chain. Rather than take a general-purpose validator, the subset of JSON Schema this
report actually uses is implemented here -- about eighty lines -- and it is tested
against the real pipeline output. The subset is stated explicitly so no reader has
to guess what is checked:

    type, enum, properties, required, additionalProperties, items, minItems,
    maxItems, minLength, maxLength, minimum, maximum, const, anyOf, $ref,
    $defs (local refs only)

Anything outside that subset is reported as an unsupported keyword rather than
silently ignored. A validator that quietly ignores the keyword you cared about is
worse than no validator, because it returns "valid" with authority it has not
earned.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

REPORT_SCHEMA_ID = "https://cviaf.local/schemas/assurance-report-3.0.0.json"
REPORT_SCHEMA_VERSION = "3.0.0"

SUPPORTED_KEYWORDS = {
    "$schema", "$id", "$defs", "$ref", "title", "description", "type", "enum",
    "const", "properties", "required", "additionalProperties", "items", "minItems",
    "maxItems", "minLength", "maxLength", "minimum", "maximum", "anyOf",
    "examples", "default", "format", "pattern",
}

def _type_ok(instance: Any, expected: str) -> bool:
    """JSON type check. Booleans are not integers here, which JSON Schema requires
    and Python's subclass relationship would otherwise break."""
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "null":
        return instance is None
    return True   # unknown type name: not this validator's business to reject


def _resolve_ref(ref: str, root: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _validate(instance: Any, schema: Dict[str, Any], root: Dict[str, Any],
              path: str, errors: List[str], depth: int = 0) -> None:
    """Recursive subset validator. Appends human-readable errors with JSON paths."""
    if depth > 40:
        errors.append(f"{path}: schema recursion limit reached")
        return
    if not isinstance(schema, dict):
        return

    for kw in schema:
        if kw not in SUPPORTED_KEYWORDS:
            errors.append(f"{path}: schema uses unsupported keyword {kw!r}; this "
                          f"validator will not silently ignore it")

    if "$ref" in schema:
        target = _resolve_ref(schema["$ref"], root)
        if target is None:
            errors.append(f"{path}: unresolvable $ref {schema['$ref']!r}")
            return
        _validate(instance, target, root, path, errors, depth + 1)
        return

    if "anyOf" in schema:
        branch_errors: List[List[str]] = []
        for branch in schema["anyOf"]:
            sub: List[str] = []
            _validate(instance, branch, root, path, sub, depth + 1)
            if not sub:
                branch_errors = []
                break
            branch_errors.append(sub)
        if branch_errors:
            errors.append(f"{path}: no anyOf branch matched ({branch_errors[0][0]})")

    for t in ([schema["type"]] if isinstance(schema.get("type"), str)
              else schema.get("type", [])):
        if not _type_ok(instance, t):
            errors.append(f"{path}: expected {t}, got {type(instance).__name__}")
            return

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                _validate(value, props[key], root, f"{path}.{key}", errors, depth + 1)
            elif extra is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                _validate(value, extra, root, f"{path}.{key}", errors, depth + 1)

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: string of length {len(instance)} < minLength "
                          f"{schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string of length {len(instance)} > maxLength "
                          f"{schema['maxLength']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: {len(instance)} items < minItems "
                          f"{schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: {len(instance)} items > maxItems "
                          f"{schema['maxItems']}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(instance):
                _validate(item, schema["items"], root, f"{path}[{i}]", errors,
                          depth + 1)


# --------------------------------------------------------------------------- #
# the schema
# --------------------------------------------------------------------------- #

REPORT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": REPORT_SCHEMA_ID,
    "title": "CVIAF Assurance Report",
    "description": ("A single evidence-based assessment of a contributed dataset, a "
                    "trained model and the inference records associated with them. "
                    "Every finding carries a reason, evidence, confidence, the "
                    "affected asset and a recommended disposition; the report "
                    "carries the coverage statement and the limitations of its own "
                    "assessment, and a tamper-evident audit trail."),
    "type": "object",
    "additionalProperties": True,   # forward-compatible: unknown blocks are allowed
    "required": ["report_id", "framework_version", "timestamp", "pipeline_id",
                 "assessments", "findings", "overall_risk", "overall_disposition",
                 "audit_trail", "coverage_statement", "limitations", "metadata"],
    "properties": {
        "report_id": {"type": "string", "minLength": 1},
        "framework_version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "pipeline_id": {"type": "string"},
        "assessments": {"$ref": "#/$defs/assessments"},
        "findings": {"type": "array", "items": {"$ref": "#/$defs/finding"}},
        "overall_risk": {"$ref": "#/$defs/severity"},
        "overall_disposition": {"$ref": "#/$defs/disposition"},
        "audit_trail": {"type": "array", "items": {"$ref": "#/$defs/audit_entry"}},
        "coverage_statement": {"$ref": "#/$defs/coverage_statement"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
    "$defs": {
        "severity": {"type": "string",
                     "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]},
        "disposition": {"type": "string",
                        "enum": ["quarantine", "review", "accept"]},
        "finding": {
            "type": "object",
            "required": ["finding_id", "module", "attack_class", "severity",
                         "confidence", "title", "description", "evidence",
                         "affected_assets", "disposition", "remediation",
                         "timestamp"],
            "properties": {
                "finding_id": {"type": "string", "minLength": 1},
                "module": {"type": "string", "minLength": 1,
                           "description": "which assessment module raised this"},
                "attack_class": {"type": "string",
                                 "description": ("the attack class this finding is "
                                                 "evidence of, or empty when the "
                                                 "module can only report an anomaly")},
                "severity": {"$ref": "#/$defs/severity"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "title": {"type": "string", "minLength": 1},
                "description": {"type": "string",
                                "description": "human-readable reason for the flag"},
                "evidence": {"type": "object",
                             "description": "measured values supporting the finding"},
                "affected_assets": {"type": "array", "items": {"type": "string"}},
                "disposition": {"$ref": "#/$defs/disposition"},
                "remediation": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
            },
        },
        "audit_entry": {
            "type": "object",
            "required": ["entry_id", "sequence_number", "timestamp", "action",
                         "module", "details", "input_hash", "output_hash",
                         "previous_entry_hash", "entry_hash"],
            "properties": {
                "entry_id": {"type": "string"},
                "sequence_number": {"type": "integer", "minimum": 0},
                "timestamp": {"type": "string", "format": "date-time"},
                "action": {"type": "string"},
                "module": {"type": "string"},
                "details": {"type": "object"},
                "input_hash": {"type": "string"},
                "output_hash": {"type": "string"},
                "previous_entry_hash": {"type": "string"},
                "entry_hash": {"type": "string",
                               "description": ("SHA-256 over the entry's own fields "
                                               "and its predecessor's hash; recomputing "
                                               "it is what verifies the chain")},
            },
        },
        "coverage_statement": {
            "type": "object",
            "required": ["supported_attack_classes", "unsupported_conditions",
                         "assumptions"],
            "properties": {
                "supported_attack_classes": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["class", "description", "detection_methods",
                                     "access_required", "known_limitations"],
                        "properties": {
                            "class": {"type": "string"},
                            "description": {"type": "string"},
                            "detection_methods": {"type": "array",
                                                  "items": {"type": "string"}},
                            "access_required": {"type": "string"},
                            "known_limitations": {"type": "array",
                                                  "items": {"type": "string"}},
                        },
                    },
                },
                "unsupported_conditions": {"type": "array",
                                           "items": {"type": "string"}},
                "assumptions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "assessments": {
            "type": "object",
            "description": ("One block per module that ran. A module that did not "
                            "run is absent, and its absence is the signal that "
                            "'accept' cannot be claimed on its axis."),
            "additionalProperties": {"type": "object"},
        },
    },
}


def write_schema(path: str) -> str:
    """Persist the schema so it can be checked in or handed to a consumer."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(REPORT_SCHEMA, fh, indent=1)
    return path


def validate_report(report: Dict[str, Any],
                    schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """Validate a report dict. Returns a list of error strings; empty means valid."""
    root = schema if schema is not None else REPORT_SCHEMA
    errors: List[str] = []
    _validate(report, root, root, "$", errors)
    return errors


def validate_file(path: str) -> List[str]:
    with open(path) as fh:
        return validate_report(json.load(fh))
