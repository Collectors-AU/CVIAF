"""Opt-in signed checkpoint and DSSE/in-toto assurance report export.

Private keys are supplied by the operator, never generated or stored here.
"""
import base64
import hashlib
import json
from pathlib import Path

PAYLOAD_TYPE = "application/vnd.in-toto+json"
PREDICATE_TYPE = "urn:cviaf:assurance-checkpoint:v1"
DOMAIN = b"CVIAF checkpoint v1\0"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def merkle_root(entries):
    """Domain-separated binary Merkle tree over ordered canonical entry hashes.

    Odd nodes are carried unchanged; the empty tree is not a valid checkpoint.
    """
    if not entries:
        raise ValueError("Cannot checkpoint an empty log")
    nodes = [hashlib.sha256(b"\x00" + bytes.fromhex(e["entry_hash"])).digest() for e in entries]
    while len(nodes) > 1:
        nodes = [hashlib.sha256(b"\x01" + nodes[i] + nodes[i + 1]).digest()
                 if i + 1 < len(nodes) else nodes[i] for i in range(0, len(nodes), 2)]
    return nodes[0].hex()


def pae(payload_type, payload):
    t = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(t)).encode() + b" " + t + b" " + str(len(payload)).encode() + b" " + payload


def _validate_chain(entries):
    # Import only at signing time. The offline verifier has its own implementation.
    from cviaf.core.types import AuditEntry
    if not entries:
        raise ValueError("Empty log")
    prev = "GENESIS"
    for i, record in enumerate(entries):
        if record["sequence_number"] != i or record["previous_entry_hash"] != prev:
            raise ValueError(f"Broken sequence/link at {i}")
        entry = AuditEntry(**{k: v for k, v in record.items() if k in AuditEntry.__dataclass_fields__})
        if entry.compute_hash() != record["entry_hash"]:
            raise ValueError(f"Broken entry hash at {i}")
        prev = record["entry_hash"]


def sign_bundle(report, entries, private_key, key_id):
    """Return a portable checkpoint and DSSE envelope, bound to full report and log.

    Anchor the resulting bundle outside the mutable log. A newly issued checkpoint
    for an earlier prefix is valid in isolation; a pinned latest count/root is needed
    to assert that this is the latest state.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("A pinned key identifier is required")
    key = load_pem_private_key(private_key, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Checkpoint signer must be Ed25519")
    _validate_chain(entries)
    if report.get("audit_trail") != entries:
        raise ValueError("Report and audit trail differ")
    checkpoint = {"version": 1, "tree_size": len(entries), "merkle_root": merkle_root(entries),
                  "last_entry_hash": entries[-1]["entry_hash"], "audit_sha256": digest(canonical(entries))}
    checkpoint["signature"] = base64.b64encode(key.sign(DOMAIN + canonical(checkpoint))).decode()
    statement = {"_type": "https://in-toto.io/Statement/v1",
                 "subject": [{"name": "assurance_report.json", "digest": {"sha256": digest(canonical(report))}}],
                 "predicateType": PREDICATE_TYPE,
                 "predicate": {"checkpoint": checkpoint}}
    payload = canonical(statement)
    envelope = {"payloadType": PAYLOAD_TYPE,
                "payload": base64.b64encode(payload).decode(),
                "signatures": [{"keyid": key_id,
                                "sig": base64.b64encode(key.sign(pae(PAYLOAD_TYPE, payload))).decode()}]}
    return {"checkpoint": checkpoint, "attestation": envelope}


def sign_files(report_path, audit_path, private_key_path, key_id, output_path):
    report = json.loads(Path(report_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    bundle = sign_bundle(report, audit, Path(private_key_path).read_bytes(), key_id)
    Path(output_path).write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle
