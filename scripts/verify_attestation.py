#!/usr/bin/env python3
"""Standalone offline CVIAF report/checkpoint verifier. No CVIAF imports.

Install cryptography separately. Trust the explicitly pinned Ed25519 public key,
not the unauthenticated keyid in the envelope. Exits nonzero on any failure.
"""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

PAYLOAD_TYPE = "application/vnd.in-toto+json"
PREDICATE_TYPE = "urn:cviaf:assurance-checkpoint:v1"
DOMAIN = b"CVIAF checkpoint v1\0"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def merkle_root(entries):
    if not entries:
        raise ValueError("Empty audit trail")
    nodes = [hashlib.sha256(b"\x00" + bytes.fromhex(e["entry_hash"])).digest() for e in entries]
    while len(nodes) > 1:
        nodes = [hashlib.sha256(b"\x01" + nodes[i] + nodes[i+1]).digest()
                 if i+1 < len(nodes) else nodes[i] for i in range(0, len(nodes), 2)]
    return nodes[0].hex()


def pae(t, p):
    t = t.encode()
    return b"DSSEv1 " + str(len(t)).encode() + b" " + t + b" " + str(len(p)).encode() + b" " + p


def verify(report, entries, bundle, public_key_pem, expected_count=None, expected_root=None):
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pub = load_pem_public_key(public_key_pem)
    if not isinstance(pub, Ed25519PublicKey):
        raise ValueError("Expected Ed25519 public key")
    envelope = bundle["attestation"]
    if envelope["payloadType"] != PAYLOAD_TYPE or len(envelope["signatures"]) != 1:
        raise ValueError("Wrong payload type or signature count")
    payload = base64.b64decode(envelope["payload"], validate=True)
    sig = base64.b64decode(envelope["signatures"][0]["sig"], validate=True)
    pub.verify(sig, pae(PAYLOAD_TYPE, payload))
    statement = json.loads(payload)
    if canonical(statement) != payload or statement["_type"] != "https://in-toto.io/Statement/v1" or statement["predicateType"] != PREDICATE_TYPE:
        raise ValueError("Invalid in-toto statement")
    if statement["subject"] != [{"name": "assurance_report.json", "digest": {"sha256": sha(canonical(report))}}]:
        raise ValueError("Report digest mismatch")
    checkpoint = statement["predicate"]["checkpoint"]
    if checkpoint != bundle["checkpoint"] or checkpoint["version"] != 1:
        raise ValueError("Checkpoint mismatch")
    signature = base64.b64decode(checkpoint["signature"], validate=True)
    unsigned = {k: v for k, v in checkpoint.items() if k != "signature"}
    pub.verify(signature, DOMAIN + canonical(unsigned))
    if not entries or report["audit_trail"] != entries or checkpoint["tree_size"] != len(entries):
        raise ValueError("Audit trail length/content mismatch (possible truncation)")
    if expected_count is not None and expected_count != len(entries):
        raise ValueError("Pinned latest tree size mismatch")
    if expected_root is not None and expected_root != checkpoint["merkle_root"]:
        raise ValueError("Pinned latest root mismatch")
    prev = "GENESIS"
    for i, e in enumerate(entries):
        if e["sequence_number"] != i or e["previous_entry_hash"] != prev:
            raise ValueError(f"Broken sequence/link at {i}")
        content = {k: e[k] for k in ("entry_id", "sequence_number", "timestamp", "action", "module", "details", "input_hash", "output_hash", "previous_entry_hash")}
        # Legacy AuditEntry uses default json.dumps separators (with spaces), sorted keys.
        h = sha(json.dumps(content, sort_keys=True, ensure_ascii=True).encode())
        if h != e["entry_hash"]:
            raise ValueError(f"Broken entry at {i}")
        prev = h
    if (checkpoint["last_entry_hash"] != prev or checkpoint["audit_sha256"] != sha(canonical(entries))
            or checkpoint["merkle_root"] != merkle_root(entries)):
        raise ValueError("Checkpoint does not match entries")
    return {"verified": True, "tree_size": len(entries), "merkle_root": checkpoint["merkle_root"],
            "signer_hint": envelope["signatures"][0].get("keyid")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--public-key", required=True, help="Trusted, pinned Ed25519 PEM public key")
    p.add_argument("--expected-count", type=int, help="Trusted latest log length; defeats rollback to older signed checkpoint")
    p.add_argument("--expected-root", help="Trusted latest checkpoint root")
    args = p.parse_args()
    try:
        result = verify(json.loads(Path(args.report).read_text()), json.loads(Path(args.audit).read_text()),
                        json.loads(Path(args.bundle).read_text()), Path(args.public_key).read_bytes(),
                        args.expected_count, args.expected_root)
        print(json.dumps(result))
        return 0
    except (Exception) as exc:
        print(f"INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
