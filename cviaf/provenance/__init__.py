"""
Inference Provenance and Output Integrity Module.

Creates verifiable cryptographic bindings among:
  - Input image hash
  - Model identifier/weight digest
  - Preprocessing and inference configuration
  - Resulting output tensor/predictions

Uses Ed25519 digital signatures (via Python's cryptography library or
the built-in nacl fallback) for offline, air-gapped operation.

Implements:
  - Hash-chain audit trail for sequential records
  - Merkle tree for batch verification
  - Nonce + sequence + timestamp controls against replay
  - Tamper detection on any component
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import struct
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cviaf.core.types import Finding, Severity, Disposition, AuditTrail


# --- Key Management ---

class OfflineKeyManager:
    """
    Manages Ed25519 signing keys for air-gapped environments.
    Keys are generated locally and stored on disk (in production,
    would use an HSM or secure enclave).
    """

    def __init__(self, key_dir: str = ".cviaf_keys"):
        self.key_dir = key_dir
        self._signing_key = None
        self._verify_key = None
        self._key_id = ""
        self._use_fallback = False

    def generate_keypair(self) -> str:
        """Generate a new Ed25519 keypair. Returns key_id."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization

            private_key = Ed25519PrivateKey.generate()
            self._signing_key = private_key
            self._verify_key = private_key.public_key()

            # Serialize for storage
            priv_bytes = private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption()
            )
            pub_bytes = self._verify_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw
            )
        except ImportError:
            # Fallback: use HMAC-SHA256 with a shared secret
            self._use_fallback = True
            priv_bytes = secrets.token_bytes(32)
            pub_bytes = priv_bytes  # In HMAC mode, same key for sign/verify
            self._signing_key = priv_bytes
            self._verify_key = priv_bytes

        self._key_id = hashlib.sha256(pub_bytes).hexdigest()[:16]

        os.makedirs(self.key_dir, exist_ok=True)
        key_path = os.path.join(self.key_dir, f"key_{self._key_id}")
        with open(key_path + ".priv", "wb") as f:
            f.write(priv_bytes)
        with open(key_path + ".pub", "wb") as f:
            f.write(pub_bytes)

        return self._key_id

    def load_keypair(self, key_id: str) -> None:
        """Load an existing keypair from disk."""
        key_path = os.path.join(self.key_dir, f"key_{key_id}")

        with open(key_path + ".priv", "rb") as f:
            priv_bytes = f.read()
        with open(key_path + ".pub", "rb") as f:
            pub_bytes = f.read()

        self._key_id = key_id

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            self._signing_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
            self._verify_key = self._signing_key.public_key()
        except (ImportError, Exception):
            self._use_fallback = True
            self._signing_key = priv_bytes
            self._verify_key = pub_bytes

    def sign(self, data: bytes) -> bytes:
        """Sign data with the private key."""
        if self._signing_key is None:
            raise RuntimeError("No keypair loaded. Call generate_keypair() or load_keypair() first.")

        if self._use_fallback:
            return hmac.new(self._signing_key, data, hashlib.sha256).digest()
        else:
            return self._signing_key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify a signature. Returns True if valid."""
        if self._verify_key is None:
            raise RuntimeError("No keypair loaded.")

        try:
            if self._use_fallback:
                expected = hmac.new(self._verify_key, data, hashlib.sha256).digest()
                return hmac.compare_digest(expected, signature)
            else:
                self._verify_key.verify(signature, data)
                return True
        except Exception:
            return False

    @property
    def key_id(self) -> str:
        return self._key_id


# --- Inference Seal ---

@dataclass
class InferenceSeal:
    """A cryptographic seal binding input, model, config, and output."""
    seal_id: str = ""
    sequence_number: int = 0
    timestamp: str = ""
    nonce: str = ""
    image_hash: str = ""
    model_digest: str = ""
    config_hash: str = ""
    output_hash: str = ""
    payload_hash: str = ""
    signature: str = ""  # hex-encoded
    key_id: str = ""
    previous_seal_hash: str = ""
    seal_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> InferenceSeal:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class InferenceProvenanceEngine:
    """
    Creates and verifies cryptographic provenance records for
    CV inference operations.
    
    Every inference record binds together:
      1. SHA-256 of the input image
      2. SHA-256 digest of model weights
      3. Hash of preprocessing/inference config
      4. Hash of the output tensor
      5. Unique nonce (anti-replay)
      6. Monotonic sequence number
      7. Timestamp
      8. Link to previous record (hash chain)
    
    Signed with Ed25519 for non-repudiation.
    """

    def __init__(self, key_manager: Optional[OfflineKeyManager] = None,
                 key_dir: str = ".cviaf_keys"):
        if key_manager:
            self.key_mgr = key_manager
        else:
            self.key_mgr = OfflineKeyManager(key_dir)
            self.key_mgr.generate_keypair()

        self._sequence = 0
        self._seals: List[InferenceSeal] = []
        self._used_nonces: set = set()

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Deterministic hash of configuration dict."""
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _hash_output(self, output: Any) -> str:
        """Hash an output tensor or prediction."""
        if isinstance(output, np.ndarray):
            # Use tobytes for exact reproducibility
            data = output.tobytes()
        elif isinstance(output, (dict, list)):
            data = json.dumps(output, sort_keys=True, default=str).encode()
        elif isinstance(output, str):
            data = output.encode()
        elif isinstance(output, bytes):
            data = output
        else:
            data = str(output).encode()
        return hashlib.sha256(data).hexdigest()

    def _hash_image(self, image_path: str = "", image_data: bytes = None) -> str:
        """Hash an input image."""
        if image_data:
            return hashlib.sha256(image_data).hexdigest()
        elif image_path and os.path.exists(image_path):
            h = hashlib.sha256()
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        else:
            return hashlib.sha256(b"").hexdigest()

    def seal_inference(
        self,
        image_path: str = "",
        image_data: bytes = None,
        model_digest: str = "",
        config: Dict[str, Any] = None,
        output: Any = None,
        output_hash: str = "",
    ) -> InferenceSeal:
        """
        Create a sealed, signed inference record.
        
        Args:
            image_path: Path to input image file
            image_data: Raw image bytes (alternative to path)
            model_digest: SHA-256 of model weights
            config: Inference configuration dict
            output: Raw output (tensor, dict, etc.)
            output_hash: Pre-computed output hash (alternative to output)
        
        Returns:
            InferenceSeal with cryptographic binding
        """
        config = config or {}

        # Compute component hashes
        img_hash = self._hash_image(image_path, image_data)
        cfg_hash = self._hash_config(config)
        out_hash = output_hash or self._hash_output(output)

        # Anti-replay controls
        nonce = secrets.token_hex(16)
        while nonce in self._used_nonces:
            nonce = secrets.token_hex(16)
        self._used_nonces.add(nonce)

        timestamp = datetime.now(timezone.utc).isoformat()
        prev_hash = self._seals[-1].seal_hash if self._seals else "GENESIS"

        # Build the payload to sign
        payload = {
            "sequence": self._sequence,
            "timestamp": timestamp,
            "nonce": nonce,
            "image_hash": img_hash,
            "model_digest": model_digest,
            "config_hash": cfg_hash,
            "output_hash": out_hash,
            "previous_seal_hash": prev_hash,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        # Sign
        signature = self.key_mgr.sign(payload_bytes)

        # Compute seal hash (covers everything including signature)
        seal_content = payload_hash + signature.hex()
        seal_hash = hashlib.sha256(seal_content.encode()).hexdigest()

        seal = InferenceSeal(
            seal_id=f"seal-{self._sequence:06d}",
            sequence_number=self._sequence,
            timestamp=timestamp,
            nonce=nonce,
            image_hash=img_hash,
            model_digest=model_digest,
            config_hash=cfg_hash,
            output_hash=out_hash,
            payload_hash=payload_hash,
            signature=signature.hex(),
            key_id=self.key_mgr.key_id,
            previous_seal_hash=prev_hash,
            seal_hash=seal_hash,
        )

        self._seals.append(seal)
        self._sequence += 1

        return seal

    def verify_seal(
        self,
        seal: InferenceSeal,
        image_path: str = "",
        image_data: bytes = None,
        model_digest: str = "",
        config: Dict[str, Any] = None,
        output: Any = None,
        output_hash: str = "",
    ) -> Dict[str, Any]:
        """
        Verify an inference seal against its claimed inputs/outputs.
        
        Returns a verification report with pass/fail for each check.
        """
        config = config or {}
        results = {
            "seal_id": seal.seal_id,
            "checks": {},
            "valid": True,
            "findings": [],
        }

        # 1. Verify image hash
        if image_path or image_data:
            computed_img_hash = self._hash_image(image_path, image_data)
            img_match = computed_img_hash == seal.image_hash
            results["checks"]["image_hash"] = {
                "match": img_match,
                "expected": seal.image_hash,
                "computed": computed_img_hash,
            }
            if not img_match:
                results["valid"] = False
                results["findings"].append(Finding(
                    module="provenance",
                    attack_class="inference_tampering",
                    severity=Severity.CRITICAL.value,
                    confidence=1.0,
                    title="Input image mismatch",
                    description="The provided image does not match the sealed image hash. The input may have been substituted.",
                    disposition=Disposition.QUARANTINE.value,
                ).to_dict())

        # 2. Verify model digest
        if model_digest:
            model_match = model_digest == seal.model_digest
            results["checks"]["model_digest"] = {
                "match": model_match,
                "expected": seal.model_digest,
                "provided": model_digest,
            }
            if not model_match:
                results["valid"] = False
                results["findings"].append(Finding(
                    module="provenance",
                    attack_class="model_substitution",
                    severity=Severity.CRITICAL.value,
                    confidence=1.0,
                    title="Model digest mismatch",
                    description="The model digest does not match the sealed record. The model may have been substituted.",
                    disposition=Disposition.QUARANTINE.value,
                ).to_dict())

        # 3. Verify config hash
        if config:
            computed_cfg_hash = self._hash_config(config)
            cfg_match = computed_cfg_hash == seal.config_hash
            results["checks"]["config_hash"] = {
                "match": cfg_match,
                "expected": seal.config_hash,
                "computed": computed_cfg_hash,
            }
            if not cfg_match:
                results["valid"] = False
                results["findings"].append(Finding(
                    module="provenance",
                    attack_class="inference_tampering",
                    severity=Severity.HIGH.value,
                    confidence=1.0,
                    title="Configuration mismatch",
                    description="Inference configuration differs from sealed record.",
                    disposition=Disposition.REVIEW.value,
                ).to_dict())

        # 4. Verify output hash
        if output is not None or output_hash:
            computed_out_hash = output_hash or self._hash_output(output)
            out_match = computed_out_hash == seal.output_hash
            results["checks"]["output_hash"] = {
                "match": out_match,
                "expected": seal.output_hash,
                "computed": computed_out_hash,
            }
            if not out_match:
                results["valid"] = False
                results["findings"].append(Finding(
                    module="provenance",
                    attack_class="output_substitution",
                    severity=Severity.CRITICAL.value,
                    confidence=1.0,
                    title="Output tampered",
                    description="The inference output does not match the sealed record. Results may have been altered post-generation.",
                    disposition=Disposition.QUARANTINE.value,
                ).to_dict())

        # 5. Verify cryptographic signature
        payload = {
            "sequence": seal.sequence_number,
            "timestamp": seal.timestamp,
            "nonce": seal.nonce,
            "image_hash": seal.image_hash,
            "model_digest": seal.model_digest,
            "config_hash": seal.config_hash,
            "output_hash": seal.output_hash,
            "previous_seal_hash": seal.previous_seal_hash,
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode()

        try:
            sig_bytes = bytes.fromhex(seal.signature)
            sig_valid = self.key_mgr.verify(payload_bytes, sig_bytes)
        except Exception:
            sig_valid = False

        results["checks"]["signature"] = {"valid": sig_valid}
        if not sig_valid:
            results["valid"] = False
            results["findings"].append(Finding(
                module="provenance",
                attack_class="inference_tampering",
                severity=Severity.CRITICAL.value,
                confidence=1.0,
                title="Invalid signature",
                description="Cryptographic signature verification failed. The seal may have been forged or tampered with.",
                disposition=Disposition.QUARANTINE.value,
            ).to_dict())

        return results

    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify the entire chain of seals for ordering integrity.
        Checks: sequence monotonicity, hash chain linkage, no gaps.
        """
        results = {
            "chain_length": len(self._seals),
            "valid": True,
            "breaks": [],
        }

        for i, seal in enumerate(self._seals):
            # Check sequence number
            if seal.sequence_number != i:
                results["valid"] = False
                results["breaks"].append({
                    "index": i,
                    "issue": f"Sequence gap: expected {i}, got {seal.sequence_number}",
                })

            # Check previous hash linkage
            expected_prev = self._seals[i - 1].seal_hash if i > 0 else "GENESIS"
            if seal.previous_seal_hash != expected_prev:
                results["valid"] = False
                results["breaks"].append({
                    "index": i,
                    "issue": "Previous seal hash mismatch - chain broken",
                })

        return results

    def detect_replay(self, seal: InferenceSeal) -> bool:
        """Check if a seal's nonce has already been used (replay attack)."""
        return seal.nonce in self._used_nonces

    def export_seals(self) -> List[Dict[str, Any]]:
        """Export all seals as serializable dicts."""
        return [s.to_dict() for s in self._seals]

    def import_seals(self, seal_dicts: List[Dict[str, Any]]) -> None:
        """Import seals from serialized dicts."""
        self._seals = [InferenceSeal.from_dict(d) for d in seal_dicts]
        self._sequence = len(self._seals)
        self._used_nonces = {s.nonce for s in self._seals}


# --- Merkle Tree for Batch Verification ---

class MerkleTree:
    """
    Merkle tree for efficient batch verification of inference records.
    Allows O(log n) proof that a specific record is part of a batch.
    """

    def __init__(self, leaves: List[str] = None):
        self.leaves: List[str] = leaves or []
        self.tree: List[List[str]] = []
        self.root: str = ""
        if self.leaves:
            self._build()

    def _hash_pair(self, left: str, right: str) -> str:
        combined = (left + right).encode()
        return hashlib.sha256(combined).hexdigest()

    def _build(self) -> None:
        """Build the Merkle tree from leaves."""
        if not self.leaves:
            self.root = hashlib.sha256(b"EMPTY").hexdigest()
            return

        # Hash the raw leaves
        current_level = [
            hashlib.sha256(leaf.encode()).hexdigest()
            for leaf in self.leaves
        ]
        self.tree = [current_level[:]]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash_pair(left, right))
            self.tree.append(next_level)
            current_level = next_level

        self.root = current_level[0] if current_level else ""

    def add_leaf(self, data: str) -> None:
        """Add a leaf and rebuild."""
        self.leaves.append(data)
        self._build()

    def get_proof(self, index: int) -> List[Dict[str, str]]:
        """
        Get the Merkle proof for a leaf at the given index.
        Returns a list of {hash, position} pairs needed to reconstruct the root.
        """
        if index >= len(self.leaves) or not self.tree:
            return []

        proof = []
        idx = index

        for level in self.tree[:-1]:  # Skip the root level
            if idx % 2 == 0:
                # Need the right sibling
                sibling_idx = idx + 1
                position = "right"
            else:
                sibling_idx = idx - 1
                position = "left"

            if sibling_idx < len(level):
                proof.append({"hash": level[sibling_idx], "position": position})
            else:
                proof.append({"hash": level[idx], "position": position})

            idx //= 2

        return proof

    def verify_proof(self, leaf_data: str, proof: List[Dict[str, str]],
                     root: str) -> bool:
        """Verify a Merkle proof for a leaf against a known root."""
        current = hashlib.sha256(leaf_data.encode()).hexdigest()

        for step in proof:
            if step["position"] == "right":
                current = self._hash_pair(current, step["hash"])
            else:
                current = self._hash_pair(step["hash"], current)

        return current == root

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "num_leaves": len(self.leaves),
            "tree_depth": len(self.tree),
        }
