"""
Inference Provenance and Output Integrity Module.

Creates verifiable cryptographic bindings among:
  - Input image hash
  - Model identifier/weight digest
  - Preprocessing and inference configuration
  - Resulting output tensor/predictions

Uses Ed25519 digital signatures (via Python's cryptography library) for offline,
air-gapped operation. An HMAC-SHA256 symmetric mode exists ONLY behind an
explicit ``allow_symmetric=True`` opt-in (CLI: ``--allow-symmetric``); it gives
tamper-evidence but not non-repudiation, seals are tagged ``alg:
"hmac-sha256"``, and a public-key-only verifier ABSTAINS on them instead of
calling them clear.

Architecture (v3, per docs/CVIAF_V3_ARCHITECTURE.md §5.3 items 4-5 and §12 P0):
  - ``OfflineKeyManager`` - algorithm-tagged key envelopes (``cviaf-key-1``),
    load-or-generate semantics. The v2 defect of regenerating a fresh keypair on
    every engine construction (which made cross-engine verification impossible)
    is fixed: construction on a non-empty key directory loads the existing key.
  - ``Sealer`` - the only object that ever touches a private key.
  - ``TrustStore`` - a verifier-side directory of PUBLIC key envelopes plus an
    optional ``revoked.json``. Air-gap distributable by construction.
  - ``Verifier`` - public keys + the trusted history only. Resolves the
    verification key from ``seal.key_id``; unknown keys fail closed as
    ``untrusted_key`` (never silently "invalid signature"), revoked keys are a
    distinct CRITICAL verdict, and replay is judged against the trusted history
    alone (nonce membership / sequence ceiling), not against everything a
    process ever issued.
  - ``InferenceProvenanceEngine`` - back-compatible single-node composition of
    Sealer + a local Verifier (holds the secret; the demo and orchestrator keep
    working). New consumers should use Sealer/Verifier directly.

Seal schema ``cviaf-seal-2`` binds ``schema``, ``alg`` and ``key_id`` INSIDE the
signed payload, so stripping or flipping the algorithm tag (an HMAC seal
relabelled Ed25519 or vice versa) breaks the signature. v1 seals (no ``schema``
field) still parse and verify against the legacy payload layout.

Implements:
  - Hash-chain audit trail for sequential records
  - Merkle tree for batch verification
  - Nonce + sequence + (advisory) timestamp controls against replay
  - Tamper detection on any component
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from cviaf.core.types import Finding, Severity, Disposition, AuditTrail


# --- Schemas and algorithms ---

SEAL_SCHEMA = "cviaf-seal-2"
KEY_SCHEMA = "cviaf-key-1"
ALG_ED25519 = "ed25519"
ALG_HMAC = "hmac-sha256"
GENESIS = "GENESIS"

_SYMMETRIC_REFUSAL = (
    "Ed25519 signing is unavailable (the 'cryptography' package is not "
    "installed, or an HMAC key was loaded). Provenance refuses to sign or load "
    "symmetric keys silently: install cviaf[full], or pass "
    "allow_symmetric=True (CLI: --allow-symmetric). Symmetric mode gives "
    "tamper-evidence only - anyone holding the verification key can forge a "
    "seal - and every seal and report is tagged accordingly."
)


# --- Hash helpers (shared by Sealer and Verifier; module-level so the exact
# --- derivation is importable and testable) ---

def _hash_config(config: Dict[str, Any]) -> str:
    """Deterministic hash of configuration dict."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()


def _hash_output(output: Any) -> str:
    """Hash an output tensor or prediction."""
    if isinstance(output, np.ndarray):
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


def _hash_image(image_path: str = "", image_data: bytes = None) -> str:
    """Hash an input image."""
    if image_data:
        return hashlib.sha256(image_data).hexdigest()
    elif image_path and os.path.exists(image_path):
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(b"").hexdigest()


def _payload_bytes_from_fields(schema: str, alg: str, key_id: str,
                               sequence: int, timestamp: str, nonce: str,
                               image_hash: str, model_digest: str,
                               config_hash: str, output_hash: str,
                               previous_seal_hash: str) -> bytes:
    """The exact canonical bytes that are signed.

    v2 seals bind schema/alg/key_id into the payload; v1 seals (no schema)
    reproduce the legacy 8-field layout so historical records still verify.
    """
    payload: Dict[str, Any] = {
        "sequence": sequence,
        "timestamp": timestamp,
        "nonce": nonce,
        "image_hash": image_hash,
        "model_digest": model_digest,
        "config_hash": config_hash,
        "output_hash": output_hash,
        "previous_seal_hash": previous_seal_hash,
    }
    if schema:
        payload = {**payload, "schema": schema, "alg": alg, "key_id": key_id}
    return json.dumps(payload, sort_keys=True).encode()


def _payload_bytes(seal: "InferenceSeal") -> bytes:
    return _payload_bytes_from_fields(
        seal.schema, seal.alg, seal.key_id, seal.sequence_number,
        seal.timestamp, seal.nonce, seal.image_hash, seal.model_digest,
        seal.config_hash, seal.output_hash, seal.previous_seal_hash,
    )


# --- Key Management ---

class OfflineKeyManager:
    """
    Manages signing keys for air-gapped environments.

    Keys are stored as algorithm-tagged JSON envelopes (``cviaf-key-1``):
    ``key_<key_id>.priv.json`` (mode 0600) always, and
    ``key_<key_id>.pub.json`` for Ed25519 only - in HMAC mode there is no
    public key, and no file that publishes the shared secret (the v2 fallback
    wrote the secret to the .pub file; that defect is removed).

    The algorithm is recorded in the envelope and honoured on load. A 32-byte
    HMAC secret is a valid Ed25519 seed, so v2 could silently reinterpret an
    HMAC key as Ed25519 when 'cryptography' was present; that re-typing is now
    impossible. Legacy bare-bytes v2 keys (``key_<id>.priv``) were only ever
    produced by the HMAC fallback and load as HMAC, behind allow_symmetric.
    """

    def __init__(self, key_dir: str = ".cviaf_keys"):
        self.key_dir = key_dir
        self._signing_key = None
        self._verify_key = None
        self._key_id = ""
        self._alg = ""

    # -- introspection ------------------------------------------------------

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def alg(self) -> str:
        return self._alg

    @property
    def is_symmetric(self) -> bool:
        return self._alg == ALG_HMAC

    @property
    def _use_fallback(self) -> bool:  # back-compat introspection
        return self.is_symmetric

    # -- generation / loading ------------------------------------------------

    def generate_keypair(self, alg: str = ALG_ED25519,
                         allow_symmetric: bool = False) -> str:
        """Generate a new keypair. Returns key_id.

        Refuses to silently downgrade: Ed25519 without the 'cryptography'
        package raises unless allow_symmetric=True, in which case an HMAC
        key is generated and every seal it makes is tagged accordingly.
        """
        if alg not in (ALG_ED25519, ALG_HMAC):
            raise ValueError(f"unknown signing algorithm: {alg!r}")

        if alg == ALG_ED25519:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey)
                from cryptography.hazmat.primitives import serialization
            except ImportError as e:
                if not allow_symmetric:
                    raise RuntimeError(_SYMMETRIC_REFUSAL) from e
                alg = ALG_HMAC

        if alg == ALG_ED25519:
            private_key = Ed25519PrivateKey.generate()
            self._signing_key = private_key
            self._verify_key = private_key.public_key()
            priv_bytes = private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption())
            pub_bytes = self._verify_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw)
        else:
            if not allow_symmetric:
                raise RuntimeError(_SYMMETRIC_REFUSAL)
            priv_bytes = secrets.token_bytes(32)
            pub_bytes = b""  # no public key exists in symmetric mode
            self._signing_key = priv_bytes
            self._verify_key = priv_bytes

        self._alg = alg
        self._key_id = hashlib.sha256(pub_bytes or priv_bytes).hexdigest()[:16]

        os.makedirs(self.key_dir, exist_ok=True)
        base = os.path.join(self.key_dir, f"key_{self._key_id}")
        if os.path.exists(base + ".priv.json"):
            raise RuntimeError(f"key {self._key_id} already exists; refusing to overwrite")

        priv_env = {
            "schema": KEY_SCHEMA,
            "alg": alg,
            "key_id": self._key_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "private_key_hex": priv_bytes.hex(),
        }
        if alg == ALG_ED25519:
            priv_env["public_key_hex"] = pub_bytes.hex()
        priv_path = base + ".priv.json"
        with open(priv_path, "w") as f:
            json.dump(priv_env, f, indent=2)
        os.chmod(priv_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600

        if alg == ALG_ED25519:
            TrustStore.write_public_key(self.key_dir, self.public_key_envelope())

        return self._key_id

    def find_existing_key(self) -> Optional[str]:
        """Return the key_id of an existing key in key_dir, or None."""
        if not os.path.isdir(self.key_dir):
            return None
        names = sorted(os.listdir(self.key_dir))
        for n in names:
            if n.startswith("key_") and n.endswith(".priv.json"):
                return n[len("key_"):-len(".priv.json")]
        for n in names:
            if n.startswith("key_") and n.endswith(".priv"):
                return n[len("key_"):-len(".priv")]
        return None

    def load_keypair(self, key_id: str, allow_symmetric: bool = False) -> None:
        """Load an existing keypair from disk, honouring its recorded algorithm."""
        base = os.path.join(self.key_dir, f"key_{key_id}")

        if os.path.exists(base + ".priv.json"):
            with open(base + ".priv.json") as f:
                env = json.load(f)
            alg = env.get("alg", "")
            if alg == ALG_HMAC and not allow_symmetric:
                raise RuntimeError(_SYMMETRIC_REFUSAL)
            priv_bytes = bytes.fromhex(env["private_key_hex"])
            if alg == ALG_ED25519:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey)
                from cryptography.hazmat.primitives import serialization
                sk = Ed25519PrivateKey.from_private_bytes(priv_bytes)
                derived_pub = sk.public_key().public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw)
                if derived_pub.hex() != env.get("public_key_hex"):
                    raise RuntimeError(
                        f"key envelope key_{key_id}.priv.json is corrupt: "
                        "public key does not match private key")
                self._signing_key = sk
                self._verify_key = sk.public_key()
            elif alg == ALG_HMAC:
                self._signing_key = priv_bytes
                self._verify_key = priv_bytes
            else:
                raise RuntimeError(f"key envelope has unknown alg: {alg!r}")
            self._alg = alg
            self._key_id = key_id
            return

        if os.path.exists(base + ".priv"):
            # Legacy v2 bare-bytes key: no algorithm marker. These files were
            # only ever produced by the HMAC fallback (the .pub sibling is a
            # copy of the same secret), so they load as HMAC and are NEVER
            # reinterpreted as an Ed25519 seed.
            if not allow_symmetric:
                raise RuntimeError(_SYMMETRIC_REFUSAL)
            with open(base + ".priv", "rb") as f:
                priv_bytes = f.read()
            self._signing_key = priv_bytes
            self._verify_key = priv_bytes
            self._alg = ALG_HMAC
            self._key_id = key_id
            return

        raise FileNotFoundError(f"no key '{key_id}' in {self.key_dir}")

    def public_key_envelope(self) -> Dict[str, str]:
        """The distributable public-half envelope (Ed25519 only)."""
        if self._alg != ALG_ED25519:
            raise RuntimeError(
                "symmetric keys have no public half to distribute; verification "
                "requires the shared secret and gives no non-repudiation")
        from cryptography.hazmat.primitives import serialization
        return {
            "schema": KEY_SCHEMA,
            "alg": ALG_ED25519,
            "key_id": self._key_id,
            "public_key_hex": self._verify_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw).hex(),
        }

    # -- operations -----------------------------------------------------------

    def sign(self, data: bytes) -> bytes:
        if self._signing_key is None:
            raise RuntimeError("No keypair loaded. Call generate_keypair() or load_keypair() first.")
        if self._alg == ALG_HMAC:
            return hmac.new(self._signing_key, data, hashlib.sha256).digest()
        return self._signing_key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        if self._verify_key is None:
            raise RuntimeError("No keypair loaded.")
        try:
            if self._alg == ALG_HMAC:
                expected = hmac.new(self._verify_key, data, hashlib.sha256).digest()
                return hmac.compare_digest(expected, signature)
            self._verify_key.verify(signature, data)
            return True
        except Exception:
            return False


# --- Trust store (verifier side; public keys only) ---

class _Ed25519VerifyingKey:
    alg = ALG_ED25519
    holds_secret = False

    def __init__(self, pub_bytes: bytes):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        self._pk = Ed25519PublicKey.from_public_bytes(pub_bytes)

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            self._pk.verify(signature, data)
            return True
        except Exception:
            return False


class TrustStore:
    """
    A verifier's root of trust: a directory of ``key_*.pub.json`` envelopes
    (transportable on removable media - air-gap compatible) plus an optional
    ``revoked.json`` ({"revoked_key_ids": [...]}). Holds no private material;
    a verifier built on it cannot sign, which is the point.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self._keys: Dict[str, _Ed25519VerifyingKey] = {}
        self._revoked: set = set()
        if os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                if name.startswith("key_") and name.endswith(".pub.json"):
                    with open(os.path.join(directory, name)) as f:
                        env = json.load(f)
                    if env.get("schema") == KEY_SCHEMA and env.get("alg") == ALG_ED25519:
                        self._keys[env["key_id"]] = _Ed25519VerifyingKey(
                            bytes.fromhex(env["public_key_hex"]))
            rpath = os.path.join(directory, "revoked.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    data = json.load(f)
                ids = data.get("revoked_key_ids", []) if isinstance(data, dict) else data
                self._revoked = set(ids)

    def resolve(self, key_id: str) -> Optional[_Ed25519VerifyingKey]:
        return self._keys.get(key_id)

    def is_revoked(self, key_id: str) -> bool:
        return key_id in self._revoked

    def known_key_ids(self) -> List[str]:
        return sorted(self._keys)

    @staticmethod
    def write_public_key(directory: str, envelope: Dict[str, str]) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"key_{envelope['key_id']}.pub.json")
        with open(path, "w") as f:
            json.dump(envelope, f, indent=2)
        return path


class _ManagerVerifyingKey:
    """Back-compat adapter: single-node engine verifying with its own manager."""
    holds_secret = True

    def __init__(self, km: OfflineKeyManager):
        self._km = km
        self.alg = km.alg

    def verify(self, data: bytes, signature: bytes) -> bool:
        return self._km.verify(data, signature)


class _LocalKeyManagerTrust:
    """Trust adapter so the legacy single-node Engine can reuse Verifier logic."""

    def __init__(self, km: OfflineKeyManager):
        self._km = km

    def resolve(self, key_id: str):
        if key_id and key_id == self._km.key_id:
            return _ManagerVerifyingKey(self._km)
        return None

    def is_revoked(self, key_id: str) -> bool:
        return False


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
    schema: str = ""     # cviaf-seal-2 and later; empty means v1
    alg: str = ""        # ed25519 | hmac-sha256; signed, so it cannot be flipped

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InferenceSeal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --- Sealer (the only object that sees a private key) ---

class Sealer:
    """
    Produces sealed, signed inference records. Holds a signing-capable
    OfflineKeyManager; nothing else in the system needs one.
    """

    def __init__(self, key_manager: OfflineKeyManager):
        if key_manager._signing_key is None:
            raise RuntimeError("Sealer needs a key manager with a signing key loaded.")
        self.key_mgr = key_manager
        self._sequence = 0
        self._seals: List[InferenceSeal] = []
        self._used_nonces: set = set()

    def seal_inference(
        self,
        image_path: str = "",
        image_data: bytes = None,
        model_digest: str = "",
        config: Dict[str, Any] = None,
        output: Any = None,
        output_hash: str = "",
    ) -> InferenceSeal:
        """Create a sealed, signed inference record (schema cviaf-seal-2)."""
        config = config or {}

        img_hash = _hash_image(image_path, image_data)
        cfg_hash = _hash_config(config)
        out_hash = output_hash or _hash_output(output)

        nonce = secrets.token_hex(16)
        while nonce in self._used_nonces:
            nonce = secrets.token_hex(16)
        self._used_nonces.add(nonce)

        timestamp = datetime.now(timezone.utc).isoformat()  # advisory; see docs §5.3.2
        prev_hash = self._seals[-1].seal_hash if self._seals else GENESIS

        payload_bytes = _payload_bytes_from_fields(
            SEAL_SCHEMA, self.key_mgr.alg, self.key_mgr.key_id,
            self._sequence, timestamp, nonce, img_hash, model_digest,
            cfg_hash, out_hash, prev_hash)
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        signature = self.key_mgr.sign(payload_bytes)
        seal_hash = hashlib.sha256((payload_hash + signature.hex()).encode()).hexdigest()

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
            schema=SEAL_SCHEMA,
            alg=self.key_mgr.alg,
        )

        self._seals.append(seal)
        self._sequence += 1
        return seal

    def export_seals(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._seals]

    def import_seals(self, seal_dicts: List[Dict[str, Any]]) -> None:
        self._seals = [InferenceSeal.from_dict(d) for d in seal_dicts]
        self._sequence = len(self._seals)
        self._used_nonces = {s.nonce for s in self._seals}


# --- Verifier (public keys + trusted history only) ---

class Verifier:
    """
    Verifies seals against a TrustStore (public keys) and a trusted history.

    Contract:
      * The verification key is resolved from seal.key_id. An unknown key is a
        distinct ``untrusted_key`` verdict (fail closed), never a silent
        "invalid signature"; a revoked key is a distinct CRITICAL verdict.
      * seal.alg must match the resolved key's algorithm (downgrade attempt).
      * A symmetric (hmac-sha256) seal presented to a public-key-only verifier
        is an ABSTENTION - "cannot assess origin" - not a clearance and not a
        forgery finding.
      * Replay is judged against the trusted history alone: a record whose
        nonce is in the history, or whose sequence number is at/below the
        history ceiling FOR ITS SIGNING KEY, is a replay IF its signature is
        valid. Ceilings are per-key because a key rotation legitimately
        restarts the sequence. A record with a broken signature is a forgery
        and never borrows the replay control's credit.
    """

    def __init__(self, trust_store, history: Iterable[Union[InferenceSeal, Dict[str, Any]]] = ()):  # TrustStore or _LocalKeyManagerTrust
        self._trust = trust_store
        self._history: List[InferenceSeal] = []
        self._nonces: set = set()
        self._max_seq = -1
        self._max_seq_by_key: Dict[str, int] = {}
        self.add_history(history)

    def add_history(self, seals: Iterable[Union[InferenceSeal, Dict[str, Any]]]) -> None:
        for s in seals:
            if isinstance(s, dict):
                s = InferenceSeal.from_dict(s)
            self._history.append(s)
            self._nonces.add(s.nonce)
            self._max_seq = max(self._max_seq, s.sequence_number)
            if s.key_id:
                self._max_seq_by_key[s.key_id] = max(
                    s.sequence_number, self._max_seq_by_key.get(s.key_id, -1))

    @property
    def history(self) -> List[InferenceSeal]:
        return list(self._history)

    @property
    def max_history_sequence(self) -> int:
        return self._max_seq

    def is_stale(self, seal: InferenceSeal) -> bool:
        """Sequence number at/below the trusted-history ceiling for ITS key.
        Per-key, so a post-rotation seal (sequence restarts under the new
        key_id) is not a false replay."""
        # A key with no history has no ceiling (-1): staleness is only ever
        # consulted after the signature verifies, which requires a trusted key,
        # so an unseen-but-trusted key (post-rotation) must not read as stale.
        ceiling = self._max_seq_by_key.get(seal.key_id, -1)
        return seal.sequence_number <= ceiling

    def detect_replay(self, seal: InferenceSeal) -> bool:
        """Nonce already present in the TRUSTED HISTORY (not in everything this
        process ever issued - the v2 semantics, which called every genuine
        record a replay)."""
        return seal.nonce in self._nonces

    # -- field checks ---------------------------------------------------------

    @staticmethod
    def _finding(attack_class: str, severity: str, title: str,
                 description: str, disposition: str) -> Dict[str, Any]:
        return Finding(
            module="provenance",
            attack_class=attack_class,
            severity=severity,
            confidence=1.0,
            title=title,
            description=description,
            disposition=disposition,
        ).to_dict()

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
        """Verify a seal against its claimed inputs/outputs and the trust store."""
        config = config or {}
        results: Dict[str, Any] = {
            "seal_id": seal.seal_id,
            "checks": {},
            "valid": True,
            "abstain": False,
            "findings": [],
        }

        if image_path or image_data:
            computed = _hash_image(image_path, image_data)
            match = computed == seal.image_hash
            results["checks"]["image_hash"] = {
                "match": match, "expected": seal.image_hash, "computed": computed}
            if not match:
                results["valid"] = False
                results["findings"].append(self._finding(
                    "inference_tampering", Severity.CRITICAL.value,
                    "Input image mismatch",
                    "The provided image does not match the sealed image hash. The input may have been substituted.",
                    Disposition.QUARANTINE.value))

        if model_digest:
            match = model_digest == seal.model_digest
            results["checks"]["model_digest"] = {
                "match": match, "expected": seal.model_digest, "provided": model_digest}
            if not match:
                results["valid"] = False
                results["findings"].append(self._finding(
                    "model_substitution", Severity.CRITICAL.value,
                    "Model digest mismatch",
                    "The model digest does not match the sealed record. The model may have been substituted.",
                    Disposition.QUARANTINE.value))

        if config:
            computed = _hash_config(config)
            match = computed == seal.config_hash
            results["checks"]["config_hash"] = {
                "match": match, "expected": seal.config_hash, "computed": computed}
            if not match:
                results["valid"] = False
                results["findings"].append(self._finding(
                    "inference_tampering", Severity.HIGH.value,
                    "Configuration mismatch",
                    "Inference configuration differs from sealed record.",
                    Disposition.REVIEW.value))

        if output is not None or output_hash:
            computed = output_hash or _hash_output(output)
            match = computed == seal.output_hash
            results["checks"]["output_hash"] = {
                "match": match, "expected": seal.output_hash, "computed": computed}
            if not match:
                results["valid"] = False
                results["findings"].append(self._finding(
                    "output_substitution", Severity.CRITICAL.value,
                    "Output tampered",
                    "The inference output does not match the sealed record. Results may have been altered post-generation.",
                    Disposition.QUARANTINE.value))

        # -- signature / trust ---------------------------------------------
        try:
            sig_bytes = bytes.fromhex(seal.signature)
        except Exception:
            sig_bytes = b""

        key = self._trust.resolve(seal.key_id) if seal.key_id else None

        if key is None:
            if seal.alg == ALG_HMAC:
                # Symmetric seal, public-key-only verifier: origin is not
                # assessable. Third outcome - neither "clear" nor "forged".
                results["checks"]["signature"] = {
                    "valid": None,
                    "status": "abstain",
                    "reason": "symmetric seal (hmac-sha256): a verifier without "
                              "the shared secret cannot assess origin",
                }
                results["abstain"] = True
                results["findings"].append(self._finding(
                    "symmetric_signing_mode", Severity.HIGH.value,
                    "Origin not assessable (symmetric seal)",
                    "This seal was made with a shared-secret HMAC. Anyone holding "
                    "the verification key could have produced it, so this verifier "
                    "abstains on origin. Tamper fields were still checked above.",
                    Disposition.REVIEW.value))
            else:
                results["checks"]["signature"] = {
                    "valid": False, "reason": "untrusted_key",
                    "key_id": seal.key_id}
                results["valid"] = False
                results["findings"].append(self._finding(
                    "inference_tampering", Severity.CRITICAL.value,
                    "Untrusted signing key",
                    f"The seal names key '{seal.key_id}', which is not in the "
                    "verifier's trust store. Failing closed: an unverifiable "
                    "origin is not a clear origin.",
                    Disposition.QUARANTINE.value))
        elif self._trust.is_revoked(seal.key_id):
            results["checks"]["signature"] = {
                "valid": False, "reason": "revoked_key", "key_id": seal.key_id}
            results["valid"] = False
            results["findings"].append(self._finding(
                "inference_tampering", Severity.CRITICAL.value,
                "Signing key revoked",
                f"The seal was made under key '{seal.key_id}', which has been "
                "revoked. Treat every record under it as suspect.",
                Disposition.QUARANTINE.value))
        elif seal.alg and seal.alg != key.alg:
            results["checks"]["signature"] = {
                "valid": False, "reason": "alg_mismatch",
                "seal_alg": seal.alg, "key_alg": key.alg}
            results["valid"] = False
            results["findings"].append(self._finding(
                "inference_tampering", Severity.CRITICAL.value,
                "Algorithm mismatch (downgrade attempt)",
                f"The seal claims alg '{seal.alg}' but the trusted key is "
                f"'{key.alg}'. Relabelling a seal's algorithm is a downgrade attack.",
                Disposition.QUARANTINE.value))
        else:
            try:
                sig_valid = key.verify(_payload_bytes(seal), sig_bytes)
            except Exception:
                sig_valid = False
            results["checks"]["signature"] = {"valid": sig_valid}
            if not sig_valid:
                results["valid"] = False
                results["findings"].append(self._finding(
                    "inference_tampering", Severity.CRITICAL.value,
                    "Invalid signature",
                    "Cryptographic signature verification failed. The seal may have been forged or tampered with.",
                    Disposition.QUARANTINE.value))

        return results

    def verify_chain(self, seals: Optional[List[InferenceSeal]] = None) -> Dict[str, Any]:
        """Verify ordering integrity: sequence monotonicity, hash linkage, no gaps.

        Detects interior deletion and reordering. SUFFIX truncation of the log
        is only detectable against an external anchor (a signed Merkle
        checkpoint, architecture doc §5.3 item 3) - state that limitation.
        """
        chain = list(seals) if seals is not None else self._history
        results = {"chain_length": len(chain), "valid": True, "breaks": []}
        for i, seal in enumerate(chain):
            if seal.sequence_number != i:
                results["valid"] = False
                results["breaks"].append({
                    "index": i,
                    "issue": f"Sequence gap: expected {i}, got {seal.sequence_number}"})
            expected_prev = chain[i - 1].seal_hash if i > 0 else GENESIS
            if seal.previous_seal_hash != expected_prev:
                results["valid"] = False
                results["breaks"].append({
                    "index": i, "issue": "Previous seal hash mismatch - chain broken"})
            # Linkage alone is not authentication: a writer can recompute all
            # unkeyed hashes. Revalidate each member against trusted key material.
            check = self.verify_seal(seal)
            if not check["valid"] or check.get("abstain"):
                results["valid"] = False
                results["breaks"].append({
                    "index": i, "issue": "Seal verification failed: " +
                    str(check["checks"].get("signature", {}))})
        return results


# --- Back-compatible single-node engine (Sealer + local Verifier) ---

class InferenceProvenanceEngine:
    """
    Single-node composition kept for the demo, orchestrator and CLI.

    v3 fix (architecture §12 P0): construction is now load-or-generate. When no
    key_manager is passed, an EXISTING key in key_dir is loaded; a fresh
    keypair is generated only when the directory is empty. Two engines pointed
    at the same directory now hold the same key, so cross-engine verification
    works. v2 regenerated unconditionally, which is why the demo flagged all
    10 seals.

    For real verifier/sealer separation use Sealer + Verifier(TrustStore)
    directly; this class's local verifier necessarily holds the secret.
    """

    def __init__(self, key_manager: Optional[OfflineKeyManager] = None,
                 key_dir: str = ".cviaf_keys", allow_symmetric: bool = False):
        if key_manager is not None:
            self.key_mgr = key_manager
        else:
            km = OfflineKeyManager(key_dir)
            existing = km.find_existing_key()
            if existing:
                km.load_keypair(existing, allow_symmetric=allow_symmetric)
            else:
                km.generate_keypair(allow_symmetric=allow_symmetric)
            self.key_mgr = km

        self.sealer = Sealer(self.key_mgr)
        self._verifier = Verifier(_LocalKeyManagerTrust(self.key_mgr))

    # -- back-compat delegations ----------------------------------------------

    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        return _hash_config(config)

    @staticmethod
    def _hash_output(output: Any) -> str:
        return _hash_output(output)

    @staticmethod
    def _hash_image(image_path: str = "", image_data: bytes = None) -> str:
        return _hash_image(image_path, image_data)

    def seal_inference(self, *args, **kwargs) -> InferenceSeal:
        return self.sealer.seal_inference(*args, **kwargs)

    def verify_seal(self, seal: InferenceSeal, *args, **kwargs) -> Dict[str, Any]:
        report = self._verifier.verify_seal(seal, *args, **kwargs)
        report["verifier_key_material"] = "local-key-manager"
        return report

    def verify_chain(self) -> Dict[str, Any]:
        return self._verifier.verify_chain(self.sealer._seals)

    def is_stale(self, seal: InferenceSeal) -> bool:
        """Sequence number at/below the trusted-history ceiling for ITS key.
        Per-key, so a post-rotation seal (sequence restarts under the new
        key_id) is not a false replay."""
        # A key with no history has no ceiling (-1): staleness is only ever
        # consulted after the signature verifies, which requires a trusted key,
        # so an unseen-but-trusted key (post-rotation) must not read as stale.
        ceiling = self._max_seq_by_key.get(seal.key_id, -1)
        return seal.sequence_number <= ceiling

    def detect_replay(self, seal: InferenceSeal) -> bool:
        # Single-node semantics unchanged from v2: nonces this node issued or
        # imported. The history-only semantics live in Verifier; consumers who
        # verify records from a sealing service should use Verifier directly.
        return seal.nonce in self.sealer._used_nonces

    def export_seals(self) -> List[Dict[str, Any]]:
        return self.sealer.export_seals()

    def import_seals(self, seal_dicts: List[Dict[str, Any]]) -> None:
        self.sealer.import_seals(seal_dicts)

    # back-compat attribute access used by older callers
    @property
    def _seals(self) -> List[InferenceSeal]:
        return self.sealer._seals

    @property
    def _sequence(self) -> int:
        return self.sealer._sequence

    @property
    def _used_nonces(self) -> set:
        return self.sealer._used_nonces


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
        if not self.leaves:
            self.root = hashlib.sha256(b"EMPTY").hexdigest()
            return
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
        self.leaves.append(data)
        self._build()

    def get_proof(self, index: int) -> List[Dict[str, str]]:
        if index >= len(self.leaves) or not self.tree:
            return []
        proof = []
        idx = index
        for level in self.tree[:-1]:
            if idx % 2 == 0:
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
