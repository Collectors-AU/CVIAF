"""Signing-mode disclosure and full-chain signature verification regressions."""

import hashlib
import json
from unittest.mock import patch

import pytest

from cviaf.provenance import InferenceProvenanceEngine, OfflineKeyManager, ALG_HMAC, InferenceSeal, _payload_bytes


def test_ed25519_seal_and_valid_chain(tmp_path):
    pytest.importorskip("cryptography")
    engine = InferenceProvenanceEngine(key_dir=str(tmp_path))
    a = engine.seal_inference(image_data=b"a", output=b"first")
    engine.seal_inference(image_data=b"b", output=b"second")
    assert a.alg == "ed25519" and a.schema == "cviaf-seal-2"
    assert engine.verify_seal(a)["valid"]
    assert engine.verify_chain()["valid"]
    reloaded = OfflineKeyManager(str(tmp_path))
    reloaded.load_keypair(engine.key_mgr.key_id)
    assert reloaded.alg == "ed25519"
    assert InferenceProvenanceEngine(key_manager=reloaded).verify_seal(a)["valid"]


def test_hmac_fallback_is_opt_in_and_public_verifier_abstains(tmp_path):
    import builtins
    original_import = builtins.__import__

    def without_crypto(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("simulated missing cryptography")
        return original_import(name, *args, **kwargs)

    key = OfflineKeyManager(str(tmp_path))
    with patch("builtins.__import__", side_effect=without_crypto):
        with pytest.raises(RuntimeError, match="Ed25519 signing is unavailable"):
            key.generate_keypair()
        key.generate_keypair(allow_symmetric=True)
    engine = InferenceProvenanceEngine(key_manager=key)
    seal = engine.seal_inference(image_data=b"a", output=b"out")
    assert seal.alg == ALG_HMAC
    assert engine.verify_seal(seal)["valid"]
    from cviaf.provenance import TrustStore, Verifier
    public_only = Verifier(TrustStore(str(tmp_path / "public")))
    assert public_only.verify_seal(seal)["abstain"]
    with pytest.raises(RuntimeError, match="symmetric"):
        OfflineKeyManager(str(tmp_path)).load_keypair(key.key_id)
    reloaded = OfflineKeyManager(str(tmp_path))
    reloaded.load_keypair(key.key_id, allow_symmetric=True)
    assert reloaded.alg == ALG_HMAC
    seal.alg = "ed25519"
    assert not engine.verify_seal(seal)["valid"]


def test_missing_crypto_does_not_reinterpret_ed25519_key_as_hmac(tmp_path):
    pytest.importorskip("cryptography")
    import builtins
    original_import = builtins.__import__
    key = OfflineKeyManager(str(tmp_path))
    key.generate_keypair()

    def without_crypto(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("simulated missing cryptography")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=without_crypto):
        with pytest.raises(ImportError):
            OfflineKeyManager(str(tmp_path)).load_keypair(key.key_id)


def test_forged_interior_seal_with_relinked_successor_fails(tmp_path):
    engine = InferenceProvenanceEngine(key_dir=str(tmp_path))
    for number in range(3):
        engine.seal_inference(image_data=b"image", output=str(number))
    assert engine.verify_chain()["valid"]
    seals = engine.export_seals()
    forged = seals[1]
    forged["output_hash"] = hashlib.sha256(b"attacker output").hexdigest()
    for i in (1, 2):
        row = seals[i]
        if i == 2:
            row["previous_seal_hash"] = seals[1]["seal_hash"]
        row["payload_hash"] = hashlib.sha256(_payload_bytes(InferenceSeal.from_dict(row))).hexdigest()
        row["signature"] = ("00" * 64 if i == 1 else
                            engine.key_mgr.sign(_payload_bytes(InferenceSeal.from_dict(row))).hex())
        row["seal_hash"] = hashlib.sha256((row["payload_hash"] + row["signature"]).encode()).hexdigest()
    verifier = InferenceProvenanceEngine(key_manager=engine.key_mgr)
    verifier.import_seals(seals)
    result = verifier.verify_chain()
    assert not result["valid"]
    assert any(b["index"] == 1 and "Seal verification" in b["issue"] for b in result["breaks"])
    assert not any(b["index"] == 2 for b in result["breaks"])
