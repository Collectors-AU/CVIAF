"""
Provenance attack matrix - 19 rows.

Adversary model (full strength): write access to the record store, recomputes
every unkeyed hash after editing, never holds the signing private key. The
verifier is public-key-only (TrustStore) with the trusted history as its only
replay reference.

Runs standalone (``python3 tests/test_provenance_matrix.py`` prints the matrix
and exits non-zero on any failed expectation) or under pytest.
"""

import builtins
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cviaf.provenance import (
    ALG_ED25519, ALG_HMAC, InferenceProvenanceEngine, InferenceSeal,
    OfflineKeyManager, Sealer, TrustStore, Verifier, _hash_output,
    _payload_bytes,
)

CONFIG = {"task": "detection", "input_size": 64, "score_thresh": 0.30,
          "nms_iou": 0.50, "preprocess": "unit_scale", "device": "cpu"}
DIGEST = hashlib.sha256(b"model-a").hexdigest()
DIGEST_OTHER = hashlib.sha256(b"model-b").hexdigest()


def _record(i):
    tag = str(i)
    n = abs(hash(tag)) % 1000
    raw = f"image-bytes-{tag}".encode()
    out = {"boxes": [[n, n + 1, n + 2, n + 3]], "scores": [0.9], "classes": [1]}
    return raw, out


def make_sealer(key_dir, allow_symmetric=False, alg=ALG_ED25519):
    km = OfflineKeyManager(key_dir)
    km.generate_keypair(alg=alg, allow_symmetric=allow_symmetric)
    return Sealer(km), km


def make_trust(km, directory, revoked=()):
    TrustStore.write_public_key(directory, km.public_key_envelope())
    if revoked:
        with open(os.path.join(directory, "revoked.json"), "w") as f:
            json.dump({"revoked_key_ids": list(revoked)}, f)
    return TrustStore(directory)


def adversary_recompute(d):
    """Full-strength forger: after editing bound fields, recompute every
    unkeyed hash so the record is internally self-consistent."""
    seal = InferenceSeal.from_dict(d)
    d["payload_hash"] = hashlib.sha256(_payload_bytes(seal)).hexdigest()
    d["seal_hash"] = hashlib.sha256(
        (d["payload_hash"] + d["signature"]).encode()).hexdigest()
    return d


class World:
    """Sealer A with a 6-record trusted history; public-key-only verifier."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="matrix-")
        self.sealer, self.km = make_sealer(os.path.join(self.dir, "keys-a"))
        self.history = []
        for i in range(6):
            raw, out = _record(i)
            seal = self.sealer.seal_inference(
                image_data=raw, model_digest=DIGEST, config=CONFIG, output=out)
            self.history.append({"seal": seal, "raw": raw, "output": out})
        trust_dir = os.path.join(self.dir, "trust")
        self.verifier = Verifier(
            make_trust(self.km, trust_dir),
            history=[h["seal"] for h in self.history])
        self.max_seq = max(h["seal"].sequence_number for h in self.history)

    def fresh_genuine(self, tag):
        raw, out = _record(f"new-{tag}")
        seal = self.sealer.seal_inference(
            image_data=raw, model_digest=DIGEST, config=CONFIG, output=out)
        return {"seal": seal, "raw": raw, "output": out}

    def present(self, verifier, seal_dict, raw, output, claimed_digest=DIGEST,
                config=CONFIG):
        seal = InferenceSeal.from_dict(seal_dict)
        res = verifier.verify_seal(seal, image_data=raw,
                                   model_digest=claimed_digest,
                                   config=config, output=output)
        sig = res["checks"].get("signature", {})
        sig_valid = bool(sig.get("valid"))
        nonce_seen = verifier.detect_replay(seal)
        stale = verifier.is_stale(seal)
        replay = bool(sig_valid and (nonce_seen or stale))
        flagged = bool((not res["valid"]) or replay)
        return {"res": res, "sig_valid": sig_valid, "replay": replay,
                "nonce_seen": nonce_seen, "stale": stale, "flagged": flagged,
                "abstain": bool(res.get("abstain")),
                "sig_reason": sig.get("reason", "")}


_WORLD = None
def world():
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD


# --- the 19 rows (each returns (ok, observed_summary)) ---

def row_01():
    w = world(); g = w.fresh_genuine("r1")
    o = w.present(w.verifier, g["seal"].to_dict(), g["raw"], g["output"])
    ok = o["res"]["valid"] and not o["replay"] and not o["abstain"]
    return ok, "accept" if ok else f"unexpected: {o}"

def row_02():
    d = tempfile.mkdtemp(prefix="matrix-r2-")
    sealer, km = make_sealer(os.path.join(d, "keys"))
    v = Verifier(make_trust(km, os.path.join(d, "trust")), history=[])
    raw, out = _record("first")
    seal = sealer.seal_inference(image_data=raw, model_digest=DIGEST,
                                 config=CONFIG, output=out)
    o = w_present = present_standalone(v, seal.to_dict(), raw, out)
    ok = o["res"]["valid"] and not o["replay"]
    return ok, "accept (empty history bootstraps)" if ok else f"unexpected: {o}"

def present_standalone(verifier, seal_dict, raw, output):
    seal = InferenceSeal.from_dict(seal_dict)
    res = verifier.verify_seal(seal, image_data=raw, model_digest=DIGEST,
                               config=CONFIG, output=output)
    sig_valid = bool(res["checks"].get("signature", {}).get("valid"))
    replay = bool(sig_valid and (
        verifier.detect_replay(seal) or verifier.is_stale(seal)))
    return {"res": res, "sig_valid": sig_valid, "replay": replay}

def row_03():
    w = world(); h = w.history[2]
    o = w.present(w.verifier, h["seal"].to_dict(), h["raw"], h["output"])
    ok = o["flagged"] and o["replay"] and o["sig_valid"] and o["nonce_seen"]
    return ok, "flagged: replay (nonce in history)" if ok else f"unexpected: {o}"

def row_04():
    w = world(); h = w.history[1]
    d = h["seal"].to_dict()
    d["sequence_number"] = w.max_seq + 1
    d["nonce"] = "f" * 32
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, h["raw"], h["output"])
    ok = o["flagged"] and not o["sig_valid"] and not o["replay"]
    return ok, "flagged: forgery (re-signed impossible); NOT credited to replay" if ok else f"unexpected: {o}"

def row_05():
    w = world(); h = w.history[0]
    d = h["seal"].to_dict()
    d["output_hash"] = _hash_output({"boxes": [[0, 0, 1, 1]], "scores": [0.1]})
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, h["raw"], h["output"])
    ok = o["flagged"] and not o["sig_valid"]
    return ok, "flagged: inference_tampering (self-consistent hashes, dead signature)" if ok else f"unexpected: {o}"

def row_06():
    w = world(); h = w.history[3]
    d = h["seal"].to_dict()
    d["model_digest"] = DIGEST_OTHER
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, h["raw"], h["output"], claimed_digest=DIGEST_OTHER)
    ok = o["flagged"] and not o["sig_valid"]
    return ok, "flagged: model re-attribution" if ok else f"unexpected: {o}"

def row_07():
    w = world(); h = w.history[4]
    d = h["seal"].to_dict()
    d["config_hash"] = "0" * 64
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, h["raw"], h["output"])
    ok = o["flagged"] and not o["sig_valid"]
    return ok, "flagged: config edit" if ok else f"unexpected: {o}"

def row_08():
    w = world(); g = w.fresh_genuine("r8")
    d = g["seal"].to_dict()
    d["output_hash"] = _hash_output({"boxes": [], "scores": [], "never": "happened"})
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, g["raw"], {"boxes": [], "scores": []})
    ok = o["flagged"] and not o["sig_valid"]
    return ok, "flagged: forged record for an inference that never happened" if ok else f"unexpected: {o}"

def row_09():
    w = world()
    # verifier history deliberately skips sequence 3 (withheld record)
    keep = [h["seal"] for i, h in enumerate(w.history) if i != 3]
    v = Verifier(make_trust(w.km, tempfile.mkdtemp(prefix="matrix-r9t-")),
                 history=keep)
    h3 = w.history[3]
    o = w.present(v, h3["seal"].to_dict(), h3["raw"], h3["output"])
    ok = (o["flagged"] and o["replay"] and o["sig_valid"] and o["stale"]
          and not o["nonce_seen"])
    return ok, "flagged: replay via stale sequence (withheld record presented late)" if ok else f"unexpected: {o}"

def row_10():
    w = world()
    chain = [h["seal"] for i, h in enumerate(w.history) if i != 2]  # interior deletion
    r = w.verifier.verify_chain(chain)
    ok = not r["valid"] and r["breaks"]
    return ok, f"flagged: chain break x{len(r['breaks'])} (interior deletion)" if ok else "MISSED interior deletion"

def row_11():
    w = world()
    chain = [h["seal"] for h in w.history]
    chain[2], chain[3] = chain[3], chain[2]
    r = w.verifier.verify_chain(chain)
    ok = not r["valid"] and r["breaks"]
    return ok, f"flagged: chain break x{len(r['breaks'])} (reorder)" if ok else "MISSED reorder"

def row_12():
    w = world()
    d = tempfile.mkdtemp(prefix="matrix-r12-")
    sealer_b, km_b = make_sealer(os.path.join(d, "keys-b"))
    raw, out = _record("b")
    seal_b = sealer_b.seal_inference(image_data=raw, model_digest=DIGEST,
                                     config=CONFIG, output=out)
    o = w.present(w.verifier, seal_b.to_dict(), raw, out)  # trust store has A only
    ok = o["flagged"] and o["sig_reason"] == "untrusted_key"
    return ok, "flagged: untrusted_key (real CVIAF key, not in this trust store)" if ok else f"unexpected: {o}"

def row_13():
    w = world(); h = w.history[0]
    d = h["seal"].to_dict()
    d["key_id"] = "deadbeefdeadbeef"
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, h["raw"], h["output"])
    ok = o["flagged"] and o["sig_reason"] == "untrusted_key"
    return ok, "flagged: untrusted_key (fabricated key id)" if ok else f"unexpected: {o}"

def row_14():
    w = world()
    v = Verifier(make_trust(w.km, tempfile.mkdtemp(prefix="matrix-r14t-"),
                            revoked=(w.km.key_id,)),
                 history=[h["seal"] for h in w.history])
    h = w.history[0]
    o = w.present(v, h["seal"].to_dict(), h["raw"], h["output"])
    ok = o["flagged"] and o["sig_reason"] == "revoked_key"
    return ok, "flagged: revoked_key (CRITICAL)" if ok else f"unexpected: {o}"

def row_15():
    w = world(); h = w.history[0]
    d = h["seal"].to_dict()
    d["alg"] = ALG_HMAC          # downgrade: relabel an Ed25519 seal as HMAC
    d = adversary_recompute(d)
    o = w.present(w.verifier, d, h["raw"], h["output"])
    ok = o["flagged"] and o["sig_reason"] == "alg_mismatch"
    return ok, "flagged: alg_mismatch (alg is inside the signed payload)" if ok else f"unexpected: {o}"

def row_16():
    w = world()
    d = tempfile.mkdtemp(prefix="matrix-r16-")
    sealer_h, km_h = make_sealer(os.path.join(d, "keys-h"), allow_symmetric=True, alg=ALG_HMAC)
    raw, out = _record("hmac")
    seal_h = sealer_h.seal_inference(image_data=raw, model_digest=DIGEST,
                                     config=CONFIG, output=out)
    # present to w's public-key-only verifier (trust store holds only A's pub)
    o = w.present(w.verifier, seal_h.to_dict(), raw, out)
    ok = o["abstain"] and not o["flagged"]
    return ok, "abstain: symmetric seal, origin not assessable (not clear, not forgery)" if ok else f"unexpected: {o}"

def row_17():
    w = world()
    d = tempfile.mkdtemp(prefix="matrix-r17-")
    sealer_b, km_b = make_sealer(os.path.join(d, "keys-b"))  # rotation
    trust_dir = os.path.join(d, "trust")
    TrustStore.write_public_key(trust_dir, w.km.public_key_envelope())
    TrustStore.write_public_key(trust_dir, km_b.public_key_envelope())
    v = Verifier(TrustStore(trust_dir), history=[h["seal"] for h in w.history])
    # a NEW pre-rotation seal under the old key (not in the verifier's history)
    raw1, out1 = _record("pre-rotation-new")
    seal_a2 = w.sealer.seal_inference(image_data=raw1, model_digest=DIGEST,
                                      config=CONFIG, output=out1)
    o1 = w.present(v, seal_a2.to_dict(), raw1, out1)
    # a NEW post-rotation seal under the rotated key: sequence restarts at 0,
    # which must NOT read as stale against the old key's ceiling
    raw2, out2 = _record("post-rotation")
    seal_b = sealer_b.seal_inference(image_data=raw2, model_digest=DIGEST,
                                     config=CONFIG, output=out2)
    o2 = w.present(v, seal_b.to_dict(), raw2, out2)
    ok = (o1["res"]["valid"] and not o1["replay"]
          and o2["res"]["valid"] and not o2["replay"])
    return ok, "accept both pre- and post-rotation seals (per-seal key_id resolution, per-key sequence ceiling)" if ok else f"unexpected: {o1} {o2}"

def row_18():
    d = tempfile.mkdtemp(prefix="matrix-r18-")
    key_dir = os.path.join(d, "keys")
    engine_a = InferenceProvenanceEngine(key_dir=key_dir)
    raw, out = _record("r18")
    seal = engine_a.seal_inference(image_data=raw, model_digest=DIGEST,
                                   config=CONFIG, output=out)
    engine_b = InferenceProvenanceEngine(key_dir=key_dir)  # verifier, same dir
    res = engine_b.verify_seal(seal, image_data=raw, model_digest=DIGEST,
                               config=CONFIG, output=out)
    same = engine_a.key_mgr.key_id == engine_b.key_mgr.key_id
    ok = res["valid"] and same
    return ok, f"cross-engine verification works (shared key_id {engine_b.key_mgr.key_id})" if ok else "REGEN BUG: engines disagree"

def row_19():
    d = tempfile.mkdtemp(prefix="matrix-r19-")
    real_import = builtins.__import__
    def blocked(name, *a, **k):
        if name.split(".")[0] == "cryptography":
            raise ImportError("blocked: simulating box without cryptography")
        return real_import(name, *a, **k)
    builtins.__import__ = blocked
    km_gen = OfflineKeyManager(d)
    kid = km_gen.generate_keypair(allow_symmetric=True)   # HMAC key
    sig0 = km_gen.sign(b"payload")
    builtins.__import__ = real_import
    # load on a box WITH cryptography, without the opt-in: must refuse
    km_strict = OfflineKeyManager(d)
    refused = False
    try:
        km_strict.load_keypair(kid)
    except RuntimeError:
        refused = True
    # with the explicit opt-in: must load as HMAC, never re-typed as Ed25519
    km_load = OfflineKeyManager(d)
    km_load.load_keypair(kid, allow_symmetric=True)
    preserved = km_load.alg == ALG_HMAC and km_load.sign(b"payload") == sig0
    no_pub_leak = not any(n.endswith(".pub") or n.endswith(".pub.json")
                          for n in os.listdir(d))
    ok = refused and preserved and no_pub_leak
    return ok, f"HMAC key stays HMAC under load (refusal w/o flag: {refused}, mode preserved: {preserved}, no secret .pub: {no_pub_leak})" if ok else f"MODE FLIP: refused={refused} preserved={preserved}"


ROWS = [
    (1,  "genuine, mid-run", row_01),
    (2,  "genuine, first-run (empty history)", row_02),
    (3,  "byte-identical replay", row_03),
    (4,  "replay with re-sequenced header", row_04),
    (5,  "in-place output edit", row_05),
    (6,  "model re-attribution", row_06),
    (7,  "config edit", row_07),
    (8,  "forged new record", row_08),
    (9,  "stale-sequence injection", row_09),
    (10, "history truncation (interior)", row_10),
    (11, "chain reorder", row_11),
    (12, "cross-key confusion", row_12),
    (13, "unknown key_id", row_13),
    (14, "revoked key", row_14),
    (15, "alg downgrade", row_15),
    (16, "HMAC seal to pubkey-only verifier", row_16),
    (17, "post-rotation genuine", row_17),
    (18, "key-reuse regression (v2 bug)", row_18),
    (19, "mode-flip regression (defect)", row_19),
]


def main():
    failures = 0
    print(f"{'row':>3}  {'case':<38} {'result':<6} observed")
    print("-" * 110)
    for n, name, fn in ROWS:
        try:
            ok, obs = fn()
        except Exception as e:
            ok, obs = False, f"EXCEPTION: {type(e).__name__}: {e}"
        failures += 0 if ok else 1
        print(f"{n:>3}  {name:<38} {'PASS' if ok else 'FAIL':<6} {obs}")
    print("-" * 110)
    print(f"{len(ROWS) - failures}/{len(ROWS)} rows as expected"
          + (f"  ({failures} FAILURES)" if failures else "  (all pass)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# --- pytest entry points ---
def _mk(n, fn):
    def _t():
        ok, obs = fn()
        assert ok, f"row {n}: {obs}"
    _t.__name__ = f"test_row_{n:02d}"
    return _t

for _n, _name, _fn in ROWS:
    globals()[f"test_row_{_n:02d}"] = _mk(_n, _fn)
