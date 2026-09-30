# Offline report attestation (experimental v-next ARM I)

The legacy audit hash chain catches edits, interior deletion and reordering but
cannot detect removal of a suffix: a shorter chain still verifies. This optional
export signs a checkpoint with a tree size, last-entry hash and domain-separated
SHA-256 Merkle root over the ordered entry hashes, and wraps the full assurance
report digest and checkpoint in a DSSE-signed in-toto Statement v1. The
checkpoint signature and DSSE signature use the operator-supplied Ed25519 key.
It does not silently fall back to HMAC. The standalone verification script has
**no CVIAF imports** and takes a separately pinned public key.

Install `cryptography` on signing and verification hosts (the verifier does not
need CVIAF or numpy). Export an Ed25519 private key in unencrypted PEM, stored
outside the repository. For instance, provision it with your offline key
management process; never publish it. Once a report and audit trail exist:

```sh
python -m cviaf.cli attest-report \
  --report out/assurance_report.json --audit out/audit_trail.json \
  --private-key /secure/assayer.pem --key-id assayer-2026 \
  --output out/attestation.json
python scripts/verify_attestation.py \
  --report out/assurance_report.json --audit out/audit_trail.json \
  --bundle out/attestation.json --public-key /trusted/assayer.pub.pem \
  --expected-count 42 --expected-root <trusted-latest-root>
```

The key ID is only an untrusted hint. Pin the public key out of band, keep a
trusted copy of the **latest** checkpoint count/root outside the mutable log,
and distribute the report, log and attestation as one versioned release. The
verifier rejects missing/reordered/interior-deleted/suffix-truncated entries
against that checkpoint, modified report content, and invalid signatures.
Without a pinned latest count/root, an adversary can replace all three files
with an *older, legitimately signed* bundle (rollback); no local signature can
prove that no newer checkpoint exists. Likewise, records written after a
checkpoint are not covered until the next checkpoint is issued and pinned.
Offline verification cannot establish key revocation or trust rotation; record
which trust configuration/date selected the pinned key. A stolen signing key
allows forging new checkpoints. This is an opt-in export and does not upgrade
old unauthenticated demo files or prove the model actually executed.

The DSSE payload type is `application/vnd.in-toto+json`; signatures cover the
DSSE PAE, not raw JSON or base64. The payload is an in-toto Statement v1 with
an assurance report SHA-256 subject and the CVIAF checkpoint predicate
`urn:cviaf:assurance-checkpoint:v1`. Standard DSSE tooling can check its
signature with the pinned public key; interpreting this custom predicate and
checking the legacy audit hashes requires a verifier such as this standalone
script. This is not an in-toto multi-functionary layout, CycloneDX ML-BOM or
C2PA implementation.

The Merkle algorithm is SHA-256(`0x00 || raw entry-hash bytes`) per leaf,
SHA-256(`0x01 || left || right`) per inner node; unpaired odd nodes are carried
unchanged. It binds the **whole ordered log** at checkpoint creation, but this
prototype does not produce compact inclusion/consistency proofs.

Regression: `python -m unittest discover -s tests -p test_attestation.py -v`.
