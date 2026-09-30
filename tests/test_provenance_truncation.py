"""Future acceptance test for Arm I's externally anchored tail attestation.

A chain alone cannot detect a deleted suffix. Arm I signs an external checkpoint
for report and audit logs, but this seal-chain API does not consume that anchor.
This xfail remains an explicit test of the unsatisfied chain-only claim; it does
not count as attestation validation. See tests/test_attestation.py for the
checkpoint verifier and pinned-count/root tests.
"""

import pytest

from cviaf.provenance import InferenceProvenanceEngine


@pytest.mark.xfail(strict=True, reason="unanchored seal chain cannot detect suffix deletion; use pinned Arm I checkpoint")
def test_deleted_tail_must_fail_verification(tmp_path):
    sealer = InferenceProvenanceEngine(key_dir=str(tmp_path))
    for n in range(3):
        sealer.seal_inference(image_data=b"image", output=str(n))
    complete = sealer.export_seals()
    verifier = InferenceProvenanceEngine(key_manager=sealer.key_mgr)
    verifier.import_seals(complete[:-1])
    assert not verifier.verify_chain()["valid"], "deleted tail must be rejected against trusted attestation"
