"""Regression tests: seeded drift detection and the drift_calibration policy.

The MMD permutation test in cviaf.drift previously drew from the global RNG,
so a clean control could flip-flop between no-finding and a borderline
finding across identical reruns. The permutation test now runs on an
explicitly seeded Generator.
"""

import json

import numpy as np
import pytest

from cviaf.drift import (
    DRIFT_CALIBRATION_SCHEMA,
    DistributionShiftAssessor,
    DriftCalibrationError,
    MMDCalculator,
    load_drift_calibration,
)


def _clean_control():
    """Reference and operational data from the SAME distribution.

    Data seeds are fixed, so any run-to-run difference in the verdict can
    only come from randomness inside the detector itself.
    """
    d = np.random.default_rng(7)
    ref = d.normal(0, 1, (200, 20)).astype(np.float32)
    op = d.normal(0, 1, (200, 20)).astype(np.float32)
    return ref, op


def test_mmd_permutation_test_is_deterministic():
    ref, op = _clean_control()
    calc = MMDCalculator(seed=0)
    r1 = calc.compute(ref, op)
    r2 = calc.compute(ref, op)
    assert r1 == r2


def test_clean_control_verdict_stable_across_20_reruns():
    ref, op = _clean_control()
    results = [
        DistributionShiftAssessor().assess(ref, op) for _ in range(20)
    ]
    verdicts = {(r["shift_detected"], r["overall_disposition"]) for r in results}
    p_values = {r["mmd"]["p_value"] for r in results}
    assert len(verdicts) == 1, f"verdict flip-flopped across reruns: {verdicts}"
    assert len(p_values) == 1, f"p-value not deterministic: {p_values}"
    # The clean control must not raise a finding.
    assert verdicts.pop()[0] is False


def test_seed_none_restores_legacy_global_rng_path():
    # seed=None keeps the old call signature working (nondeterministic).
    ref, op = _clean_control()
    out = MMDCalculator(seed=None).compute(ref, op)
    assert "p_value" in out


def test_load_drift_calibration_roundtrip(tmp_path):
    ref, op = _clean_control()
    assessment = DistributionShiftAssessor().assess(ref, op)
    path = tmp_path / "drift_calibration.json"
    payload = {
        "schema": DRIFT_CALIBRATION_SCHEMA,
        "seed": 0,
        "assessment": assessment,
    }
    path.write_text(json.dumps(payload, default=str))
    loaded = load_drift_calibration(str(path))
    assert loaded["assessment"]["mmd"]["p_value"] == assessment["mmd"]["p_value"]


def test_load_drift_calibration_fails_closed(tmp_path):
    # missing file
    with pytest.raises(DriftCalibrationError):
        load_drift_calibration(str(tmp_path / "nope.json"))
    # malformed JSON
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(DriftCalibrationError):
        load_drift_calibration(str(bad))
    # wrong schema
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": "something-else", "assessment": {}}))
    with pytest.raises(DriftCalibrationError):
        load_drift_calibration(str(wrong))
    # missing payload
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": DRIFT_CALIBRATION_SCHEMA}))
    with pytest.raises(DriftCalibrationError):
        load_drift_calibration(str(empty))
