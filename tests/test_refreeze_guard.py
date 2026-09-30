"""Tests for the guarded re-freeze of the pinned assure calibration protocol.

The guard exists because the two ways to handle a stale pin are both bad without it:
leaving it stale red-tests the tree, and hand-editing the JSON launders a real
calibration change in as maintenance. So the classifier has to distinguish:

  * the source-digest pin and the payload digest (the reason to re-freeze),
  * floating-point drift at the 1e-16 level (same calibration, different rounding),
  * anything else (the calibration moved -- refuse, and let a human decide).

The tuple-vs-list case is not hypothetical: `fit_protocol` holds its seed lists as
tuples while the fixture stores them as JSON lists, and the first version of the
comparator called that "not a numeric difference" and refused a safe re-freeze.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.refreeze_assure_protocol import (  # noqa: E402
    _max_abs_delta,
    classify_diff,
    render,
)

FROZEN = {
    "schema": "cviaf.assure-protocol.v2",
    "alpha": 0.05,
    "asset_size": 240,
    "fit_seeds": [30000, 30001, 30002, 30003],
    "effect_floor": 0.15068358000857712,
    "effect_null": [0.11727273752035723, 0.10612385419413017],
    "detector_versions": {"detectors": "aaaa", "drift": "bbbb"},
    "sha256": "0" * 64,
}


def _refit(**over):
    out = {k: (v[:] if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in FROZEN.items()}
    out.update(over)
    return out


def test_identical_payload_is_already_current():
    verdict = classify_diff(FROZEN, _refit())
    assert verdict["ok"] is True
    assert verdict["allowed"] == {} and verdict["substantive"] == {}


def test_a_stale_detector_pin_is_a_reason_to_refreeze():
    refit = _refit(detector_versions={"detectors": "cccc", "drift": "bbbb"},
                   sha256="f" * 64)
    verdict = classify_diff(FROZEN, refit)
    assert verdict["ok"] is True
    assert set(verdict["allowed"]) == {"detector_versions", "sha256"}
    assert verdict["substantive"] == {}


def test_float_noise_below_tolerance_is_allowed():
    refit = _refit(effect_null=[0.1172727375203572, 0.10612385419413016])
    verdict = classify_diff(FROZEN, refit)
    assert verdict["ok"] is True
    assert verdict["allowed"]["effect_null"]["max_abs_delta"] < 1e-9


def test_a_moved_calibration_number_refuses():
    """A different effect floor is a different checker, not a maintenance edit."""
    verdict = classify_diff(FROZEN, _refit(effect_floor=0.1400))
    assert verdict["ok"] is False
    assert "effect_floor" in verdict["substantive"]
    assert "REFUSED" in render(verdict)


def test_seed_tuples_are_not_a_difference():
    """fit_protocol yields tuples; the fixture holds JSON lists."""
    refit = _refit(fit_seeds=(30000, 30001, 30002, 30003))
    verdict = classify_diff(FROZEN, refit)
    assert verdict["ok"] is True
    assert verdict["substantive"] == {}


def test_a_changed_seed_or_count_refuses():
    assert classify_diff(FROZEN, _refit(fit_seeds=[30000, 30001, 30002, 30009]))["ok"] is False
    assert classify_diff(FROZEN, _refit(fit_seeds=[30000, 30001, 30002]))["ok"] is False


def test_a_key_on_one_side_only_refuses():
    hostile = _refit()
    hostile.pop("effect_floor")
    assert classify_diff(FROZEN, hostile)["ok"] is False
    assert classify_diff(hostile, FROZEN)["ok"] is False


def test_render_names_the_reason_for_each_difference():
    verdict = classify_diff(FROZEN, _refit(alpha=0.10, detector_versions={"detectors": "zz",
                                                                         "drift": "bbbb"}))
    text = render(verdict)
    assert "allowed" in text and "SUBSTANTIVE" in text
    assert "alpha" in text


@pytest.mark.parametrize("a,b,expected", [
    (1.0, 1.0, 0.0),
    ([1.0, 2.0], (1.0, 2.0), 0.0),
    ([1.0, 2.0], [1.0, 2.5], 0.5),
    ({"a": [1, 2]}, {"a": [1, 3]}, 1.0),
    ("x", "y", None),
    ([1, 2], [1, 2, 3], None),
    (1, True, None),
])
def test_max_abs_delta_contract(a, b, expected):
    assert _max_abs_delta(a, b) == expected
