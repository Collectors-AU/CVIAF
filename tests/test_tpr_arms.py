"""Tests for the TPR-at-frozen-thresholds pass, named after what each one prevents.

The dangerous failure in this pass is not a crash, it is a *flattering* number: a TPR
that looks measured but was silently recalibrated, or a degenerate rule's inability to
fire reported as a 0% detection rate. Both would be believed.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tpr_arms import (  # noqa: E402
    SCHEMA,
    cmd_evaluate,
    exact_upper,
    wilson,
)

SIGNALS = ["ctc_mean_clean", "ctc_peak_clean", "ctc_q95_clean", "refdiv_mean_clean"]


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _write_results(tmp, rows):
    """rows: list of (model_id, kind, {signal: value})"""
    np = pytest.importorskip("numpy")
    path = os.path.join(tmp, "results_tpr_arms.npz")
    np.savez_compressed(
        path,
        model_id=np.array([r[0] for r in rows]),
        seed=np.array([1000 + i for i in range(len(rows))], dtype="int64"),
        kind=np.array([r[1] for r in rows]),
        **{s: np.array([r[2][s] for r in rows], dtype="float64") for s in SIGNALS},
    )
    return path


def _frozen_report(tmp, thresholds):
    path = os.path.join(tmp, "frozen.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema": "cviaf.fpr-tpr-report.v1",
                "alpha": 0.05,
                "rules": {
                    s: {"threshold": float(t), "status": "fpr_only"}
                    for s, t in thresholds.items()
                },
            },
            fh,
        )
    return path


def _run(tmp, rows, thresholds, min_positives=20):
    _write_results(tmp, rows)
    out = os.path.join(tmp, "tpr.json")
    rc = cmd_evaluate(_Args(
        results=tmp,
        frozen_report=_frozen_report(tmp, thresholds),
        out=out,
        min_positives=min_positives,
    ))
    assert rc == 0
    return json.load(open(out, encoding="utf-8"))


def test_tpr_uses_the_frozen_threshold_and_never_recalibrates():
    """The whole point of the pass: positives are judged at the FPR half's operating point.

    If the evaluator ever recomputed a threshold from the arms, the FPR would silently
    become a property of the attack set and the two numbers would stop being comparable.
    This asserts the threshold is taken verbatim by moving it and watching the count move.
    """
    tmpdir = tempfile.mkdtemp()
    scores = [{"ctc_mean_clean": 0.99, "ctc_peak_clean": 1.0, "ctc_q95_clean": 1.0,
               "refdiv_mean_clean": 0.9} for _ in range(30)]
    rows = [(f"weight_tamper_r0.25_s{i}", "weight_tamper", scores[i]) for i in range(30)]

    low = _run(tmpdir, rows, {"ctc_mean_clean": 0.98, "refdiv_mean_clean": 0.8})
    assert low["rules"]["ctc_mean_clean"]["pooled"]["caught"] == 30

    tmpdir2 = tempfile.mkdtemp()
    high = _run(tmpdir2, rows, {"ctc_mean_clean": 0.995, "refdiv_mean_clean": 0.8})
    assert high["rules"]["ctc_mean_clean"]["pooled"]["caught"] == 0, (
        "raising the frozen threshold must lower the count; anything else means the "
        "evaluator is choosing its own threshold"
    )


def test_a_degenerate_rule_is_reported_as_degenerate_not_as_zero():
    """`ctc_peak` measures FPR 0.0000 because its threshold is the corpus ceiling.

    Such a rule cannot fire on an attack either. Reporting `TPR 0.0` would credit it with
    a decision it never made, and would make the two dead rules look like the two live
    ones on a misses-vs-hits axis.
    """
    tmpdir = tempfile.mkdtemp()
    scores = [{"ctc_mean_clean": 0.5, "ctc_peak_clean": 1.0, "ctc_q95_clean": 1.0,
               "refdiv_mean_clean": 0.1} for _ in range(30)]
    rows = [(f"substitution_r0.25_s{i}", "substitution", scores[i]) for i in range(30)]
    out = _run(tmpdir, rows, {"ctc_peak_clean": 1.0, "ctc_mean_clean": 0.9})

    peak = out["rules"]["ctc_peak_clean"]
    assert peak["status"] == "degenerate_threshold_at_ceiling"
    assert peak["pooled"]["tpr"] is None, "degenerate means undefined, not 0.0"
    assert peak["pooled"]["caught"] == 0
    assert "cannot fire" in peak["note"]


def test_small_denominator_reports_a_bound_and_says_so():
    """A class with too few arms cannot support a rate - the FPR side's rule, applied here."""
    tmpdir = tempfile.mkdtemp()
    scores = [{"ctc_mean_clean": 0.99, "ctc_peak_clean": 1.0, "ctc_q95_clean": 1.0,
               "refdiv_mean_clean": 0.9} for _ in range(10)]
    rows = [(f"substitution_r0.25_s{i}", "substitution", scores[i]) for i in range(10)]
    out = _run(tmpdir, rows, {"refdiv_mean_clean": 0.8}, min_positives=20)

    cell = out["rules"]["refdiv_mean_clean"]["per_kind"]["substitution"]
    assert cell["conclusion"] == "insufficient_denominator"
    assert cell["n_below_floor"] is True


def test_evaluate_flags_a_clean_model_inside_the_arm_set():
    """An arm set containing a clean model inflates the denominator with a non-attack."""
    tmpdir = tempfile.mkdtemp()
    scores = [{"ctc_mean_clean": 0.5, "ctc_peak_clean": 1.0, "ctc_q95_clean": 1.0,
               "refdiv_mean_clean": 0.1} for _ in range(21)]
    rows = [(f"substitution_r0.25_s{i}", "substitution", scores[i]) for i in range(20)]
    rows.append(("clean_none_fixed_s99999", "clean", scores[20]))
    out = _run(tmpdir, rows, {"ctc_mean_clean": 0.9})

    assert out["schema"] == SCHEMA
    assert any("clean" in p for p in out["problems"])


def test_a_no_op_tamper_is_refused_rather_than_recorded():
    """A tamper that leaves the weights byte-identical is not an attack.

    Recorded, it would put a clean model into the TPR denominator while the report claims
    it is an attack - the exact reverse of the honesty rule this pass runs on.
    """
    from scripts.tpr_arms import build_arm  # noqa: F401  (import proves the guard's home)

    tmpdir = tempfile.mkdtemp()
    from cviaf.lab.detector import DetectorConfig, TinyDetector

    clean = TinyDetector(DetectorConfig())
    clean.trained = True
    clean_dir = os.path.join(tmpdir, "clean_none_fixed_s1")
    os.makedirs(clean_dir)
    clean.save(os.path.join(clean_dir, "weights.npz"))
    base = {
        "seeds": {"detector": 1},
        "spec": {"model_id": "x"},
        "artifact": {"weights_digest": clean.digest(), "backbone_digest": "b" * 64},
        "metrics": {"clean_quality": {"f1": 0.5}},
        "quality_flags": {},
        "ground_truth": {"kind": "clean"},
    }
    # magnitude 0 leaves the head untouched, so digest() must equal the clean digest.
    with pytest.raises(SystemExit) as excinfo:
        build_arm(
            __import__("pathlib").Path(clean_dir), base, "weight_tamper", 0.0, "head",
            __import__("pathlib").Path(tmpdir) / "out", "tpr_arms",
        )
    assert "identical weights" in str(excinfo.value)


def test_a_zero_count_gets_a_positive_upper_bound_from_both_conventions():
    """Two intervals, both of which must refuse to say "zero".

    The first version of this test asserted Wilson and Clopper-Pearson agree to 2%. They
    do not, and should not - Wilson is a score interval, the exact bound is conservative.
    What actually matters is the property the lane depends on: neither reports a rate of
    zero, and the exact bound is the tighter of the two at zero hits.
    """
    lo, hi = wilson(0, 20)
    exact = exact_upper(0, 20)
    assert lo == 0.0
    assert 0.0 < exact < hi, "the exact bound must be positive and no wider than Wilson"
    assert hi == pytest.approx(0.161, abs=0.002)
