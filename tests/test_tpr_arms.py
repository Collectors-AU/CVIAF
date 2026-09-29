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
    SCHEMA_LADDER,
    cmd_evaluate,
    cmd_evaluate_ladder,
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


def _ladder(tmp, cells, thresholds, min_positives=20, inert_cells=()):
    """cells: {(kind, dose): [(model_id, {signal: value}), ...]} -> the ladder receipt.

    The dose lives in the build registry, not in the scored npz, exactly as it does in a
    real run - so the fixture has to write both or it would not test the join that makes
    a per-dose table possible.
    """
    rows = []
    registry = []
    for (kind, dose), entries in cells.items():
        for mid, scores in entries:
            rows.append((mid, kind, scores))
            registry.append({
                "model_id": mid,
                "kind": kind,
                "magnitude": float(dose),
                "f1_relative_drop": 0.0 if (kind, dose) in inert_cells else 0.5,
                "effect_weak": (kind, dose) in inert_cells,
            })
    _write_results(tmp, rows)
    reg_path = os.path.join(tmp, "registry.jsonl")
    with open(reg_path, "w", encoding="utf-8") as fh:
        for r in registry:
            fh.write(json.dumps(r) + "\n")

    out = os.path.join(tmp, "ladder.json")
    rc = cmd_evaluate_ladder(_Args(
        results=tmp,
        registry=reg_path,
        frozen_report=_frozen_report(tmp, thresholds),
        out=out,
        min_positives=min_positives,
        skipped=None,
    ))
    assert rc == 0
    return json.load(open(out, encoding="utf-8"))


def _scores(**kw):
    base = {"ctc_mean_clean": 0.5, "ctc_peak_clean": 1.0, "ctc_q95_clean": 1.0,
            "refdiv_mean_clean": 0.1}
    base.update(kw)
    return base


def test_the_ladder_reports_each_class_and_dose_separately_and_never_pools():
    """A pooled rate survives exactly one follow-up question. This one refuses to be askable.

    Two cells: one class caught perfectly, one never caught. A pooled rate would be exactly
    0.5 and would hide both facts at once, so 0.5 must not appear as a rate anywhere - and
    the refusal has to be a named field, so a future reader sees it was a decision.
    """
    tmpdir = tempfile.mkdtemp()
    hit = _scores(refdiv_mean_clean=0.95)
    miss = _scores(refdiv_mean_clean=0.1)
    cells = {
        ("substitution", 0.1): [(f"substitution_r0.1_s{i}", hit) for i in range(25)],
        ("bias_lift", 0.1): [(f"bias_lift_r0.1_s{i}", miss) for i in range(25)],
    }
    out = _ladder(tmpdir, cells, {"refdiv_mean_clean": 0.8})

    assert out["schema"] == SCHEMA_LADDER
    assert out["pooled_estimate"] is None
    assert "across attack classes" in out["pooled_refusal"]
    assert "across doses" in out["pooled_refusal"]
    assert out["cells"]["substitution"]["0.1"]["rules"]["refdiv_mean_clean"]["tpr"] == 1.0
    assert out["cells"]["bias_lift"]["0.1"]["rules"]["refdiv_mean_clean"]["tpr"] == 0.0
    assert "tpr" not in out["rules"]["refdiv_mean_clean"], (
        "a rule entry carries the threshold, not a rate: every rate belongs to a cell"
    )
    rates = [cell["rules"]["refdiv_mean_clean"]["tpr"]
             for kind in out["cells"] for cell in out["cells"][kind].values()]
    assert 0.5 not in rates, "the pooled rate of these two cells must not exist here"


def test_a_degenerate_rule_is_null_in_every_ladder_cell():
    """Degeneracy is a property of the rule, so it cannot be true in one cell and not another."""
    tmpdir = tempfile.mkdtemp()
    cells = {
        ("weight_tamper", 0.25): [(f"weight_tamper_r0.25_s{i}", _scores()) for i in range(25)],
        ("weight_tamper", 1.0): [(f"weight_tamper_r1_s{i}", _scores()) for i in range(25)],
    }
    out = _ladder(tmpdir, cells, {"ctc_peak_clean": 1.0, "refdiv_mean_clean": 0.8})

    assert out["rules"]["ctc_peak_clean"]["status"] == "degenerate_threshold_at_ceiling"
    for dose in ("0.25", "1"):
        cell = out["cells"]["weight_tamper"][dose]["rules"]["ctc_peak_clean"]
        assert cell["tpr"] is None, "undefined is not zero, in any cell"
        assert cell["conclusion"] == "degenerate_rule"


def test_the_ladder_declares_what_one_unit_of_each_dose_axis_means():
    """0.25 is a noise scale, a prune fraction and a logit lift. The units must travel."""
    tmpdir = tempfile.mkdtemp()
    cells = {
        ("bias_lift", 0.25): [(f"bias_lift_r0.25_s{i}", _scores()) for i in range(25)],
        ("substitution", 0.25): [(f"substitution_r0.25_s{i}", _scores()) for i in range(25)],
    }
    out = _ladder(tmpdir, cells, {"refdiv_mean_clean": 0.8})

    assert set(out["units_per_kind"]) == {"bias_lift", "substitution"}
    for kind, unit in out["units_per_kind"].items():
        assert unit and isinstance(unit, str), f"{kind} must declare the unit of its dose axis"
    assert out["doses"] == [0.25]


def test_a_ladder_cell_below_the_floor_is_flagged_and_gives_a_bound():
    """Dose 1.0 of pruning leaves 7 scorable arms out of 99 - a ratio, not a rate."""
    tmpdir = tempfile.mkdtemp()
    cells = {
        ("substitution", 1.0): [(f"substitution_r1_s{i}", _scores(refdiv_mean_clean=0.95))
                                for i in range(7)],
    }
    out = _ladder(tmpdir, cells, {"refdiv_mean_clean": 0.8}, min_positives=20)

    # `n` belongs to the cell (it is shared by every rule), the conclusion to the rule.
    assert out["cells"]["substitution"]["1"]["n"] == 7
    cell = out["cells"]["substitution"]["1"]["rules"]["refdiv_mean_clean"]
    assert cell["caught"] == 7
    assert cell["n_below_floor"] is True
    assert cell["conclusion"] == "insufficient_denominator"
    assert cell["ci95_wilson"][0] < 1.0, "7 of 7 is not evidence that the rate is 1"


def test_the_ladder_counts_arms_whose_attack_did_not_move_behaviour():
    """A family that leaves f1 untouched is a weight change, not a damaged model.

    Every bias_lift arm measured f1_relative_drop 0.000 at every dose, so a cell of them
    tests whether a rule notices a weight perturbation - not whether it notices harm. The
    receipt has to say how much of a cell is inert or the detection rate there reads as a
    statement about behaviour.
    """
    tmpdir = tempfile.mkdtemp()
    cells = {
        ("bias_lift", 0.5): [(f"bias_lift_r0.5_s{i}", _scores()) for i in range(25)],
        ("weight_tamper", 0.5): [(f"weight_tamper_r0.5_s{i}", _scores()) for i in range(25)],
    }
    out = _ladder(tmpdir, cells, {"refdiv_mean_clean": 0.8}, inert_cells={("bias_lift", 0.5)})

    assert out["cells"]["bias_lift"]["0.5"]["n_behaviour_inert"] == 25
    assert out["cells"]["weight_tamper"]["0.5"]["n_behaviour_inert"] == 0
    assert "did not move it" in out["behaviour_metric"]


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
