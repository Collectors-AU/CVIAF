"""Tests for retiring rules that cannot fire (`fpr_tpr.signal_degeneracy`).

Each test is a failure this lane hit or would have hit:

  * a rule whose threshold no asset can reach being quoted beside a working rule's
    FPR, where its 0.000 reads as the best number in the table;
  * the same rule being *deleted* from the report instead of moved, which loses the
    ledger's continuity and makes the saturation invisible again;
  * a healthy rule being retired by a loose criterion -- a false retirement silently
    removes a working detector from every headline;
  * a warning that states the flag without the mechanism, so a reader cannot tell a
    saturated statistic from an unlucky calibration half.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.fpr_tpr import (  # noqa: E402
    assign_splits,
    evaluate_corpus,
    render_report,
    signal_degeneracy,
)

ATTACK_KINDS = ["oga", "oda", "weight_tamper"]


def _ledger(n: int = 400, seed: int = 3):
    """`live` is noise; `sat` is a saturated max-statistic pinned at 1.0 for most models."""
    import numpy as np
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        scores = {"live": float(rng.normal()),
                  "sat": 1.0 if rng.random() < 0.6 else float(rng.uniform(0, 0.9))}
        records.append({"model_id": f"clean_m{i}", "corpus": "shard0", "kind": "clean",
                        "is_positive": False, "split": "unassigned", "scores": scores})
    return {"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": 0.05,
            "higher_is_more_anomalous": True,
            "positive_kinds": list(ATTACK_KINDS), "negative_kinds": ["clean"],
            "provenance": {"producer": "test", "corpora": ["shard0"]},
            "records": records}


def test_a_rule_whose_threshold_is_the_corpus_maximum_is_not_quoted_as_specificity():
    led = assign_splits(_ledger(), seed=0)
    deg = signal_degeneracy(led["records"], "sat", 0.05, True)
    assert deg["at_corpus_extreme"] is True
    assert deg["retire"] is True
    assert deg["p_floor"] > 0.05
    report = evaluate_corpus(_ledger(), alpha=0.05, min_negatives=20, split_seed=0)
    assert "sat" in report["appendix_rules"]
    assert "sat" not in report["quotable_rules"]
    text = render_report(report)
    main_table = [ln for ln in text.split("APPENDIX")[0].splitlines()
                  if ln.startswith("sat ")]
    assert main_table == []             # no row in the headline table
    assert "APPENDIX" in text and any(ln.strip().startswith("sat ") for ln in
                                     text.splitlines())
    assert report["headline_note"] and "sat" in report["headline_note"]


def test_a_retired_rule_stays_in_the_report_and_the_ledger_for_continuity():
    report = evaluate_corpus(_ledger(), alpha=0.05, min_negatives=20, split_seed=0)
    assert "sat" in report["rules"]                       # measured, not deleted
    rule = report["rules"]["sat"]
    assert rule["fpr"]["denominator"] == report["denominators"]["evaluation_negatives"]
    assert rule["threshold"] is not None
    assert report["per_kind"]["sat"].get("retired") is True
    assert report["appendix_rules"]["sat"]["threshold"] == rule["threshold"]


def test_the_warning_states_the_mechanism_not_only_the_flag():
    """A reader must be able to tell 'saturated statistic' from 'short calibration'."""
    led = assign_splits(_ledger(), seed=0)
    deg = signal_degeneracy(led["records"], "sat", 0.05, True)
    assert "cannot" in deg["warning"]
    assert "range" in deg["warning"]
    assert deg["calibration_ties_at_extreme"] > 100
    assert deg["corpus_extreme"] == deg["threshold"]


def test_a_rule_that_can_fire_is_never_retired():
    """A false retirement removes a working detector from every headline silently."""
    led = assign_splits(_ledger(), seed=0)
    deg = signal_degeneracy(led["records"], "live", 0.05, True)
    assert deg["at_corpus_extreme"] is False
    assert deg["retire"] is False
    report = evaluate_corpus(_ledger(), alpha=0.05, min_negatives=20, split_seed=0)
    assert "live" in report["quotable_rules"]
    assert "live" in render_report(report).split("APPENDIX")[0]


def test_a_calibration_ceiling_threshold_is_warned_but_still_quotable():
    """The rule can fire on a held-out asset, so it is not dead -- but its FPR is a
    bound by construction and the report has to say so."""
    led = assign_splits(_ledger(), seed=0)
    # A held-out asset above the calibration ceiling: the rule CAN fire, so it stays
    # quotable even though its threshold is the largest calibration score.
    ev = next(r for r in led["records"] if r["split"] == "evaluation")
    ev["scores"]["sat"] = 5.0
    deg = signal_degeneracy(led["records"], "sat", 0.05, True)
    assert deg["at_calibration_ceiling"] is True
    assert deg["at_corpus_extreme"] is False
    assert deg["retire"] is False
    assert deg["warning"] and "cannot fire" in deg["warning"]


def test_degeneracy_needs_a_calibration_half_and_says_so():
    led = _ledger(n=20)
    for rec in led["records"]:
        rec["split"] = "evaluation"
    with pytest.raises(ValueError) as exc:
        signal_degeneracy(led["records"], "sat", 0.05, True)
    assert "calibration" in str(exc.value)


def test_a_lower_is_more_anomalous_signal_is_retired_from_the_other_end():
    """The direction contract is part of the criterion: a floor-pinned signal is as
    dead as a ceiling-pinned one, and a check that only looks upward misses it."""
    import numpy as np
    rng = np.random.default_rng(7)
    records = [{"model_id": f"m{i}", "corpus": "s", "kind": "clean",
                "is_positive": False, "split": "unassigned",
                "scores": {"low": 0.0 if rng.random() < 0.7 else float(rng.uniform(0.1, 1.0))}}
               for i in range(300)]
    led = assign_splits({"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": 0.05,
                         "higher_is_more_anomalous": False,
                         "positive_kinds": list(ATTACK_KINDS), "negative_kinds": ["clean"],
                         "provenance": {"producer": "test"},
                         "records": records}, seed=0)
    deg = signal_degeneracy(led["records"], "low", 0.05, False)
    assert deg["at_corpus_extreme"] is True
    assert deg["retire"] is True
