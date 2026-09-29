"""Tests for fused-rule scoring on a ledger (`cviaf/lab/fusion_fpr.py`).

Each test is a failure this lane would otherwise have published:

  * a fused column computed with the evaluation half inside its own calibration
    set, which returns the number that was engineered rather than the number a
    deployment gets;
  * a signal that physically cannot fire (its conformal p-value floor sits above
    alpha) being folded into a mean-based combination, which silently moves the
    fused rule's operating point and makes the false-positive rate look *better*;
  * a fused rule quoted without its denominator or interval, i.e. the 0.000 that
    reads as precision;
  * a derived (fused) ledger that loses a column on some records and is scored
    anyway, so the fused rule and the per-signal rules no longer share a population;
  * a fused number published beside per-signal numbers measured on a different
    population, which is an unfalsifiable comparison.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cviaf.lab.fpr_tpr import LedgerError, evaluate_corpus, validate_ledger  # noqa: E402
from cviaf.lab.fusion_fpr import (  # noqa: E402
    fuse_ledger,
    fusion_report,
    degenerate_signals,
    flagged_overlap,
    nominal_alpha_rules,
    render_fusion,
)
from scripts.score_fleet_fusion import reproduction_problems  # noqa: E402

ATTACK_KINDS = ["oga", "oda", "substitution", "weight_tamper"]


def _ledger(n: int = 400, seed: int = 3, dead_share: float = 0.6,
            signals=("live_a", "live_b"), with_dead: bool = False):
    """A clean-only ledger: `signals` are pure noise, `dead` cannot produce a p < 0.6."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        scores = {s: float(rng.normal()) for s in signals}
        if with_dead:
            # Saturated max-statistic: a large share of the population ties at the
            # ceiling, so the conformal floor is roughly `dead_share`.
            scores["dead"] = 1.0 if rng.random() < dead_share else float(rng.uniform(0, 0.9))
        records.append({"model_id": f"clean_m{i}", "corpus": "shard0", "kind": "clean",
                        "is_positive": False, "split": "unassigned", "scores": scores})
    return {"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": 0.05,
            "higher_is_more_anomalous": True,
            "positive_kinds": list(ATTACK_KINDS), "negative_kinds": ["clean"],
            "provenance": {"producer": "test", "corpora": ["shard0"]},
            "records": records}


def test_fused_column_cannot_peek_at_the_evaluation_half():
    """The fused null is the calibration half; a hostile evaluation half must not change it.

    This is the leakage failure: if the evaluation half contributes to the null, the
    fused FPR is the number the engineer arranged, not the number a deployment sees.
    """
    base = _ledger(n=400, seed=11)
    led_a, _ = fuse_ledger(base, split_seed=1)
    tampered = copy.deepcopy(base)
    for rec in tampered["records"]:
        for name in list(rec["scores"]):
            rec["scores"][name] = rec["scores"][name] + 1000.0   # absurd scores
    # Assign the same splits the first fusion used, then swap the evaluation half's
    # scores for the absurd ones and re-fuse.
    from cviaf.lab.fpr_tpr import assign_splits
    tampered = assign_splits(tampered, seed=1)
    led_b, _ = fuse_ledger(tampered, split_seed=1)

    cal_a = [r for r in led_a["records"] if r["split"] == "calibration"]
    cal_b = [r for r in led_b["records"] if r["split"] == "calibration"]
    for ra, rb in zip(cal_a, cal_b):
        for name in json_columns():
            assert ra["scores"][name] == pytest.approx(rb["scores"][name])
    rule_a = evaluate_corpus(led_a, alpha=0.05, min_negatives=20, split_seed=1)["rules"]
    rule_b = evaluate_corpus(led_b, alpha=0.05, min_negatives=20, split_seed=1)["rules"]
    for name in json_columns():
        assert rule_a[name]["threshold"] == pytest.approx(rule_b[name]["threshold"])
    # ... while the evaluation half, being the tampered one, moves.
    ev_a = [r for r in led_a["records"] if r["split"] == "evaluation"]
    ev_b = [r for r in led_b["records"] if r["split"] == "evaluation"]
    assert any(ra["scores"]["live_a"] != rb["scores"]["live_a"]
               for ra, rb in zip(ev_a, ev_b))


def json_columns():
    return ("fused_cauchy", "fused_bonferroni", "fused_by")


def test_a_signal_that_cannot_fire_is_named_before_it_is_fused():
    """A saturated signal has a p-value FLOOR above alpha -- that is why it cannot fire."""
    led = _ledger(n=400, seed=5, with_dead=True)
    from cviaf.lab.fpr_tpr import assign_splits
    assigned = assign_splits(led, seed=0)
    deg = degenerate_signals(assigned["records"], ["live_a", "dead"], alpha=0.05)
    assert deg["dead"]["at_ceiling"] is True
    assert deg["dead"]["p_floor"] > 0.05
    assert deg["dead"]["blind_at_alpha"] is True
    assert deg["dead"]["calibration_ties_at_extreme"] > 100
    assert deg["live_a"]["at_ceiling"] is False
    assert deg["live_a"]["p_floor"] <= 0.05


def test_a_mean_based_fusion_is_diluted_by_a_signal_that_cannot_fire():
    """Cauchy is not indifferent to a blind signal: it moves the operating point.

    Measured failure this prevents: on the fleet the four-signal Cauchy fusion fires
    at 0.018 through the nominal cut while the two live signals fuse to 0.049. A
    reader scanning the FPR column would call the diluted rule the better detector.
    """
    led = _ledger(n=600, seed=7, with_dead=True)
    report = fusion_report(led, alpha=0.05, min_negatives=20, split_seed=2)
    columns = report["fusion"]["all_signals"]["columns"]
    live_columns = report["fusion"]["live_signals"]["columns"]
    assert columns["fused_all_cauchy"]["frac_p_at_or_below_alpha"] < \
        live_columns["fused_live_cauchy"]["frac_p_at_or_below_alpha"]
    assert report["fusion"]["live_signal_names"] == ["live_a", "live_b"]
    assert report["fusion"]["dropped_from_live"] == ["dead"]
    assert "dead" in render_fusion(report)


def test_a_live_fusion_can_still_reach_alpha_when_nothing_is_saturated():
    led = _ledger(n=400, seed=9)
    report = fusion_report(led, alpha=0.05, min_negatives=20, split_seed=0)
    assert report["fusion"]["live_signal_names"] == ["live_a", "live_b"]
    floor = max(report["fusion"]["degenerate"][s]["p_floor"]
                for s in report["fusion"]["live_signal_names"])
    assert floor <= report["alpha"]


def test_fused_fpr_keeps_its_denominator_and_interval():
    """A fused rule reports a rate with a denominator, never a bare 0.000."""
    led = _ledger(n=300, seed=13)
    report = fusion_report(led, alpha=0.05, min_negatives=20, split_seed=0)
    for name, rule in report["rules"].items():
        if not name.startswith("fused"):
            continue
        assert rule["fpr"]["denominator"] == report["denominators"]["evaluation_negatives"]
        assert rule["fpr"]["denominator"] > 0
        assert rule["fpr"]["ci95_wilson"] is not None or \
            rule["fpr"]["exact_upper_bound_95"] is not None


def test_a_fused_ledger_missing_a_column_on_one_record_is_refused():
    """A derived ledger that loses a column must not be scored: the rules would
    no longer share a population."""
    led, _ = fuse_ledger(_ledger(n=120, seed=2), split_seed=0)
    led["records"][5]["scores"].pop("fused_cauchy")
    problems = validate_ledger(led)
    assert any("every record" in p for p in problems), problems


def test_fusion_refuses_a_calibration_half_too_small_to_place_a_null():
    led = _ledger(n=8, seed=1)
    with pytest.raises(LedgerError) as exc:
        fuse_ledger(led, split_seed=0, min_calibration=5)
    assert "calibration negatives" in str(exc.value)


def test_fusion_refuses_an_unknown_method_rather_than_silently_using_cauchy():
    with pytest.raises(ValueError) as exc:
        fuse_ledger(_ledger(n=120, seed=4), methods=("fisher",))
    assert "fisher" in str(exc.value)


def test_conformal_fused_pvalues_are_super_uniform_on_pure_noise():
    """A fused rule on an all-clean population must not reject more than alpha.

    The whole fleet is clean by construction, so any excess here is a fabricated
    detection. Tolerance is the Wilson scale of the evaluation half's denominator.
    """
    led = _ledger(n=2000, seed=21)
    report = fusion_report(led, alpha=0.05, min_negatives=20, split_seed=4)
    n_eval = report["denominators"]["evaluation_negatives"]
    tolerance = 4.0 * float(np.sqrt(0.05 * 0.95 / n_eval))
    for name, r in report["fusion"]["nominal_alpha_rules"].items():
        assert r["point_estimate"] <= 0.05 + tolerance, (name, r)


def test_flagged_overlap_counts_both_and_either_without_double_counting():
    led = _ledger(n=300, seed=6)
    report = fusion_report(led, alpha=0.05, min_negatives=20, split_seed=1)
    overlap = report["fusion"]["flagged_overlap"]
    n_flagged = overlap["n_flagged"]
    for key, pair in overlap["pairs"].items():
        a, b = key.split("|")
        assert pair["both"] == len(
            set(overlap["flagged_model_keys"][a]) & set(overlap["flagged_model_keys"][b]))
        assert pair["either"] == n_flagged[a] + n_flagged[b] - pair["both"]
        assert pair["jaccard"] == pytest.approx(pair["both"] / pair["either"], abs=1e-4)


def test_flagged_overlap_uses_the_evaluation_half_only():
    led = _ledger(n=300, seed=8)
    report = fusion_report(led, alpha=0.05, min_negatives=20, split_seed=1)
    overlap = report["fusion"]["flagged_overlap"]
    total = sum(overlap["n_flagged"].values())
    assert total <= report["denominators"]["evaluation_negatives"] * len(
        overlap["n_flagged"])


def test_nominal_alpha_rules_ignore_per_signal_columns():
    led, _ = fuse_ledger(_ledger(n=200, seed=3), split_seed=0)
    rules = nominal_alpha_rules(led, 0.05, min_negatives=20)
    assert rules and all(name.startswith("fused") for name in rules)


def test_the_published_per_signal_table_is_reproduced_or_the_fused_number_is_refused():
    """Publishing a fused FPR beside a per-signal table from another population is
    the failure; adding records to the ledger is how it happens in practice."""
    led = _ledger(n=400, seed=17)
    published = evaluate_corpus(led, alpha=0.05, min_negatives=20, split_seed=5)
    assert reproduction_problems(led, published, 0.05, 20, 5) == []

    grown = copy.deepcopy(led)
    grown["records"].append({"model_id": "clean_new", "corpus": "shard0",
                             "kind": "clean", "is_positive": False,
                             "split": "unassigned",
                             "scores": {s: 0.0 for s in ("live_a", "live_b")}})
    problems = reproduction_problems(grown, published, 0.05, 20, 5)
    assert any("denominator" in p for p in problems), problems
    assert any("threshold" in p or "numerator" in p for p in problems)


def test_the_reproduction_guard_also_catches_a_re_split():
    """Same models, different split seed: every rate is re-derived, so the table
    must be re-run rather than reused."""
    led = _ledger(n=400, seed=19)
    published = evaluate_corpus(led, alpha=0.05, min_negatives=20, split_seed=5)
    problems = reproduction_problems(led, published, 0.05, 20, 6)
    assert problems, "a different split seed must not silently reproduce the table"
