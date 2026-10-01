"""Tests for the fleet population check (`scripts/fpr_population_check.py`).

Each test is a failure this lane actually hit or would have hit:

  * an averaged FPR over shards that are not one population (the whole point);
  * a split-stability criterion that refuses every honest measurement (range vs CI
    width, which my first draft did);
  * a chi-square tail that is silently wrong (it gates the verdict, and a wrong tail
    means a wrong `p`);
  * attributing a reference-relative signal's low FPR to the shard that holds the
    reference model without saying so.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fpr_population_check import (  # noqa: E402
    chi2_sf,
    homogeneity,
    judge,
    population_report,
    reference_corpus,
    render,
)


def _ledger(shards):
    """A minimal valid ledger: shards is {corpus: [score, ...]} with kind 'clean'."""
    records = []
    for corpus, scores in shards.items():
        for i, score in enumerate(scores):
            records.append({"model_id": f"{os.path.basename(corpus)}_m{i}", "corpus": corpus,
                            "kind": "clean", "is_positive": False, "split": "unassigned",
                            "scores": {"s": score}})
    # The producer declares the full attack vocabulary as positives even when the
    # ledger holds none: the polarity of a kind is a property of the schema, not of
    # which arms happen to be on disk today (validated by validate_ledger).
    return {"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": 0.05,
            "higher_is_more_anomalous": True, "positive_kinds": ["weight_tamper"],
            "negative_kinds": ["clean"], "provenance": {"producer": "test"},
            "records": records}


def _shard(n, high):
    """n negatives: `high` of them anomalous-looking, rest low."""
    return [1.0 - i * 1e-6 for i in range(high)] + [0.5 + i * 1e-6 for i in range(n - high)]


def _shard_saturated(n, stuck_fraction=0.2):
    """A shard whose scores pile up at 1.0 (the CTC-peak saturation, in miniature).

    A threshold taken as the alpha-quantile of a calibration half that is >=5% ties at
    1.0 *is* 1.0, and the rule fires on ``score > threshold``: nothing clears it, so the
    shard scores an FPR of exactly 0. A continuous shard scores ~alpha. Neither shard
    is wrong -- which is precisely why the two must not be averaged.
    """
    stuck = int(n * stuck_fraction)
    return [1.0] * stuck + [0.1 + (i + 1) * 1e-6 for i in range(n - stuck)]


# --------------------------------------------------------------------------- #
# the chi-square tail gates every verdict, so it gets pinned to published values
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("statistic,df,expected", [
    (3.841458820694124, 1, 0.05),
    (11.070497693516352, 5, 0.05),
    (15.086272469388985, 5, 0.01),
    (1.0, 1, 0.31731050786291404),
    (0.0, 3, 1.0),
])
def test_chi2_sf_matches_published_tables(statistic, df, expected):
    assert chi2_sf(statistic, df) == pytest.approx(expected, abs=1e-6)


def test_chi2_sf_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0)


# --------------------------------------------------------------------------- #
# homogeneity
# --------------------------------------------------------------------------- #

def test_homogeneity_accepts_equal_rates():
    result = homogeneity([(50, 1000)] * 5)
    assert result["verdict"] == "homogeneous"
    assert result["p_value"] > 0.05


def test_homogeneity_detects_one_odd_shard():
    # 5% everywhere, 0.5% in one shard: the pooled rate describes neither group.
    result = homogeneity([(50, 1000), (50, 1000), (5, 1000), (50, 1000)])
    assert result["verdict"] == "heterogeneous"
    assert result["p_value"] < 0.05


def test_homogeneity_with_all_zero_trials_says_not_measurable():
    result = homogeneity([(0, 0), (0, 0)])
    assert result["status"] == "not_measurable"
    assert result["verdict"] if "verdict" in result else True


def test_homogeneity_pooled_zero_cannot_scatter():
    """All groups at 0/500 is exact agreement, not a division-by-zero."""
    result = homogeneity([(0, 500)] * 4)
    assert result["status"] == "measured"
    assert result["pooled_rate"] == 0.0
    assert result["verdict"] == "homogeneous"
    assert result["p_value"] == 1.0


def test_homogeneity_needs_two_groups():
    assert homogeneity([(5, 100)])["status"] == "not_measurable"


# --------------------------------------------------------------------------- #
# shard heterogeneity, end to end
# --------------------------------------------------------------------------- #

def test_heterogeneous_shards_refuse_a_pooled_average():
    """One saturated shard and two continuous ones are not one population."""
    ledger = _ledger({"w1": _shard_saturated(600),
                      "w2": _shard(600, 30), "w3": _shard(600, 30)})
    report = population_report(ledger, seeds=[0, 1], min_negatives=20)
    block = report["shard_heterogeneity"]["signals"]["s"]
    rates = {r["corpus"]: r["rate"] for r in block["per_shard"]}
    assert rates["w1"] == 0.0 and rates["w2"] > 0.0
    assert block["verdict"] == "heterogeneous"
    assert block["homogeneity"]["p_value"] < 0.05
    verdicts = judge(report)
    assert verdicts["one_population"] is False
    assert any("scatter by more than binomial noise" in r for r in verdicts["refusals"])


def test_homogeneous_shards_allow_a_pooled_average():
    ledger = _ledger({"w1": _shard(600, 30), "w2": _shard(600, 30), "w3": _shard(600, 30)})
    report = population_report(ledger, seeds=[0, 1], min_negatives=20)
    verdicts = judge(report)
    assert verdicts["one_population"] is True, verdicts["refusals"]


def test_marginal_shard_is_a_warning_not_a_refusal():
    """A per-shard interval that misses the pooled rate must not veto the average.

    The fleet has six shards; at 95% per shard the chance that at least one interval
    misses the pooled rate is ~26% *under exact homogeneity*. Refusing on that would
    refuse every fleet big enough to be worth measuring, and it is what my second
    draft did to the real 7,503-model ledger (Q=8.0, p=0.155 -- homogeneous).
    """
    ledger = _ledger({"w1": _shard_saturated(600),
                      "w2": _shard(600, 30), "w3": _shard(600, 30)})
    report = population_report(ledger, seeds=[0], min_negatives=20)
    verdicts = judge(report)
    assert verdicts["one_population"] is False  # Q here IS significant
    ledger2 = _ledger({f"w{i}": _shard(600, 30) for i in range(6)})
    report2 = population_report(ledger2, seeds=[0], min_negatives=20)
    verdicts2 = judge(report2)
    assert verdicts2["one_population"] is True
    assert all("excludes the pooled rate" in w for w in verdicts2["warnings"]) or \
        verdicts2["warnings"] == []


def test_reference_shard_is_flagged_not_celebrated():
    """A reference-relative signal is lowest in the shard holding the reference."""
    ledger = _ledger({"w1": _shard_saturated(600),
                      "w2": _shard(600, 30), "w3": _shard(600, 30)})
    ledger["provenance"] = {"producer": "test",
                            "reference_dir": "/tmp/runs/w1/ref_model"}
    assert reference_corpus(ledger) == "w1"
    report = population_report(ledger, seeds=[0], min_negatives=20)
    block = report["shard_heterogeneity"]["signals"]["s"]
    assert block["reference_shard_is_lowest"] is True
    assert block["lowest_shard"]["corpus"] == "w1"
    assert "reference shard" in render(report)


def test_reference_corpus_is_none_when_the_producer_did_not_say():
    ledger = _ledger({"w1": _shard(100, 5), "w2": _shard(100, 5)})
    assert reference_corpus(ledger) is None


# --------------------------------------------------------------------------- #
# split stability: overdispersion, not range-vs-interval
# --------------------------------------------------------------------------- #

def test_split_stability_is_reported_per_seed_and_deterministic():
    ledger = _ledger({"w1": _shard(600, 30), "w2": _shard(600, 30)})
    first = population_report(ledger, seeds=[0, 1, 2])
    second = population_report(ledger, seeds=[0, 1, 2])
    assert first["split_stability"] == second["split_stability"]
    block = first["split_stability"]["signals"]["s"]
    assert [row["seed"] for row in block["per_seed"]] == [0, 1, 2]
    assert 0.0 <= block["min"] <= block["max"] <= 1.0


def test_split_stability_does_not_refuse_binomial_noise():
    """The first draft compared the range to the CI width and refused everything.

    A rate of 5% on 3000 negatives has SE ~0.004; five draws of a binomial span about
    2.3 SE by construction, which is the same order as the interval width. That is not
    instability, and the overdispersion test must not call it so.
    """
    ledger = _ledger({"w1": _shard(3000, 150)})
    report = population_report(ledger, seeds=[0, 1, 2, 3, 4], min_negatives=20)
    block = report["split_stability"]["signals"]["s"]
    assert block["homogeneity"]["verdict"] == "homogeneous"
    assert judge(report)["split_stable"] is True


def test_verdicts_are_written_for_a_reader():
    ledger = _ledger({"w1": _shard(500, 25), "w2": _shard(500, 25)})
    report = population_report(ledger, seeds=[0], min_negatives=20)
    verdicts = judge(report)
    assert verdicts["one_population"] is True
    assert verdicts["split_stable"] is True
    for key in ("one_population", "split_stable", "refusals", "warnings"):
        assert key in verdicts
