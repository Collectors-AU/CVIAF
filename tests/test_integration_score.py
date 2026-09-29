"""Tests for the tolerant scorer (`scripts/integration.py score`).

The failure these exist for is not hypothetical: box 5's 5,000-model run died at row
1,467 because one clean model's CTC score is undefined on every held-out image, so the
producer emitted only `refdiv_mean_clean`, and the adapter's "all four signals" check
raised inside the worker pool. 3,533 finished rows never became a results file, and the
box was reported as "already scored".

So each test below is one way that a single model can either take a shard down or, just
as bad, be counted as a row:

  * an incomplete model raising instead of being classified;
  * an incomplete model being kept in the ledger because "refdiv is a signal";
  * a raised scorer being swallowed so quietly that nothing says which model failed;
  * a resumed run re-scoring what it already has, then double-counting it;
  * a re-issued shard still advertising the models that were dropped, so the merge
    validates membership against a shard nobody scored.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.integration import (  # noqa: E402
    SCORE_SCHEMA,
    classify_signals,
    score_records,
    scored_shard,
)

SIGNALS = ["ctc_mean_clean", "ctc_q95_clean", "ctc_peak_clean", "refdiv_mean_clean"]
FULL = {
    "ctc_mean_clean": 0.7,
    "ctc_q95_clean": 1.0,
    "ctc_peak_clean": 1.0,
    "refdiv_mean_clean": 0.5,
}


def item(model_id, seed, kind="clean"):
    return {"model_id": model_id, "seed": seed, "kind": kind, "path": f"/corpus/{model_id}",
            "manifest": {"model_id": model_id, "ground_truth": {"kind": kind}}}


def test_classify_signals_separates_undefined_from_error():
    assert classify_signals(FULL, SIGNALS) == "complete"
    assert classify_signals({"refdiv_mean_clean": 0.5}, SIGNALS) == "incomplete"
    assert classify_signals({"ctc_mean_clean": float("nan"), "refdiv_mean_clean": 1.0}, SIGNALS) == "incomplete"
    assert classify_signals({}, SIGNALS) == "error"
    assert classify_signals(None, SIGNALS) == "error"
    assert classify_signals(None, SIGNALS, error="ValueError: boom") == "error"


def test_score_records_classifies_an_incomplete_model_instead_of_raising():
    """One model with undefined CTC must not take the shard down with it."""
    models = [item(f"clean_none_fixed_s{81619 + i}", 81619 + i) for i in range(4)]

    def fake_score(model_dir, manifest):
        if model_dir.name.endswith("s81621"):
            return {"refdiv_mean_clean": 1.0}  # exactly what the producer emitted
        return FULL

    report = score_records(models, fake_score, SIGNALS, workers=1)

    assert report["schema"] == SCORE_SCHEMA
    assert report["n_models"] == 4
    assert report["n_scored"] == 3
    assert report["n_incomplete"] == 1
    assert report["n_error"] == 0
    assert report["skipped"][0]["model_id"] == "clean_none_fixed_s81621"
    assert "omitted" in report["skipped"][0]["reason"]


def test_score_records_names_the_model_when_the_scorer_raises():
    """A swallowed exception with no model id is unreproducible."""
    models = [item("clean_none_fixed_s81619", 81619)]

    def boom(model_dir, manifest):
        raise RuntimeError("reference model has no heads")

    report = score_records(models, boom, SIGNALS, workers=1)

    assert report["n_error"] == 1
    assert report["skipped"][0]["reason"] == "RuntimeError: reference model has no heads"
    assert report["rows"][0]["status"] == "error"


def test_score_records_keeps_incomplete_rows_out_of_the_scored_set():
    """`refdiv` alone is not a CTC signal; the row must not be counted as scored."""
    models = [item("clean_none_fixed_s81621", 81621)]

    report = score_records(models, lambda d, m: {"refdiv_mean_clean": 1.0}, SIGNALS, workers=1)

    assert report["n_scored"] == 0
    assert report["rows"][0]["status"] == "incomplete"
    assert report["rows"][0]["scores"] == {"refdiv_mean_clean": 1.0}


def test_score_records_retries_a_previously_errored_model_on_resume():
    """An error is usually systemic; treating it as done makes it permanent.

    Written from a real run: box 5's rescore first executed while the corpus was
    mid-move, so all 5,000 rows came out `error`. Resuming with those rows treated as
    done would have reported 0 scored on the second attempt and looked like a clean skip.
    """
    models = [item(f"clean_none_fixed_s{80152 + i}", 80152 + i) for i in range(3)]
    resume = {
        m["model_id"]: {
            "model_id": m["model_id"],
            "seed": m["seed"],
            "status": "error",
            "reason": "FileNotFoundError: corpus was mid-move",
            "scores": None,
        }
        for m in models
    }
    seen = []

    def counting_score(model_dir, manifest):
        seen.append(model_dir.name)
        return FULL

    report = score_records(models, counting_score, SIGNALS, workers=1, resume=resume)

    assert len(seen) == 3, "a previously errored model must be attempted again"
    assert report["n_retried"] == 3
    assert report["n_scored"] == 3


def test_score_records_does_not_retry_an_incomplete_model():
    """Undefined-on-every-image is a property of the model, not of the run."""
    models = [item("clean_none_fixed_s81621", 81621)]
    resume = {
        "clean_none_fixed_s81621": {
            "model_id": "clean_none_fixed_s81621",
            "seed": 81621,
            "status": "incomplete",
            "reason": "scorer omitted ctc_mean_clean, ctc_q95_clean, ctc_peak_clean",
            "scores": {"refdiv_mean_clean": 1.0},
        }
    }
    seen = []

    def counting_score(model_dir, manifest):
        seen.append(model_dir.name)
        return FULL

    report = score_records(models, counting_score, SIGNALS, workers=1, resume=resume)

    assert seen == [], "an incomplete model must not be scored again"
    assert report["n_retried"] == 0
    assert report["n_incomplete"] == 1


def test_score_records_resumes_without_rescoring_the_models_it_has():
    """The registry is the checkpoint; scoring 4,000 rows twice is 4,000 rows of waste."""
    models = [item(f"clean_none_fixed_s{81619 + i}", 81619 + i) for i in range(3)]
    seen = []

    def counting_score(model_dir, manifest):
        seen.append(model_dir.name)
        return FULL

    first = score_records(models, counting_score, SIGNALS, workers=1)
    resume = {r["model_id"]: r for r in first["rows"]}
    second = score_records(models, counting_score, SIGNALS, workers=1, resume=resume)

    assert seen == [m["model_id"] for m in models], "resume must not rescore"
    assert second["n_scored"] == 3
    assert second["n_models"] == 3


def test_score_records_keeps_the_rows_written_before_a_failing_model():
    """A model that fails mid-shard must not deny the 1,467 rows scored before it."""
    models = [item(f"clean_none_fixed_s{81619 + i}", 81619 + i) for i in range(5)]
    checkpoints = []

    def flaky(model_dir, manifest):
        if model_dir.name.endswith("s81622"):
            raise RuntimeError("boom")
        return FULL

    report = score_records(
        models, flaky, SIGNALS, workers=1, checkpoint=lambda done: checkpoints.append(len(done))
    )

    assert report["n_scored"] == 4, "every model but the failing one is scored"
    assert report["n_error"] == 1
    assert report["skipped"][0]["model_id"] == "clean_none_fixed_s81622"
    assert [r["model_id"] for r in report["rows"]] == [m["model_id"] for m in models]
    assert checkpoints == [1, 2, 3, 4, 5], "a checkpoint after every model"


def test_scored_shard_names_the_models_it_dropped():
    """A re-issued shard must not still advertise models nobody could score."""
    shard = {
        "schema": "cviaf.analysis-shard.v1",
        "shard_id": "seed_80152_85151",
        "seed_min": 80152,
        "seed_max": 80154,
        "count": 3,
        "models": [
            {"model_id": f"clean_none_fixed_s{80152 + i}", "seed": 80152 + i,
             "path": f"clean_null_lab_b5w1/clean_none_fixed_s{80152 + i}"}
            for i in range(3)
        ],
    }
    report = {
        "rows": [
            {"model_id": "clean_none_fixed_s80152", "seed": 80152, "status": "complete"},
            {"model_id": "clean_none_fixed_s80153", "seed": 80153, "status": "incomplete"},
            {"model_id": "clean_none_fixed_s80154", "seed": 80154, "status": "complete"},
        ]
    }

    out = scored_shard(shard, report)

    assert out["count"] == 2
    assert [m["model_id"] for m in out["models"]] == ["clean_none_fixed_s80152", "clean_none_fixed_s80154"]
    assert out["unscorable_model_ids"] == ["clean_none_fixed_s80153"]
    assert out["seed_min"] == 80152 and out["seed_max"] == 80154, "bounds stay the plan's"
    assert shard["count"] == 3, "the original shard is not mutated"


def test_scored_shard_with_every_model_scored_is_unchanged_in_membership():
    shard = {
        "schema": "cviaf.analysis-shard.v1",
        "shard_id": "seed_150_151",
        "seed_min": 150,
        "seed_max": 151,
        "count": 2,
        "models": [
            {"model_id": f"clean_none_fixed_s{150 + i}", "seed": 150 + i, "path": f"models/clean_none_fixed_s{150 + i}"}
            for i in range(2)
        ],
    }
    report = {"rows": [
        {"model_id": "clean_none_fixed_s150", "seed": 150, "status": "complete"},
        {"model_id": "clean_none_fixed_s151", "seed": 151, "status": "complete"},
    ]}

    out = scored_shard(shard, report)

    assert out["count"] == 2
    assert out["unscorable_model_ids"] == []
