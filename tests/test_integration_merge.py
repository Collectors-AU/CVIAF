"""Tests for the result merge preflight (`scripts/integration.py merge-results`).

The merge is the last step before a number becomes quotable, so the preflight is what
decides whether the population inside the ledger is the population on the label:

  * a shard with no results file being merged as if it contributed (box 5's run died at
    row 1,467 and left a registry but no npz);
  * an npz that no longer matches the digest its report published;
  * a shard whose bytes changed after it was scored, so the checkpoint identity that
    "proves" it was scored does not actually prove that;
  * shards that disagree about the reference model, which are not one population even if
    the counts add up;
  * shards that disagree about n_eval/backgrounds, the same failure one level down;
  * a final report that quotes a TPR from a corpus that contains no positives.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.integration import (  # noqa: E402
    MERGE_SCHEMA,
    merge_plans,
    preflight_results,
    summarise_fpr_report,
)


def write_shard(path, shard_id, seed_min, count, worker="b1w1"):
    seeds = list(range(seed_min, seed_min + count))
    shard = {
        "schema": "cviaf.analysis-shard.v1",
        "shard_id": shard_id,
        "seed_min": seeds[0],
        "seed_max": seeds[-1],
        "count": count,
        "preferred_box": None,
        "local_count": 0,
        "models": [
            {
                "model_id": f"clean_none_fixed_s{seed}",
                "seed": seed,
                "path": f"clean_null_lab_{worker}/clean_none_fixed_s{seed}",
                "manifest_sha256": "a" * 64,
                "weights_sha256": "b" * 64,
            }
            for seed in seeds
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(shard), encoding="utf-8")
    return Path(path)


def scored(directory, shard_id, plan_dir, count, reference="clean_none_fixed_s5800", meta_extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz = directory / f"results_{shard_id}.npz"
    npz.write_bytes(b"npz-bytes-" + shard_id.encode())
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    (directory / f"report_{shard_id}.json").write_text(
        json.dumps(
            {
                "schema": "cviaf.analysis-result.v1",
                "shard_id": shard_id,
                "count": count,
                "mode": "adapter",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    meta = {
        "shard_sha256": hashlib.sha256(
            (Path(plan_dir) / "shards" / f"{shard_id}.json").read_bytes()
        ).hexdigest(),
        "mode": "adapter",
        "adapter": "real_score_adapter:score",
        "reference": {"model_id": reference, "manifest_sha256": "f" * 64, "weights_sha256": "4" * 64},
        "n_eval": "40",
        "backgrounds": "4",
    }
    if meta_extra:
        meta.update(meta_extra)
    (directory / f"registry_{shard_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return digest


def build(tmp_path, sources):
    specs = []
    for index, (source_id, count) in enumerate(sources):
        seed_min = 60152 + index * 1000
        shard = write_shard(
            tmp_path / f"{source_id}.json", f"seed_{seed_min}_{seed_min + count - 1}", seed_min, count
        )
        specs.append((source_id, shard))
    plan_dir = tmp_path / "combined"
    plan = merge_plans(specs, plan_dir, excluded=["clean_none_fixed_s5800"])
    return plan, plan_dir


def test_preflight_refuses_a_shard_with_no_results_file(tmp_path):
    """A registry is not a result: box 5 had 1,467 rows and no npz."""
    plan, plan_dir = build(tmp_path, [("box1", 3), ("box5", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3)
    # box5's shard is deliberately left without a results file

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is False
    assert report["n_shards_with_results"] == 1
    assert report["n_shards_missing"] == 1
    assert any("missing" in p and "npz" in p for p in report["problems"])
    assert [s["ready"] for s in report["shards"]] == [True, False]


def test_preflight_refuses_an_npz_that_no_longer_matches_its_report(tmp_path):
    """The report's digest is the only link between a row set and its bytes."""
    plan, plan_dir = build(tmp_path, [("box1", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3)
    (results / "results_seed_60152_60154.npz").write_bytes(b"rewritten")

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is False
    assert any("npz digest does not match" in p for p in report["problems"])


def test_preflight_refuses_a_shard_that_changed_after_it_was_scored(tmp_path):
    """A checkpoint that pins a shard nobody scored proves nothing."""
    plan, plan_dir = build(tmp_path, [("box1", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3)
    shard_path = plan_dir / "shards" / "seed_60152_60154.json"
    data = json.loads(shard_path.read_text())
    data["models"] = data["models"][:2]
    data["count"] = 2
    shard_path.write_text(json.dumps(data))

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is False
    assert any("changed after it was scored" in p for p in report["problems"])


def test_preflight_refuses_shards_that_disagree_about_the_reference(tmp_path):
    """Two references are two populations; the counts must not be pooled."""
    plan, plan_dir = build(tmp_path, [("box1", 3), ("box3", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3, reference="clean_none_fixed_s5800")
    scored(results, "seed_61152_61154", plan_dir, 3, reference="clean_none_fixed_s10264")

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is False
    assert any("disagree about the reference" in p for p in report["problems"])


def test_preflight_refuses_shards_that_disagree_about_scorer_settings(tmp_path):
    """n_eval 40 and n_eval 4 are not the same measurement."""
    plan, plan_dir = build(tmp_path, [("box1", 3), ("box3", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3)
    scored(results, "seed_61152_61154", plan_dir, 3, meta_extra={"n_eval": "4"})

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is False
    assert any("disagree about scorer settings" in p for p in report["problems"])


def test_preflight_refuses_a_stub_mode_result(tmp_path):
    """A stub run's numbers are fabricated scaffolding, never a measurement."""
    plan, plan_dir = build(tmp_path, [("box1", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3, meta_extra={"mode": "stub"})

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is False
    assert any("is not a real score" in p for p in report["problems"])


def test_preflight_counts_a_complete_merge_and_its_denominators(tmp_path):
    """Every shard scored: the report has to say how many models that is."""
    plan, plan_dir = build(tmp_path, [("box1", 3), ("box3", 4)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3)
    scored(results, "seed_61152_61155", plan_dir, 4)

    report = preflight_results(plan, results, plan_dir)

    assert report["ready"] is True
    assert report["schema"] == MERGE_SCHEMA
    assert report["n_shards_with_results"] == 2
    assert report["n_models_scored"] == 7
    assert report["n_models_in_plan"] == 7
    assert report["problems"] == []


def test_preflight_reports_the_reference_it_validated(tmp_path):
    """The reference is pinned; a reader must be able to see which one was used."""
    plan, plan_dir = build(tmp_path, [("box1", 3)])
    results = tmp_path / "all-results"
    scored(results, "seed_60152_60154", plan_dir, 3)

    report = preflight_results(plan, results, plan_dir)

    assert list(report["references"]) == ["clean_none_fixed_s5800"]


def test_summarise_fpr_report_refuses_to_invent_a_tpr(tmp_path):
    """A clean-null corpus has no positives; a TPR column would be fabricated."""
    native = tmp_path / "fpr_tpr_report.json"
    native.write_text(
        json.dumps(
            {
                "ledger_schema": "cviaf.fpr-tpr-ledger.v1",
                "alpha": 0.05,
                "min_negatives": 30,
                "positive_kinds": [],
                "denominators": {"negatives": 7503, "positives": 0},
                "rules": {
                    "ctc_mean_clean": {"threshold": 0.9847, "fpr": 0.048, "tpr": None, "status": "ok"}
                },
                "fpr_measured": True,
                "positives_measured": False,
                "headline": "FPR measured on 7503 clean assets",
            }
        ),
        encoding="utf-8",
    )

    published = summarise_fpr_report(native, {"status": "EVALUATED"})

    assert published["fpr_measured"] is True
    assert published["positives_measured"] is False
    assert published["rules"]["ctc_mean_clean"]["tpr"] is None
    assert "fabricated" in published["tpr"]
    assert published["denominators"]["positives"] == 0
