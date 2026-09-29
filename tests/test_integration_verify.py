"""Tests for the integration verify step (`scripts/integration.py verify`).

Verification is the step that decides which models are allowed into the FPR ledger, so
the tests here are about the ways that decision can be wrong or destructive:

  * a model whose weights are not the bytes the box's pinned plan scored being averaged
    in anyway (the plan says 5,000 scored, the corpus says 4,999 of those plus a
    stranger);
  * a model that fails validation being *deleted* rather than set aside, which loses the
    evidence of what went wrong;
  * this pass moving files out of a directory it does not own (`runs/` is the user's
    training output, not scratch space);
  * a source with fewer models than its plan claims passing as complete;
  * plan entries with no model behind them passing as complete;
  * a native schema failure being reported as a plan-digest mismatch, which sends the
    reader looking in the wrong place.
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
    VERIFY_SCHEMA,
    plan_entries,
    quarantine_dir,
    render_verify,
    verify_judge,
    verify_model_dir,
    verify_source,
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_model(root, model_id, seed=60152, weights=b"weights", manifest_extra=None):
    directory = Path(root) / model_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "lab_version": "lab-1.0.0",
        "model_id": model_id,
        "created_utc": "2026-09-29T00:00:00Z",
        "spec": {"kind": "clean"},
        "spec_digest": hashlib.sha256(model_id.encode()).hexdigest()[:16],
        "dataset_digests": {"train_clean": "d" * 64, "eval_clean": "e" * 64},
        "attack_digest": "b" * 16,
        "seeds": {"scene": seed + 1, "attack": 11, "detector": seed},
        "artifact": {"weights_digest": "c" * 64, "head_digest": "f" * 64},
        "ground_truth": {"kind": "clean"},
        "metrics": {"clean_quality": {"f1": 0.5}, "attack_success_rate": {}},
        "quality_flags": {},
        "timing_seconds": 1.0,
        "environment": {},
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if weights is not None:
        (directory / "weights.npz").write_bytes(weights)
    return directory


def pinned_entry(directory: Path):
    return {
        "model_id": Path(directory).name,
        "seed": 60152,
        "path": f"clean_null_lab_b1w1/{Path(directory).name}",
        "manifest_sha256": sha(Path(directory) / "manifest.json"),
        "weights_sha256": sha(Path(directory) / "weights.npz"),
    }


def test_verify_quarantines_a_model_whose_weights_are_not_the_pinned_bytes(tmp_path):
    """The plan scored 5,000 specific files; a re-export is not one of them."""
    root = tmp_path / "corpus"
    model = write_model(root, "clean_none_fixed_s60152", weights=b"pinned")
    entry = pinned_entry(model)
    (model / "weights.npz").write_bytes(b"re-exported")

    problems = verify_model_dir(model, entry)

    assert any("weights_sha256" in p for p in problems)


def test_verify_quarantines_a_model_whose_manifest_is_not_the_pinned_manifest(tmp_path):
    """A re-serialised manifest is a different artefact, even with identical weights."""
    root = tmp_path / "corpus"
    model = write_model(root, "clean_none_fixed_s60152")
    entry = pinned_entry(model)
    manifest = json.loads((model / "manifest.json").read_text())
    manifest["timing_seconds"] = 26.99
    (model / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    problems = verify_model_dir(model, entry)

    assert any("manifest_sha256" in p for p in problems)


def test_verify_never_deletes_a_quarantined_model(tmp_path):
    """Quarantine moves the evidence aside; deleting it would hide the failure."""
    root = tmp_path / "corpus"
    quarantine = tmp_path / "quarantine"
    model = write_model(root, "clean_none_fixed_s60152", weights=b"wrong")

    summary = verify_source(
        "box1-corpus-v2",
        root,
        plan={"clean_none_fixed_s60152": {"weights_sha256": "0" * 64}},
        quarantine_root=quarantine,
        owned_root=root,
    )

    assert summary["n_failed"] == 1
    target = Path(summary["failures"][0]["quarantine"]["target"])
    assert target.is_dir()
    assert (target / "manifest.json").is_file(), "the manifest must survive the move"
    assert (target / "weights.npz").read_bytes() == b"wrong"
    assert not model.exists()


def test_verify_does_not_move_models_out_of_a_source_it_does_not_own(tmp_path):
    """`runs/` is the user's training output; a bad model is recorded, not relocated."""
    runs = tmp_path / "runs" / "clean_null_local_w1"
    model = write_model(runs, "clean_none_fixed_s10264", seed=10264, weights=b"bad")
    quarantine = tmp_path / "quarantine"

    summary = verify_source(
        "runs/clean_null_local_w1",
        runs,
        plan={"clean_none_fixed_s10264": {"weights_sha256": "0" * 64}},  # forced mismatch
        quarantine_root=quarantine,
        owned_root=tmp_path / "cviaf-analysis",
    )

    assert summary["n_failed"] == 1
    failure = summary["failures"][0]
    assert failure["quarantine"]["moved"] is False
    assert "not owned" in failure["quarantine"]["reason"]
    assert model.is_dir(), "an in-place source must never be emptied"
    assert not (quarantine / "runs/clean_null_local_w1").exists()


def test_quarantine_refuses_to_move_outside_the_owned_root(tmp_path):
    """A verification failure is not permission to relocate somebody else's files."""
    outside = tmp_path / "elsewhere" / "clean_none_fixed_s60152"
    write_model(tmp_path / "elsewhere", "clean_none_fixed_s60152")

    action = quarantine_dir(outside, tmp_path / "quarantine", "src", tmp_path / "owned")

    assert action["moved"] is False
    assert outside.is_dir()


def test_verify_reports_a_schema_failure_as_a_schema_failure(tmp_path):
    """A missing manifest field must not be reported as a plan-digest mismatch."""
    root = tmp_path / "corpus"
    model = write_model(root, "clean_none_fixed_s60152")
    manifest = json.loads((model / "manifest.json").read_text())
    del manifest["dataset_digests"]
    (model / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    problems = verify_model_dir(model)

    assert problems, "the native validator must reject a manifest missing a required field"
    assert not any("plan" in p for p in problems)


def test_verify_passes_a_source_that_matches_its_pinned_plan(tmp_path):
    root = tmp_path / "corpus"
    models = [write_model(root, f"clean_none_fixed_s{60152 + i}") for i in range(3)]
    plan = {m.name: pinned_entry(m) for m in models}

    summary = verify_source("box1-corpus-v2", root, plan=plan)
    report = verify_judge({"schema": VERIFY_SCHEMA, "sources": [summary]})

    assert summary["n_verified"] == 3
    assert summary["n_failed"] == 0
    assert report["ok"] is True
    assert report["exit_code"] == 0


def test_verify_fails_when_a_pinned_plan_entry_has_no_model_on_disk(tmp_path):
    """The plan says 5,000 models; four of them missing is not a rounding error."""
    root = tmp_path / "corpus"
    model = write_model(root, "clean_none_fixed_s60152")
    plan = {model.name: pinned_entry(model)}
    plan["clean_none_fixed_s60153"] = {"weights_sha256": "0" * 64, "manifest_sha256": "1" * 64}

    summary = verify_source("box1-corpus-v2", root, plan=plan)
    report = verify_judge({"schema": VERIFY_SCHEMA, "sources": [summary]})

    assert summary["n_plan_entries_not_seen"] == 1
    assert report["exit_code"] == 7
    assert any("no model on disk" in p for p in report["problems"])


def test_verify_fails_when_a_model_is_not_in_the_pinned_plan(tmp_path):
    """An extra model in a byte-pinned corpus has never been scored by anybody."""
    root = tmp_path / "corpus"
    model = write_model(root, "clean_none_fixed_s60152")
    write_model(root, "clean_none_fixed_s60153")
    plan = {model.name: pinned_entry(model)}

    summary = verify_source("box1-corpus-v2", root, plan=plan)
    report = verify_judge({"schema": VERIFY_SCHEMA, "sources": [summary]})

    assert summary["n_missing_from_plan"] == 1
    assert report["exit_code"] == 7


def test_verify_counts_a_failed_model_as_excluded_and_says_so(tmp_path):
    """One bad model must shrink the denominator visibly, not silently."""
    root = tmp_path / "corpus"
    models = [write_model(root, f"clean_none_fixed_s{60152 + i}") for i in range(5)]
    plan = {m.name: pinned_entry(m) for m in models}
    (models[2] / "weights.npz").write_bytes(b"tampered")

    summary = verify_source("box1-corpus-v2", root, plan=plan)
    report = verify_judge({"schema": VERIFY_SCHEMA, "sources": [summary]})

    assert summary["n_verified"] == 4
    assert summary["n_failed"] == 1
    assert report["exit_code"] == 7
    assert any("4 of" not in p and "1 of 5" in p for p in report["problems"])


def test_plan_entries_merge_several_shards(tmp_path):
    """box-plan7 holds two shards; both belong to the same plan."""
    shards = []
    for index, (low, high) in enumerate(((90152, 90153), (95152, 95153))):
        shard = {
            "schema": "cviaf.analysis-shard.v1",
            "shard_id": f"seed_{low}_{high}",
            "seed_min": low,
            "seed_max": high,
            "count": 2,
            "models": [
                {"model_id": f"clean_none_fixed_s{low + i}", "seed": low + i,
                 "path": f"clean_null_lab_b7w1/clean_none_fixed_s{low + i}",
                 "manifest_sha256": "a" * 64, "weights_sha256": "b" * 64}
                for i in range(2)
            ],
        }
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")
        shards.append(path)

    entries = plan_entries(shards)

    assert set(entries) == {
        "clean_none_fixed_s90152", "clean_none_fixed_s90153",
        "clean_none_fixed_s95152", "clean_none_fixed_s95153",
    }


def test_render_verify_shows_counts_and_the_verdict(tmp_path):
    root = tmp_path / "corpus"
    model = write_model(root, "clean_none_fixed_s60152")
    summary = verify_source("box1-corpus-v2", root, plan={model.name: pinned_entry(model)})
    text = render_verify(verify_judge({"schema": VERIFY_SCHEMA, "sources": [summary]}))

    assert VERIFY_SCHEMA in text
    assert "verified 1/1" in text
    assert "verdict: PASS" in text
