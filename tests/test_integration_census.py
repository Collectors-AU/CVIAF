"""Tests for the integration census (`scripts/integration.py`).

Each test is a failure this lane would actually hit while assembling the multi-source
clean-null corpus, and each one is a failure that stays invisible unless the census
refuses to smooth it over:

  * the same detector seed present in two sources, silently deduped by a merge that
    just wanted the counts to add up;
  * a non-lab source drifting into the lab boxes' seed range, so the "8 disjoint lab
    boxes" claim quietly stops being true;
  * a re-manifest of one model looking like two models (and the reverse: two different
    models wearing one seed);
  * an archive that is not the bytes that were declared for it;
  * a model directory with no weights passing the census and failing much later;
  * a seed range with holes reported as one tidy min..max;
  * lab boxes that are simply absent being reported as nothing at all.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.integration import (  # noqa: E402
    CENSUS_SCHEMA,
    LAB_SEED_MAX,
    LAB_SEED_MIN,
    build_report,
    discover_sources,
    duplicate_values,
    find_collisions,
    render,
    scan_archive,
    scan_dir,
    seed_of,
    seed_ranges,
    summarize,
)


def write_model(root, model_id, seed=None, kind="clean", weights=b"weights"):
    directory = Path(root) / model_id
    directory.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = seed_of(model_id) or 0
    manifest = {
        "model_id": model_id,
        "seeds": {"detector": seed, "scene": seed + 1, "attack": 0},
        "ground_truth": {"kind": kind},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if weights is not None:
        (directory / "weights.npz").write_bytes(weights)
    return directory


def source(root, source_id, kind="mac_local"):
    return {"id": source_id, "kind": kind, "path": str(root), "entries": scan_dir(root)}


def judge(sources, selected=None):
    selected = selected if selected is not None else [s["id"] for s in sources]
    return build_report(sources, archives=[], gaps=[], selected=selected)


def test_census_does_not_silently_dedupe_a_seed_claimed_by_two_sources(tmp_path):
    """Two sources, one seed, different weights: that is two models, not one."""
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_model(left, "clean_none_fixed_s10264", seed=10264, weights=b"model-a")
    write_model(right, "clean_none_fixed_s10264", seed=10264, weights=b"model-b")

    report = judge([source(left, "left"), source(right, "right")])

    assert report["exit_code"] == 4
    assert report["collisions"][0]["verdict"] == "different_weights"
    assert report["collisions_fatal"], "a real clash must be fatal, not a note"
    # both sources still report their own count: nothing was removed to make it fit
    assert [s["n_models"] for s in report["sources"]] == [1, 1]
    assert any("cannot both be scored" in p for p in report["problems"])


def test_census_stops_when_a_non_lab_source_spans_the_lab_seed_range(tmp_path):
    """A range crossing 60152..100151 means the lab shards are no longer disjoint."""
    root = tmp_path / "local"
    write_model(root, "clean_none_fixed_s10264", seed=10264)
    write_model(root, "clean_none_fixed_s70000", seed=70000)

    report = judge([source(root, "runs/clean_null_local_w1")])

    assert report["stop_on_lab_overlap"] is True
    assert report["exit_code"] == 3
    assert any("STOP, do not dedupe" in p for p in report["problems"])


def test_census_flags_a_single_seed_inside_the_lab_range_too(tmp_path):
    """One stray lab seed in a non-lab source is the same failure as a whole range."""
    root = tmp_path / "local"
    write_model(root, f"clean_none_fixed_s{LAB_SEED_MIN + 5}", seed=LAB_SEED_MIN + 5)

    report = judge([source(root, "runs/clean_null_local_w1")])

    assert report["stop_on_lab_overlap"] is True
    assert report["exit_code"] == 3


def test_lab_sources_are_allowed_inside_the_lab_range(tmp_path):
    """The lab boxes are supposed to be there; the rule is about non-lab sources."""
    root = tmp_path / "b1"
    write_model(root, "clean_none_fixed_s60152", seed=60152)

    report = judge([source(root, "box1-corpus-v2", kind="lab_corpus")])

    assert report["stop_on_lab_overlap"] is False
    assert report["exit_code"] == 0


def test_census_records_a_re_manifest_seed_collision_as_a_warning_not_a_failure(tmp_path):
    """Same seed and byte-identical weights is one model exported twice."""
    left = tmp_path / "local_10250"
    right = tmp_path / "clean_null_local_w1"
    write_model(left, "clean_none_fixed_s10264", seed=10264, weights=b"same-bytes")
    write_model(right, "clean_none_fixed_s10264", seed=10264, weights=b"same-bytes")

    report = judge([source(left, "runs/local_10250"), source(right, "runs/clean_null_local_w1")])

    assert report["collisions"][0]["verdict"] == "identical_weights"
    assert report["exit_code"] == 0
    assert not report["collisions_fatal"]
    assert any("re-manifest" in w for w in report["warnings"])


def test_census_treats_a_stale_unselected_copy_as_a_warning_only(tmp_path):
    """A directory that is not part of the merge must not fail the merge."""
    staged = tmp_path / "mac-corpus"
    stale = tmp_path / "local_10250"
    write_model(staged, "clean_none_fixed_s10264", seed=10264, weights=b"same-bytes")
    write_model(stale, "clean_none_fixed_s10264", seed=10264, weights=b"same-bytes")

    report = judge([source(staged, "staged/mac-corpus"), source(stale, "runs/local_10250")],
                   selected=["staged/mac-corpus"])

    assert report["exit_code"] == 0
    assert any("stale copy" in w for w in report["warnings"])


def test_census_fails_when_an_archive_digest_disagrees_with_the_declared_bytes(tmp_path, monkeypatch):
    """An archive that is not the promised bytes is a hard failure, never a note."""
    from scripts import integration

    archive = tmp_path / "git-1056.tar.gz"
    inner = tmp_path / "payload" / "models" / "clean_none_fixed_s10150"
    write_model(tmp_path / "payload" / "models", "clean_none_fixed_s10150", seed=10150)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner / "manifest.json", arcname="models/clean_none_fixed_s10150/manifest.json")
        tar.add(inner / "weights.npz", arcname="models/clean_none_fixed_s10150/weights.npz")

    monkeypatch.setattr(
        integration,
        "EXPECTED_ARCHIVES",
        {"git-1056.tar.gz": ("0" * 64, archive.stat().st_size, 1056)},
    )
    sources, archives, gaps = discover_sources(tmp_path, None)

    assert archives[0]["digest_ok"] is False
    assert archives[0]["n_manifests"] == 1
    report = build_report(sources, archives, gaps, selected=[s["id"] for s in sources])
    assert report["exit_code"] == 2
    assert any("sha256" in p for p in report["problems"])
    assert any("!= declared 1056" in p for p in report["problems"])


def test_census_fails_a_missing_reassembled_archive_instead_of_scoring_around_it(tmp_path, monkeypatch):
    """If the fleet archive was never reassembled, the run must say so."""
    from scripts import integration

    monkeypatch.setattr(
        integration, "EXPECTED_ARCHIVES", {"fleet-all-6250.tar.gz": ("a" * 64, 10, 6250)}
    )
    sources, archives, gaps = discover_sources(tmp_path, None)

    assert any(gap["what"] == "fleet-all-6250.tar.gz" for gap in gaps)


def test_census_counts_manifest_dirs_even_when_weights_are_missing(tmp_path):
    """The census counts what is there and names what is missing; verify quarantines."""
    root = tmp_path / "corpus"
    write_model(root, "clean_none_fixed_s10150", seed=10150, weights=None)
    write_model(root, "clean_none_fixed_s10151", seed=10151)

    report = judge([source(root, "runs/staging")])

    assert report["sources"][0]["n_models"] == 2
    assert report["sources"][0]["n_missing_weights"] == 1
    assert any("no weights.npz" in w for w in report["warnings"])


def test_census_fails_on_a_duplicate_seed_inside_one_source(tmp_path):
    """One source cannot hold the same detector seed twice, whatever the id says."""
    root = tmp_path / "corpus"
    write_model(root, "clean_none_fixed_s10150", seed=10150)
    write_model(root, "clean_none_fixed_s10150_copy", seed=10150, weights=b"other")

    report = judge([source(root, "runs/staging")])

    assert report["exit_code"] == 4
    assert any("duplicate detector seeds" in p for p in report["problems"])


def test_seed_ranges_keep_a_gap_visible_instead_of_reporting_min_to_max():
    """A hole in a corpus is the difference between 11,637 models and the 24,607th seed."""
    assert seed_ranges([1, 2, 3, 7, 8, 20]) == [[1, 3], [7, 8], [20, 20]]
    assert seed_ranges([]) == []
    assert seed_ranges([5]) == [[5, 5]]


def test_census_reports_lab_boxes_missing_from_the_integration_folder(tmp_path):
    """Two absent boxes are 10,000 models, and the report must name them."""
    cviaf = tmp_path / "models_results" / "cviaf"
    (cviaf / "box-plan4" / "shards").mkdir(parents=True)
    shard = {
        "schema": "cviaf.analysis-shard.v1",
        "shard_id": "seed_75152_80151",
        "seed_min": 75152,
        "seed_max": 80151,
        "count": 1,
        "models": [
            {"model_id": "clean_none_fixed_s75152", "seed": 75152, "path": "clean_null_lab_b4w1/clean_none_fixed_s75152"}
        ],
    }
    (cviaf / "box-plan4" / "shards" / "seed_75152_80151.json").write_text(json.dumps(shard))

    sources, archives, gaps = discover_sources(tmp_path, None)
    whats = " | ".join(gap["what"] for gap in gaps)

    plans = [s for s in sources if s["kind"] == "lab_plan"]
    assert len(plans) == 1
    assert summarize(plans[0])["n_models"] == 1
    assert "lab box 2 (seeds 65152..70151)" in whats
    assert "lab box 3 (seeds 70152..75151)" in whats
    # the one box whose plan is present must not be reported as a gap
    assert "lab box 4" not in whats


def test_census_sees_lab_box_plans_that_contain_two_shards(tmp_path):
    """box-plan7 holds box 8's shard too; the census must show both, not one."""
    cviaf = tmp_path / "models_results" / "cviaf"
    shards_dir = cviaf / "box-plan7" / "shards"
    shards_dir.mkdir(parents=True)
    for shard_id, low, high in (("seed_90152_95151", 90152, 95151), ("seed_95152_100151", 95152, 100151)):
        (shards_dir / f"{shard_id}.json").write_text(
            json.dumps(
                {
                    "schema": "cviaf.analysis-shard.v1",
                    "shard_id": shard_id,
                    "seed_min": low,
                    "seed_max": high,
                    "count": 1,
                    "models": [{"model_id": f"clean_none_fixed_s{low}", "seed": low, "path": f"clean_null_lab_b7w1/clean_none_fixed_s{low}"}],
                }
            )
        )

    sources, _, _ = discover_sources(tmp_path, None)
    plan = [s for s in sources if s["kind"] == "lab_plan"][0]
    summary = summarize(plan)

    assert summary["shards"] == ["seed_90152_95151.json", "seed_95152_100151.json"]
    assert summary["n_models"] == 2
    assert summary["seed_ranges"] == [[90152, 90152], [95152, 95152]]


def test_duplicate_values_reports_each_offending_value_once():
    assert duplicate_values([1, 1, 2, 3, 3, 3]) == [1, 3]
    assert duplicate_values(["a"]) == []


def test_seed_of_ignores_ids_without_a_seed_suffix():
    assert seed_of("clean_none_fixed_s60152") == 60152
    assert seed_of("clean_none_fixed") is None
    # a suffix that is not the seed must not be read as one
    assert seed_of("clean_none_fixed_s5800_v2") is None
    assert seed_of("results_seed_60152_65151.npz") is None


def test_scan_archive_derives_seeds_from_member_names_without_extracting(tmp_path):
    """The census sizes the archive without unpacking 94 MB into the tree."""
    archive = tmp_path / "box5-corpus.tar.gz"
    inner = write_model(tmp_path / "payload", "clean_none_fixed_s80152", seed=80152)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner / "manifest.json", arcname="box-corpus/clean_null_lab_b5w1/clean_none_fixed_s80152/manifest.json")
        tar.add(inner / "weights.npz", arcname="box-corpus/clean_null_lab_b5w1/clean_none_fixed_s80152/weights.npz")

    entries = scan_archive(archive)

    assert len(entries) == 1
    assert entries[0]["seed"] == 80152
    assert entries[0]["rel"] == "clean_null_lab_b5w1/clean_none_fixed_s80152"


def test_render_states_the_verdict_and_the_missing_sources(tmp_path):
    """The operator reads the text render, so the verdict has to be in it."""
    root = tmp_path / "corpus"
    write_model(root, "clean_none_fixed_s10150", seed=10150)
    report = judge([source(root, "runs/staging")])
    text = render(report)

    assert CENSUS_SCHEMA in text
    assert "verdict: PASS" in text
    assert "fatal=0" in text
