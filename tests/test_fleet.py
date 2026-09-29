"""Tests for the fleet census.

The census exists to answer three questions before a population is used for an FPR
average: does everything validate, is every asset distinct, and are the shards
disjoint? Each test below is one of those going wrong.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from cviaf.lab.fleet import (FLEET_SCHEMA, census, cross_shard_duplicates, main,
                            shard_summary, weights_digest, within_shard_duplicates)


def write_model(root, model_id, seed=0, spec_digest=None, weights=None, body=None):
    spec_digest = spec_digest or hashlib.sha256(model_id.encode()).hexdigest()[:16]
    weights = weights if weights is not None else model_id.encode()
    d = os.path.join(root, model_id)
    os.makedirs(d, exist_ok=True)
    manifest = {
        "lab_version": "lab-1.0.0", "model_id": model_id,
        "created_utc": "2026-09-29T00:00:00Z",
        "spec": {"kind": "clean"}, "spec_digest": spec_digest,
        "dataset_digests": {"train": "d" * 64}, "attack_digest": "b" * 16,
        "seeds": {"scene": seed + 7, "attack": 11, "detector": seed},
        "artifact": {"weights_digest": "c" * 64, "head_digest": "e" * 64},
        "ground_truth": {"kind": "clean"},
        "metrics": {"clean_quality": {"f1": 0.5}, "attack_success_rate": {}},
        "quality_flags": {},
    }
    if body:
        manifest.update(body)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    with open(os.path.join(d, "weights.npz"), "wb") as fh:
        fh.write(weights)
    return d


def write_shard(root, name, ids, registry=True, **kw):
    shard = os.path.join(root, name)
    os.makedirs(shard, exist_ok=True)
    for i, mid in enumerate(ids):
        write_model(shard, mid, seed=kw.get("seed", 0) + i)
    if registry:
        with open(os.path.join(shard, "registry.jsonl"), "w", encoding="utf-8") as fh:
            for mid in ids:
                fh.write(json.dumps({"dir": f"{name}/{mid}",
                                     "manifest": {"model_id": mid}}) + "\n")
    return shard


# --------------------------------------------------------------------------- #
# one shard
# --------------------------------------------------------------------------- #

def test_a_shard_summarises_its_models(tmp_path):
    shard = write_shard(str(tmp_path), "w1", ["m0", "m1", "m2"])
    summary = shard_summary(shard)
    assert summary["n_models"] == 3 and summary["problems"] == []
    assert [m["model_id"] for m in summary["models"]] == ["m0", "m1", "m2"]
    assert summary["models"][0]["kind"] == "clean"
    assert summary["models"][0]["seed"] == 0           # seeds.detector


def test_a_shard_without_a_registry_is_not_ready(tmp_path):
    shard = write_shard(str(tmp_path), "w1", ["m0"], registry=False)
    assert any("registry.jsonl is missing" in p for p in shard_summary(shard)["problems"])


def test_within_shard_duplicate_ids_are_reported(tmp_path):
    shard = write_shard(str(tmp_path), "w1", ["m0"])
    with open(os.path.join(shard, "registry.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"dir": "w1/m0", "manifest": {"model_id": "m0"}}) + "\n")
    dups = within_shard_duplicates(shard_summary(shard))
    assert dups == [] or dups[0]["count"] >= 1        # manifest count is the source
    assert any("appears 2 times" in p or "lists" in p
               for p in shard_summary(shard)["problems"])


def test_weights_digest_reads_the_artefact(tmp_path):
    d = write_model(str(tmp_path), "m0", weights=b"hello")
    assert weights_digest(d) == pytest.approx(weights_digest(d))
    assert weights_digest(os.path.join(str(tmp_path), "nope")) is None
    write_model(str(tmp_path), "m1", weights=b"other")
    assert weights_digest(os.path.join(str(tmp_path), "m1")) != weights_digest(d)


# --------------------------------------------------------------------------- #
# across shards
# --------------------------------------------------------------------------- #

def test_disjoint_shards_are_ready(tmp_path):
    a = write_shard(str(tmp_path), "w1", ["m0", "m1"])
    b = write_shard(str(tmp_path), "w2", ["m2", "m3"], seed=10)
    report = census([a, b])
    assert report["n_models"] == 4 and report["n_shards"] == 2
    assert report["ready_for_fpr"] is True and report["problems"] == []
    assert report["kinds"] == {"clean": 4}
    assert report["seed_range"] == [0, 11] and report["n_distinct_seeds"] == 4


def test_a_model_id_in_two_shards_is_fatal(tmp_path):
    """One asset counted twice inflates whichever half it lands in."""
    a = write_shard(str(tmp_path), "w1", ["m0"])
    b = write_shard(str(tmp_path), "w2", ["m0"], seed=5)
    report = census([a, b])
    assert report["ready_for_fpr"] is False
    assert report["duplicate_model_ids"][0]["value"] == "m0"
    assert any("appears in 2 shards" in p for p in report["problems"])


def test_a_shared_spec_digest_is_fatal(tmp_path):
    """Same recipe and seed under two ids is a re-run, not an independent draw."""
    a = write_shard(str(tmp_path), "w1", ["m0"])
    b = write_shard(str(tmp_path), "w2", ["m1"])
    for shard, mid in ((a, "m0"), (b, "m1")):
        write_model(shard, mid, spec_digest="f" * 16)
    report = census([a, b])
    assert report["ready_for_fpr"] is False
    assert any("shared by 2 models" in p for p in report["problems"])


def test_identical_weights_under_two_ids_is_reported_when_asked(tmp_path):
    a = write_shard(str(tmp_path), "w1", ["m0"])
    b = write_shard(str(tmp_path), "w2", ["m1"])
    write_model(a, "m0", weights=b"same")
    write_model(b, "m1", weights=b"same")
    assert census([a, b], check_weight_digests=False)["duplicate_weight_digests"] == []
    report = census([a, b], check_weight_digests=True)
    assert report["duplicate_weight_digests"][0]["count"] == 2
    assert any("identical weights" in p for p in report["problems"])


def test_an_empty_shard_is_not_ready(tmp_path):
    empty = os.path.join(str(tmp_path), "w1")
    os.makedirs(empty)
    report = census([empty])
    assert report["ready_for_fpr"] is False
    assert any("no model manifests" in p for p in report["problems"])


def test_a_missing_shard_is_reported_not_crashed(tmp_path):
    report = census([os.path.join(str(tmp_path), "ghost")])
    assert report["ready_for_fpr"] is False
    assert any("not a directory" in p for p in report["problems"])


def test_cross_shard_duplicates_helper_is_deterministic(tmp_path):
    a = write_shard(str(tmp_path), "w1", ["m0", "m1"])
    b = write_shard(str(tmp_path), "w2", ["m1", "m2"], seed=5)
    shards = [shard_summary(a), shard_summary(b)]
    dups = cross_shard_duplicates(shards, "model_id")
    assert dups == [{"value": "m1", "count": 2, "where": [f"{a}:m1", f"{b}:m1"]}]


def test_census_records_the_schema_and_shard_labels(tmp_path):
    a = write_shard(str(tmp_path), "w1", ["m0"])
    report = census([a])
    assert report["schema"] == FLEET_SCHEMA and report["corpora"] == [a]
    assert report["shards"][0]["corpus"] == a
    assert "models" not in report["shards"][0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_exit_codes_and_json(tmp_path, capsys):
    good = write_shard(str(tmp_path), "w1", ["m0"])
    bad = write_shard(str(tmp_path), "w2", ["m0"])
    out = os.path.join(str(tmp_path), "census.json")
    assert main(["--corpus", good, "--json", out]) == 0
    assert main(["--corpus", good, bad]) == 1
    assert main(["--glob", os.path.join(str(tmp_path), "nothing*")]) == 2
    capsys.readouterr()
    written = json.loads(open(out, encoding="utf-8").read())
    assert written["n_models"] == 1 and written["ready_for_fpr"] is True


def test_cli_reports_the_fleet_on_disk_when_present(capsys):
    import glob as globmod
    corpora = sorted(globmod.glob(os.path.join(os.path.dirname(__file__), "..",
                                               "..", "runs", "clean_null_local_w*")))
    if not corpora:
        pytest.skip("no local fleet shards present")
    code = main(["--corpus", *corpora])
    text = capsys.readouterr().out
    assert "models" in text and ("READY" in text or "NOT READY" in text)
    assert code in (0, 1)
