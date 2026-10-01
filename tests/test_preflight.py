"""Tests for the scoring-pass preflight (`cviaf/lab/preflight.py`).

Each test is a way a multi-hour pass was lost, or would have been:

  * a shard that is missing, or present but still being written (no registry), so the
    pass starts and dies an hour in;
  * a tampered kind inside what is supposed to be a clean-only null, where every
    attacked model is then counted as a false positive;
  * resuming into a ledger whose alpha moved, which mixes two thresholds into one
    rate instead of failing loudly;
  * an absent corpus digest read as "verified" -- a check passing because it could not
    look, which is the same class of bug as an undeclared kind defaulting to negative;
  * six shards scored against six different reference models, which nothing in the
    report would ever say;
  * a signal-name typo, which produces an empty column rather than an error.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from cviaf.lab.fleet import census  # noqa: F401  (import-time sanity)
from cviaf.lab.preflight import KNOWN_SIGNALS, preflight, render


def write_model(root, model_id, seed=0, kind="clean"):
    d = os.path.join(root, model_id)
    os.makedirs(d, exist_ok=True)
    manifest = {
        "lab_version": "lab-1.0.0", "model_id": model_id,
        "created_utc": "2026-09-29T00:00:00Z",
        "spec": {"kind": kind},
        "spec_digest": hashlib.sha256(model_id.encode()).hexdigest()[:16],
        "dataset_digests": {"train": "d" * 64}, "attack_digest": "b" * 16,
        "seeds": {"scene": seed + 7, "attack": 11, "detector": seed},
        "artifact": {"weights_digest": flake_digest(model_id),
                     "head_digest": "e" * 64},
        "ground_truth": {"kind": kind},
        "metrics": {"clean_quality": {"f1": 0.5}, "attack_success_rate": {}},
        "quality_flags": {},
    }
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    with open(os.path.join(d, "weights.npz"), "wb") as fh:
        fh.write(model_id.encode())
    return d


def flake_digest(model_id):
    return hashlib.sha256(("w" + model_id).encode()).hexdigest()


def write_shard(root, name, ids, registry=True, kind="clean"):
    shard = os.path.join(root, name)
    os.makedirs(shard, exist_ok=True)
    for i, mid in enumerate(ids):
        write_model(shard, mid, seed=i, kind=kind)
    if registry:
        with open(os.path.join(shard, "registry.jsonl"), "w", encoding="utf-8") as fh:
            for mid in ids:
                fh.write(json.dumps({"dir": f"{name}/{mid}",
                                     "manifest": {"model_id": mid}}) + "\n")
    return shard


def names(prefix, n):
    return [f"{prefix}_{i}" for i in range(n)]


@pytest.fixture()
def fleet(tmp_path):
    root = str(tmp_path)
    # The reference model lives INSIDE w1 and is registered there, which is how the
    # real fleet is laid out. That is a warning case, not an invalid shard.
    a = write_shard(root, "w1", [*names("a", 3), "ref_model"])
    b = write_shard(root, "w2", names("b", 3))
    return {"shards": [a, b], "reference": os.path.join(a, "ref_model"),
            "w1": a, "tmp": root}


def test_a_green_fleet_records_the_population_the_reference_and_the_command(fleet, tmp_path):
    out = str(tmp_path / "ledger.json")
    report = preflight(fleet["shards"], out, reference=fleet["reference"],
                       signals=list(KNOWN_SIGNALS), min_free_gb=0.0)
    assert report["green"], report["failures"]
    assert report["fleet"]["n_models"] == 7
    assert report["schema"] == "cviaf.preflight.v1"
    check = next(c for c in report["checks"] if c["name"] == "reference_pinned")
    assert "weights_digest" in check["detail"] and "ref_model" in check["detail"]
    command = report["launch_command"]
    for token in (out, *fleet["shards"], "--resume", "--reference", fleet["reference"]):
        assert token in command
    assert render(report).count("ok") >= 4


def test_a_missing_shard_is_refused_before_anything_is_launched(fleet, tmp_path):
    report = preflight([*fleet["shards"], str(tmp_path / "nope")], str(tmp_path / "l.json"),
                       min_free_gb=0.0)
    assert report["green"] is False
    assert "corpora_readable" in report["failures"]
    assert "REFUSING TO LAUNCH" in report["verdict"]


def test_a_shard_still_being_written_is_refused_with_the_repair(fleet, tmp_path):
    half = write_shard(fleet["tmp"], "w3", names("c", 2), registry=False)
    report = preflight([half], str(tmp_path / "l.json"), min_free_gb=0.0)
    assert report["green"] is False
    check = next(c for c in report["checks"] if c["name"] == "corpora_readable")
    assert "registry.jsonl" in check["detail"]
    assert "rebuild-registry" in check["remedy"]


def test_a_tampered_kind_in_the_fpr_null_is_refused(fleet, tmp_path):
    """Every attacked model in a clean-only run is counted as a false positive."""
    poisoned = write_shard(fleet["tmp"], "w4", ["t_0", "t_1"], registry=True,
                           kind="weight_tamper")
    report = preflight([*fleet["shards"], poisoned], str(tmp_path / "l.json"),
                       min_free_gb=0.0)
    assert "clean_only_population" in report["failures"]
    check = next(c for c in report["checks"] if c["name"] == "clean_only_population")
    assert "weight_tamper" in check["detail"]


def test_resuming_into_a_ledger_whose_alpha_moved_is_refused(fleet, tmp_path):
    ledger = tmp_path / "l.json"
    ledger.write_text(json.dumps(_ledger(fleet["shards"], alpha=0.01)), encoding="utf-8")
    report = preflight(fleet["shards"], str(tmp_path / "next.json"),
                       ledger=str(ledger), alpha=0.05, min_free_gb=0.0)
    assert "ledger_contract" in report["failures"]
    assert "alpha" in next(c for c in report["checks"]
                           if c["name"] == "ledger_contract")["detail"]


def test_an_absent_corpus_digest_is_a_warning_not_a_pass(fleet, tmp_path):
    """A check that passes because it could not look is worse than no check."""
    ledger = tmp_path / "l.json"
    ledger.write_text(json.dumps(_ledger(fleet["shards"])), encoding="utf-8")
    report = preflight(fleet["shards"], str(tmp_path / "next.json"), ledger=str(ledger),
                       reference=fleet["reference"], min_free_gb=0.0)
    check = next(c for c in report["checks"]
                 if c["name"] == "ledger_population_unchanged")
    assert check["status"] == "warn"
    assert "cannot verify" in check["detail"]
    assert report["green"] is True          # a warning must not block the pass
    assert "ledger_population_unchanged" in report["warnings"]


def test_a_shard_rewritten_since_the_ledger_is_refused(fleet, tmp_path):
    from cviaf.lab.fpr_tpr import corpus_snapshot
    ledger = tmp_path / "l.json"
    snaps = {s: corpus_snapshot(s) for s in fleet["shards"]}
    ledger.write_text(json.dumps(_ledger(fleet["shards"], snapshots=snaps)),
                      encoding="utf-8")
    report = preflight(fleet["shards"], str(tmp_path / "next.json"), ledger=str(ledger),
                       reference=fleet["reference"], min_free_gb=0.0)
    assert report["green"] is True, report["failures"]   # nothing moved yet
    # Now the writer appends a model to a shard that was already scored.
    write_shard(fleet["tmp"], "w1", [*names("a", 3), "ref_model", "a_new"])
    report = preflight(fleet["shards"], str(tmp_path / "next.json"), ledger=str(ledger),
                       reference=fleet["reference"], min_free_gb=0.0)
    assert "ledger_population_unchanged" in report["failures"]
    assert "NEW ledger" in next(c for c in report["checks"]
                                if c["name"] == "ledger_population_unchanged")["remedy"]


def test_a_multi_shard_refdiv_run_without_one_reference_is_refused(fleet, tmp_path):
    report = preflight(fleet["shards"], str(tmp_path / "l.json"),
                       signals=["refdiv_mean_clean"], min_free_gb=0.0)
    assert "reference_pinned" in report["failures"]
    assert "ONE --reference" in next(c for c in report["checks"]
                                     if c["name"] == "reference_pinned")["remedy"]
    single = preflight([fleet["shards"][0]], str(tmp_path / "l.json"),
                       signals=["refdiv_mean_clean"], min_free_gb=0.0)
    assert single["green"] is True           # one shard picks its own by construction


def test_a_reference_inside_a_scored_shard_is_recorded_as_a_warning(fleet, tmp_path):
    report = preflight(fleet["shards"], str(tmp_path / "l.json"),
                       reference=fleet["reference"], min_free_gb=0.0)
    check = next(c for c in report["checks"]
                 if c["name"] == "reference_outside_population")
    assert check["status"] == "warn"
    assert "w1" in check["detail"]
    assert report["green"] is True
    outside = os.path.join(fleet["tmp"], "refs", "r")
    os.makedirs(os.path.dirname(outside), exist_ok=True)
    write_model(os.path.dirname(outside), "r", seed=5)
    clean = preflight(fleet["shards"], str(tmp_path / "l.json"), reference=outside,
                      min_free_gb=0.0)
    assert next(c for c in clean["checks"]
                if c["name"] == "reference_outside_population")["status"] == "pass"


def test_a_signal_name_typo_is_refused_rather_than_producing_an_empty_column(fleet, tmp_path):
    report = preflight(fleet["shards"], str(tmp_path / "l.json"),
                       signals=["ctc_mean_clean", "ctc_meen_clean"], min_free_gb=0.0)
    assert "signals" in report["failures"]
    assert "ctc_meen_clean" in next(c for c in report["checks"]
                                    if c["name"] == "signals")["detail"]


def test_insufficient_disk_is_refused_with_the_requirement(fleet, tmp_path):
    report = preflight(fleet["shards"], str(tmp_path / "l.json"), min_free_gb=1e9)
    assert "disk_space" in report["failures"]
    assert "free at least" in next(c for c in report["checks"]
                                   if c["name"] == "disk_space")["remedy"]


def test_the_estimated_time_is_labelled_with_its_basis(fleet, tmp_path):
    report = preflight(fleet["shards"], str(tmp_path / "l.json"), min_free_gb=0.0,
                       workers=2)
    check = next(c for c in report["checks"] if c["name"] == "runtime_estimate")
    assert check["status"] == "warn"          # informative, never blocking
    assert "basis" in check["detail"]
    assert report["estimated_minutes"] is not None


def _ledger(shards, alpha=0.05, snapshots=None, n_per_shard=1):
    records = []
    for shard in shards:
        for i in range(n_per_shard):
            records.append({"model_id": f"m{os.path.basename(shard)}_{i}",
                            "corpus": os.path.abspath(shard), "kind": "clean",
                            "is_positive": False, "split": "unassigned",
                            "scores": {"ctc_mean_clean": 0.5}})
    prov = {"producer": "test"}
    if snapshots:
        prov["corpus_snapshots"] = snapshots
    return {"schema": "cviaf.fpr-tpr-ledger.v1", "alpha": alpha,
            "higher_is_more_anomalous": True, "positive_kinds": ["oga"],
            "negative_kinds": ["clean"], "provenance": prov, "records": records}
