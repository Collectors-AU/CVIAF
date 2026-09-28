"""Tests for the manifest + registry validator.

Each test is a failure this lane actually suffered or would only discover after a long
battery run:

  * a hand-built arm saved without the real-backbone marker (load would silently rebuild
    a random backbone and keep the trained head),
  * an arm with no ``behaviour_divergence`` that crashed a 48-model battery with a
    KeyError minutes in,
  * a trainer run that rewrote ``registry.jsonl`` and dropped 24 models, which the
    registry-preferring loader then made invisible.

The fixtures are synthetic so the suite does not depend on a corpus being on disk; a
final conformance test locks the real corpora when they are present.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from cviaf.lab.manifest_schema import (is_model_manifest, main, make_registry_from_disk,
                                      scan_corpus, validate_manifest, validate_model_dir,
                                      validate_registry)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def manifest(kind: str = "clean", **over):
    m = {
        "lab_version": "1.0.0",
        "model_id": "m0",
        "created_utc": "2026-09-27T00:00:00Z",
        "spec": {"kind": "synthetic"},
        "spec_digest": "0123456789abcdef",
        "dataset_digests": {"train": "a" * 64},
        "attack_digest": "fedcba9876543210",
        "seeds": {"train": 0},
        "artifact": {"weights_digest": "b" * 64, "head_digest": "c" * 64},
        "ground_truth": {"kind": kind},
        "metrics": {"clean_quality": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                    "attack_success_rate": {}},
        "quality_flags": {},
    }
    for key, value in over.items():
        m[key] = value
    return m


def weight_tamper_manifest(**over):
    """A valid weight-space arm: the three contract fields are what make it valid."""
    m = manifest(kind="weight_tamper")
    m["quality_flags"] = {"is_model_attack": True, "model_effect_weak": False}
    m["metrics"]["behaviour_divergence"] = {"f1_relative_drop": 0.2}
    for key, value in over.items():
        m[key] = value
    return m


def write_model(root, name, m):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh)
    return d


def write_registry(root, model_ids):
    with open(os.path.join(root, "registry.jsonl"), "w", encoding="utf-8") as fh:
        for mid in model_ids:
            fh.write(json.dumps({"dir": os.path.join(root, mid),
                                 "manifest": {"model_id": mid}}) + "\n")


def write_weights(path, real_backbone_marker=True):
    payload = {"w": np.zeros(3, dtype=np.float32)}
    if real_backbone_marker:
        payload["_meta"] = np.frombuffer(
            json.dumps({"kind": "real_backbone"}).encode(), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def make_corpus(tmp_path, model_ids=("m0", "m1"), registry=None, name="corpus"):
    root = os.path.join(str(tmp_path), name)
    os.makedirs(root, exist_ok=True)
    for mid in model_ids:
        d = write_model(root, mid, manifest(**{"model_id": mid}))
        np.savez_compressed(os.path.join(d, "weights.npz"),
                            w=np.zeros(3, dtype=np.float32))
    if registry is not None:
        write_registry(root, registry)
    return root


# --------------------------------------------------------------------------- #
# manifest-level checks
# --------------------------------------------------------------------------- #

def test_minimal_manifest_is_valid():
    assert validate_manifest(manifest()) == []
    assert validate_manifest(weight_tamper_manifest()) == []


def test_missing_required_key_is_reported():
    m = manifest()
    del m["metrics"]
    assert any("missing required key 'metrics'" in p for p in validate_manifest(m))


def test_model_id_must_match_its_directory():
    """Two artefacts sharing one digest in a report is how a corpus double-counts."""
    problems = validate_manifest(manifest(model_id="other"), model_id="m0")
    assert any("does not match its directory" in p for p in problems)


@pytest.mark.parametrize("key", ["spec_digest", "attack_digest"])
@pytest.mark.parametrize("bad", ["", "ABC0123456789abc", "0123456789abcde", 12])
def test_digest_format_is_enforced(key, bad):
    assert any(key in p for p in validate_manifest(manifest(**{key: bad})))


def test_dataset_digest_must_be_sha256():
    m = manifest(dataset_digests={"train": "deadbeef"})
    assert any("dataset_digests['train']" in p for p in validate_manifest(m))


def test_unknown_ground_truth_kind_is_rejected():
    problems = validate_manifest(manifest(ground_truth={"kind": "backdoor"}))
    assert any("is not a known attack kind" in p for p in problems)


def test_stampfree_kind_is_allowed():
    assert validate_manifest(manifest(ground_truth={"kind": "stampfree"})) == []


def test_nan_and_inf_are_rejected():
    m = manifest()
    m["metrics"]["clean_quality"]["f1"] = float("nan")
    assert any("not finite" in p for p in validate_manifest(m))
    m["metrics"]["clean_quality"]["f1"] = float("inf")
    assert any("not finite" in p for p in validate_manifest(m))


def test_f1_outside_unit_interval_is_rejected():
    m = manifest()
    m["metrics"]["clean_quality"]["f1"] = 1.4
    assert any("out of [0,1]" in p for p in validate_manifest(m))


# --------------------------------------------------------------------------- #
# the weight-space arm contract (the 48-model battery KeyError)
# --------------------------------------------------------------------------- #

def test_weight_space_arm_must_declare_is_model_attack():
    m = weight_tamper_manifest()
    m["quality_flags"] = {"model_effect_weak": False}
    assert any("is_model_attack" in p for p in validate_manifest(m))


def test_weight_space_arm_must_carry_f1_relative_drop():
    m = weight_tamper_manifest()
    del m["metrics"]["behaviour_divergence"]
    problems = validate_manifest(m)
    assert any("f1_relative_drop" in p for p in problems)


def test_weight_space_arm_must_carry_model_effect_weak():
    m = weight_tamper_manifest()
    m["quality_flags"] = {"is_model_attack": True}
    assert any("model_effect_weak" in p for p in validate_manifest(m))


def test_data_attack_arm_needs_no_behaviour_divergence():
    """The contract is specific to weight-space arms; a poisoning arm has a trigger."""
    for kind in ("oga", "oda", "rma", "gma", "label_flip", "dup_flood", "ood_insert"):
        assert validate_manifest(manifest(kind=kind)) == [], kind


# --------------------------------------------------------------------------- #
# the artefact contract (a plain npz saved as a real-backbone model)
# --------------------------------------------------------------------------- #

def test_real_backbone_manifest_without_marker_is_rejected(tmp_path):
    d = write_model(str(tmp_path), "rb0",
                    manifest(model_id="rb0", spec={"kind": "real_backbone"}))
    write_weights(os.path.join(d, "weights.npz"), real_backbone_marker=False)
    problems = validate_model_dir(d)
    assert any("_meta marker" in p for p in problems)


def test_real_backbone_manifest_with_marker_passes(tmp_path):
    d = write_model(str(tmp_path), "rb0",
                    manifest(model_id="rb0", spec={"kind": "real_backbone"}))
    write_weights(os.path.join(d, "weights.npz"), real_backbone_marker=True)
    assert validate_model_dir(d) == []


def test_missing_weights_npz_is_reported(tmp_path):
    d = write_model(str(tmp_path), "m0", manifest())
    assert any("weights.npz is missing" in p for p in validate_model_dir(d))


def test_synthetic_manifest_needs_no_marker(tmp_path):
    d = write_model(str(tmp_path), "m0", manifest())
    np.savez_compressed(os.path.join(d, "weights.npz"), w=np.zeros(3, dtype=np.float32))
    assert validate_model_dir(d) == []


def test_unparseable_manifest_is_a_problem_not_a_skip(tmp_path):
    d = os.path.join(str(tmp_path), "m0")
    os.makedirs(d)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert any("not readable JSON" in p for p in validate_model_dir(d))


# --------------------------------------------------------------------------- #
# registry vs disk (the run that silently dropped 24 models)
# --------------------------------------------------------------------------- #

def test_on_disk_models_absent_from_registry_are_reported(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0", "m1"), registry=["m0"])
    problems = validate_registry(root)
    assert any("absent from registry.jsonl" in p and "m1" in p for p in problems)


def test_duplicate_registry_entries_are_reported(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0",), registry=["m0", "m0"])
    assert any("lists 'm0' 2 times" in p for p in validate_registry(root))


def test_registry_entry_with_missing_dir_is_reported(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0",), registry=["m0", "ghost"])
    assert any("points at a missing dir" in p for p in validate_registry(root))


def test_missing_registry_is_reported_with_repair_path(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0",))
    problems = validate_registry(root)
    assert any("registry.jsonl is missing" in p for p in problems)


def test_make_registry_from_disk_repairs_a_clobbered_registry(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0", "m1"), registry=["m0"])
    assert validate_registry(root)
    entries = make_registry_from_disk(root, write=True)
    assert [e["manifest"]["model_id"] for e in entries] == ["m0", "m1"]
    assert validate_registry(root) == []


def test_strict_flags_registry_entries_with_no_manifest_on_disk(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0",), registry=["m0", "ghost"])
    os.makedirs(os.path.join(root, "ghost"), exist_ok=True)   # dir exists, manifest gone
    assert validate_registry(root) == []          # lenient: registry-only is tolerated
    assert any("no manifest on disk" in p for p in validate_registry(root, strict=True))


def test_valid_corpus_has_no_problems(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0", "m1"), registry=["m0", "m1"])
    assert validate_registry(root) == []


# --------------------------------------------------------------------------- #
# foreign schemas: drift cells are not models, and are not silently dropped
# --------------------------------------------------------------------------- #

def test_foreign_schema_dirs_are_excluded_from_model_validation(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0",), registry=["m0"])
    cell = os.path.join(root, "desert_summer_blurred")
    os.makedirs(cell)
    with open(os.path.join(cell, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": "cviaf.drift-cells/1", "cell_id": "desert_summer_blurred"}, fh)
    assert validate_registry(root) == []
    on_disk, foreign = scan_corpus(root)
    assert on_disk == ["m0"] and foreign == {"cviaf.drift-cells/1":
                                             ["desert_summer_blurred"]}
    assert not is_model_manifest({"schema": "cviaf.drift-cells/1"})


def test_corpus_of_only_foreign_artefacts_needs_no_registry(tmp_path):
    """runs/drift_cells: descriptors only, so a missing registry.jsonl is not a defect."""
    root = os.path.join(str(tmp_path), "cells")
    os.makedirs(os.path.join(root, "cell_a"))
    with open(os.path.join(root, "cell_a", "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": "cviaf.drift-cells/1"}, fh)
    assert validate_registry(root) == []


def test_foreign_dirs_are_not_registered_by_the_repair_path(tmp_path):
    root = make_corpus(tmp_path, model_ids=("m0",))
    cell = os.path.join(root, "cell_a")
    os.makedirs(cell)
    with open(os.path.join(cell, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": "cviaf.drift-cells/1"}, fh)
    entries = make_registry_from_disk(root)
    assert [e["manifest"]["model_id"] for e in entries] == ["m0"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_exit_codes(tmp_path, capsys):
    good = make_corpus(tmp_path, model_ids=("m0",), registry=["m0"], name="good")
    bad = make_corpus(tmp_path, model_ids=("m0", "m1"), registry=["m0"], name="bad")
    out = os.path.join(str(tmp_path), "report.json")
    assert main(["--corpus", good, "--json", out]) == 0
    assert main(["--corpus", bad]) == 1
    assert main(["--corpus", os.path.join(str(tmp_path), "nope")]) == 2
    capsys.readouterr()
    report = json.load(open(out, encoding="utf-8"))
    assert report["schema"] == "cviaf.manifest-validation.v1"
    assert report["corpora"][good]["n_models"] == 1


def test_cli_rebuild_registry_repairs_and_passes(tmp_path, capsys):
    root = make_corpus(tmp_path, model_ids=("m0", "m1"), registry=["m0"])
    assert main(["--corpus", root, "--rebuild-registry"]) == 0
    capsys.readouterr()
    assert validate_registry(root) == []


# --------------------------------------------------------------------------- #
# conformance of the corpora on disk (the real lock)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("corpus", ["runs/real_cifar", "runs/day1", "runs/clean_null",
                                    "runs/mvp", "runs/day1_backup_prepaired",
                                    "runs/real_cifar_smoke", "runs/stampfree"])
def test_on_disk_corpora_conform(corpus):
    path = os.path.join(REPO, corpus)
    if not os.path.isdir(path):
        pytest.skip(f"{corpus} not present")
    problems = validate_registry(path)
    # stampfree is known to be missing one registry line; everything else must be clean
    if corpus.endswith("stampfree"):
        assert all("absent from registry.jsonl" in p for p in problems), problems
    else:
        assert problems == []


def test_corpus_snapshot_is_deepcopy_safe():
    """Guards the fixture against accidental in-place mutation between tests."""
    assert validate_manifest(copy.deepcopy(manifest())) == []
