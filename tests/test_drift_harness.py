"""Tests for the drift harness.

The harness's job is to stop a significant p-value from being read as a finding, so
these tests are mostly about the two nulls and the decisions built on them: a shift
smaller than the same metric's value on a no-shift resample is not drift, and a
metric that is at its maximum on every contrast is not discriminating.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cviaf.lab.drift_harness import (AXES, CELL_SCHEMA, DECISIONS, HARNESS_SCHEMA,
                                    METRICS, NULL_DEFINITION, evaluate_contrast,
                                    load_cells, main, render, run, split_half,
                                    validate_drift_cell)

REF_SCENE = {"terrain": "desert", "season": "summer", "illumination": 1.0, "gamma": 1.0,
             "sensor_noise": 0.02, "sensor_blur": 0, "objects_per_image": [1, 3], "seed": 7}


def cell(cell_id="c", role="shifted", scene=None, moved=None, pair=None, n=240,
         counts=None):
    counts = counts if counts is not None else {"lab_alpha": n}
    return {
        "schema": CELL_SCHEMA, "cell_id": cell_id, "role": role,
        "scene_spec": dict(scene if scene is not None else REF_SCENE),
        "declared_axes_moved": dict(moved or {}), "n_images": n, "seed_offset": 1,
        "contributors": sorted(counts), "contributor_counts": counts,
        "image_shape": [n, 8, 8, 3], "digests": {"dataset": "a" * 64,
                                                 "pixels_uint8_sha256": "b" * 64},
        "object_count": 1, "class_counts": [1], "paired_cell": pair,
    }


def ref_cell():
    return cell("reference_desert_summer_clean", role="reference")


# --------------------------------------------------------------------------- #
# the definitions themselves
# --------------------------------------------------------------------------- #

def test_every_metric_definition_is_complete():
    assert METRICS, "no metrics defined"
    for name, d in METRICS.items():
        for key in ("unit", "direction", "formula", "minimum_n", "p_value",
                    "supports", "cannot"):
            assert key in d, f"{name} is missing {key}"
        assert d["direction"] in ("higher_is_more_shift", "lower_is_more_shift")
        assert d["minimum_n"] >= 2
        assert d["supports"] and d["cannot"], f"{name} does not state its limits"


def test_the_null_definition_names_both_nulls():
    assert "control_value" in NULL_DEFINITION and "self_null" in NULL_DEFINITION


# --------------------------------------------------------------------------- #
# cell manifests
# --------------------------------------------------------------------------- #

def test_a_valid_shifted_cell_has_no_problems():
    shifted = cell("blur", scene={**REF_SCENE, "sensor_blur": 3},
                   moved={"sensor_blur": 3}, pair="control")
    assert validate_drift_cell(shifted, cell_ids=["blur", "control"],
                               reference=ref_cell(), pair=None) == []


def test_axes_list_matches_the_scene_spec():
    assert AXES == tuple(REF_SCENE)


@pytest.mark.parametrize("key", ["schema", "cell_id", "role", "scene_spec",
                                 "declared_axes_moved", "n_images", "digests"])
def test_missing_required_key_is_reported(key):
    bad = cell()
    del bad[key]
    assert any(f"missing required key {key!r}" in p for p in validate_drift_cell(bad))


def test_wrong_schema_is_reported():
    bad = cell()
    bad["schema"] = "cviaf.drift-cells/2"
    assert any("is not" in p for p in validate_drift_cell(bad))


def test_unknown_role_is_reported():
    bad = cell(role="baseline")
    assert any("unknown role" in p for p in validate_drift_cell(bad))


def test_shifted_cell_without_a_declared_move_is_reported():
    bad = cell(moved={})
    assert any("no axis is declared as moved" in p for p in validate_drift_cell(bad))


def test_declared_move_must_match_the_scene_spec():
    bad = cell(scene={**REF_SCENE, "sensor_blur": 1}, moved={"sensor_blur": 3})
    assert any("declared_axes_moved['sensor_blur']" in p
               for p in validate_drift_cell(bad))


def test_declared_axis_must_be_a_real_scene_axis():
    bad = cell(moved={"brightness": 2})
    assert any("is not a scene axis" in p for p in validate_drift_cell(bad))


def test_an_undeclared_move_away_from_the_reference_domain_is_reported():
    """The failure that made the drift corpus a tautology: a silent extra shift."""
    bad = cell("forest", scene={**REF_SCENE, "terrain": "forest"}, moved={},
               role="shifted")
    bad["declared_axes_moved"] = {"sensor_blur": 0}
    bad["scene_spec"]["sensor_blur"] = 0
    problems = validate_drift_cell(bad, reference=ref_cell())
    assert any("differs from the reference domain" in p and "'terrain'" in p
               for p in problems)


def test_a_control_that_moves_axes_is_reported():
    bad = cell("control", role="no_shift_control",
               scene={**REF_SCENE, "sensor_blur": 3}, moved={"sensor_blur": 3},
               pair="blur")
    assert any("a control that moves is not a control" in p
               for p in validate_drift_cell(bad))


def test_a_control_is_checked_against_its_pair_not_the_reference():
    """The first bug in this validator: it flagged every legitimate control."""
    shifted = cell("forest", scene={**REF_SCENE, "terrain": "forest"},
                   moved={"terrain": "forest"}, pair="forest_control")
    control = cell("forest_control", role="no_shift_control",
                   scene={**REF_SCENE, "terrain": "forest"}, moved={}, pair="forest")
    assert validate_drift_cell(control, cell_ids=["forest", "forest_control"],
                               reference=ref_cell(), pair=shifted) == []


def test_a_control_that_does_not_resample_its_pair_is_reported():
    shifted = cell("forest", scene={**REF_SCENE, "terrain": "forest"},
                   moved={"terrain": "forest"}, pair="forest_control")
    control = cell("forest_control", role="no_shift_control", scene=REF_SCENE, moved={},
                   pair="forest")
    problems = validate_drift_cell(control, cell_ids=["forest", "forest_control"],
                                   reference=ref_cell(), pair=shifted)
    assert any("is not a control" in p for p in problems)


def test_a_control_without_a_pair_is_reported():
    orphan = cell("control", role="no_shift_control", moved={}, pair="nowhere")
    problems = validate_drift_cell(orphan, cell_ids=["control"], reference=ref_cell(),
                                   pair=None)
    assert any("must name the cell it resamples" in p for p in problems)


def test_paired_cell_must_exist():
    bad = cell(pair="ghost")
    assert any("is not among the cells" in p
               for p in validate_drift_cell(bad, cell_ids=["c"]))


def test_contributor_counts_must_sum_to_n_images():
    bad = cell(n=100, counts={"lab_alpha": 60})
    assert any("contributor_counts sum to 60" in p for p in validate_drift_cell(bad))


def test_contributor_counts_must_match_the_contributor_list():
    bad = cell(n=100, counts={"lab_alpha": 50, "lab_beta": 50})
    bad["contributors"] = ["lab_alpha"]
    assert any("do not match contributors" in p for p in validate_drift_cell(bad))


def test_image_shape_must_agree_with_n_images():
    bad = cell(n=100)
    bad["image_shape"] = [99, 8, 8, 3]
    assert any("image_shape[0]" in p for p in validate_drift_cell(bad))


def test_digests_must_be_sha256():
    bad = cell()
    bad["digests"] = {"dataset": "short", "pixels_uint8_sha256": "b" * 64}
    assert any("digests['dataset']" in p for p in validate_drift_cell(bad))


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def _features(mean, n=80, seed=0, dim=8):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, 1.0, size=(n, dim))


def test_identical_distributions_are_not_drift():
    """Same array three times: p=1 and every metric at or below its floor."""
    ref = _features(0.0)
    out = evaluate_contrast(ref, ref.copy(), control=ref.copy(), metrics=["mmd2_rbf"])
    entry = out["metrics"]["mmd2_rbf"]
    assert entry["decision"] == "no_drift" and entry["p_value"] > 0.05


def test_a_large_shift_clears_the_floor_and_is_called_drift():
    ref = _features(0.0)
    op = _features(3.0, seed=1)
    control = _features(0.0, seed=2)      # an honest resample of the same domain
    out = evaluate_contrast(ref, op, control=control, metrics=["mmd2_rbf",
                                                               "wasserstein_1_mean"])
    for name in ("mmd2_rbf", "wasserstein_1_mean"):
        assert out["metrics"][name]["decision"] == "drift", name
        assert out["metrics"][name]["margin_over_floor"] > 0.0


def test_a_shift_smaller_than_the_same_metrics_noise_is_under_determined():
    """The whole point: a significant p-value is not a finding."""
    ref = _features(0.0, n=120)
    op = _features(0.02, n=120, seed=3)        # tiny shift, p will be large or small
    control = _features(0.5, n=120, seed=4)    # control is MORE shifted than the cell
    out = evaluate_contrast(ref, op, control=control, metrics=["wasserstein_1_mean"])
    entry = out["metrics"]["wasserstein_1_mean"]
    assert entry["decision"] in ("under_determined", "no_drift")
    if entry["decision"] == "under_determined":
        assert "does not exceed the noise floor" in entry["reason"] \
            or "contradictory evidence" in entry["reason"]


def test_without_a_control_the_decision_is_under_determined_however_significant():
    ref = _features(0.0)
    op = _features(3.0, seed=1)
    out = evaluate_contrast(ref, op, control=None, metrics=["mmd2_rbf"])
    entry = out["metrics"]["mmd2_rbf"]
    assert entry["p_value"] <= 0.05 and entry["decision"] == "under_determined"
    assert "no no-shift control" in entry["reason"]
    assert entry["control_value"] is None
    # the self-null is still measurable; it is just not enough to attribute anything
    assert entry["noise_floor"] is not None


def test_too_few_samples_is_reported_as_insufficient_not_as_a_decision():
    ref = _features(0.0, n=20)
    out = evaluate_contrast(ref, _features(3.0, n=20, seed=1), control=ref.copy(),
                            metrics=["mmd2_rbf"])
    assert out["metrics"]["mmd2_rbf"]["decision"] == "insufficient_n"


def test_a_saturated_metric_is_flagged():
    """ks_max is 1.0 on every contrast in the real corpus, so it cannot discriminate."""
    # disjoint supports: every dimension is perfectly separated, which is what
    # happens on the real corpus for every cell including the no-shift controls
    ref = _features(0.0, n=80)
    op = _features(50.0, n=80, seed=5)
    out = evaluate_contrast(ref, op, control=_features(0.0, n=80, seed=6),
                            metrics=["ks_max"])
    entry = out["metrics"]["ks_max"]
    assert entry["value"] == 1.0 and entry["saturated"] is True
    # and when the control is just as separable, saturation costs the metric the
    # decision instead of producing a finding -- which is what happens on the real
    # corpus, where every cell including every control reads exactly 1.0
    out2 = evaluate_contrast(ref, op, control=_features(50.0, n=80, seed=7),
                             metrics=["ks_max"])
    entry2 = out2["metrics"]["ks_max"]
    assert entry2["noise_floor"] == 1.0
    assert entry2["decision"] == "under_determined"


def test_the_self_null_is_the_noise_scale_of_the_metric():
    ref = _features(0.0, n=80)
    out = evaluate_contrast(ref, _features(0.2, n=80, seed=7),
                            control=_features(0.0, n=80, seed=8),
                            metrics=["mmd2_rbf"])
    assert abs(out["metrics"]["mmd2_rbf"]["self_null_value"]) < 0.2


def test_split_half_is_deterministic_and_disjoint():
    ref = _features(0.0, n=81)
    a, b = split_half(ref)
    assert len(a) == len(b) == 40
    assert not np.array_equal(a, b)
    a2, b2 = split_half(ref)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def test_permutation_pvalues_are_reproducible_for_a_seed():
    ref = _features(0.0, n=60, seed=11)
    op = _features(0.5, n=60, seed=12)
    first = evaluate_contrast(ref, op, control=_features(0.0, n=60, seed=13),
                              n_permutations=40, seed=99, metrics=["mmd2_rbf"])
    second = evaluate_contrast(ref, op, control=_features(0.0, n=60, seed=13),
                               n_permutations=40, seed=99, metrics=["mmd2_rbf"])
    assert first["metrics"]["mmd2_rbf"]["p_value"] == second["metrics"]["mmd2_rbf"]["p_value"]


def test_metric_subsetting_limits_what_is_computed():
    ref = _features(0.0)
    out = evaluate_contrast(ref, ref.copy(), control=ref.copy(), metrics=["ks_max"])
    assert set(out["metrics"]) == {"ks_max"}


# --------------------------------------------------------------------------- #
# end to end on a synthetic corpus
# --------------------------------------------------------------------------- #

def build_corpus(root, n=40, size=8):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    cells = [ref_cell()]
    for cid, shift in (("forest", 0.6), ("forest_control", 0.0)):
        scene = {**REF_SCENE, "terrain": "forest"} if shift else {**REF_SCENE,
                                                                  "terrain": "forest"}
        moved = {"terrain": "forest"} if shift else {}
        c = cell(cid, role="shifted" if shift else "no_shift_control", scene=scene,
                 moved=moved, pair="forest_control" if shift else "forest", n=n)
        c["image_shape"] = [n, size, size, 3]
        cells.append(c)
    for c in cells:
        d = os.path.join(root, c["cell_id"])
        os.makedirs(d, exist_ok=True)
        base = 0.0 if c["role"] == "reference" or c["cell_id"].endswith("control") else shift
        images = np.clip(rng.normal(base, 0.1, size=(n, size, size, 3)) * 255, 0,
                         255).astype(np.uint8)
        np.savez_compressed(os.path.join(d, "images_uint8.npz"), images=images)
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": "cviaf.drift-cells/1", "cells": cells}, fh)
    return root


def test_load_cells_reports_a_missing_index(tmp_path):
    cells, problems = load_cells(str(tmp_path))
    assert cells == [] and "no cell index" in problems[0]


def test_load_cells_reports_an_index_without_a_reference(tmp_path):
    root = os.path.join(str(tmp_path), "c")
    os.makedirs(root)
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"cells": [cell("blur", scene={**REF_SCENE, "sensor_blur": 3},
                                  moved={"sensor_blur": 3})]}, fh)
    _, problems = load_cells(root)
    assert any("no cell has role 'reference'" in p for p in problems)


def test_run_produces_a_report_with_definitions_and_null_discipline(tmp_path):
    root = build_corpus(os.path.join(str(tmp_path), "cells"))
    report = run(root, max_images=40, n_permutations=25)
    assert report["schema"] == HARNESS_SCHEMA and report["status"] == "skeleton"
    assert report["reference_cell"] == "reference_desert_summer_clean"
    assert set(report["metric_definitions"]) == set(METRICS)
    assert report["attribution"]["verdict"] == "under_determined"
    assert set(report["per_cell"]) == {"forest", "forest_control"}
    for cell_report in report["per_cell"].values():
        for name, entry in cell_report["metrics"].items():
            assert entry["decision"] in DECISIONS
            assert entry["unit"] == METRICS[name]["unit"]
    text = render(report)
    assert "drift harness" in text and "attribution: under_determined" in text


def test_run_refuses_an_unknown_metric(tmp_path):
    root = build_corpus(os.path.join(str(tmp_path), "cells"))
    with pytest.raises(ValueError, match="unknown metric"):
        run(root, max_images=16, n_permutations=10, metrics=["mmd2_rbf", "vibes"])


def test_run_refuses_a_corpus_whose_cells_do_not_validate(tmp_path):
    root = build_corpus(os.path.join(str(tmp_path), "cells"))
    with open(os.path.join(root, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    index["cells"][1]["declared_axes_moved"] = {"terrain": "swamp"}
    index["cells"][1]["scene_spec"]["terrain"] = "forest"
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    with pytest.raises(ValueError, match="manifest problems"):
        run(root, max_images=16, n_permutations=10)


def test_cli_writes_a_report_and_checks_exit_codes(tmp_path, capsys):
    root = build_corpus(os.path.join(str(tmp_path), "cells"))
    out = os.path.join(str(tmp_path), "report.json")
    code = main(["--corpus", root, "--json", out, "--max-images", "40",
                 "--permutations", "25"])
    assert code == 0
    capsys.readouterr()
    written = json.load(open(out, encoding="utf-8"))
    assert written["schema"] == HARNESS_SCHEMA
    assert main(["--corpus", os.path.join(str(tmp_path), "nope")]) == 2
    capsys.readouterr()


def test_the_committed_corpus_validates(monkeypatch):
    corpus = os.path.join(os.path.dirname(__file__), "..", "runs", "drift_cells")
    if not os.path.isdir(corpus):
        pytest.skip("runs/drift_cells not present")
    cells, problems = load_cells(corpus)
    assert cells and problems == [], problems
