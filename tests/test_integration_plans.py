"""Tests for the combined plan (`scripts/integration.py merge-plans`).

The combined plan is the only place where the whole population is described at once, so
these tests are about the contradictions that would otherwise be averaged into it:

  * the same detector seed in two sources, silently deduped to make the counts add up;
  * the same model id twice;
  * the same shard id twice - box 7's plan directory shipped with a copy of box 8's
    shard inside it, which is exactly this bug in the wild;
  * a shard that still contains the reference model (it must never score itself);
  * a shard that still contains a model this pass already excluded;
  * a shard whose bytes changed on the way into the plan, which would break the worker's
    checkpoint identity without breaking anything visibly.
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
    PLAN_SCHEMA,
    PlanError,
    merge_plans,
    render_plan,
)


def write_shard(path, shard_id, seed_min, count, worker="b1w1", start=None, prefix=""):
    seeds = list(range(seed_min if start is None else start, (seed_min if start is None else start) + count))
    shard = {
        "schema": "cviaf.analysis-shard.v1",
        "shard_id": shard_id,
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "count": count,
        "preferred_box": None,
        "local_count": 0,
        "models": [
            {
                "model_id": f"{prefix}clean_none_fixed_s{seed}",
                "seed": seed,
                "path": f"clean_null_lab_{worker}/{prefix}clean_none_fixed_s{seed}",
                "manifest_sha256": "a" * 64,
                "weights_sha256": "b" * 64,
            }
            for seed in seeds
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(shard), encoding="utf-8")
    return Path(path)


def test_combined_plan_refuses_two_sources_that_share_a_detector_seed(tmp_path):
    """Two sources, one seed: the plan must stop, not keep the first and move on."""
    # different ids, same detector seed: a re-trained model wearing another box's seed
    first = write_shard(tmp_path / "b1" / "shard.json", "seed_60152_60154", 60152, 3)
    second = write_shard(
        tmp_path / "b3" / "shard.json", "seed_60154_60156", 60154, 3, prefix="retrain_"
    )

    with pytest.raises(PlanError) as excinfo:
        merge_plans([("box1", first), ("box3", second)], tmp_path / "out")

    assert "duplicate detector seed 60154" in str(excinfo.value)
    assert "refusing to dedupe" in str(excinfo.value)


def test_combined_plan_copies_shard_bytes_unchanged(tmp_path):
    """The worker hashed these bytes; the merge must validate against the same bytes."""
    shard = write_shard(tmp_path / "b1" / "shard.json", "seed_60152_60154", 60152, 3)

    plan = merge_plans([("box1", shard)], tmp_path / "out")

    copied = tmp_path / "out" / "shards" / "seed_60152_60154.json"
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    assert plan["sources"][0]["shard_sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()


def test_combined_plan_refuses_a_shard_that_still_contains_the_reference_model(tmp_path):
    """The reference model cannot be in any scored shard, not even one."""
    shard = write_shard(tmp_path / "fleet" / "shard.json", "seed_150_152", 150, 3)
    data = json.loads(shard.read_text())
    data["models"][0]["model_id"] = "clean_none_fixed_s5800"
    shard.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PlanError) as excinfo:
        merge_plans([("fleet6250", shard)], tmp_path / "out", reference="clean_none_fixed_s5800")

    assert "reference model" in str(excinfo.value)


def test_combined_plan_refuses_a_shard_that_still_contains_an_excluded_model(tmp_path):
    """A model this pass excluded (undefined CTC) must not re-enter through a shard."""
    shard = write_shard(tmp_path / "b5" / "shard.json", "seed_80152_80154", 80152, 3)

    with pytest.raises(PlanError) as excinfo:
        merge_plans(
            [("box5", shard)], tmp_path / "out", excluded=["clean_none_fixed_s80153"]
        )

    assert "still contains excluded model clean_none_fixed_s80153" in str(excinfo.value)


def test_combined_plan_refuses_a_duplicate_shard_id(tmp_path):
    """box-plan7 shipped a copy of box 8's shard; the plan must not count it twice."""
    first = write_shard(tmp_path / "b7" / "shard.json", "seed_95152_95154", 95152, 3)
    second = write_shard(tmp_path / "b8" / "shard.json", "seed_95152_95154", 95152, 3)

    with pytest.raises(PlanError) as excinfo:
        merge_plans([("box7", first), ("box8", second)], tmp_path / "out")

    assert "duplicate shard id seed_95152_95154" in str(excinfo.value)


def test_combined_plan_keeps_each_source_its_own_shard_set(tmp_path):
    """Seven lab boxes plus the fleet stay seven plus one, not one merged pool."""
    specs = []
    for index in range(3):
        start = 60152 + index * 10
        specs.append(
            (
                f"box{index + 1}",
                write_shard(
                    tmp_path / f"b{index + 1}" / "shard.json",
                    f"seed_{start}_{start + 9}",
                    start,
                    10,
                ),
            )
        )
    specs.append(("fleet6250", write_shard(tmp_path / "fleet" / "shard.json", "seed_150_159", 150, 10)))

    plan = merge_plans(specs, tmp_path / "out")

    assert len(plan["shards"]) == 4
    assert plan["corpus_count"] == 40
    assert [s["id"] for s in plan["sources"]] == ["box1", "box2", "box3", "fleet6250"]
    assert (tmp_path / "out" / "shards").is_dir()
    assert len(list((tmp_path / "out" / "shards").glob("*.json"))) == 4
    assert plan["schema"] == PLAN_SCHEMA


def test_combined_plan_refuses_a_shard_whose_bounds_disagree_with_its_models(tmp_path):
    """A shard that claims seeds 60152..65151 but holds three of them is a lie."""
    shard = write_shard(tmp_path / "b1" / "shard.json", "seed_60152_60154", 60152, 3)
    data = json.loads(shard.read_text())
    data["seed_max"] = 65151
    shard.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PlanError) as excinfo:
        merge_plans([("box1", shard)], tmp_path / "out")

    assert "do not match the models" in str(excinfo.value)


def test_combined_plan_refuses_a_non_empty_output_directory(tmp_path):
    """A stale plan directory is how last week's shards get merged with this week's."""
    shard = write_shard(tmp_path / "b1" / "shard.json", "seed_60152_60154", 60152, 3)
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.json").write_text("{}")

    with pytest.raises(PlanError) as excinfo:
        merge_plans([("box1", shard)], out)

    assert "not empty" in str(excinfo.value)


def test_combined_plan_records_exclusions(tmp_path):
    """The reference exclusion is part of the plan, not an unstated convention."""
    shard = write_shard(tmp_path / "fleet" / "shard.json", "seed_150_152", 150, 3)

    plan = merge_plans(
        [("fleet6250", shard)],
        tmp_path / "out",
        excluded=["clean_none_fixed_s5800"],
        reference="clean_none_fixed_s5800",
    )

    assert plan["excluded_model_ids"] == ["clean_none_fixed_s5800"]
    assert plan["unique_detector_seeds"] == 3


def test_render_plan_lists_every_source_and_the_total(tmp_path):
    shard = write_shard(tmp_path / "b1" / "shard.json", "seed_60152_60154", 60152, 3)
    plan = merge_plans([("box1", shard)], tmp_path / "out")
    text = render_plan(plan)

    assert "box1" in text
    assert "corpus_count=3" in text
    assert "shards=1" in text
