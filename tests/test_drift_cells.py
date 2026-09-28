"""Drift cells: the declarations a battery would rely on must be checkable.

The interesting failure here is a control that is not a control -- a resample whose
SceneSpec silently drifted from its cell's, which would turn a false-alarm bound into a
tautology. These tests fail on that, not on the numbers.
"""
import json
import os

import pytest

from cviaf.lab import drift_cells
from cviaf.lab.synth import SceneSpec

COMMITTED = os.path.join("runs", "drift_cells", "index.json")


def test_spec_for_moves_only_declared_axes():
    cell = drift_cells.cells()[2]
    spec = drift_cells.spec_for(cell, 0)
    assert isinstance(spec, SceneSpec)
    base = dict(drift_cells.REFERENCE)
    got = spec.to_dict()
    for axis in ("terrain", "season", "illumination", "gamma", "sensor_noise",
                 "sensor_blur", "objects_per_image", "seed"):
        if axis in cell["moved"]:
            assert got[axis] == cell["moved"][axis], axis
        else:
            assert got[axis] == base[axis], f"{axis} moved but was not declared"
    # every declared move names a real axis
    for axis in cell["moved"]:
        assert axis in base


def test_every_declared_cell_has_a_distinct_name_and_a_move():
    cells = drift_cells.cells()
    assert len(cells) == 5
    assert len({c["cell"] for c in cells}) == len(cells)
    for c in cells:
        assert c["moved"], f"{c['cell']} declares no moved axis"
        assert set(c["moved"]) <= {"terrain", "season", "illumination", "gamma",
                                  "sensor_noise", "sensor_blur"}


def test_seed_bands_do_not_collide():
    offsets = [drift_cells.REFERENCE_OFFSET]
    for i in range(5):
        offsets += [drift_cells.SHIFT_BASE + i * drift_cells.BAND,
                    drift_cells.CONTROL_BASE + i * drift_cells.BAND]
    assert len(set(offsets)) == len(offsets)
    assert min(offsets) > 700_000


@pytest.mark.skipif(not os.path.isfile(COMMITTED), reason="drift cells not generated")
def test_committed_cells_are_consistent_and_paired():
    st = drift_cells.status(os.path.dirname(COMMITTED))
    assert st["problems"] == [], st["problems"]
    assert st["consistent"]
    assert st["n_shifted"] == 5 and st["n_controls"] == 5
    assert st["all_digests_unique"]
    index = json.load(open(COMMITTED))
    by_id = {c["cell_id"]: c for c in index["cells"]}
    for cell in index["cells"]:
        if cell["role"] != "shifted":
            continue
        control = by_id[cell["paired_cell"]]
        assert control["scene_spec"] == cell["scene_spec"]
        assert control["seed_offset"] != cell["seed_offset"]
        assert control["declared_axes_moved"] == {}
