"""A generated natural-drift calibration must not be silently accepted if edited."""
import json
import pytest
from cviaf.drift.attribution import NaturalDriftCalibration


def test_natural_drift_calibration_digest_roundtrip_and_tamper(tmp_path):
    cal = NaturalDriftCalibration(nulls={"dimension_concentration": [.1, .2]},
                                   scenario_descriptions=["clean:shift"], seed=7)
    path = tmp_path / "drift_calibration.json"
    cal.save(str(path))
    assert NaturalDriftCalibration.load(str(path)).digest() == cal.digest()
    data = json.loads(path.read_text())
    data["nulls"]["dimension_concentration"][0] = .9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tampered"):
        NaturalDriftCalibration.load(str(path))
