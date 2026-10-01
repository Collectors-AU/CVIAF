"""A partial assessment must not read as a clean pipeline."""
import argparse
from unittest.mock import patch

import pytest

from cviaf.cli import cmd_assess
from cviaf.governance import GovernanceEngine
from cviaf.orchestrator import CVIAFOrchestrator


def args(**kwargs):
    defaults = dict(dataset=None, model=None, format="auto", contributor="unknown",
                    images_dir="", output=None, access_level="white-box", skip="",
                    pipeline_id="")
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_empty_findings_require_review():
    assert GovernanceEngine().compute_overall_risk() == ("LOW", "review")


def test_assess_requires_dataset_and_model():
    with pytest.raises(ValueError, match="requires both"):
        cmd_assess(args())
    with pytest.raises(ValueError, match="requires both"):
        cmd_assess(args(dataset="some.json"))


def test_missing_pillow_rejected_without_placeholders(tmp_path):
    from cviaf.formats import DatasetSample
    with patch("cviaf.formats.load_dataset", return_value=[DatasetSample(image_path="x", labels=[1])]):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with pytest.raises(RuntimeError, match="Pillow is required"):
                cmd_assess(args(dataset="d.json", model="m.onnx"))


def test_missing_images_rejected(tmp_path):
    from cviaf.formats import DatasetSample
    with patch("cviaf.formats.load_dataset", return_value=[DatasetSample(image_path="missing.png", labels=[1])]):
        with pytest.raises(FileNotFoundError, match="Incomplete assessment"):
            cmd_assess(args(dataset="d.json", model="m.onnx"))


def test_model_load_failure_rejected(tmp_path):
    from PIL import Image
    img = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(img)
    from cviaf.formats import DatasetSample
    with patch("cviaf.formats.load_dataset", return_value=[DatasetSample(image_path=str(img), labels=[1])]):
        with patch("cviaf.formats.model_loader.load_model", side_effect=ValueError("bad model")):
            with pytest.raises(RuntimeError, match="Could not load requested model"):
                cmd_assess(args(dataset="d.json", model="bad.onnx"))


def test_orchestrator_does_not_accept_unassessed_pipeline(tmp_path):
    report = CVIAFOrchestrator(output_dir=str(tmp_path)).run_full_assessment()
    assert report.overall_disposition == "review"
    assert set(report.metadata["incomplete_assessments"]) == {"data", "model", "provenance", "drift"}
