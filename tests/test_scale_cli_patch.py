"""Safety regressions for the legacy real-data assess route."""
import argparse
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from cviaf.cli import cmd_assess
from cviaf.formats import DatasetSample


def args(path):
    return argparse.Namespace(dataset='fixture',format='auto',contributor='x',images_dir='',
            image_size=None,model=path,access_level='black-box',output='/tmp/unused',
            pipeline_id='',skip='')

class CLIPatchTests(unittest.TestCase):
    def test_multi_box_refused_before_model_load(self):
        with patch('cviaf.formats.load_dataset',return_value=[DatasetSample('a',labels=[1,2])]):
            with self.assertRaisesRegex(ValueError,'exactly one label'):cmd_assess(args('missing'))
    def test_missing_image_fails_instead_of_silent_skip(self):
        with patch('cviaf.formats.load_dataset',return_value=[DatasetSample('/missing/cviaf.jpg',labels=[1])]):
            with self.assertRaises(FileNotFoundError):cmd_assess(args('missing'))
    def test_load_error_propagates_no_report(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.png'; Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(path)
            sample=DatasetSample(str(path),labels=[1])
            with patch('cviaf.formats.load_dataset',return_value=[sample]), \
                 patch('cviaf.formats.model_loader.load_model',side_effect=ValueError('bad checkpoint')):
                with self.assertRaisesRegex(RuntimeError,'bad checkpoint'):cmd_assess(args(str(path)))

if __name__=='__main__':unittest.main()
