import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scale_harness.core import (Detections, DecodedAdapter, freeze_manifest,
                                check_export_parity, gate_asr)


def boxes(score=.9):
    return Detections(np.array([[1., 1., 4., 4.]], dtype=np.float32),
                      np.array([score]), np.array([2], dtype=np.int64))

class Static:
    def __init__(self, changed=False): self.changed = changed
    def predict(self, image): return boxes(.9 if (self.changed and image[0,0,0]) else .1)

class HarnessTests(unittest.TestCase):
    def test_manifest_freeze_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            artifact = Path(d)/'model'; artifact.write_bytes(b'weights')
            m = dict(splits={k:[str(i)] for i,k in enumerate(('train','calibration','development','effect_gate','final'))},
                     artifacts={'model':str(artifact)}, artifact_digests={'model':hashlib.sha256(b'weights').hexdigest()})
            out = Path(d)/'manifest.json'
            self.assertEqual(freeze_manifest(m,out),freeze_manifest(m,out))
            m['splits']['final']=['0']
            with self.assertRaises(ValueError): freeze_manifest(m,out)
    def test_decoder_rejects_raw_logits_and_bad_boxes(self):
        img=np.zeros((6,6,3),np.uint8)
        a=DecodedAdapter(lambda batch:np.array([.1,.9]),lambda raw,shape:raw,lambda im:im)
        with self.assertRaises(TypeError): a.predict(img)
        b=DecodedAdapter(lambda batch:None,lambda raw,shape:Detections(np.array([[1,1,9,9]]),np.array([.9]),np.array([1])),lambda im:im)
        with self.assertRaises(ValueError): b.predict(img)
    def test_export_parity(self):
        image=np.zeros((6,6,3),np.uint8)
        self.assertEqual(check_export_parity(Static(),{'onnx':Static()},{'im':image}),{'onnx/im':1})
        with self.assertRaises(AssertionError): check_export_parity(Static(),{'onnx':Static(True)},{'im':np.ones_like(image)})
    def test_null_subtraction_and_unassessable(self):
        clean=np.zeros((6,6,3),np.uint8); trig=clean.copy(); trig[0,0,0]=1
        pairs={'a':(clean,trig),'b':(clean,trig)}
        success=lambda before,after:after.scores[0] > before.scores[0]+.5
        eligible=lambda det:True
        weak=gate_asr(Static(True),Static(True),pairs,eligible,success,floor=.5,min_eligible=2)
        self.assertEqual((weak['status'],weak['asr_net']),('weak',0.))
        good=gate_asr(Static(True),Static(),pairs,eligible,success,floor=.5,min_eligible=2)
        self.assertEqual((good['status'],good['asr_net']),('implanted',1.))
        self.assertEqual(gate_asr(Static(True),Static(),pairs,eligible,success,floor=.5,min_eligible=3)['status'],'unassessable')

if __name__=='__main__': unittest.main()

class FormatParityTests(unittest.TestCase):
    def test_coco_yolo_boxes_and_class_mapping(self):
        from PIL import Image
        from scale_harness.formats import parity
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); ci=root/'coco'; yi=root/'yolo'/'images'; yl=root/'yolo'/'labels'
            for x in (ci,yi,yl):x.mkdir(parents=True)
            pix=np.zeros((10,10,3),dtype=np.uint8)
            Image.fromarray(pix).save(ci/'a.png'); Image.fromarray(pix).save(yi/'a.png')
            (yl/'a.txt').write_text('0 0.5 0.5 0.4 0.4\n')
            ann={'images':[{'id':1,'file_name':'a.png','width':10,'height':10}],
                 'annotations':[{'image_id':1,'category_id':7,'bbox':[3,3,4,4]}],
                 'categories':[{'id':7,'name':'thing'}]}
            (root/'coco.json').write_text(json.dumps(ann))
            self.assertEqual(parity(root/'coco.json',ci,root/'yolo',{7:0})['a']['boxes'],1)
            with self.assertRaises(AssertionError):parity(root/'coco.json',ci,root/'yolo',{7:1})
