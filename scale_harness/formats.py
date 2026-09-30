"""Cross-format image/box parity, with explicit COCO-to-YOLO class mapping."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from cviaf.formats import load_dataset


def parity(coco_json, coco_images, yolo_root, coco_to_yolo, box_tolerance=1e-4):
    coco = load_dataset(str(coco_json), format='coco', images_dir=str(coco_images))
    yolo = load_dataset(str(yolo_root), format='yolo')
    if len(coco) != len(yolo):
        raise AssertionError(f'image counts differ: COCO={len(coco)} YOLO={len(yolo)}')
    by_stem = {Path(s.image_path).stem:s for s in yolo}
    if len(by_stem) != len(yolo):
        raise AssertionError('duplicate YOLO image stems')
    result = {}
    for source in coco:
        stem = Path(source.image_path).stem
        if stem not in by_stem:
            raise AssertionError(f'missing YOLO image {stem}')
        target = by_stem[stem]
        with Image.open(source.image_path) as im:
            source_pixels = np.asarray(im.convert('RGB'))
        with Image.open(target.image_path) as im:
            target_pixels = np.asarray(im.convert('RGB'))
        if not np.array_equal(source_pixels,target_pixels):
            raise AssertionError(f'image pixel mismatch {stem}')
        def canonical(sample, translate):
            return sorted((translate[b.class_id], *[float(x) for x in
                            (b.x_center,b.y_center,b.width,b.height)]) for b in sample.bboxes)
        try:
            a=canonical(source,coco_to_yolo); b=canonical(target,{i:i for i in target.labels})
        except KeyError as exc:
            raise AssertionError(f'unmapped COCO category {exc} in {stem}') from exc
        if len(a)!=len(b) or any(x[0]!=y[0] or not np.allclose(x[1:],y[1:],atol=box_tolerance,rtol=0)
                                     for x,y in zip(a,b)):
            raise AssertionError(f'annotation mismatch {stem}')
        result[stem]={'sha256_pixels':hashlib.sha256(source_pixels.tobytes()).hexdigest(),
                      'boxes':len(a)}
    return result
