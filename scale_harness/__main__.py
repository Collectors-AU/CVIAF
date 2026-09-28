"""Mac-friendly smoke commands; model-specific decoders remain explicit Python code."""
import argparse
import json
from pathlib import Path
from .core import freeze_manifest
from .formats import parity

p=argparse.ArgumentParser()
s=p.add_subparsers(dest='action',required=True)
a=s.add_parser('freeze'); a.add_argument('manifest'); a.add_argument('out')
a=s.add_parser('format-parity'); a.add_argument('coco_json'); a.add_argument('coco_images'); a.add_argument('yolo_root'); a.add_argument('class_map_json')
args=p.parse_args()
if args.action=='freeze':
    print(json.dumps({'manifest_sha256':freeze_manifest(json.loads(Path(args.manifest).read_text()),args.out)}))
else:
    result=parity(args.coco_json,args.coco_images,args.yolo_root,
                  {int(k):int(v) for k,v in json.loads(Path(args.class_map_json).read_text()).items()})
    print(json.dumps({'image_count':len(result),'per_image':result},indent=2))
