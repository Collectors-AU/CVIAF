"""Lab replay capsules. HMAC is a lab transport seal, NOT non-repudiation.

The audit key and enrolled model must be kept outside the inference service. The
service may hold the record seal key; replay tests its claims, not its honesty.
"""
import hashlib
import hmac
import json
import math

import numpy as np


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def output_view(prediction):
    """Stable detections; preserve unrounded values to compare replay tolerance."""
    rows = [[int(l), *[float(v) for v in b], float(s)]
            for b, s, l in zip(prediction['boxes'], prediction['scores'], prediction['labels'])]
    return sorted(rows, key=lambda row: (row[0], *row[1:],))


def quantize(rows, spec):
    box, score = spec['box_step'], spec['score_step']
    if box <= 0 or score <= 0 or not all(math.isfinite(x) for row in rows for x in row):
        raise ValueError('invalid quantization or non-finite output')
    return [[row[0], *[int(math.floor(v / box + 0.5)) for v in row[1:5]],
             int(math.floor(row[5] / score + 0.5))] for row in rows]


def seal_record(image, prediction, model_digest, config, environment, record_id,
                seal_key, quantization=None):
    """The image is external to this capsule; the verifier resolves it by ID."""
    spec = quantization or {'box_step': 1e-3, 'score_step': 1e-4}
    rows = output_view(prediction)
    record = {'version': 1, 'record_id': str(record_id), 'input_sha256':
              hashlib.sha256(np.ascontiguousarray(image, dtype=np.float32).tobytes()).hexdigest(),
              'model_digest': model_digest, 'config': config, 'config_sha256': digest(config),
              'environment': environment, 'quantization': spec,
              'output': rows, 'output_sha256': digest(quantize(rows, spec))}
    record['seal'] = hmac.new(seal_key, canonical(record), hashlib.sha256).hexdigest()
    return record
