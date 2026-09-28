"""Standalone verifier: does not import the writer. Run in an isolated process.

This lab adapter accepts a callback that loads an enrolled immutable model and a
callback that retrieves input by record ID. Production must provide a trusted
DSSE verification and independently provisioned enrolled model and audit key.
"""
import hashlib
import hmac
import json
import math
import random

import numpy as np


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _quantized(rows, spec):
    b, s = spec['box_step'], spec['score_step']
    if b <= 0 or s <= 0:
        raise ValueError('invalid tolerance')
    return [[int(r[0]), *[int(math.floor(float(v) / b + .5)) for v in r[1:5]],
             int(math.floor(float(r[5]) / s + .5))] for r in rows]


def sample_size(q, miss_probability, n):
    if not (0 < q < 1 and 0 < miss_probability < 1 and n > 0):
        raise ValueError('q, miss_probability and n must be in range')
    return min(n, math.ceil(math.log(miss_probability) / math.log1p(-q)))


def verify(records, load_enrolled_model, load_input, seal_key, audit_key,
           q=.05, miss_probability=.05, audit_seed=None, audit_output_key=None):
    """Audit random records, with seed committed after ledger is frozen.

    The independent audit_seed must be secret until records are frozen. Without
    one, the audit key derives a reproducible unpredictable seed from the ledger.
    The returned audit seal uses its own key, never the inference seal key.
    """
    records = list(records)
    n = len(records)
    k = sample_size(q, miss_probability, n)
    seen = set()
    for r in records:
        rid = r['record_id']
        if rid in seen:
            raise ValueError('duplicate record ID')
        seen.add(rid)
        signed = {key: val for key, val in r.items() if key != 'seal'}
        if not hmac.compare_digest(r['seal'], hmac.new(seal_key, _bytes(signed), hashlib.sha256).hexdigest()):
            raise ValueError('invalid record seal')
        if r['config_sha256'] != _hash(r['config']) or r['output_sha256'] != _hash(_quantized(r['output'], r['quantization'])):
            raise ValueError('invalid signed field digest')
        if r['version'] != 1:
            raise ValueError('unsupported capsule version')
    ledger_hash = hashlib.sha256(_bytes(records)).hexdigest()
    seed_material = (audit_seed if audit_seed is not None else
                     hmac.new(audit_key, ledger_hash.encode(), hashlib.sha256).digest())
    seed = int.from_bytes(hashlib.sha256(seed_material + ledger_hash.encode()).digest(), 'big')
    indices = sorted(random.Random(seed).sample(range(n), k))
    findings = []
    for i in indices:
        r = records[i]
        model = load_enrolled_model(r['model_digest'])
        if model.digest() != r['model_digest']:
            raise ValueError('enrolled model digest mismatch')
        image = np.asarray(load_input(r['record_id']), np.float32)
        if hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != r['input_sha256']:
            raise ValueError('input digest mismatch')
        cfg = r['config']
        out = model.predict(image, **cfg)
        actual = sorted([[int(l), *[float(v) for v in b], float(s)]
                         for b, s, l in zip(out['boxes'], out['scores'], out['labels'])],
                        key=lambda row: (row[0], *row[1:],))
        claimed = r['output']
        tol = r['quantization']
        equal = len(actual) == len(claimed) and all(
            a[0] == b[0] and all(abs(x - y) <= tol['box_step'] for x, y in zip(a[1:5], b[1:5]))
            and abs(a[5] - b[5]) <= tol['score_step'] for a, b in zip(actual, claimed))
        if not equal:
            findings.append(r['record_id'])
    result = {'version': 1, 'ledger_sha256': ledger_hash, 'seed_commitment':
              hashlib.sha256(seed_material).hexdigest(), 'n': n, 'k': k,
              'audited_fraction': k / n, 'q': q, 'miss_probability_bound': (1-q)**k,
              'sampled_indices': indices, 'mismatched_record_ids': findings,
              'status': 'fail' if findings else 'pass',
              'limitation': 'sampling claim assumes unpredictable audit choice and at least q fraction forged'}
    if audit_output_key is not None:
        result['audit_seal'] = hmac.new(audit_output_key, _bytes(result), hashlib.sha256).hexdigest()
    return result
