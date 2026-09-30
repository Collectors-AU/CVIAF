"""Live v4 bridge from the existing lab assurance run to planned detectors."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from cviaf.plugins import CalibrationRef, CostModel, Manifest, Registry, Result
from cviaf.planner import Context, plan, completeness
from cviaf.schema_v3 import validate_report
from cviaf.schema_v3.coverage import BASE_CELLS, generate_coverage

class LabProbe:
    """One actual detector-native AB-1 probe, calibrated on disjoint clean scenes.

    This is a demonstration calibration only: empirical p-values are not a measured
    attack power floor, and the matrix must not claim full coverage.
    """
    def __init__(self, name: str, keys: tuple[str,...], digest: str, seconds: float):
        self.name = name
        self.manifest = Manifest('model.trace.' + name, 'model_integrity', keys,
            frozenset({'model.black_box'}), CostModel(seconds=seconds),
            CalibrationRef('AB1.cal.clean', digest), 'p_value',
            frozenset({'LAB_NUMPY'}), frozenset({'AB1.probe'}),
            residual_risk='TinyDetector laboratory probe only; no attack-class power floor measured')

    def run(self, assets, battery_slice, budget):
        from cviaf.lab.detectors import trace_ctc, trace_ftc, make_backgrounds
        images = assets['images']; model = assets['model']
        cal = battery_slice['images']
        backgrounds = make_backgrounds(3, seed=7)
        def statistic(xs):
            if self.name == 'ctc':
                return trace_ctc(model, xs, backgrounds)['score']
            return trace_ftc(model, xs, stride=16)['score']
        # Compare means over equally sized clean calibration blocks, rather than
        # in-sample point scores. Calibration blocks are disjoint from the target.
        n = len(images)
        blocks = [cal[i:i+n] for i in range(0, len(cal)-n+1, n)]
        if len(blocks) < 2:
            return Result(self.manifest.id, 'skipped', reason_code='REFERENCE_MISSING',
                          human_reason='too few disjoint clean calibration blocks',
                          coverage_impact='no calibrated probe result')
        start = time.monotonic()
        observed = float(np.nanmean(statistic(images)))
        null = [float(np.nanmean(statistic(block))) for block in blocks]
        if not np.isfinite(observed) or not all(np.isfinite(null)):
            return Result(self.manifest.id, 'skipped', reason_code='NOT_APPLICABLE',
                          human_reason='no usable object detections in a probe block',
                          coverage_impact='probe cannot score this model')
        p_value = (1 + sum(x >= observed for x in null)) / (len(null)+1)
        return Result(self.manifest.id, 'ran', statistic=observed,
                      p_value=float(p_value), calibration_ref=self.manifest.calibration.digest,
                      cost_actual=time.monotonic()-start, evidence=(assets['evidence_pointer'],))

def run(corpus: str, model_id: str, out: str, access_level: str='black-box',
        threat_classes: set[str] | None=None, budget_seconds: float=180.) -> dict:
    """Run legacy engine and a fresh planned probe, then validate a separate v3 report."""
    from cviaf.lab.pipeline import assure_model
    from cviaf.lab.evaluate import load_registry, train_spec_from_manifest
    from cviaf.lab.train import ModelArtifact, build_splits
    from cviaf.core.types import AuditTrail, hash_bytes, hash_dict
    Path(out).mkdir(parents=True, exist_ok=True)
    entry = next((e for e in load_registry(corpus) if e['manifest']['model_id'] == model_id), None)
    if entry is None: raise ValueError('model not found in corpus')
    spec = train_spec_from_manifest(entry['manifest'])
    splits = build_splits(spec)
    # Only unmodified holdout images. No attack kind, poison indices, ASR, or
    # triggered examples flow into the detector or calibration.
    cal = np.asarray(splits.cal_clean.images, dtype=np.float32)
    target = np.asarray(splits.eval_clean.images[:8], dtype=np.float32)
    calibration_digest = hashlib.sha256(cal.tobytes()).hexdigest()
    checks = Registry()
    checks.register(LabProbe('ctc', ('poisoning.backdoor.badet.OGA','poisoning.backdoor.badet.RMA',
                                     'poisoning.backdoor.badet.GMA'), calibration_digest, 55))
    checks.register(LabProbe('ftc', ('poisoning.backdoor.badet.ODA',), calibration_digest, 75))
    classes = threat_classes or set(BASE_CELLS)
    available = {'model.black_box','dataset','ref_dist','log','ops'}
    access = set(available)
    if access_level == 'white-box': access.add('model.white_box'); available.add('model.white_box')
    ctx = Context(frozenset(classes), frozenset(access), frozenset({'LAB_NUMPY','COCO','YOLO'}),
                  frozenset(available), frozenset({'AB1.cal.clean','AB1.probe'}), budget_seconds)
    p = plan(checks.manifests(), ctx)
    legacy = assure_model(corpus, model_id=model_id, out_dir=os.path.join(out,'legacy'),
                          access_level=access_level, log=lambda _: None)
    evidence_path = os.path.join(out, 'legacy', 'assurance_report.json')
    evidence_pointer = 'file:' + os.path.abspath(evidence_path)
    results = checks.execute(p.selected, {'images':target,'model':ModelArtifact.load(entry['dir']).model,
                                            'evidence_pointer': evidence_pointer},
                              {'AB1.cal.clean': {'images':cal}}, {})
    complete = completeness(p, results)
    matrix = generate_coverage(p, checks.manifests(), results)
    # Planned execution is not sufficient to accept: without measured floors every
    # asserted attack class remains partial, even if all planned checks ran.
    gaps = [r['attack_class'] for r in matrix if r['status'] not in ('covered', 'not_applicable')]
    complete['blocking_skips'] = sorted(set(complete['blocking_skips'] +
        [f'{k}: {next(row["status"] for row in matrix if row["attack_class"] == k)}' for k in gaps]))
    complete['accept_permitted'] = bool(complete['accept_permitted'] and not gaps)
    v4_audit = AuditTrail()
    v4_audit.append('plan_generated','assurance_planner',details=p.to_dict(),
                    input_hash=hash_dict({'calibration_digest':calibration_digest,'model_id':model_id}),
                    output_hash=hash_dict(p.to_dict()))
    v4_audit.append('detectors_executed','plugin_registry',
                    details={'results':[vars(r) for r in results], 'coverage_statuses':
                             {r['attack_class']:r['status'] for r in matrix}},
                    input_hash=hash_dict(p.to_dict()),output_hash=hash_dict(complete))
    assert v4_audit.verify_chain()[0]
    # Full legacy findings are preserved, explicitly uncalibrated. A v2 float
    # confidence must never be silently relabelled a v3 conformal confidence.
    v2 = legacy['report'].to_dict()
    v3 = dict(v2)
    v3.update(schema_version='3.0', framework_version='3.0.0',
        threat_model_declared={'attack_classes_in_scope':sorted(classes),
            'access_granted':{'dataset':'full','model':access_level,'log':'present'},
            'assumptions':['lab-generated holdout serves as clean calibration; not field validation',
                           'the null blocks are clean and disjoint from the target'],
            'declared_by':'operator','policy_ref':'AB1.policy:unavailable'},
        battery={'id':'AB-1-demo','digest':calibration_digest,
                 'slices_used':['AB1.cal.clean','AB1.probe']},
        plan=p.to_dict(), assessment_completeness=complete,
        hypothesis_posteriors=[], findings=[], legacy_findings=v2['findings'],
        coverage_matrix=matrix, loss_matrix_used={},
        overall_disposition=('review' if v2['overall_disposition']=='accept' else v2['overall_disposition']),
        detector_results=[vars(r) for r in results],
        planner_audit_trail=v4_audit.to_dict(),
        limitations=v2['limitations']+['v2 findings remain uncalibrated in legacy_findings; no v3 findings asserted',
                                      'No measured attack-class detection floors; all ran cells remain partial.'])
    errors = validate_report(v3)
    if errors: raise ValueError('invalid v3 report: ' + '; '.join(errors[:8]))
    dest = Path(out) / 'assurance_report_v3.json'
    dest.write_text(json.dumps(v3, indent=2, default=str)+'\n')
    (Path(out)/'coverage_matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    (Path(out)/'planner_audit_trail.json').write_text(json.dumps(v4_audit.to_dict(),indent=2)+'\n')
    return {'report_path':str(dest), 'selected':p.selected,
            'results':[(r.detector,r.status,r.p_value,r.reason_code) for r in results],
            'completeness':complete, 'legacy_disposition':v2['overall_disposition'],
            'disposition':v3['overall_disposition'], 'coverage_statuses':{r['attack_class']:r['status'] for r in matrix}}

def main():
    parser=argparse.ArgumentParser(description='Run planned v4 lab assurance without replacing v2 output')
    parser.add_argument('--corpus', default='runs/mvp')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--access-level',choices=('black-box','white-box'),default='black-box')
    parser.add_argument('--budget-seconds', type=float, default=180)
    parser.add_argument('--attack-class', action='append',dest='classes')
    args=parser.parse_args()
    print(json.dumps(run(args.corpus,args.model,args.out,args.access_level,
                         set(args.classes) if args.classes else None,args.budget_seconds),indent=2))
