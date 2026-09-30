"""Run-generated matrix keyed to NIST AML family and BadDet codes.

Coverage status is evidence-driven, not a static promise of detector power.
"""
from __future__ import annotations
from typing import Iterable
from cviaf.plugins import Manifest, Result
from cviaf.planner import Plan

# External taxonomy reference is provenance, not a claim of certified alignment.
TAXONOMY_REF = 'https://csrc.nist.gov/pubs/ai/100/2/e2025/final'
# Supported key vocabulary can be extended by callers without core edits.
BASE_CELLS = {
    'poisoning.backdoor.badet.OGA': ('model_integrity', 'NIST poisoning.backdoor / BadDet OGA'),
    'poisoning.backdoor.badet.RMA': ('model_integrity', 'NIST poisoning.backdoor / BadDet RMA'),
    'poisoning.backdoor.badet.GMA': ('model_integrity', 'NIST poisoning.backdoor / BadDet GMA'),
    'poisoning.backdoor.badet.ODA': ('model_integrity', 'NIST poisoning.backdoor / BadDet ODA'),
    'poisoning.backdoor.physical': ('model_integrity', 'NIST poisoning.backdoor / BadDet+'),
    'poisoning.availability.clean-label': ('data_integrity', 'NIST poisoning.availability'),
    'poisoning.targeted.label-flip': ('data_integrity', 'NIST poisoning.targeted'),
    'poisoning.availability.dup-flood': ('data_integrity', 'NIST poisoning.availability'),
    'poisoning.availability.ood-inject': ('data_integrity', 'NIST poisoning.availability'),
    'evasion.test-time': ('model_integrity', 'NIST evasion'),
    'privacy.*': ('governance', 'NIST privacy'),
    'supply-chain.model-substitution': ('provenance', 'model substitution'),
    'supply-chain.runtime-compromise': ('provenance', 'runtime compromise'),
    'supply-chain.tamper-log': ('provenance', 'tamper log'),
    'supply-chain.replay': ('provenance', 'replay'),
    'drift.covariate': ('drift', 'covariate shift'),
    'drift.concept': ('drift', 'concept shift'),
    'drift.vs-manipulation': ('drift', 'shift versus manipulation'),
}
# Never mark these fully covered by a single detector run.
PARTIAL_CEILINGS = {'poisoning.backdoor.physical', 'poisoning.availability.clean-label',
                    'drift.vs-manipulation'}
DESIGN_EXCLUSIONS = {'evasion.test-time': 'Integrity assessment does not certify per-input robustness',
                     'privacy.*': 'Privacy assessment outside this integrity threat model'}

def generate_coverage(plan: Plan, manifests: Iterable[Manifest], results: Iterable[Result],
                      detection_floors: dict[str, dict] | None = None,
                      calibration_sets: dict[str, str] | None = None,
                      evidence_pointers: dict[str, str] | None = None) -> list[dict]:
    """Produce one cell per declared key. Unexecuted checks get no coverage credit.

    `detection_floors` keyed by detector; missing measured power/floor caps to partial.
    A floor has metric, value, power and alpha; a mere point estimate cannot certify coverage.
    """
    by_id = {m.id: m for m in manifests}
    runs = tuple(results)
    if any(i not in by_id for i in plan.selected):
        raise ValueError('missing selected detector manifest')
    if len({r.detector for r in runs}) != len(runs) or {r.detector for r in runs} != set(plan.selected):
        raise ValueError('results do not match selected plan')
    floors = detection_floors or {}
    calibration_sets = calibration_sets or {}
    evidence_pointers = evidence_pointers or {}
    rows = []
    for key in plan.attack_classes_in_scope:
        candidate = [by_id[i] for i in plan.selected if i in by_id and key in by_id[i].attack_classes]
        ran = [m for m in candidate if next(r for r in runs if r.detector == m.id).status == 'ran']
        floor = [(m.id, floors.get(m.id) or next(r for r in runs if r.detector == m.id).detection_floor) for m in ran]
        measured = [(i, f) for i, f in floor if isinstance(f, dict) and {'metric', 'value', 'power', 'alpha'} <= f.keys()]
        if key in DESIGN_EXCLUSIONS:
            status = 'not_applicable' if key == 'privacy.*' else 'not_covered'
        elif measured:
            status = 'partial' if key in PARTIAL_CEILINGS else 'covered'
        else:
            status = 'partial' if ran else 'not_covered'
        module, label = BASE_CELLS.get(key, (ran[0].module if ran else (candidate[0].module if candidate else 'unassigned'), key))
        rows.append({'module': module, 'attack_class': key, 'taxonomy': label,
                     'taxonomy_ref': TAXONOMY_REF, 'status': status,
                     'method': [m.id for m in ran],
                     'access_required': sorted(set().union(*(m.access_required for m in candidate))),
                     'calibration_set': [calibration_sets.get(m.id, m.calibration.slice_id) for m in ran],
                     'measured_detection_floor': [dict(f, detector=i) for i, f in measured],
                     'residual_risk': DESIGN_EXCLUSIONS.get(key) or ('; '.join(m.residual_risk for m in ran) if ran else 'No executed check supports this class'),
                     'evidence_pointer': [pointer for m in ran for pointer in (
                         [evidence_pointers[m.id]] if m.id in evidence_pointers else
                         list(next(r for r in runs if r.detector == m.id).evidence))],
                     'skipped_checks': [vars(r) for r in plan.rejected if r.detector in by_id and key in by_id[r.detector].attack_classes] +
                                       [{'detector': r.detector, 'reason': r.reason_code} for r in runs if r.status != 'ran' and r.detector in by_id and key in by_id[r.detector].attack_classes]})
    return rows
