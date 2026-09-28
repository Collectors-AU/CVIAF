"""Additive v3 assurance report validation with cross-field safety invariants."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
from .coverage import generate_coverage, BASE_CELLS

SCHEMA_PATH = Path(__file__).with_name('assurance-report-v3.schema.json')
REPORT_SCHEMA = json.loads(SCHEMA_PATH.read_text())

def validate_report(report: Any) -> list[str]:
    """No third-party validator required. Returns JSON-path errors, never silently passes.

    This checks the normative v3 fields and safety invariants; the companion JSON
    Schema gives consumers the full structural contract. It does not validate
    legacy v2 fields, whose existing validator remains authoritative.
    """
    errors = []
    def err(path, reason): errors.append(f'{path}: {reason}')
    def obj(parent, key, path):
        value = parent.get(key) if isinstance(parent, dict) else None
        if not isinstance(value, dict): err(path, 'required object'); return {}
        return value
    def arr(parent, key, path):
        value = parent.get(key) if isinstance(parent, dict) else None
        if not isinstance(value, list): err(path, 'required array'); return []
        return value
    def text(parent, key, path):
        if not isinstance(parent.get(key), str) or not parent[key]: err(path, 'required nonempty string')
    def number(value, path, lo=0, hi=None):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < lo or (hi is not None and value > hi):
            err(path, f'expected finite number in [{lo}, {hi}]')
    if not isinstance(report, dict): return ['$: required object']
    if report.get('schema_version') != '3.0': err('$.schema_version', 'must equal 3.0')
    for k in ('report_id', 'timestamp', 'framework_version'): text(report, k, '$.'+k)
    threat = obj(report, 'threat_model_declared', '$.threat_model_declared')
    scope = arr(threat, 'attack_classes_in_scope', '$.threat_model_declared.attack_classes_in_scope')
    if not scope or any(not isinstance(k, str) or not k for k in scope) or len(scope) != len(set(scope)):
        err('$.threat_model_declared.attack_classes_in_scope', 'nonempty unique string keys required')
    obj(threat, 'access_granted', '$.threat_model_declared.access_granted')
    assumptions = arr(threat, 'assumptions', '$.threat_model_declared.assumptions')
    if not assumptions or any(not isinstance(x, str) or not x for x in assumptions): err('$.threat_model_declared.assumptions', 'explicit nonempty assumptions required')
    battery = obj(report, 'battery', '$.battery')
    for k in ('id', 'digest'): text(battery, k, '$.battery.'+k)
    arr(battery, 'slices_used', '$.battery.slices_used')
    plan = obj(report, 'plan', '$.plan')
    text(plan, 'planner_version', '$.plan.planner_version')
    selected = arr(plan, 'selected', '$.plan.selected')
    if len(selected) != len(set(selected)) or any(not isinstance(x, str) or not x for x in selected):
        err('$.plan.selected', 'unique nonempty detector ids required')
    rejected = arr(plan, 'rejected', '$.plan.rejected')
    for i, r in enumerate(rejected):
        if not isinstance(r, dict) or any(not r.get(k) for k in ('detector', 'reason', 'coverage_impact')): err(f'$.plan.rejected[{i}]', 'missing detector/reason/coverage_impact')
    complete = obj(report, 'assessment_completeness', '$.assessment_completeness')
    for k in ('applicable', 'executed'):
        v = complete.get(k)
        if type(v) is not int or v < 0: err('$.assessment_completeness.'+k, 'nonnegative integer required')
    number(complete.get('fraction'), '$.assessment_completeness.fraction', hi=1)
    if type(complete.get('accept_permitted')) is not bool: err('$.assessment_completeness.accept_permitted', 'boolean required')
    skips = arr(complete, 'blocking_skips', '$.assessment_completeness.blocking_skips')
    a, e = complete.get('applicable'), complete.get('executed')
    if type(a) is int and type(e) is int and a >= 0:
        if e > a or (isinstance(complete.get('fraction'), (float,int)) and abs(complete['fraction'] - (e/a if a else 0)) > 1e-6):
            err('$.assessment_completeness', 'fraction/count mismatch')
        if complete.get('accept_permitted') and (e != a or skips or rejected): err('$.assessment_completeness.accept_permitted', 'cannot accept with incomplete checks')
    for i, posterior in enumerate(arr(report, 'hypothesis_posteriors', '$.hypothesis_posteriors')):
        p = f'$.hypothesis_posteriors[{i}]'
        if not isinstance(posterior, dict): err(p, 'object required'); continue
        text(posterior, 'hypothesis', p+'.hypothesis')
        number(posterior.get('posterior'), p+'.posterior', hi=1)
        arr(posterior, 'evidence_refs', p+'.evidence_refs')
    obj(report, 'assessments', '$.assessments')
    findings = arr(report, 'findings', '$.findings')
    coverage = arr(report, 'coverage_matrix', '$.coverage_matrix')
    cells = {}
    for i, row in enumerate(coverage):
        p = f'$.coverage_matrix[{i}]'
        if not isinstance(row, dict): err(p, 'object required'); continue
        for k in ('module','attack_class','taxonomy','residual_risk'): text(row, k, p+'.'+k)
        if row.get('status') not in ('covered','partial','not_covered','not_applicable'): err(p+'.status', 'invalid status')
        for k in ('method','access_required','calibration_set','measured_detection_floor','evidence_pointer','skipped_checks'): arr(row, k, p+'.'+k)
        key = row.get('attack_class')
        if isinstance(key, str):
            if key in cells: err(p, 'duplicate taxonomy key')
            cells[key] = row
        if row.get('status') == 'covered' and (not row.get('method') or not row.get('measured_detection_floor')):
            err(p+'.status', 'covered needs executed method and measured floor')
    if all(isinstance(k, str) for k in scope) and set(scope) != set(cells):
        err('$.coverage_matrix', 'must contain exactly the declared attack keys')
    if report.get('overall_disposition') == 'accept' and not complete.get('accept_permitted'):
        err('$.overall_disposition', 'accept forbidden when assessment incomplete')
    for i, finding in enumerate(findings):
        p = f'$.findings[{i}]'
        if not isinstance(finding, dict): err(p, 'object required'); continue
        for k in ('finding_id','module','attack_class','method','access_mode'): text(finding,k,p+'.'+k)
        if finding.get('attack_class') not in cells: err(p+'.attack_class', 'missing coverage cell')
        confidence = obj(finding,'confidence',p+'.confidence')
        number(confidence.get('value'),p+'.confidence.value',hi=1)
        for k in ('type','calibration_ref'): text(confidence,k,p+'.confidence.'+k)
        floor = obj(finding,'detection_floor',p+'.detection_floor')
        text(floor,'metric',p+'.detection_floor.metric')
        number(floor.get('value'),p+'.detection_floor.value')
        number(floor.get('power'),p+'.detection_floor.power',hi=1)
        number(floor.get('alpha'),p+'.detection_floor.alpha',hi=1)
        number(finding.get('battery_coverage_fraction'),p+'.battery_coverage_fraction',hi=1)
        if not arr(finding,'assumptions',p+'.assumptions'): err(p+'.assumptions', 'at least one assumption required')
        obj(finding,'evidence',p+'.evidence')
        if finding.get('attack_class') in cells and isinstance(finding.get('battery_coverage_fraction'),(int,float)):
            if finding['battery_coverage_fraction'] > complete.get('fraction',0) + 1e-6:
                err(p+'.battery_coverage_fraction','cannot exceed report completeness fraction')
    obj(report,'loss_matrix_used','$.loss_matrix_used')
    return errors
