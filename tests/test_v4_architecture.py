"""Offline contract tests: python3 -m unittest discover -s tests -p test_v4_architecture.py"""
import copy
import unittest
from cviaf.plugins import CalibrationRef, CostModel, Manifest, Registry, Result
from cviaf.planner import Context, plan, completeness
from cviaf.schema_v3 import validate_report
from cviaf.schema_v3.coverage import generate_coverage

KEY = 'poisoning.backdoor.badet.ODA'

def manifest(id='trace', classes=(KEY,), access=frozenset({'model.black_box'}), seconds=2):
    return Manifest(id, 'model_integrity', classes, access, CostModel(seconds=seconds),
                    CalibrationRef('AB1.probe', 'sha256:1234'), 'p_value',
                    frozenset({'ONNX'}), residual_risk='adaptive triggers not certified')

def context(**kwargs):
    data = dict(attack_classes=frozenset({KEY}), access=frozenset({'model.black_box'}),
                formats=frozenset({'ONNX'}), assets=frozenset({'model.black_box'}),
                battery_slices=frozenset({'AB1.probe'}), seconds=10)
    data.update(kwargs)
    return Context(**data)

class GoodPlugin:
    manifest = manifest()
    def run(self, assets, battery_slice, budget):
        return Result('trace', 'ran', 2.4, p_value=.02, calibration_ref='sha256:1234',
                      cost_actual=1.5, evidence=('artifact://probe-1',))

class BadPlugin(GoodPlugin):
    def run(self, assets, battery_slice, budget):
        return Result('trace', 'ran', 2.4, p_value=.02, calibration_ref='wrong',
                      cost_actual=1.5, evidence=('artifact://probe-1',))

class Contracts(unittest.TestCase):
    def test_access_and_format_skip_are_recorded(self):
        m = manifest(access=frozenset({'model.white_box'}))
        p = plan([m], context())
        self.assertEqual(p.rejected[0].reason, 'ACCESS_DENIED')
        self.assertFalse(completeness(p, [])['accept_permitted'])
        self.assertEqual(plan([manifest()], context(formats=frozenset({'TorchScript'}))).rejected[0].reason, 'ASSET_ABSENT')

    def test_budget_optimizes_weighted_coverage(self):
        p = plan([manifest('a', ('a',), seconds=5), manifest('b', ('b',), seconds=5)],
                 context(attack_classes=frozenset({'a','b'}), seconds=5, risk_weights={'b':2}))
        self.assertEqual(p.selected, ('b',))
        self.assertEqual(p.rejected[0].reason, 'BUDGET_EXHAUSTED')

    def test_execution_and_fail_closed(self):
        r = Registry(); r.register(GoodPlugin())
        with self.assertRaises(ValueError): r.register(GoodPlugin())
        p = plan(r.manifests(), context())
        results = r.execute(p.selected, {}, {'AB1.probe':{}}, {})
        self.assertEqual(results[0].status, 'ran')
        self.assertTrue(completeness(p, results)['accept_permitted'])
        bad = Registry(); bad.register(BadPlugin())
        failed = bad.execute(p.selected, {}, {'AB1.probe':{}}, {})
        self.assertEqual(failed[0].status, 'error')
        self.assertFalse(completeness(p, failed)['accept_permitted'])
        with self.assertRaises(ValueError): Result('bad', 'ran', .4, p_value=.2)

    def test_matrix_no_static_coverage(self):
        m = manifest(); p = plan([m], context())
        r = Registry(); r.register(GoodPlugin()); outcome = r.execute(p.selected, {}, {'AB1.probe':{}}, {})
        row = generate_coverage(p, [m], outcome)[0]
        self.assertEqual(row['status'], 'partial')
        floor = {'metric':'trigger_area_px2','value':196,'power':.82,'alpha':.05}
        row = generate_coverage(p, [m], outcome, {'trace':floor}, evidence_pointers={'trace':'artifact://probe-1'})[0]
        self.assertEqual(row['status'], 'covered')
        self.assertEqual(row['evidence_pointer'], ['artifact://probe-1'])
        skipped = (Result('trace','skipped',reason_code='ACCESS_DENIED',human_reason='no access',coverage_impact='no coverage'),)
        self.assertEqual(generate_coverage(p, [m], skipped)[0]['status'], 'not_covered')

    def test_report_cross_field_gate(self):
        m=manifest(); p=plan([m],context()); r=Registry(); r.register(GoodPlugin())
        outcome=r.execute(p.selected,{}, {'AB1.probe':{}},{})
        floor={'metric':'trigger_area_px2','value':196,'power':.82,'alpha':.05}
        report = {'schema_version':'3.0','report_id':'run-1','timestamp':'2026-09-28T00:00:00Z',
                  'framework_version':'3.0.0','threat_model_declared':{
                      'attack_classes_in_scope':[KEY], 'access_granted':{'model':'black-box'},
                      'assumptions':['calibration held out']},
                  'battery':{'id':'AB-1','digest':'sha256:abcd','slices_used':['AB1.probe']},
                  'plan':p.to_dict(), 'assessment_completeness':completeness(p,outcome),
                  'hypothesis_posteriors':[], 'assessments':{}, 'findings':[{
                      'finding_id':'F1','module':'model_integrity','attack_class':KEY,
                      'method':'trace','access_mode':'black-box',
                      'confidence':{'value':.91,'type':'conformal_p_value','calibration_ref':'sha256:1234'},
                      'detection_floor':floor,'battery_coverage_fraction':1.,
                      'assumptions':['held-out calibration'],'evidence':{'artifact':'artifact://probe-1'}}],
                  'coverage_matrix':generate_coverage(p,[m],outcome,{'trace':floor}),
                  'loss_matrix_used':{},'overall_disposition':'accept'}
        self.assertEqual(validate_report(report), [])
        broken=copy.deepcopy(report); broken['assessment_completeness']['accept_permitted']=False
        self.assertTrue(any('overall_disposition' in e for e in validate_report(broken)))
        broken=copy.deepcopy(report); broken['findings'][0]['confidence']['value']=1.3
        self.assertTrue(any('confidence.value' in e for e in validate_report(broken)))

if __name__ == '__main__': unittest.main()
