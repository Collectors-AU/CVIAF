"""MVP corpus integration smoke test (requires the repository's scikit-learn deps)."""
import os
import tempfile
import unittest
from cviaf.v4_assurance import run
from cviaf.schema_v3 import validate_report
import json

CORPUS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs', 'mvp'))

@unittest.skipUnless(os.path.isfile(os.path.join(CORPUS, 'registry.jsonl')), 'MVP corpus missing')
class Live(unittest.TestCase):
    def test_oda_budget_skip_and_audit(self):
        with tempfile.TemporaryDirectory() as out:
            result = run(CORPUS, 'oda_patch_on_object_s5', out,
                         threat_classes={'poisoning.backdoor.badet.ODA',
                                         'poisoning.backdoor.badet.OGA'}, budget_seconds=60)
            with open(result['report_path']) as fh: report = json.load(fh)
            self.assertEqual(validate_report(report), [])
            self.assertEqual(report['coverage_matrix'][0]['status'], 'not_covered')
            self.assertFalse(report['assessment_completeness']['accept_permitted'])
            self.assertTrue(any('BUDGET_EXHAUSTED' in x for x in report['assessment_completeness']['blocking_skips']))
            self.assertEqual(len(report['planner_audit_trail']), 2)
            with open(os.path.join(out,'legacy','assurance_report.json')) as fh: legacy = json.load(fh)
            self.assertEqual(report['legacy_findings'], legacy['findings'])
            self.assertTrue(os.path.isfile(os.path.join(out,'coverage_matrix.json')))

if __name__ == '__main__': unittest.main()
