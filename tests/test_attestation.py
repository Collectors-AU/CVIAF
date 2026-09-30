"""Standalone, adversarial checkpoint/attestation regression tests."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

from cviaf.core.types import AuditTrail
from cviaf.provenance.attestation import sign_bundle

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        key = Ed25519PrivateKey.generate()
        self.private = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        self.public = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        (self.folder / 'pub.pem').write_bytes(self.public)
        trail = AuditTrail()
        for i in range(6):
            trail.append('step', 'test', {'n': i})
        self.audit = trail.to_dict()
        self.report = {'audit_trail': self.audit, 'findings': [{'severity': 'LOW'}]}
        self.bundle = sign_bundle(self.report, self.audit, self.private, 'test-key')

    def check(self, report=None, audit=None, bundle=None, extra=()):
        for name, obj in (('report', report if report is not None else self.report),
                          ('audit', audit if audit is not None else self.audit),
                          ('bundle', bundle if bundle is not None else self.bundle)):
            (self.folder / f'{name}.json').write_text(json.dumps(obj))
        return subprocess.run([sys.executable, str(ROOT/'scripts/verify_attestation.py'),
                               '--report', str(self.folder/'report.json'), '--audit', str(self.folder/'audit.json'),
                               '--bundle', str(self.folder/'bundle.json'), '--public-key', str(self.folder/'pub.pem'),
                               *extra], cwd=self.folder, capture_output=True, text=True)

    def test_valid_offline_and_pinned_latest(self):
        result = self.check(extra=('--expected-count', '6', '--expected-root', self.bundle['checkpoint']['merkle_root']))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"tree_size": 6', result.stdout)

    def test_suffix_truncation_caught(self):
        truncated = self.audit[:-1]
        self.assertNotEqual(self.check(report={**self.report, 'audit_trail': truncated}, audit=truncated).returncode, 0)

    def test_reorder_and_interior_deletion_caught(self):
        for broken in (self.audit[:2] + self.audit[3:],
                       self.audit[:2] + [self.audit[3], self.audit[2]] + self.audit[4:]):
            self.assertNotEqual(self.check(report={**self.report, 'audit_trail': broken}, audit=broken).returncode, 0)

    def test_tamper_report_and_signature_caught(self):
        report = copy.deepcopy(self.report)
        report['findings'][0]['severity'] = 'CRITICAL'
        self.assertNotEqual(self.check(report=report).returncode, 0)
        bundle = copy.deepcopy(self.bundle)
        bundle['attestation']['signatures'][0]['sig'] = 'AAAA'
        self.assertNotEqual(self.check(bundle=bundle).returncode, 0)

    def test_wrong_key_caught(self):
        key = Ed25519PrivateKey.generate()
        (self.folder/'pub.pem').write_bytes(key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
        self.assertNotEqual(self.check().returncode, 0)

    def test_old_valid_checkpoint_requires_pinned_latest(self):
        old_audit = self.audit[:-1]
        old_report = {**self.report, 'audit_trail': old_audit}
        old_bundle = sign_bundle(old_report, old_audit, self.private, 'test-key')
        self.assertEqual(self.check(report=old_report, audit=old_audit, bundle=old_bundle).returncode, 0)
        self.assertNotEqual(self.check(report=old_report, audit=old_audit, bundle=old_bundle,
                                       extra=('--expected-count', '6')).returncode, 0)

    def test_signer_rejects_invalid_chain(self):
        broken = self.audit[:2] + self.audit[3:]
        with self.assertRaises(ValueError):
            sign_bundle({'audit_trail': broken}, broken, self.private, 'test-key')


if __name__ == '__main__':
    unittest.main()
