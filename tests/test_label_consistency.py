import numpy as np
import pytest
from cviaf.lab.label_consistency import LabelConsistencyGate
from cviaf.lab.synth import SceneSpec, build_dataset
from cviaf.lab.poison import AttackSpec, inject
from cviaf.lab.baseline import cviaf_data_asset_verdict


def test_label_gate_and_dataset_verdict():
    ref = [build_dataset(32, SceneSpec(seed=100+i)) for i in range(3)]
    cal = [build_dataset(32, SceneSpec(seed=200+i)) for i in range(39)]
    gate = LabelConsistencyGate.fit(ref, cal)
    clean = build_dataset(32, SceneSpec(seed=300))
    attack, _ = inject(clean, AttackSpec(kind='label_flip', rate=.25, seed=5))
    # An annotation-only flip does not alter the pixels/boxes.
    np.testing.assert_array_equal(clean.images, attack.images)
    assert gate.assess(clean)['flagged'] is False
    verdict = cviaf_data_asset_verdict('asset', {}, dataset=attack, label_gate=gate)
    assert verdict.flagged and verdict.attack_class == 'label_inconsistency'
    assert 'label_consistency' in verdict.evidence


def test_missing_reference_abstains_and_partial_pair_fails():
    ds = build_dataset(12, SceneSpec(seed=90))
    assert cviaf_data_asset_verdict('asset', {}).abstained
    with pytest.raises(ValueError):
        cviaf_data_asset_verdict('asset', {}, dataset=ds)
    with pytest.raises(ValueError):
        LabelConsistencyGate.fit([ds], [ds])
