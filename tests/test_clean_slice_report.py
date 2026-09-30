"""Do not turn zero observed clean flags into a zero population FPR claim."""
from scripts.clean_slice_report import upper, report


def test_exact_one_sided_zero_count_and_slices():
    assert round(upper(0, 32), 4) == .0894
    assert upper(0, 59) <= .05
    rows = [dict(kind='clean', contributes_poison=False,
                 calibration_stratum_match=True,
                 cviaf={'flagged': False, 'abstained': False}) for _ in range(32)]
    out = report({'data_axis': {'per_model': rows}})
    assert out['slices'][0]['n_clean_assets'] == 32
    assert .08 < out['slices'][0]['fpr_upper_95_if_independent'] < .10
