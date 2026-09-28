import numpy as np

from cviaf.lab.null_suite import fft_energy, pair_metric, pvalues


def test_identical_scores_are_chance_and_not_rejected_at_five_percent():
    x = np.arange(20, dtype=float)
    m = pair_metric(x, x)
    assert m['auroc'] == .5
    assert m['tpr_at_5fpr'] == .05


def test_fft_only_depends_on_images():
    images = np.ones((2, 64, 64, 3), dtype=float)
    stamped = images.copy()
    stamped[:, :8, :8, :] = 0
    assert np.all(fft_energy(stamped) > fft_energy(images))


def test_stamped_null_calibration_fuses_with_and_without_ftc():
    cells = ('clean_unstamped', 'clean_stamped', 'backdoored_unstamped',
             'backdoored_stamped', 'peer_clean_stamped')
    data = {c: {s: np.arange(20, dtype=float) for s in ('ctc', 'refdiv', 'ftc', 'fft')}
            for c in cells}
    p = pvalues(data, data)
    assert len(p['clean_stamped']['with_ftc']) == 20
    assert np.all(p['clean_stamped']['with_ftc'] >= p['clean_stamped']['without_ftc'])
