"""The four fine-tuning measures must satisfy the spectral identities."""
import numpy as np
import pychm
from tests.test_anchors import _rand_point


def test_measure_identities():
    rng = np.random.default_rng(7)
    m = pychm.Model(); checked = 0
    for _ in range(300):
        t = m.tuning(_rand_point(rng))
        if t is None:
            continue
        assert t['KL'] == t['I']                                  # KL = I (Gaussian)
        assert abs(t['I'] - 0.5*np.log1p(t['HOT']**2)) < 1e-9     # I = 1/2 log(1+|J|^2)
        assert t['BG'] <= t['HOT'] + 1e-9                         # max <= L2 norm
        checked += 1
        if checked >= 5:
            break
    assert checked >= 1, "no EWSB points found to test measures"
