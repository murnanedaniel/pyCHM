"""Phase-0 machinery tests. (Tuned-point accuracy awaits the eigenvalue fermion route,
Phase 1 -- the current form-factor-fermion + eigenvalue-gauge mix is not normalisation-
consistent on the near-cancellation that sets a tuned vacuum; see README.)"""
import numpy as np
import pychm

P0 = dict(mU=2.0, mUt=1.2, mD=1.8, mDt=1.0, mYu=0.9, Yu=1.2, mYd=0.7, Yd=0.4,
          Lq=1.5, Lt=1.6, Lb=0.3, f=0.9, f1=1.5, fX=3.0,
          g=0.671, gp=0.358, grho=5.0, gX=3.0)


def _rand_point(rng):
    return dict(mU=rng.uniform(0.6, 2), mUt=rng.uniform(0.6, 2), mD=rng.uniform(0.8, 2.5),
                mDt=rng.uniform(0.6, 2), mYu=rng.uniform(0.3, 2), Yu=rng.uniform(-2, 2),
                mYd=rng.uniform(0.3, 2), Yd=rng.uniform(-1, 1), Lq=rng.uniform(1.0, 3),
                Lt=rng.uniform(1.0, 3), Lb=rng.uniform(0.05, 1), f=rng.uniform(0.7, 1.3),
                f1=rng.uniform(1.0, 2.0), fX=rng.uniform(2, 4), g=0.671, gp=0.358,
                grho=rng.uniform(3, 5), gX=3.0)


def test_mZ_exact():
    # m_Z = (1/2) sqrt(g^2+g'^2) v, independent of EWSB details
    s = pychm.spectrum.spectrum(P0)
    if s is not None:
        assert abs(s['mZ'] - 0.0912) < 1e-3


def test_potential_runs():
    g, b, d = pychm.Model().potential_coeffs(P0)
    assert all(np.isfinite([g, b, d]))


def test_ewsb_fires_for_some_points():
    rng = np.random.default_rng(1)
    m = pychm.Model(); broke = sum(m.spectrum(_rand_point(rng)) is not None for _ in range(300))
    assert broke > 0, "engine never breaks EWSB -- something is wrong"


def test_spectrum_keys_and_finiteness():
    rng = np.random.default_rng(1)
    m = pychm.Model()
    for _ in range(300):
        s = m.spectrum(_rand_point(rng))
        if s is None:
            continue
        assert set(s) >= {'xi', 'f', 'mt', 'mb', 'mh', 'mW', 'mZ'}
        assert 0 < s['xi'] < 1 and all(np.isfinite(list(s.values())))
        return
    raise AssertionError("no EWSB points found")
