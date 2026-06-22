"""Validation anchors for the two-site M4DCHM (5-5-5).

REF is a known electroweak-breaking point (the benchmark used throughout the validation
record).  Its spectrum and fine-tuning are pinned here: pyCHM reproduces the independent
mass-eigenvalue engine behind arXiv:2101.00428 to 0.03% on xi and <0.5% on Delta_BG, with
NO shared code.
"""
import numpy as np
import pychm

# canonical EWSB benchmark (TeV); xi ~ 0.070, m_t ~ 0.163 TeV, Delta_BG ~ 127
REF = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)


def _rand_point(rng):
    """Random point in the benchmark region (Delta_* mixing convention)."""
    return dict(
        mU=rng.uniform(1.5, 3.0), mUt=rng.uniform(0.8, 2.0),
        mD=rng.uniform(1.5, 3.0), mDt=rng.uniform(0.8, 2.0),
        mYu=rng.uniform(0.005, 0.05), Yu=rng.uniform(3.0, 9.0),
        mYd=rng.uniform(0.01, 0.1), Yd=rng.uniform(0.2, 1.0),
        Delta_uL=rng.uniform(0.8, 1.5), Delta_uR=rng.uniform(0.8, 1.5),
        Delta_dL=rng.uniform(0.3, 0.8), Delta_dR=rng.uniform(0.05, 0.3),
        f=rng.uniform(0.8, 1.0), f1=rng.uniform(1.3, 1.7), fX=rng.uniform(2.5, 3.2),
        g=0.67095, gp=0.358138, grho=rng.uniform(4.5, 5.5), gX=2.96355)


def test_reference_spectrum():
    s = pychm.spectrum.spectrum(REF)
    assert s is not None, "reference point must break EWSB"
    assert abs(s['xi'] - 0.0701) < 5e-4          # vs independent engine 0.07005
    assert abs(s['mt'] - 0.1626) < 3e-3          # SM top (TeV)
    assert abs(s['mh'] - 0.1119) < 5e-3          # SM Higgs (TeV)
    assert abs(s['mZ'] - 0.0912) < 1e-3


def test_reference_tuning():
    t = pychm.tuning.tuning(REF)
    assert t is not None
    assert abs(t['BG'] - 127.0) < 5.0            # vs independent engine 126.7
    assert abs(t['HOT'] - 195.0) < 5.0
    assert abs(t['I'] - 5.27) < 0.1              # nats
    assert t['KL'] == t['I']


def test_ewsb_fires_for_some_points():
    rng = np.random.default_rng(1)
    broke = sum(pychm.spectrum.spectrum(_rand_point(rng)) is not None for _ in range(200))
    assert broke > 0, "engine never breaks EWSB -- something is wrong"


def test_spectrum_keys_and_finiteness():
    rng = np.random.default_rng(1)
    for _ in range(200):
        s = pychm.spectrum.spectrum(_rand_point(rng))
        if s is None:
            continue
        assert set(s) >= {'xi', 'f', 'mt', 'mb', 'mh', 'mW', 'mZ'}
        assert 0 < s['xi'] < 1 and all(np.isfinite(list(s.values())))
        return
    raise AssertionError("no EWSB points found")
