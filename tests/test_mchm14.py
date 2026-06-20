"""Validation anchors for the two-site M4DCHM 14-14-10 (q_L and t_R in the symmetric 14
of SO(5), b_R in the 10).

REF14 is the M4DCHM_14_14_10 reference benchmark.  pyCHM reproduces the independent
mass-eigenvalue engine (pypngb) on this point with NO shared code: the s_h-dependent
mass matrices and the closed-form Coleman-Weinberg potential are validated against
pypngb's `masses.py` / `obs/veff.py` to

    sh  : 0.0002%   (pyCHM 0.163619 vs pypngb 0.163618)
    m_t : <0.01%    (pyCHM 127.834 GeV vs pypngb 127.834 GeV)
    m_h : 0.06%     (pyCHM 338.31 GeV vs pypngb 338.53 GeV, compared at the oracle f)
    Delta_BG : ~0.1% (pyCHM 16.20 vs pypngb 16.17, d ln f / d ln x convention)

NB: pyCHM rescales f so that v = f sqrt(xi) = 246.22 GeV (V_EW); pypngb keeps f fixed.
Both give identical xi, m_t and identical V''(s_h); m_h below is the rescaled-f value
returned by the public API.  The physics (V'' at fixed dimensionful params) agrees with
pypngb to 0.06%.
"""
import numpy as np
import pychm

# M4DCHM_14_14_10 reference benchmark (TeV / dimensionless); xi ~ 0.0268, m_t ~ 0.128 TeV
REF14 = dict(
    mQ=3.965100934, mU=2.397190093, mD=1.482462237,
    mYu=0.1478991938, Yu=0.4989142322, Ytu=2.538912347, Yd=0.5293788277,
    Delta_q=2.850102384, Delta_u=1.915943759, Delta_d=0.2222079908,
    f=1.4396717743254574, f1=1.8934618718580186, fX=2.2076564878795057,
    g=0.6709494105248374, gp=0.3581380846874656, grho=5.525258191589697,
    gX=4.696070132753916)


def test_reference_spectrum():
    s = pychm.Model('14-14-10').spectrum(REF14)
    assert s is not None, "reference point must break EWSB"
    assert abs(s['xi'] - 0.0267711) < 2.7e-5         # vs pypngb 0.0267710 (<0.1%)
    assert abs(np.sqrt(s['xi']) - 0.163618) < 1.6e-4  # sh, vs pypngb 0.163618 (<0.1%)
    assert abs(s['mt'] - 0.127834) < 1.3e-3          # SM top (TeV), vs pypngb 0.127834 (<1%)
    assert abs(s['mb'] - 0.0025884) < 1e-4           # SM bottom (TeV)


def test_reference_tuning():
    t = pychm.Model('14-14-10').tuning(REF14)
    assert t is not None
    assert abs(t['BG'] - 16.17) < 0.16              # vs pypngb 16.17 (d ln f/d ln x, <1%)
    assert t['KL'] == t['I']


def test_mchm14_matrices_match_oracle_dims():
    """The s_h-dependent mass matrices have the dimensions pypngb's masses.py defines."""
    sh = 0.1636
    assert pychm.mchm14.mass_U(REF14, sh).shape == (19, 19)
    assert pychm.mchm14.mass_D(REF14, sh).shape == (12, 12)
    assert pychm.mchm14.mass2_W(REF14, sh).shape == (4, 4)
    assert pychm.mchm14.mass2_Z(REF14, sh).shape == (7, 7)


def test_default_model_is_555():
    """The default model must remain 5-5-5 so the existing anchors are unaffected."""
    assert pychm.Model().representation == '5-5-5'
