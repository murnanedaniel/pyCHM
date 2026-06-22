"""Validation anchors for the two-site M4DCHM 14-1-10 (q_L in the symmetric 14 of SO(5),
t_R an SO(4) singlet, b_R in the 10).

REF14_1_10 is the M4DCHM_14_1_10 reference benchmark.  pyCHM reproduces the independent
mass-eigenvalue engine (pypngb) on this point with NO shared code: the s_h-dependent mass
matrices are bit-identical to pypngb's `masses.py`, and the closed-form Coleman-Weinberg
potential reproduces pypngb's vacuum / spectrum / fine-tuning to

    sh       : 0.035%   (pyCHM 0.220088 vs pypngb 0.220165)
    m_t      : 0.034%   (pyCHM 153.742 GeV vs pypngb 153.794 GeV)
    m_b      : 0.034%   (pyCHM 2.6457 GeV vs pypngb 2.6466 GeV)
    m_h      : <0.01%   (pyCHM vs pypngb at the oracle f, matched scheme; see note)
    Delta_BG : 0.038%   (pyCHM 7.324 vs pypngb 7.321, d ln f/d ln x convention; argmax Delta_q)

NB: pyCHM rescales f so that v = f sqrt(xi) = 246.22 GeV (V_EW); pypngb keeps f fixed.  Both
give identical xi, m_t and identical V''(s_h); the m_h returned by the public API is the
rescaled-f value.  Compared at the oracle f with a matched finite-difference scheme the V''
(hence m_h) agrees to <0.01%.
"""
import numpy as np
import pychm

# M4DCHM_14_1_10 reference benchmark (TeV / dimensionless); xi ~ 0.0484, m_t ~ 0.154 TeV.
# Resolved from pypngb's parWithMin (functions_14_1_10_log: each x = exp(log_x), in GeV -> TeV).
REF14_1_10 = dict(
    mQ=1.5774575432442343, mU=0.053295037861913734, mD=3.9639579843374463,
    Yu=3.213522953043455, Yd=1.5033041681268544,
    Delta_q=0.8229454568382461, Delta_u=3.843401100178204, Delta_d=0.19968869968780683,
    f=1.1329813572955627, f1=1.5049448725076554, fX=1.4519742141888066,
    g=0.6709494105248374, gp=0.3581380846874656, grho=3.3254933316652062,
    gX=9.7650801141742978)


def test_reference_spectrum():
    s = pychm.Model('14-1-10').spectrum(REF14_1_10)
    assert s is not None, "reference point must break EWSB"
    assert abs(s['xi'] - 0.0484725) < 4.9e-5          # vs pypngb 0.0484725 (<0.1%)
    assert abs(np.sqrt(s['xi']) - 0.220165) < 2.3e-4  # sh, vs pypngb 0.220165 (<0.1%)
    assert abs(s['mt'] - 0.153794) < 1.6e-3           # SM top (TeV), vs pypngb 0.153794 (<1%)
    assert abs(s['mb'] - 0.0026466) < 1e-4            # SM bottom (TeV)


def test_reference_tuning():
    t = pychm.Model('14-1-10').tuning(REF14_1_10)
    assert t is not None
    assert abs(t['BG'] - 7.321) < 0.074               # vs pypngb 7.321 (d ln f/d ln x, <1%)
    assert t['KL'] == t['I']


def test_mchm14_1_10_matrices_match_oracle_dims():
    """The s_h-dependent mass matrices have the dimensions pypngb's masses.py defines."""
    sh = 0.2201
    assert pychm.mchm14_1_10.mass_U(REF14_1_10, sh).shape == (14, 14)
    assert pychm.mchm14_1_10.mass_D(REF14_1_10, sh).shape == (9, 9)
    assert pychm.mchm14_1_10.mass2_W(REF14_1_10, sh).shape == (4, 4)
    assert pychm.mchm14_1_10.mass2_Z(REF14_1_10, sh).shape == (7, 7)


def test_default_model_is_555():
    """The default model must remain 5-5-5 so the existing anchors are unaffected."""
    assert pychm.Model().representation == '5-5-5'
