"""The generic assembler reproduces the hand-coded 5-5-5 fermion mass matrices entry-for-entry
(s_h dependence from ccwz, not transcribed), and hence the full validated spectrum and tuning."""
import numpy as np
import pychm
from pychm import mchm5, assemble

REF = dict(mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391, mYu=0.00754655, Yu=7.95104,
           mYd=0.03112, Yd=0.60624, Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056,
           Delta_dR=0.106068, f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138,
           grho=5.44665, gX=2.96355)


def test_assembler_matches_handcoded_matrices():
    for sh in (0.05, 0.2, 0.4, 0.6, 0.85):
        AU = assemble.assemble(assemble.spec_555(REF, up=True), sh)
        AD = assemble.assemble(assemble.spec_555(REF, up=False), sh)
        assert np.max(np.abs(AU - mchm5.mass_U(REF, sh))) < 1e-12
        assert np.max(np.abs(AD - mchm5.mass_D(REF, sh))) < 1e-12


def test_assembled_model_reproduces_spectrum_and_tuning():
    # The assembled and hand-coded mass matrices are identical to machine precision
    # (test above), but the electroweak vacuum is a near-cancellation, so it amplifies the
    # ~1e-16 difference in float operation order into ~1e-6 on xi and ~1e-4 on m_h.  That
    # amplification IS the fine-tuning; the tolerances here reflect it honestly.
    hand = pychm.Model('5-5-5')
    asm = pychm.Model('5-5-5-assembled')
    sh, sa = hand.spectrum(REF), asm.spectrum(REF)
    for k in ('xi', 'mt', 'mb', 'f'):
        assert np.isclose(sh[k], sa[k], rtol=1e-5), (k, sh[k], sa[k])
    assert np.isclose(sh['mh'], sa['mh'], rtol=1e-3)        # 2nd derivative: more amplified
    th, ta = hand.tuning(REF), asm.tuning(REF)
    assert np.isclose(th['BG'], ta['BG'], rtol=1e-3)
    assert np.isclose(th['I'], ta['I'], rtol=1e-3)
