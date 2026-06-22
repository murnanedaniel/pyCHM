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


def test_assembled_potential_matches_handcoded():
    """The Coleman-Weinberg potential curve V(s_h) -- the full pipeline at FIXED s_h, with no
    minimisation -- agrees between the assembled and hand-coded 5-5-5 to machine precision.
    This is the rigorous end-to-end check; it is not subject to vacuum amplification."""
    from pychm import routes
    shs = np.linspace(0.0, 0.32, 13)
    Vh = routes.potential_curve(REF, shs, model='5-5-5')
    Va = routes.potential_curve(REF, shs, model='5-5-5-assembled')
    assert np.allclose(Vh, Va, rtol=1e-6, atol=1e-12 * (abs(Vh).max() + 1e-30))


def test_assembled_vacuum_is_consistent():
    """End-to-end: the assembled model breaks EWSB at the same vacuum as the hand-coded one.
    The agreement is bounded by the tuned vacuum amplifying the ~1e-16 matrix float-order
    difference (that amplification IS the fine-tuning); a few-percent tolerance reflects it
    honestly and robustly across numpy/scipy builds."""
    sh = pychm.Model('5-5-5').spectrum(REF)
    sa = pychm.Model('5-5-5-assembled').spectrum(REF)
    for k in ('xi', 'mt', 'mb', 'mh', 'f'):
        assert np.isclose(sh[k], sa[k], rtol=3e-2), (k, sh[k], sa[k])


# ---- 14-1-10: q_L in the 14, b_R in the 10, t_R an SO(4) singlet --------------------- #
REF_14_1_10 = dict(
    mQ=1.5774575432442343, mU=0.053295037861913734, mD=3.9639579843374463,
    Yu=3.213522953043455, Yd=1.5033041681268544, Delta_q=0.8229454568382461,
    Delta_u=3.843401100178204, Delta_d=0.19968869968780683, f=1.1329813572955627,
    f1=1.5049448725076554, fX=1.4519742141888066, g=0.6709494105248374,
    gp=0.3581380846874656, grho=3.3254933316652062, gX=2.6398012309529406)


def test_14_1_10_matrices_match_handcoded():
    """Assembled from embeddings in the 14 (q_L) and 10 (b_R) -- two more SO(5) irreps --
    with the dressing from ccwz; reproduces the hand-coded 14x14/9x9 matrices entry-for-entry."""
    from pychm import mchm14_1_10
    asm = assemble.model_14_1_10
    for sh in (0.05, 0.2, 0.4, 0.6, 0.85):
        assert np.max(np.abs(asm.mass_U(REF_14_1_10, sh) - mchm14_1_10.mass_U(REF_14_1_10, sh))) < 1e-12
        assert np.max(np.abs(asm.mass_D(REF_14_1_10, sh) - mchm14_1_10.mass_D(REF_14_1_10, sh))) < 1e-12


def test_14_1_10_potential_and_vacuum():
    from pychm import routes
    shs = np.linspace(0.0, 0.32, 13)
    Vh = routes.potential_curve(REF_14_1_10, shs, model='14-1-10')
    Va = routes.potential_curve(REF_14_1_10, shs, model='14-1-10-assembled')
    assert np.allclose(Vh, Va, rtol=1e-6, atol=1e-12 * (abs(Vh).max() + 1e-30))
    sh = pychm.Model('14-1-10').spectrum(REF_14_1_10)
    sa = pychm.Model('14-1-10-assembled').spectrum(REF_14_1_10)
    for k in ('xi', 'mt', 'mb', 'mh', 'f'):
        assert np.isclose(sh[k], sa[k], rtol=3e-2), (k, sh[k], sa[k])


# ---- 14-14-10: q_L and t_R both in the 14, b_R in the 10 (the 19x19 up sector) ------- #
def test_14_14_10_matches_handcoded():
    from pychm import mchm14
    from tests.test_mchm14 import REF14
    asm = assemble.model_14_14_10
    for sh in (0.05, 0.2, 0.4, 0.6, 0.85):
        assert np.max(np.abs(asm.mass_U(REF14, sh) - mchm14.mass_U(REF14, sh))) < 1e-12
        assert np.max(np.abs(asm.mass_D(REF14, sh) - mchm14.mass_D(REF14, sh))) < 1e-12
    from pychm import routes
    shs = np.linspace(0.0, 0.32, 13)
    Vh = routes.potential_curve(REF14, shs, model='14-14-10')
    Va = routes.potential_curve(REF14, shs, model='14-14-10-assembled')
    assert np.allclose(Vh, Va, rtol=1e-6, atol=1e-12 * (abs(Vh).max() + 1e-30))
    sh = pychm.Model('14-14-10').spectrum(REF14)
    sa = pychm.Model('14-14-10-assembled').spectrum(REF14)
    for k in ('xi', 'mt', 'mb', 'mh', 'f'):
        assert np.isclose(sh[k], sa[k], rtol=3e-2), (k, sh[k], sa[k])
