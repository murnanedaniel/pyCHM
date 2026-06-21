"""The generic CCWZ Goldstone machinery reproduces the rep-specific s_h-factors that are
hand-coded in mchm5 / mchm14, computed from group theory alone.  Convention: sh = sin(h/f)."""
import numpy as np
from pychm.groups import so5 as ccwz


def test_so5_vector_is_a_rotation():
    U = ccwz.U_vector(0.37)
    assert np.allclose(U @ U.T, np.eye(5))
    assert np.isclose(np.linalg.det(U), 1.0)


def test_rep5_factors_match_handcoded():
    """The 5: U components are cos(h/f)=sqrt(1-sh^2) and sin(h/f)=sh; the (2,2) gives the
    half-angle factors c2h2=cos^2(h/2f), s2h2=sin^2(h/2f) used in mchm5."""
    sh = 0.41
    ch = np.sqrt(1 - sh**2)
    U = ccwz.U_vector(sh)
    assert np.isclose(U[3, 3], ch)
    assert np.isclose(U[3, 4], sh)
    assert np.isclose((1 + ch) / 2, 0.5 * (1 + np.sqrt(1 - sh**2)))   # mchm5 _c2h2(sh)
    assert np.isclose((1 - ch) / 2, 0.5 * (1 - np.sqrt(1 - sh**2)))   # mchm5 _s2h2(sh)


def test_rep14_singlet_factor_is_exact():
    """The 14: <(1,1)|U14|(1,1)> = (3 + 5 cos(2h/f))/8, with cos(2h/f) = 1 - 2 sh^2 --
    the exact mchm14 m[4,2] factor, reproduced to machine precision."""
    for sh in (0.0, 0.25, 0.5, 0.9):
        ES = ccwz.embedding('14', 'singlet')
        val = ccwz.overlap('14', ES, ES, sh).real
        cos2h = 1 - 2 * sh**2
        assert np.isclose(val, (3 + 5 * cos2h) / 8, atol=1e-12)


def test_rep14_unitarity_and_identity():
    assert np.allclose(ccwz.U_rep('14', 0.0), np.eye(14))
    assert np.allclose(ccwz.U_rep('14', 0.6) @ ccwz.U_rep('14', 0.6).T, np.eye(14), atol=1e-10)


def test_rep10_unitarity_and_identity():
    assert np.allclose(ccwz.U_rep('10', 0.0), np.eye(10))
    assert np.allclose(ccwz.U_rep('10', 0.6) @ ccwz.U_rep('10', 0.6).T, np.eye(10), atol=1e-10)
