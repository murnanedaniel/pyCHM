"""The generic CCWZ Goldstone machinery reproduces the rep-specific s_h-factors that are
hand-coded in mchm5 / mchm14, computed from group theory alone."""
import numpy as np
import pychm.ccwz as ccwz


def test_so5_vector_is_a_rotation():
    U = ccwz.U_vector(0.37)
    assert np.allclose(U @ U.T, np.eye(5))          # orthogonal
    assert np.isclose(np.linalg.det(U), 1.0)         # proper rotation


def test_rep5_factors_match_handcoded():
    """The 5: U components ARE cos(h/f), sin(h/f); the (2,2) gives the half-angle factors."""
    th = 0.41
    U = ccwz.U_vector(th)
    assert np.isclose(U[3, 3], np.cos(th))
    assert np.isclose(U[3, 4], np.sin(th))
    # mchm5's c2h2 = cos^2(h/2f), s2h2 = sin^2(h/2f)
    assert np.isclose((1 + np.cos(th)) / 2, np.cos(th / 2) ** 2)
    assert np.isclose((1 - np.cos(th)) / 2, np.sin(th / 2) ** 2)


def test_rep14_singlet_factor_is_exact():
    """The 14: <(1,1)|U14|(1,1)> = (3 + 5 cos(2h/f))/8 -- the exact mchm14 m[4,2] factor,
    a normalisation-independent diagonal overlap, reproduced to machine precision."""
    for th in (0.0, 0.25, 0.5, 0.9, 1.3):
        ES = ccwz.embedding('14', 'singlet')
        val = ccwz.overlap('14', ES, ES, th)
        assert np.isclose(val, (3 + 5 * np.cos(2 * th)) / 8, atol=1e-12)


def test_rep14_unitarity_and_identity():
    assert np.allclose(ccwz.U_rep('14', 0.0), np.eye(14))
    U = ccwz.U_rep('14', 0.6)
    assert np.allclose(U @ U.T, np.eye(14), atol=1e-10)   # orthogonal in the 14


def test_rep10_unitarity_and_identity():
    assert np.allclose(ccwz.U_rep('10', 0.0), np.eye(10))
    U = ccwz.U_rep('10', 0.6)
    assert np.allclose(U @ U.T, np.eye(10), atol=1e-10)   # orthogonal in the 10
