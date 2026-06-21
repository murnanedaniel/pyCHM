"""Stage 4: spinorial representations of SO(5) ~= Sp(4) -- the 4 (Dirac spinor) and the 16.

These are not reached by the tensor construction; they come from the Clifford algebra.  The 4
is the MCHM4 partner representation; validated by the Clifford algebra, unitarity, the spinor
(half-angle) homomorphism, and the SO(4) branchings 4 = (2,1)+(1,2), 16 = (2,1)+(1,2)+(2,3)+(3,2).
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from pychm.symbolic import spinors as S


def test_clifford_algebra():
    assert S._check_clifford()


def test_spinor_4_unitary_identity_homomorphism():
    assert np.allclose(S.U_spinor(0.0), np.eye(4))
    for s in (0.2, 0.5, 0.85):
        U = S.U_spinor(s)
        assert np.allclose(U.conj().T @ U, np.eye(4), atol=1e-12)
    th1, th2 = 0.31, 0.47                                  # spinor rotations compose in theta
    assert np.allclose(S.U_spinor(np.sin(th1)) @ S.U_spinor(np.sin(th2)),
                       S.U_spinor(np.sin(th1 + th2)), atol=1e-12)


def test_spinor_half_angle():
    """U_4 is the half-angle (spinor) rotation: U_4(theta)^2 ~ rotation by 2*theta in content."""
    s = 0.5
    th = np.arcsin(s)
    U = S.U_spinor(s)
    # eigenvalues are exp(+/- i theta/2) (each doubly degenerate for the 4)
    evals = np.angle(np.linalg.eigvals(U))
    assert np.allclose(np.sort(np.abs(evals)), np.full(4, th / 2), atol=1e-9)


def test_so4_branchings():
    c4 = S.so4_content_4()
    assert c4 == {(0.5, 0.0): 1, (0.0, 0.5): 1}
    assert sum(int((2 * a + 1) * (2 * b + 1)) * m for (a, b), m in c4.items()) == 4

    c16 = S.so4_content_16()
    assert c16 == {(0.5, 0.0): 1, (0.0, 0.5): 1, (1.0, 0.5): 1, (0.5, 1.0): 1}
    assert sum(int((2 * a + 1) * (2 * b + 1)) * m for (a, b), m in c16.items()) == 16


def test_U16_unitary_identity():
    assert np.allclose(S.U_16(0.0), np.eye(16), atol=1e-10)
    for s in (0.3, 0.7):
        U = S.U_16(s)
        assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-9)


def test_spinor_compatible_with_qL_and_tR():
    """The 4 hosts q_L = (2,1) and t_R/b_R = (1,2): the MCHM4 embedding."""
    from pychm.symbolic import decompose as D
    M4 = S.so4_generators_spinor()
    content = D.so4_content_gen(M4)
    assert (0.5, 0.0) in content and (0.0, 0.5) in content
