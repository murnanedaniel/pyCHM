"""NMCHM spinors: the Weyl 4 and 4bar of SO(6) ~= SU(4), built from the 8-dim Clifford
algebra and validated by internal consistency and the thesis branchings.

Thesis (Ch.7): the 4 of SU(4) has no SU(2)_L x SU(2)_R bidoublet, so it cannot host the SM
q_L -- the reason the NM4DCHM uses the 6, not the 4.  We check exactly that branching here:
the 4 -> SO(5) spinor 4 -> SO(4) (1/2,0)+(0,1/2) (no (1/2,1/2) bidoublet).
"""
import numpy as np
import pytest

pytest.importorskip("sympy")

from pychm.groups import so6_spinors as S


def test_clifford_and_chirality():
    """Six 8x8 gammas obey {Gamma^A,Gamma^B}=2 delta^{AB}; chirality anticommutes with each
    and squares to 1 (so it splits 8 = 4 + 4bar)."""
    assert S._check_clifford()


@pytest.mark.parametrize("sign,name", [(+1, '4'), (-1, '4bar')])
def test_weyl_goldstone_unitary_identity_homomorphism(sign, name):
    I = np.eye(4)
    assert np.allclose(S.U4(0.0, 0.0, sign=sign), I, atol=1e-12)
    for th, ts in [(0.3, 0.2), (0.7, -0.4)]:
        U = S.U4(th, ts, sign=sign)
        assert np.allclose(U @ U.conj().T, I, atol=1e-12)
    a, b = 0.21, 0.33                                    # one-parameter subgroup (ts=0)
    assert np.allclose(S.U4(a, 0, sign=sign) @ S.U4(b, 0, sign=sign),
                       S.U4(a + b, 0, sign=sign), atol=1e-12)


@pytest.mark.parametrize("sign", [+1, -1])
def test_spinor_branchings_match_thesis(sign):
    """4 (and 4bar) -> SO(5) spinor 4 (Casimir 5/2) -> SO(4) (1/2,0)+(0,1/2): no bidoublet,
    so the 4 cannot embed the SM q_L (thesis Ch.7)."""
    assert S.so5_content_4(sign) == [('4', 4)]
    assert S.so4_content_4(sign) == {(0.5, 0.0): 1, (0.0, 0.5): 1}
    # the SO(5) quadratic Casimir on the 4 is 5/2 (matches so6.SO5_CASIMIR calibration)
    C = S.so5_casimir_4(sign)
    assert np.allclose(np.linalg.eigvalsh((C + C.conj().T) / 2), 2.5, atol=1e-9)


def test_weyl_half_angle():
    """The Weyl Goldstone along the Higgs direction (ts=0) is a half-angle rotation:
    eigenphases +/- theta/2 (the SO(4) (1/2,0)+(0,1/2) content)."""
    th = 0.6
    phases = np.sort(np.angle(np.linalg.eigvals(S.U4(th, 0.0))))
    assert np.allclose(np.abs(phases), th / 2, atol=1e-9)


def test_chiral_subspaces_complete():
    """The 4 and 4bar are orthogonal and together fill the Dirac 8 (chi = +1 and -1 eigenspaces)."""
    c4 = S._chiral_cols(+1)
    c4b = S._chiral_cols(-1)
    assert c4.shape == (8, 4) and c4b.shape == (8, 4)
    assert np.allclose(c4.conj().T @ c4b, 0, atol=1e-12)            # orthogonal
    P = c4 @ c4.conj().T + c4b @ c4b.conj().T
    assert np.allclose(P, np.eye(8), atol=1e-12)                    # complete
