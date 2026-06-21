"""Spinorial representations of SO(6) ~= SU(4): the Weyl spinors 4 and 4bar.

SO(6) has rank 3 and is the double cover SU(4); its two chiral spinor irreps (the 4 and the
conjugate 4bar of SU(4)) are not reached by the tensor construction and need the 8-dimensional
Clifford algebra.  We build the six Euclidean gammas Gamma^A (8x8, {Gamma^A,Gamma^B}=2 delta)
from the five SO(5) gammas of `spinors` (so the conventions nest), form the generators
Sigma^{AB} = -(i/4)[Gamma^A, Gamma^B], split 8 = 4 + 4bar by the chirality chi = sigma_3 (x) 1,
and lift the SO(6)/SO(5) Goldstone (the Higgs+singlet) to the 4.

The SO(4) and SO(5) Casimir machinery of `decompose`/`so6` applies, fed the spinor generators:
the 4 of SO(6) branches to the 4 of SO(5) (thesis: the 4 has no SU(2)_LxSU(2)_R bidoublet),
which in turn is (2,1)+(1,2) under SO(4).
"""
import numpy as np
from scipy.linalg import expm

from . import so6
from . import decompose as _D
from .spinors import GAMMA as _G5            # five 4x4 Euclidean SO(5) gammas

_s1 = np.array([[0, 1], [1, 0]], dtype=complex)
_s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_s3 = np.array([[1, 0], [0, -1]], dtype=complex)
_I4 = np.eye(4, dtype=complex)

# Six 8x8 Euclidean gammas, indexed 0..5 to match so6's index convention (SO(5) on 0..4,
# coset = index 5):  Gamma^a = sigma1 (x) gamma5^a (a=0..4),  Gamma^5 = sigma2 (x) 1.
GAMMA = [np.kron(_s1, _G5[a]) for a in range(5)] + [np.kron(_s2, _I4)]
CHI = np.kron(_s3, _I4)                       # chirality: +1 -> 4, -1 -> 4bar


def _check_clifford(tol=1e-12):
    for a in range(6):
        for b in range(6):
            anti = GAMMA[a] @ GAMMA[b] + GAMMA[b] @ GAMMA[a]
            if not np.allclose(anti, 2 * (a == b) * np.eye(8), atol=tol):
                return False
    # chirality anticommutes with every gamma and squares to 1
    return all(np.allclose(CHI @ g + g @ CHI, 0, atol=tol) for g in GAMMA) \
        and np.allclose(CHI @ CHI, np.eye(8), atol=tol)


def Sigma(A, B):
    """SO(6) generator in the Dirac spinor 8: Sigma^{AB} = -(i/4)[Gamma^A, Gamma^B] (Hermitian)."""
    return -0.25j * (GAMMA[A] @ GAMMA[B] - GAMMA[B] @ GAMMA[A])


def _chiral_cols(sign):
    """Orthonormal columns spanning the chi = sign eigenspace (the 4 for +1, the 4bar for -1)."""
    w, V = np.linalg.eigh(CHI)
    return V[:, np.abs(w - sign) < 1e-9]


def Sigma_chiral(A, B, sign=+1):
    """Sigma^{AB} restricted to the chiral 4 (sign=+1) or 4bar (sign=-1)."""
    cols = _chiral_cols(sign)
    return cols.conj().T @ Sigma(A, B) @ cols


def U4(th, ts, hat=3, sign=+1):
    """Goldstone matrix in the Weyl 4 (or 4bar): the (Higgs, singlet) coset rotation
    exp(i(th Sigma^{hat,5} + ts Sigma^{4,5})) projected onto the chiral subspace."""
    cols = _chiral_cols(sign)
    gen = th * Sigma(hat, so6.COSET) + ts * Sigma(4, so6.COSET)
    return cols.conj().T @ expm(1j * gen) @ cols


def _so_generators(sign, pairs):
    """The chiral-4 generators Sigma^{ab} for the given index pairs (a dict keyed by (a,b))."""
    cols = _chiral_cols(sign)
    return {(a, b): cols.conj().T @ Sigma(a, b) @ cols for (a, b) in pairs}


def so5_casimir_4(sign=+1):
    """SO(5) quadratic Casimir on the chiral 4 (should be the 4's value, 5/2)."""
    g = _so_generators(sign, so6.UNBROKEN)
    return sum(G @ G for G in g.values())


def so5_content_4(sign=+1):
    """SO(5) branching of the chiral 4 -> the SO(5) spinor 4."""
    return so6.so5_content_gen(so5_casimir_4(sign))


def so4_content_4(sign=+1):
    """SO(4) = SU(2)xSU(2) branching of the chiral 4 -> (1/2,0) + (0,1/2)."""
    M = _so_generators(sign, [(mu, nu) for mu in range(4) for nu in range(mu + 1, 4)])
    return _D.so4_content_gen(M)
