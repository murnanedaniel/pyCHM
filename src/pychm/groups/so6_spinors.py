"""SO(6) ~= SU(4) Weyl spinors (the 4 and 4bar), as an instance on the engine.

Six 8x8 Euclidean gammas are built from the five SO(5) gammas (`symbolic.spinors.GAMMA`) tensored
with Pauli matrices, split by chirality into the 4/4bar; the Goldstone is the coset rotation
projected onto a chiral subspace.  Branchings use the shared Casimir machinery (`groups.branch`).
This is also the cross-check oracle for the SU(4)/Sp(4) coset (SU(4) fundamental = this 4).
"""
import numpy as np
from scipy.linalg import expm

from . import so6, branch
from . import decompose as _D
from .spinors import GAMMA as _G5            # five 4x4 Euclidean SO(5) gammas

_s1 = np.array([[0, 1], [1, 0]], dtype=complex)
_s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_s3 = np.array([[1, 0], [0, -1]], dtype=complex)
_I4 = np.eye(4, dtype=complex)

GAMMA = [np.kron(_s1, _G5[a]) for a in range(5)] + [np.kron(_s2, _I4)]
CHI = np.kron(_s3, _I4)                       # chirality: +1 -> 4, -1 -> 4bar


def _check_clifford(tol=1e-12):
    for a in range(6):
        for b in range(6):
            anti = GAMMA[a] @ GAMMA[b] + GAMMA[b] @ GAMMA[a]
            if not np.allclose(anti, 2 * (a == b) * np.eye(8), atol=tol):
                return False
    return all(np.allclose(CHI @ g + g @ CHI, 0, atol=tol) for g in GAMMA) \
        and np.allclose(CHI @ CHI, np.eye(8), atol=tol)


def Sigma(A, B):
    """SO(6) generator in the Dirac spinor 8: Sigma^{AB} = -(i/4)[Gamma^A, Gamma^B] (Hermitian)."""
    return -0.25j * (GAMMA[A] @ GAMMA[B] - GAMMA[B] @ GAMMA[A])


def _chiral_cols(sign):
    w, V = np.linalg.eigh(CHI)
    return V[:, np.abs(w - sign) < 1e-9]


def Sigma_chiral(A, B, sign=+1):
    cols = _chiral_cols(sign)
    return cols.conj().T @ Sigma(A, B) @ cols


def U4(th, ts, hat=3, sign=+1):
    """Goldstone in the Weyl 4 (or 4bar): exp(i(th Sigma^{hat,5} + ts Sigma^{4,5})) on the chiral
    subspace."""
    cols = _chiral_cols(sign)
    gen = th * Sigma(hat, so6.COSET) + ts * Sigma(4, so6.COSET)
    return cols.conj().T @ expm(1j * gen) @ cols


def _so_generators(sign, pairs):
    cols = _chiral_cols(sign)
    return {(a, b): cols.conj().T @ Sigma(a, b) @ cols for (a, b) in pairs}


def so5_casimir_4(sign=+1):
    g = _so_generators(sign, so6.UNBROKEN)
    return sum(G @ G for G in g.values())


def so5_content_4(sign=+1):
    return branch.content_from_casimir(so5_casimir_4(sign), branch.SO5_CASIMIR)


def so4_content_4(sign=+1):
    M = _so_generators(sign, [(mu, nu) for mu in range(4) for nu in range(mu + 1, 4)])
    return _D.so4_content_gen(M)
