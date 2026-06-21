"""Spinorial representations of SO(5) ~= Sp(4): the 4 (Dirac spinor) and 16.

SO(5) has rank 2 and is the double cover Sp(4); its spinor irreps are not reached by the
tensor construction and need the Clifford algebra.  Five Euclidean gamma matrices Gamma^A
(4x4) with {Gamma^A, Gamma^B} = 2 delta^{AB} give the SO(5) generators Sigma^{AB} = (i/4)
[Gamma^A, Gamma^B] in the 4, the Goldstone matrix U_4(h) = exp(i (h/f) Sigma^{hat,4}), and --
by the conventions of `ccwz` -- the SO(4) branching 4 = (2,1) + (1,2).  The 16 is built as the
spinor (x) vector irrep (4 (x) 5 = 16 + 4) projected onto its 16.

The same SO(4) Casimir machinery as `decompose` applies, fed the spinor generators.
"""
import numpy as np
from scipy.linalg import expm

from . import decompose as _D
from .. import ccwz as _ccwz

# Pauli matrices
_s0 = np.eye(2, dtype=complex)
_s1 = np.array([[0, 1], [1, 0]], dtype=complex)
_s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_s3 = np.array([[1, 0], [0, -1]], dtype=complex)

# Five 4x4 Euclidean gamma matrices, indexed 0..4 to match ccwz's SO(5) index convention.
GAMMA = [np.kron(_s1, _s0), np.kron(_s2, _s0), np.kron(_s3, _s1),
         np.kron(_s3, _s2), np.kron(_s3, _s3)]


def _check_clifford(tol=1e-12):
    for a in range(5):
        for b in range(5):
            anti = GAMMA[a] @ GAMMA[b] + GAMMA[b] @ GAMMA[a]
            if not np.allclose(anti, 2 * (a == b) * np.eye(4), atol=tol):
                return False
    return True


def Sigma(A, B):
    """SO(5) generator in the spinor 4: Sigma^{AB} = -(i/4)[Gamma^A, Gamma^B] (Hermitian).

    The sign matches ccwz's vector convention T^{ab}_{cd} = -i(d_ac d_bd - d_ad d_bc), so that
    spinor and vector generators close the SO(4) algebra with the *same* structure constants
    (required when they are combined, as in the 16 = 4 (x) 5 construction)."""
    return -0.25j * (GAMMA[A] @ GAMMA[B] - GAMMA[B] @ GAMMA[A])


def U_spinor(sh, hat=3):
    """Goldstone matrix in the spinor 4 (sh = sin(h/f)): exp(i theta Sigma^{hat,4}),
    theta = arcsin(sh).  A spinorial (half-angle) rotation along the broken generator."""
    theta = np.arcsin(np.clip(sh, -1.0, 1.0))
    return expm(1j * theta * Sigma(hat, 4))


def so4_generators_spinor():
    """The six SO(4) generators Sigma^{mu,nu} (mu<nu in 0..3) in the spinor 4."""
    return {(mu, nu): Sigma(mu, nu) for mu in range(4) for nu in range(mu + 1, 4)}


def so4_content_4():
    """SO(4) branching of the spinor 4 = (2,1) + (1,2)  ->  {(1/2,0):1, (0,1/2):1}."""
    return _D.so4_content_gen(so4_generators_spinor())


# ---- the 16 = (4 (x) 5) projected onto the 16 ------------------------------------------ #
def _vector_gens():
    """The six SO(4) generators in the vector 5 (Hermitian)."""
    M = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            T = np.zeros((5, 5), dtype=complex)
            T[mu, nu] = -1j
            T[nu, mu] = 1j
            M[(mu, nu)] = T
    return M


def _nullspace(A, tol=1e-9):
    u, s, vh = np.linalg.svd(A)
    return vh[np.sum(s > tol):].conj().T


def _gen_16():
    """SO(4) generators in the 16 = gamma-traceless vector-spinor Psi_A (Gamma^A Psi_A = 0),
    the irreducible part of 4 (x) 5 = 16 + 4.  Basis index = alpha*5 + A (spinor (x) vector).
    Returns (generators dict, projector cols spanning the 16)."""
    Msp = so4_generators_spinor()
    Mv = _vector_gens()
    # gamma-trace map 20 -> 4:  out_alpha = sum_{A,beta} (Gamma^A)_{alpha,beta} Psi_{beta,A}
    Ttr = np.zeros((4, 20), dtype=complex)
    for al in range(4):
        for be in range(4):
            for A in range(5):
                Ttr[al, be * 5 + A] = GAMMA[A][al, be]
    cols = _nullspace(Ttr)                          # 20 x 16
    M16 = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            G = np.kron(Msp[(mu, nu)], np.eye(5)) + np.kron(np.eye(4), Mv[(mu, nu)])
            M16[(mu, nu)] = cols.conj().T @ G @ cols
    return M16, cols


def so4_content_16():
    """SO(4) branching of the 16."""
    M16, _ = _gen_16()
    return _D.so4_content_gen(M16)


def U_16(sh, hat=3):
    """Goldstone matrix in the 16 (sh = sin(h/f)): the rotation on 4 (x) 5 = U_4 (x) U_5
    restricted to the gamma-traceless 16 subspace."""
    _, cols = _gen_16()
    U20 = np.kron(U_spinor(sh, hat), _ccwz.U_vector(sh, hat))
    return cols.conj().T @ U20 @ cols
