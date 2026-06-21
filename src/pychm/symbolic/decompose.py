"""SO(4) = SU(2)_L x SU(2)_R decomposition of an SO(5) tensor irrep.

Given an orthonormal basis of a tensor irrep (from `tensors`), find its SO(4) sub-multiplets
(j_L, j_R): the Goldstone vev breaks SO(5) -> SO(4) and the elementary fermions are SO(4)
multiplets, so the admissible ("compatible") partner representations are exactly the SO(5)
irreps whose SO(4) branching contains the elementary field's (j_L, j_R).  This generalises the
hand-placed embeddings of the 5/10/14 to any representation.

SO(4) acts on indices 0..3 (index 4 is an SO(4) scalar), matching ccwz's conventions.
"""
import numpy as np

from .tensors import lift

# SU(2)_L/R from the SO(4) generators M_{mu,nu} (mu<nu in 0..3) via 't Hooft combinations:
#   J^L_i = 1/2 ( 1/2 eps_{ijk} M_{jk} + M_{0i} ),   J^R_i = 1/2 ( 1/2 eps_{ijk} M_{jk} - M_{0i} )
_EPS = {(1, 2): 3, (2, 3): 1, (3, 1): 2, (2, 1): -3, (3, 2): -1, (1, 3): -2}


def _gen_vector(a, b, n=5):
    """Hermitian SO(n) generator T^{ab} in the vector: (T)_{cd} = -i(d_ac d_bd - d_ad d_bc)."""
    T = np.zeros((n, n), dtype=complex)
    T[a, b] = -1j
    T[b, a] = 1j
    return T


def _gen_rep(basis, a, b):
    """Lift the vector generator T^{ab} to the irrep spanned by `basis` (as a derivation):
    G[p,q] = <E_p | sum_axes T^{ab}-on-that-axis E_q>."""
    n = len(basis)
    Uv = _gen_vector(a, b)
    G = np.zeros((n, n), dtype=complex)
    for q, Eq in enumerate(basis):
        # derivation: apply the generator to each tensor index and sum
        acc = np.zeros_like(Eq, dtype=complex)
        for ax in range(Eq.ndim):
            t = np.tensordot(Uv, Eq, axes=([1], [ax]))
            acc = acc + np.moveaxis(t, 0, ax)
        for p, Ep in enumerate(basis):
            G[p, q] = np.vdot(Ep, acc)
    return G


def _su2_casimirs(basis):
    """Return (C_L, C_R), the SU(2)_L and SU(2)_R Casimir matrices in the irrep."""
    M = {(mu, nu): _gen_rep(basis, mu, nu) for mu in range(4) for nu in range(mu + 1, 4)}

    def Msym(a, b):
        return M[(a, b)] if a < b else -M[(b, a)]

    def J(sign):
        # J^{L/R}_i = 1/2 ( L_i +/- K_i ),  L_i = eps_{ijk} M_{jk} (rotations), K_i = M_{0i}.
        Js = []
        for i in (1, 2, 3):
            j, k = [(2, 3), (3, 1), (1, 2)][i - 1]
            Js.append(0.5 * (Msym(j, k) + sign * Msym(0, i)))
        return Js

    JL, JR = J(+1), J(-1)
    CL = sum(j @ j for j in JL)
    CR = sum(j @ j for j in JR)
    return CL, CR


def _round_j(casimir_eig):
    """Map a Casimir eigenvalue j(j+1) back to j (half-integer)."""
    j = (-1 + np.sqrt(1 + 4 * max(0.0, casimir_eig.real))) / 2
    return round(2 * j) / 2


def so4_decompose(basis, tol=1e-6):
    """List the SO(4) sub-multiplets of the irrep as [((jL, jR), projector_columns), ...].

    projector_columns is an (n x d) matrix whose columns are an orthonormal basis of that
    (jL, jR) subspace (n = dim irrep, d = (2jL+1)(2jR+1))."""
    CL, CR = _su2_casimirs(basis)
    # CL and CR commute and are Hermitian -> simultaneously diagonalise.
    wL, VL = np.linalg.eigh(CL)
    # rotate CR into the CL eigenbasis and diagonalise block-wise by grouping equal wL
    out = []
    used = np.zeros(len(wL), dtype=bool)
    for i in range(len(wL)):
        if used[i]:
            continue
        grp = np.where(np.abs(wL - wL[i]) < tol)[0]
        used[grp] = True
        Vg = VL[:, grp]
        CRsub = Vg.conj().T @ CR @ Vg
        wR, VR = np.linalg.eigh(CRsub)
        # group equal wR
        seen = np.zeros(len(wR), dtype=bool)
        for a in range(len(wR)):
            if seen[a]:
                continue
            g2 = np.where(np.abs(wR - wR[a]) < tol)[0]
            seen[g2] = True
            cols = Vg @ VR[:, g2]
            jL, jR = _round_j(wL[i]), _round_j(wR[a])
            out.append(((jL, jR), cols))
    out.sort(key=lambda t: (t[0][0], t[0][1]))
    return out


def so4_content(basis, tol=1e-6):
    """Multiplicity table {(jL, jR): count} of the SO(4) branching of the irrep."""
    table = {}
    for (jL, jR), cols in so4_decompose(basis, tol):
        table[(jL, jR)] = table.get((jL, jR), 0) + 1
    return table


def compatible(basis, jL, jR, tol=1e-6):
    """True iff the irrep contains the SO(4) multiplet (jL, jR) -- i.e. an elementary fermion
    in (jL, jR) can be embedded in (mixed with a partner of) this representation."""
    return (jL, jR) in so4_content(basis, tol)
