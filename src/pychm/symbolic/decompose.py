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


def so4_generators(basis):
    """The six SO(4) generators (indices 0..3) lifted to the tensor irrep spanned by `basis`."""
    return {(mu, nu): _gen_rep(basis, mu, nu) for mu in range(4) for nu in range(mu + 1, 4)}


def _su2_casimirs(M):
    """Return (C_L, C_R), the SU(2)_L and SU(2)_R Casimir matrices, from the six SO(4)
    generators M[(mu,nu)] (mu<nu in 0..3) of any representation (tensor or spinor)."""
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


def so4_decompose_gen(M, tol=1e-6):
    """SO(4) sub-multiplets [((jL, jR), projector_columns), ...] from the six SO(4) generators
    M[(mu,nu)] of any representation (tensor or spinor)."""
    CL, CR = _su2_casimirs(M)
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


def so4_decompose(basis, tol=1e-6):
    """SO(4) sub-multiplets of the tensor irrep spanned by `basis`.  See so4_decompose_gen."""
    return so4_decompose_gen(so4_generators(basis), tol)


def _content(decomp):
    table = {}
    for (jL, jR), cols in decomp:
        table[(jL, jR)] = table.get((jL, jR), 0) + 1
    return table


def so4_content(basis, tol=1e-6):
    """Multiplicity table {(jL, jR): count} of the SO(4) branching of the tensor irrep."""
    return _content(so4_decompose(basis, tol))


def so4_content_gen(M, tol=1e-6):
    """Multiplicity table {(jL, jR): count} from the six SO(4) generators of any rep."""
    return _content(so4_decompose_gen(M, tol))


def compatible(basis, jL, jR, tol=1e-6):
    """True iff the irrep contains the SO(4) multiplet (jL, jR) -- i.e. an elementary fermion
    in (jL, jR) can be embedded in (mixed with a partner of) this representation."""
    return (jL, jR) in so4_content(basis, tol)


def channel_weights_sym(rep, embedding, hat=3):
    """Exact squared SO(4)-channel projections W_{(jL,jR)}(theta) of the Goldstone-dressed
    `embedding` in the tensor irrep `rep` ('10' or '14').

    The Higgs vev dresses the elementary embedding E as U_R(theta) E = Uv E Uv^T; this lands in
    the rep, where it splits into SO(4) sub-multiplets.  W_{(jL,jR)} is the squared length of the
    (jL,jR) component -- a closed trigonometric polynomial -- and sum_channels W = 1 (unitarity).
    These channel weights are exactly the thesis Pi^(a) form-factor weights up to one overall
    coupling normalization (purely group-theoretic Clebsch factors otherwise).

    Returns {(jL, jR): sympy_expr_in(theta)}.  Pure group theory: no coupling/d-factors.
    """
    import sympy as sp
    from . import core

    Uv = core.U_vector_sym(hat)
    E = sp.Matrix(embedding)
    dressed = Uv * E * Uv.T
    sym_basis = core.rep_basis_sym(rep)
    coord = sp.Matrix([core._frob(b, dressed) for b in sym_basis])     # coords in the rep basis

    num_basis = [np.array(sp.matrix2numpy(b, dtype=float)) for b in sym_basis]
    out = {}
    for (jL, jR), cols in so4_decompose(num_basis):
        P = sp.nsimplify(sp.Matrix((cols @ cols.conj().T).real), rational=True)  # exact projector
        out[(jL, jR)] = sp.simplify((coord.T * P * coord)[0])
    return out

