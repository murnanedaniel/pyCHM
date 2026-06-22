"""SO(6)/SO(5) coset (NMCHM) as an instance of the group-agnostic engine.

The generator construction, the closed-form Goldstone (`rodrigues_exp`), the index-lift of
generators to a rep, and the Casimir/Hodge branchings all come from `groups.lie/reps/branch` -- this
module holds only the SO(6)-specific content: the coset convention (SO(5) on 0..4, coset index 5),
the irrep tower (6/15/20'/10/10bar), the fermion embeddings of the 6 (thesis eq. 633), and the
closed-form (h,s) dressing/channel weights (thesis Ch.7, eq. 474).
"""
import numpy as np
import sympy as sp

from . import tensors, decompose as _D
from . import lie, reps, branch

N = 6                       # SO(6) acts on 6-dimensional indices
COSET = 5                   # the SO(6)/SO(5) direction (index 5); SO(5) on 0..4
UNBROKEN = [(a, b) for a in range(5) for b in range(a + 1, 5)]   # 10 = SO(5)
BROKEN = [(a, COSET) for a in range(5)]                          # 5 = SO(6)/SO(5)
_R2 = np.sqrt(2.0)
SO5_CASIMIR = branch.SO5_CASIMIR


def gen_vector(A, B, herm=True):
    """SO(6) generator T^{AB} in the vector 6 (engine `lie.so_generator`)."""
    return lie.so_generator(A, B, N, herm=herm)


# --------------------------------------------------------------------------------------- #
#  Vector Goldstone (numeric + closed-form symbolic), via the shared Rodrigues engine
# --------------------------------------------------------------------------------------- #
def U6_vector(th, ts, hat=3):
    """Numeric 6x6 Goldstone in the vector: exp(th L^{hat,5} + ts L^{4,5})."""
    G = th * gen_vector(hat, COSET, herm=False) + ts * gen_vector(4, COSET, herm=False)
    return lie.rodrigues_exp(G)


def U6_vector_sym(th, ts, hat=3):
    """Closed-form sympy 6x6 Goldstone in the vector (Rodrigues)."""
    G = th * lie.so_generator_sym(hat, COSET, N) + ts * lie.so_generator_sym(4, COSET, N)
    return sp.simplify(lie.rodrigues_exp(G))


def goldstone_vacuum(th, ts, hat=3):
    """Phi = U6 . e_5: the Goldstone field configuration (thesis eq. 474)."""
    e5 = sp.zeros(N, 1)
    e5[COSET] = 1
    return sp.simplify(U6_vector_sym(th, ts, hat) * e5)


# --------------------------------------------------------------------------------------- #
#  Tensor irreps and the Goldstone lift to them
# --------------------------------------------------------------------------------------- #
def rep_basis(rep):
    """Orthonormal numpy tensor basis of an SO(6) irrep.
    '6' vector · '15' antisym-2 (adjoint) · "20'" sym-traceless-2 · '20'/'10'/'10bar' antisym-3."""
    if rep == '6':
        return [np.eye(N)[i] for i in range(N)]
    if rep == '15':
        return tensors.tensor_basis('antisym', 2, N)
    if rep == "20'":
        return tensors.tensor_basis('sym', 2, N)
    if rep in ('20', '10', '10bar'):
        return tensors.tensor_basis('antisym', 3, N)
    raise ValueError(f"unknown SO(6) rep {rep!r}")


def U6_rep(rep, th, ts, hat=3):
    """Goldstone matrix in the SO(6) irrep `rep` (numeric).  '10'/'10bar' are returned in the
    (anti-)self-dual basis (10x10 complex)."""
    Uv = U6_vector(th, ts, hat)
    if rep == '6':
        return Uv
    basis = rep_basis(rep)
    M = tensors.U_rep_tensor(basis, Uv)
    if rep in ('10', '10bar'):
        cols = branch.selfdual_cols(basis, N, sign=+1 if rep == '10' else -1)
        return cols.conj().T @ M.astype(complex) @ cols
    return M


# --------------------------------------------------------------------------------------- #
#  Branchings: SO(6) irrep -> SO(5) and -> SO(4)
# --------------------------------------------------------------------------------------- #
def so5_casimir(basis):
    """SO(5) quadratic Casimir C5 = sum_{a<b in 0..4}(T^{ab})^2 on the irrep `basis`."""
    gens = [reps._lift_generator(basis, gen_vector(a, b)) for (a, b) in UNBROKEN]
    return sum(g @ g for g in gens)


def so5_content(basis, tol=1e-6):
    """SO(5) branching [(name, dim), ...] of the tensor irrep spanned by `basis`."""
    return branch.content_from_casimir(so5_casimir(basis), SO5_CASIMIR, tol)


def so4_content(basis, tol=1e-6):
    """SO(4)=SU(2)xSU(2) branching {(jL,jR): mult} of the SO(6) irrep (indices 0..3)."""
    M = {(mu, nu): reps._lift_generator(basis, gen_vector(mu, nu))
         for mu in range(4) for nu in range(mu + 1, 4)}
    return branch.so4_content(M, tol)


# --------------------------------------------------------------------------------------- #
#  Fermion embeddings in the 6 (thesis eq. 633) and the Goldstone-dressed overlaps
# --------------------------------------------------------------------------------------- #
def embedding(rep, kind):
    """Orthonormal embedding tensor for an SO(4) sub-multiplet of the 6 (= 4 + 1 + 1):
    'fourplet_i' (i=0..3) the bidoublet e_i; 'singlet5' the SO(5)-vector singlet e_4; 'singlet6'
    the SO(5)-singlet e_5 (thesis eq. 633)."""
    if rep == '6':
        e = np.zeros(N, dtype=complex)
        if kind == 'singlet5':
            e[4] = 1.0
        elif kind == 'singlet6':
            e[COSET] = 1.0
        else:
            e[int(kind.split('_')[1])] = 1.0
        return e
    raise ValueError(f"embedding not defined for rep {rep!r}")


def overlap(rep, bra, ket, th, ts, hat=3, U=None):
    """<bra| U_rep(th,ts) |ket>: the (Higgs, singlet)-dressed mixing factor between embeddings."""
    if U is None:
        U = U6_vector(th, ts, hat)
    if rep == '6':
        return complex(np.vdot(bra, U @ ket))
    raise ValueError(f"overlap not defined for rep {rep!r}")


def channel_weights6(EqL, th, ts, hat=3):
    """Squared SO(4)-channel projections of the (h,s)-dressed q_L embedding in the 6 (sum=|E|^2)."""
    U = U6_vector(th, ts, hat)
    dressed = U @ np.asarray(EqL, dtype=complex)
    return {('2,2',): float(np.vdot(dressed[:4], dressed[:4]).real),
            ('1_5',): float(abs(dressed[4])**2),
            ('1_6',): float(abs(dressed[5])**2)}


# --------------------------------------------------------------------------------------- #
#  Closed-form (symbolic) dressing in the 6 (derived, not fitted)
# --------------------------------------------------------------------------------------- #
def _trig(expr):
    return sp.simplify(sp.trigsimp(sp.expand_trig(sp.expand(expr))))


def _U6_clean_sym(th, ts, hat):
    """Guard-free closed-form vector Goldstone for the symbolic derivations (engine Rodrigues)."""
    G = th * lie.so_generator_sym(hat, COSET, N) + ts * lie.so_generator_sym(4, COSET, N)
    return lie.rodrigues_exp_clean(G)


def overlap6_sym(bra, ket, th, ts, hat=3):
    """<bra| U6(th,ts) |ket> in closed trigonometric form (sympy)."""
    U = _U6_clean_sym(th, ts, hat)
    bra, ket = sp.Matrix(bra), sp.Matrix(ket)
    return _trig(sum(sp.conjugate(bra[i]) * (U * ket)[i] for i in range(N)))


def channel_weights6_sym(EqL, th, ts, hat=3):
    """Closed-form squared SO(4)-channel weights of the (h,s)-dressed q_L embedding in the 6."""
    U = _U6_clean_sym(th, ts, hat)
    d = U * sp.Matrix(EqL)
    return {('2,2',): _trig(sum(sp.conjugate(d[i]) * d[i] for i in range(4))),
            ('1_5',): _trig(sp.conjugate(d[4]) * d[4]),
            ('1_6',): _trig(sp.conjugate(d[5]) * d[5])}
