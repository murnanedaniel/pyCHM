"""SO(5)/SO(4) coset (MCHM) as an instance of the group-agnostic engine.

Merges the former `pychm.ccwz` (numeric Goldstone dressing) and `pychm.symbolic.core` (the closed
symbolic mirror) into one coset module; the generator construction comes from `groups.lie`, the
branchings from `groups.decompose`.  SO(5)-specific content: the coset convention (SO(4) on 0..3,
coset index 4), the 5/10/14 irreps and their embeddings, the closed-form `s_h`-dressing and the SO(4)
channel weights.  The vector Goldstone keeps its planar `s_h = sin(h/f)` form (no `arcsin`
round-trip) -- the SO(5) specialisation of the engine's Rodrigues rotation.
"""
import numpy as np
import sympy as sp

from . import lie
from . import tensors, decompose as _D

# ===================================================================================== #
#  NUMERIC SO(5) in the vector (5)  (was pychm.ccwz)
# ===================================================================================== #
def gen_vector(A, B):
    """Antisymmetric SO(5) generator T^{AB} in the vector (engine `lie.so_generator`)."""
    return lie.so_generator(A, B, 5, herm=True)


UNBROKEN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # SO(4)
BROKEN = [(0, 4), (1, 4), (2, 4), (3, 4)]                     # SO(5)/SO(4)


def U_vector(sh, hat=3):
    """Goldstone matrix in the 5 as a function of sh = sin(h/f): a planar rotation by h/f in the
    (hat,4) plane, in closed form with c = sqrt(1-sh^2) (exact, no arcsin round-trip)."""
    U = np.eye(5)
    c, s = np.sqrt(max(0.0, 1.0 - sh * sh)), sh
    U[hat, hat] = c; U[4, 4] = c; U[hat, 4] = s; U[4, hat] = -s
    return U


def _sym_traceless_basis():
    B = []
    for i in range(5):
        for j in range(i + 1, 5):
            M = np.zeros((5, 5)); M[i, j] = M[j, i] = 1 / np.sqrt(2); B.append(M)
    for d in (np.diag([1., -1, 0, 0, 0]), np.diag([1., 1, -2, 0, 0]) / np.sqrt(3),
              np.diag([1., 1, 1, -3, 0]) / np.sqrt(6), np.diag([1., 1, 1, 1, -4]) / np.sqrt(10)):
        B.append(d / np.sqrt((d * d).sum()))
    return B   # 14 orthonormal symmetric traceless matrices


def _antisym_basis():
    B = []
    for i in range(5):
        for j in range(i + 1, 5):
            M = np.zeros((5, 5)); M[i, j] = 1 / np.sqrt(2); M[j, i] = -1 / np.sqrt(2); B.append(M)
    return B   # 10 orthonormal antisymmetric matrices


def U_rep(rep, sh, hat=3):
    """Goldstone matrix in an SO(5) irrep (sh = sin(h/f)).  rep is '5', '10', '14', or a tensor
    descriptor ('sym'/'antisym', k) for an arbitrary rank-k irrep (delegated to `tensors`)."""
    Uv = U_vector(sh, hat)
    if rep == '5':
        return Uv
    if rep in ('10', '14'):
        basis = _sym_traceless_basis() if rep == '14' else _antisym_basis()
        n = len(basis)
        M = np.zeros((n, n))
        for a, Ea in enumerate(basis):
            UEU = Uv @ Ea @ Uv.T
            for b, Eb in enumerate(basis):
                M[b, a] = np.sum(Eb * UEU)
        return M
    sym, rank = rep
    return tensors.U_rep_tensor(tensors.tensor_basis(sym, rank, 5), Uv)


def embedding(rep, kind):
    """Orthonormal tensor for an SO(4) sub-multiplet of `rep`.
    kind: 'singlet' = (1,1); 'fourplet_i' (i=0..3) = the (2,2) vector component."""
    if rep == '5':
        if kind == 'singlet':
            e = np.zeros(5); e[4] = 1.0; return e
        i = int(kind.split('_')[1]); e = np.zeros(5); e[i] = 1.0; return e
    if rep == '14':
        if kind == 'singlet':
            return np.diag([1., 1, 1, 1, -4]) / np.sqrt(20)         # (1,1)
        i = int(kind.split('_')[1])                                  # (2,2): index i with the 5-leg
        M = np.zeros((5, 5)); M[i, 4] = M[4, i] = 1 / np.sqrt(2); return M
    raise ValueError(rep)


def overlap(rep, bra, ket, sh, hat=3, Uv=None):
    """<bra| U_rep |ket> (sh = sin(h/f)): the Higgs-dressed mixing factor between embeddings."""
    if Uv is None:
        Uv = U_vector(sh, hat)
    if rep == '5':
        return complex(np.vdot(bra, Uv @ ket))
    return complex(np.sum(np.conjugate(bra) * (Uv @ ket @ Uv.T)))


# ===================================================================================== #
#  SYMBOLIC SO(5) (was pychm.symbolic.core)
# ===================================================================================== #
theta = sp.symbols('theta', real=True)
_s, _c = sp.sin(theta), sp.cos(theta)
sh = sp.symbols('sh', real=True)
ch = sp.sqrt(1 - sh**2)
_R2 = sp.sqrt(2)


def U_vector_sym(hat=3):
    """5x5 Goldstone matrix in the vector (5): a planar rotation by theta in the (hat,4) plane."""
    U = sp.eye(5)
    U[hat, hat] = _c
    U[4, 4] = _c
    U[hat, 4] = _s
    U[4, hat] = -_s
    return U


def _sym_traceless_basis_sym():
    B = []
    for i in range(5):
        for j in range(i + 1, 5):
            M = sp.zeros(5, 5)
            M[i, j] = M[j, i] = 1 / _R2
            B.append(M)
    diags = (sp.Matrix([1, -1, 0, 0, 0]),
             sp.Matrix([1, 1, -2, 0, 0]) / sp.sqrt(3),
             sp.Matrix([1, 1, 1, -3, 0]) / sp.sqrt(6),
             sp.Matrix([1, 1, 1, 1, -4]) / sp.sqrt(10))
    for d in diags:
        norm = sp.sqrt(sum(x**2 for x in d))
        B.append(sp.diag(*(d / norm)))
    return B   # 14 matrices


def _antisym_basis_sym():
    B = []
    for i in range(5):
        for j in range(i + 1, 5):
            M = sp.zeros(5, 5)
            M[i, j] = 1 / _R2
            M[j, i] = -1 / _R2
            B.append(M)
    return B   # 10 matrices


def rep_basis_sym(rep):
    """Orthonormal tensor basis of the irrep (sympy). '14' = symmetric traceless, '10' = antisym."""
    if rep == '14':
        return _sym_traceless_basis_sym()
    if rep == '10':
        return _antisym_basis_sym()
    raise ValueError(f"rep_basis_sym: unsupported rep {rep!r} (use '10' or '14')")


def to_closed_trig(expr):
    """Reduce a polynomial in (sin theta, cos theta) to canonical closed trigonometric form."""
    return sp.trigsimp(sp.expand_trig(sp.expand(expr)))


def _frob(bra, M):
    """Frobenius overlap sum_ij conj(bra_ij) * M_ij for equal-shape matrices."""
    return sum(sp.conjugate(bra[i, j]) * M[i, j]
               for i in range(bra.rows) for j in range(bra.cols))


def U_rep_sym(rep, hat=3):
    """Goldstone matrix lifted to irrep `rep`, as an n x n sympy Matrix in closed trig form."""
    Uv = U_vector_sym(hat)
    if rep == '5':
        return Uv
    basis = rep_basis_sym(rep)
    n = len(basis)
    M = sp.zeros(n, n)
    for a, Ea in enumerate(basis):
        UEU = Uv * Ea * Uv.T
        for b, Eb in enumerate(basis):
            M[b, a] = to_closed_trig(_frob(Eb, UEU))
    return M


def overlap_sym(rep, bra, ket, hat=3):
    """<bra| U_rep |ket> in closed trig form (symbolic counterpart of `overlap`)."""
    Uv = U_vector_sym(hat)
    bra = sp.Matrix(bra)
    ket = sp.Matrix(ket)
    if rep == '5':
        expr = _frob(bra, Uv * ket)
    else:
        expr = _frob(bra, Uv * ket * Uv.T)
    return to_closed_trig(expr)


def embedding_sym(rep, kind):
    """Orthonormal tensor for an SO(4) sub-multiplet of `rep` (symbolic `embedding`)."""
    if rep == '5':
        e = sp.zeros(5, 1)
        if kind == 'singlet':
            e[4] = 1
        else:
            e[int(kind.split('_')[1])] = 1
        return e
    if rep == '14':
        if kind == 'singlet':
            return sp.diag(1, 1, 1, 1, -4) / sp.sqrt(20)
        i = int(kind.split('_')[1])
        M = sp.zeros(5, 5)
        M[i, 4] = M[4, i] = 1 / _R2
        return M
    raise ValueError(rep)


def as_sh_expr(expr):
    """Rewrite a theta-expression in pyCHM's sh = sin(h/f) variable (ch = sqrt(1 - sh**2))."""
    e = sp.expand_trig(expr)
    e = e.subs({sp.cos(theta): ch, sp.sin(theta): sh})
    return sp.simplify(e)


def lambdify_sh(expr):
    """Compile a closed-form theta-expression to a fast numpy callable f(sh)."""
    return sp.lambdify(sh, as_sh_expr(expr), modules='numpy')


# ===================================================================================== #
#  SO(4)-channel weights of a dressed embedding (was symbolic.decompose.channel_weights_sym)
# ===================================================================================== #
def channel_weights_sym(rep, embedding, hat=3):
    """Exact squared SO(4)-channel projections W_{(jL,jR)}(theta) of the Goldstone-dressed
    `embedding` in the tensor irrep `rep` ('10' or '14').  sum_channels W = 1 (unitarity).  These
    are the thesis Pi^(a) form-factor weights up to one overall coupling normalisation."""
    Uv = U_vector_sym(hat)
    E = sp.Matrix(embedding)
    dressed = Uv * E * Uv.T
    sym_basis = rep_basis_sym(rep)
    coord = sp.Matrix([_frob(b, dressed) for b in sym_basis])
    num_basis = [np.array(sp.matrix2numpy(b, dtype=float)) for b in sym_basis]
    out = {}
    for (jL, jR), cols in _D.so4_decompose(num_basis):
        P = sp.nsimplify(sp.Matrix((cols @ cols.conj().T).real), rational=True)
        out[(jL, jR)] = sp.simplify((coord.T * P * coord)[0])
    return out
