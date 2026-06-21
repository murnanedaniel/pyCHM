"""Symbolic CCWZ core: closed-form Goldstone dressing in SO(5) irreps.

Mirrors `pychm.ccwz` exactly (same conventions, same basis ordering and normalisation) but
with sympy, so that lambdifying any result reproduces the numeric `ccwz` bit-for-bit.

Conventions (identical to ccwz): SO(4) on indices 0..3, coset direction = index 4, Higgs vev
along the broken generator T^{(hat,4)} with hat=3 by default; theta = h/f is the rotation
angle, s = sin(theta) = pyCHM's sh, c = cos(theta) = ch = sqrt(1 - sh**2).
"""
import sympy as sp

# Rotation angle theta = h/f and the trig pair that appears everywhere.
theta = sp.symbols('theta', real=True)
_s, _c = sp.sin(theta), sp.cos(theta)

# pyCHM's runtime variable sh = sin(h/f); ch = cos(h/f) = sqrt(1 - sh**2).
sh = sp.symbols('sh', real=True)
ch = sp.sqrt(1 - sh**2)

_R2 = sp.sqrt(2)


def U_vector_sym(hat=3):
    """5x5 Goldstone matrix in the vector (5): a planar rotation by theta in the (hat,4) plane.
    Symbolic counterpart of ccwz.U_vector (with sh = sin theta)."""
    U = sp.eye(5)
    U[hat, hat] = _c
    U[4, 4] = _c
    U[hat, 4] = _s
    U[4, hat] = -_s
    return U


def _sym_traceless_basis():
    """14 orthonormal symmetric-traceless 5x5 matrices, in ccwz._sym_traceless_basis order."""
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


def _antisym_basis():
    """10 orthonormal antisymmetric 5x5 matrices, in ccwz._antisym_basis order."""
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
        return _sym_traceless_basis()
    if rep == '10':
        return _antisym_basis()
    raise ValueError(f"rep_basis_sym: unsupported rep {rep!r} (use '10' or '14')")


def to_closed_trig(expr):
    """Reduce a polynomial in (sin theta, cos theta) to canonical closed trigonometric form."""
    return sp.trigsimp(sp.expand_trig(sp.expand(expr)))


def _frob(bra, M):
    """Frobenius overlap sum_ij conj(bra_ij) * M_ij for equal-shape matrices."""
    return sum(sp.conjugate(bra[i, j]) * M[i, j]
               for i in range(bra.rows) for j in range(bra.cols))


def U_rep_sym(rep, hat=3):
    """Goldstone matrix lifted to irrep `rep`, as an n x n sympy Matrix in closed trig form.
    Same algorithm as ccwz.U_rep: M[b,a] = <E_b | Uv E_a Uv^T>."""
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
    """<bra| U_rep |ket> in closed trig form (symbolic counterpart of ccwz.overlap).

    For rep '5', bra/ket are length-5 column vectors (sympy Matrix, shape (5,1)); for the
    tensor reps they are 5x5 matrices in the same basis ccwz uses."""
    Uv = U_vector_sym(hat)
    bra = sp.Matrix(bra)
    ket = sp.Matrix(ket)
    if rep == '5':
        expr = _frob(bra, Uv * ket)
    else:
        expr = _frob(bra, Uv * ket * Uv.T)
    return to_closed_trig(expr)


def embedding_sym(rep, kind):
    """Orthonormal tensor for an SO(4) sub-multiplet of `rep` (symbolic ccwz.embedding).
    kind: 'singlet' = (1,1); 'fourplet_i' (i=0..3) = the (2,2) vector component."""
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
    e = sp.expand_trig(expr)                       # cos(2 theta) -> cos^2 - sin^2, etc.
    e = e.subs({sp.cos(theta): ch, sp.sin(theta): sh})
    return sp.simplify(e)


def lambdify_sh(expr):
    """Compile a closed-form theta-expression to a fast numpy callable f(sh)."""
    return sp.lambdify(sh, as_sh_expr(expr), modules='numpy')
