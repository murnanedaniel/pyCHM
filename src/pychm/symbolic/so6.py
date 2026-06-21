"""SO(6)/SO(5) coset (NMCHM): generators, the five Goldstones (Higgs doublet + singlet),
and the Goldstone dressing of every relevant SO(6) irrep.

This is the Next-to-Minimal counterpart of `pychm.ccwz`/`pychm.symbolic.core`.  The global
SO(6) is spontaneously broken to SO(5); the coset carries **5 pNGBs = the Higgs doublet
(h1..h4) + one real SO(5)-singlet `s`** (thesis arXiv:2606.18364, Ch.7, eq. 474).

Conventions (extending ccwz's SO(5) ones):
  * SO(5) -- the unbroken group -- acts on indices 0..4; the coset direction is index 5.
  * The 5 broken generators are the bidoublet X_B^a (a=0..3, a planar rotation in the (a,5)
    plane) -> the Higgs doublet, and the singlet X_S (the (4,5) plane) -> the singlet `s`
    (thesis eq. 632, with the 6th index = our index 5).
  * In unitary gauge a vev (h along the hat-th doublet generator, s along X_S) gives the
    vector Goldstone matrix U6 = exp(th X_B^hat + ts X_S): a rotation supported on the three
    indices {hat, 4, 5}.  It is rank-2, so Rodrigues gives it in closed form (no series).

`th = h/f`, `ts = s/f`.  Setting ts=0 leaves a single planar (hat,5) rotation, the SO(6)
analogue of ccwz.U_vector (with the coset index 4 -> 5).
"""
import numpy as np
import sympy as sp

from . import tensors
from . import decompose as _D

N = 6                       # SO(6) acts on 6-dimensional indices
COSET = 5                   # the SO(6)/SO(5) direction (index 5); SO(5) on 0..4


# --------------------------------------------------------------------------------------- #
#  Generators in the vector (6) and the Goldstone matrix
# --------------------------------------------------------------------------------------- #
def gen_vector(A, B, herm=True):
    """SO(6) generator T^{AB} in the vector 6.  herm=True -> Hermitian (-i convention, as
    ccwz.gen_vector); herm=False -> the real antisymmetric rotation generator L^{AB}."""
    T = np.zeros((N, N), dtype=complex if herm else float)
    if herm:
        T[A, B] = -1j
        T[B, A] = 1j
    else:
        T[A, B] = 1.0
        T[B, A] = -1.0
    return T


# the SO(5)/coset split of the 15 SO(6) generators
UNBROKEN = [(a, b) for a in range(5) for b in range(a + 1, 5)]   # 10 = SO(5)
BROKEN = [(a, COSET) for a in range(5)]                          # 5 = SO(6)/SO(5)


def _rodrigues(G):
    """exp(G) for a real antisymmetric G of rank<=2 (G**3 = -Theta**2 G), in closed form."""
    if isinstance(G, sp.Matrix):
        Theta2 = -(G * G).trace() / 2                 # sum of squares of the (single) frequency
        Theta = sp.sqrt(Theta2)
        I = sp.eye(N)
        a = sp.Piecewise((sp.sin(Theta) / Theta, sp.Ne(Theta, 0)), (sp.Integer(1), True))
        b = sp.Piecewise(((1 - sp.cos(Theta)) / Theta2, sp.Ne(Theta2, 0)), (sp.Rational(1, 2), True))
        return I + a * G + b * (G * G)
    Theta2 = max(-(G @ G).trace() / 2, 0.0)            # clamp round-off below zero
    Theta = np.sqrt(Theta2)
    a = 1.0 if Theta < 1e-15 else np.sin(Theta) / Theta
    b = 0.5 if Theta2 < 1e-30 else (1 - np.cos(Theta)) / Theta2
    return np.eye(N) + a * G + b * (G @ G)


def U6_vector(th, ts, hat=3):
    """Numeric 6x6 Goldstone matrix in the vector: exp(th L^{hat,5} + ts L^{4,5})."""
    G = th * gen_vector(hat, COSET, herm=False) + ts * gen_vector(4, COSET, herm=False)
    return _rodrigues(G)


def _gen_vector_sym(A, B):
    """Exact real antisymmetric rotation generator L^{AB} as a sympy Matrix."""
    L = sp.zeros(N, N)
    L[A, B] = 1
    L[B, A] = -1
    return L


def U6_vector_sym(th, ts, hat=3):
    """Closed-form sympy 6x6 Goldstone matrix in the vector (Rodrigues)."""
    G = th * _gen_vector_sym(hat, COSET) + ts * _gen_vector_sym(4, COSET)
    return sp.simplify(_rodrigues(G))


def goldstone_vacuum(th, ts, hat=3):
    """Phi = U6 . e_5: the Goldstone field configuration (thesis eq. 474), as a length-6 vector."""
    e5 = sp.zeros(N, 1)
    e5[COSET] = 1
    return sp.simplify(U6_vector_sym(th, ts, hat) * e5)


# --------------------------------------------------------------------------------------- #
#  Tensor irreps of SO(6) and the Goldstone lift to them
# --------------------------------------------------------------------------------------- #
#   '6'   : vector                       (dim 6)
#   '15'  : antisymmetric rank-2 = adjoint(dim 15)
#   "20'" : symmetric traceless rank-2   (dim 20)
#   '20'  : antisymmetric rank-3         (dim 20) = 10 + 10bar  (self-dual + anti-self-dual)
#   '10','10bar' : the (anti-)self-dual parts of the 3-form     (dim 10 each, complex)
def rep_basis(rep):
    """Orthonormal numpy tensor basis of an SO(6) irrep (see module table)."""
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
    """Goldstone matrix in the SO(6) irrep `rep` (numeric).  For the (anti-)self-dual 3-forms
    '10'/'10bar' it is returned in the self-dual basis (10x10 complex)."""
    Uv = U6_vector(th, ts, hat)
    if rep == '6':
        return Uv
    basis = rep_basis(rep)
    M = tensors.U_rep_tensor(basis, Uv)               # real orthogonal on the tensor basis
    if rep in ('10', '10bar'):
        cols = _selfdual_cols(basis, sign=+1 if rep == '10' else -1)
        return cols.conj().T @ M.astype(complex) @ cols
    return M


# ----- self-dual / anti-self-dual split of the antisymmetric 3-form ---------------------- #
def _levi_civita6():
    """Totally antisymmetric epsilon tensor in 6 dimensions (dense, shape (6,)*6)."""
    from itertools import permutations
    eps = np.zeros((N,) * N)
    for p in permutations(range(N)):
        eps[p] = tensors._perm_sign(p)
    return eps


def _hodge3(T, eps):
    """(*T)_{ijk} = 1/3! eps_{ijklmn} T_{lmn} for a 3-form T."""
    return np.tensordot(eps, T, axes=([3, 4, 5], [0, 1, 2])) / 6.0


def _selfdual_cols(basis, sign):
    """Columns spanning the (sign*i)-eigenspace of the Hodge dual on the 3-form `basis`."""
    eps = _levi_civita6()
    n = len(basis)
    H = np.zeros((n, n), dtype=complex)               # dual in the orthonormal basis
    for a, Ea in enumerate(basis):
        Da = _hodge3(Ea, eps)
        for b, Eb in enumerate(basis):
            H[b, a] = np.vdot(Eb, Da)
    w, V = np.linalg.eig(H)                            # eigenvalues +/- i
    keep = np.where(np.abs(w - 1j * sign) < 1e-9)[0]
    Q, _ = np.linalg.qr(V[:, keep])                    # orthonormalise the eigenspace
    return Q


# --------------------------------------------------------------------------------------- #
#  Branchings: SO(6) irrep -> SO(5), and -> SO(4) (reusing the SU(2)xSU(2) Casimirs)
# --------------------------------------------------------------------------------------- #
def _gen_rep(basis, a, b):
    """Lift the vector generator T^{ab} (Hermitian) to the tensor irrep spanned by `basis`."""
    n = len(basis)
    Tv = gen_vector(a, b, herm=True)
    G = np.zeros((n, n), dtype=complex)
    for q, Eq in enumerate(basis):
        Eq = np.asarray(Eq, dtype=complex)
        acc = np.zeros_like(Eq)
        for ax in range(Eq.ndim):
            t = np.tensordot(Tv, Eq, axes=([1], [ax]))
            acc = acc + np.moveaxis(t, 0, ax)
        for p, Ep in enumerate(basis):
            G[p, q] = np.vdot(np.asarray(Ep, dtype=complex), acc)
    return G


# SO(5) irreps by quadratic-Casimir value in the normalisation C5 = sum_{a<b in 0..4}(T^{ab})^2
# (Hermitian generators).  Calibrated from the SO(6) branchings: singlet/vector from the 6,
# adjoint from the 15, symmetric-traceless from the 20', spinor from the 4 (so6_spinors).
SO5_CASIMIR = {0.0: ('1', 1), 2.5: ('4', 4), 4.0: ('5', 5), 6.0: ('10', 10), 10.0: ('14', 14)}


def so5_casimir(basis):
    """The SO(5) quadratic Casimir C5 = sum_{a<b in 0..4} (T^{ab})^2 on the irrep `basis`."""
    gens = [_gen_rep(basis, a, b) for (a, b) in UNBROKEN]
    return sum(g @ g for g in gens)


def so5_content_gen(C, tol=1e-6):
    """SO(5) branching [(name, dim), ...] from the quadratic-Casimir matrix C5 of a rep.

    Each distinct Casimir eigenvalue identifies one SO(5) irrep (via SO5_CASIMIR); the
    eigenspace dimension divided by the irrep dimension is its multiplicity, so degenerate
    branchings (e.g. the 3-form 20 = 10 + 10bar, two adjoints) are reported correctly."""
    w = np.linalg.eigvalsh((C + C.conj().T) / 2).real
    out, used = [], np.zeros(len(w), dtype=bool)
    for i in range(len(w)):
        if used[i]:
            continue
        grp = np.where(np.abs(w - w[i]) < tol)[0]
        used[grp] = True
        cval = round(float(np.mean(w[grp])), 4)
        match = min(SO5_CASIMIR, key=lambda v: abs(v - cval))
        if abs(match - cval) < 1e-3:
            name, d = SO5_CASIMIR[match]
            out += [(name, d)] * (len(grp) // d)        # expand multiplicity
        else:
            out.append((f'C5={cval}', len(grp)))
    out.sort(key=lambda t: t[1])
    return out


def so5_content(basis, tol=1e-6):
    """SO(5) branching [(name, dim), ...] of the tensor irrep spanned by `basis`."""
    return so5_content_gen(so5_casimir(basis), tol)


def so4_content(basis, tol=1e-6):
    """SO(4) = SU(2)xSU(2) branching {(jL,jR): mult} of the SO(6) irrep (indices 0..3).

    Unlike `decompose.so4_content_gen` (which counts the number of distinct (jL,jR) Casimir
    blocks), this counts the true multiplicity: a Casimir-degenerate block of (jL,jR) with
    dimension d > (2jL+1)(2jR+1) is d / (2jL+1)(2jR+1) copies of that multiplet -- needed for
    SO(6) reps where a multiplet recurs (e.g. the 20' has 2x(1/2,1/2) and 3x(0,0))."""
    M = {(mu, nu): _gen_rep(basis, mu, nu) for mu in range(4) for nu in range(mu + 1, 4)}
    table = {}
    for (jL, jR), cols in _D.so4_decompose_gen(M, tol):
        mult = round(cols.shape[1] / ((2 * jL + 1) * (2 * jR + 1)))
        table[(jL, jR)] = table.get((jL, jR), 0) + mult
    return table


# --------------------------------------------------------------------------------------- #
#  Fermion embeddings in the 6 (thesis eq. 633) and the Goldstone-dressed overlaps
# --------------------------------------------------------------------------------------- #
_R2 = np.sqrt(2.0)


def embedding(rep, kind):
    """Orthonormal embedding tensor for an SO(4) sub-multiplet of an SO(6) rep.

    For the 6 (= 4 + 1 + 1 under SO(4)): kind 'fourplet_i' (i=0..3) = the bidoublet component
    e_i; 'singlet5' = the SO(5)-vector singlet e_4; 'singlet6' = the SO(5)-singlet e_5 (the
    direction the Higgs+singlet rotate into).  This is the basis of thesis eq. 633."""
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
    """Squared SO(4)-channel projections of the (h,s)-dressed q_L embedding in the 6.

    The dressed embedding U6 . E lands in the 6 = (2,2) + 1 + 1; this returns the squared
    length in each SO(4) channel {(jL,jR)/singlet}, summing to |E|^2 (=1 for a unit E).  These
    are the NMCHM analogue of `decompose.channel_weights_sym` and reduce to the MCHM5 vector
    weights at ts=0."""
    U = U6_vector(th, ts, hat)
    dressed = U @ np.asarray(EqL, dtype=complex)
    out = {}
    out[('2,2',)] = float(np.vdot(dressed[:4], dressed[:4]).real)   # bidoublet (indices 0..3)
    out[('1_5',)] = float(abs(dressed[4])**2)                       # e4 singlet
    out[('1_6',)] = float(abs(dressed[5])**2)                       # e5 singlet
    return out


# --------------------------------------------------------------------------------------- #
#  Closed-form (symbolic) dressing in the 6 -- the NMCHM analogue of symbolic.core/decompose:
#  every overlap is a closed trigonometric function of (theta_h, theta_s), derived not fitted.
# --------------------------------------------------------------------------------------- #
def _trig(expr):
    return sp.simplify(sp.trigsimp(sp.expand_trig(sp.expand(expr))))


def _U6_clean_sym(th, ts, hat):
    """Guard-free closed-form vector Goldstone (Theta = sqrt(th^2+ts^2) > 0) for the symbolic
    derivations -- no Piecewise, so the closed trig forms simplify (e.g. the weights sum to 1)."""
    Th = sp.sqrt(th**2 + ts**2)
    G = th * _gen_vector_sym(hat, COSET) + ts * _gen_vector_sym(4, COSET)
    return sp.eye(N) + sp.sin(Th) / Th * G + (1 - sp.cos(Th)) / Th**2 * (G * G)


def overlap6_sym(bra, ket, th, ts, hat=3):
    """<bra| U6(th,ts) |ket> in closed trigonometric form (sympy), bra/ket length-6 vectors."""
    U = _U6_clean_sym(th, ts, hat)
    bra = sp.Matrix(bra)
    ket = sp.Matrix(ket)
    expr = sum(sp.conjugate(bra[i]) * (U * ket)[i] for i in range(N))
    return _trig(expr)


def channel_weights6_sym(EqL, th, ts, hat=3):
    """Closed-form squared SO(4)-channel weights of the (h,s)-dressed q_L embedding in the 6
    (symbolic counterpart of channel_weights6): {channel: trig_expr_in(th,ts)}, summing to |E|^2.

    Pure group theory -- the NMCHM analogue of decompose.channel_weights_sym.  At ts=0 the
    singlet-6 (coset) channel reduces to the MCHM5 vector weight sin^2(theta_h) * |E_coset|^2."""
    U = _U6_clean_sym(th, ts, hat)
    d = U * sp.Matrix(EqL)
    return {
        ('2,2',): _trig(sum(sp.conjugate(d[i]) * d[i] for i in range(4))),   # bidoublet (0..3)
        ('1_5',): _trig(sp.conjugate(d[4]) * d[4]),                          # e4 singlet
        ('1_6',): _trig(sp.conjugate(d[5]) * d[5]),                          # e5 singlet (coset)
    }
