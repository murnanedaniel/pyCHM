"""SU(5)/SO(5) coset (littlest-Higgs / Ferretti real-rep) on the group-agnostic engine.

A type-AI symmetric space: unbroken SO(5) (10), coset = the **14** (symmetric-traceless 2-tensor of
SO(5)), whose 14 pNGBs are, under custodial SO(4) ⊂ SO(5), `(3,3)+(2,2)+(1,1)` = a complex **triplet**
`phi` + a Higgs **doublet** `h` + a **singlet** `eta`.  The coset is built abstractly by
`coset.su_so_coset(5)` (validated in tests); for the model we also expose the coset generators in the
aligned SO(5) basis (the fundamental 5 carries SO(4) on indices 0..3, the custodial direction index
4), so the Higgs/singlet/triplet directions are explicit rather than read off the Gell-Mann basis.
"""
import numpy as np

from . import coset as _coset, reps as _reps, decompose as _D

_R2 = np.sqrt(2.0)


def fundamental():
    """The SU(5)/SO(5) coset in the fundamental 5 (so(5)=10, coset=14)."""
    return _coset.su_so_coset(5)


def sym15():
    """The symmetric **15** of SU(5) (NO trace removal) as a Coset -- the fuller fermion-partner rep
    (branches 14 + 1 under SO(5))."""
    return _reps.tensor_rep(fundamental(), 'sym', 2, group='SU')


# ----- the coset generators in the aligned SO(5) basis (real symmetric-traceless 5x5) ----------- #
def _sym(i, j):
    """Real symmetric generator e_i e_j^T + e_j e_i^T (off-diagonal, traceless)."""
    M = np.zeros((5, 5), dtype=complex)
    M[i, j] = M[j, i] = 1.0
    return M


HIGGS = [_sym(i, 4) for i in range(4)]          # the (2,2): index i in 0..3 with the custodial index 4
TRIPLET = ([_sym(i, j) for i in range(4) for j in range(i + 1, 4)]      # off-diag 4x4 (6)
           + [np.diag([1., -1, 0, 0, 0]).astype(complex),               # + 3 traceless diagonals in 0..3
              np.diag([1., 1, -2, 0, 0]).astype(complex) / np.sqrt(3),
              np.diag([1., 1, 1, -3, 0]).astype(complex) / np.sqrt(6)])  # = the (3,3), dim 9
SINGLET = np.diag([1., 1, 1, 1, -4]).astype(complex) / np.sqrt(20)       # the (1,1), the SO(5) singlet


def goldstone_5(th, te=0.0, tp=0.0, hat=3, ti=0):
    """Goldstone in the fundamental 5 for a (Higgs, singlet, triplet) vev: U = exp(i(th H_hat +
    te SINGLET + tp T_ti)).  H_hat is the hat-th Higgs generator (the EWSB direction), T_ti a
    representative triplet generator (its curvature gives the single custodial triplet mass)."""
    from scipy.linalg import expm
    G = th * HIGGS[hat] + te * SINGLET + tp * TRIPLET[ti]
    return expm(1j * G)


def goldstone_rep(basis, th, te=0.0, tp=0.0, hat=3, ti=0):
    """Goldstone in a tensor rep spanned by `basis`: lift the coset generators (as derivations) and
    exponentiate.  Used for the symmetric-15 partner tower."""
    from scipy.linalg import expm
    H = _reps._lift_generator(basis, HIGGS[hat])
    S = _reps._lift_generator(basis, SINGLET)
    T = _reps._lift_generator(basis, TRIPLET[ti])
    return expm(1j * (th * H + te * S + tp * T))


# ----- physical labelling of the 14 (validates the (3,3)+(2,2)+(1,1) content) ------------------- #
def _so4_generators_5():
    """The six custodial SO(4) generators (indices 0..3) in the fundamental 5 (Hermitian)."""
    M = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            T = np.zeros((5, 5), dtype=complex)
            T[mu, nu] = -1j
            T[nu, mu] = 1j
            M[(mu, nu)] = T
    return M


def coset_so4_content():
    """SO(4) = SU(2)xSU(2) content of the 14 coset (the pNGB multiplets), via the adjoint action of
    the custodial SO(4) on the coset generators.  Returns {(jL,jR): multiplicity}."""
    cos = fundamental()
    so4 = _so4_generators_5()
    nb = len(cos.broken)
    B = np.array([g.ravel() for g in cos.broken])
    coeffs = lambda M: np.linalg.lstsq(B.T, M.ravel(), rcond=None)[0]   # complex: [T,X]=i*(real)
    adj = {}
    for key, Tg in so4.items():
        A = np.zeros((nb, nb), dtype=complex)
        for a, Xa in enumerate(cos.broken):
            A[:, a] = coeffs(Tg @ Xa - Xa @ Tg)        # the adjoint generator on the 14 (Hermitian)
        adj[key] = A
    table = {}
    for (jL, jR), cols in _D.so4_decompose_gen(adj):
        m = round(cols.shape[1] / ((2 * jL + 1) * (2 * jR + 1)))
        table[(jL, jR)] = table.get((jL, jR), 0) + m
    return table
