"""Group-agnostic Lie-algebra primitives shared by every coset.

The composite-Higgs cosets pyCHM supports are symmetric spaces G/H; in unitary gauge only a few
pNGBs are switched on, so the vector Goldstone is always exp(sum_a theta_a X^a) of a low-rank
generator.  This module provides:

  * `so_generator` / `so_generator_sym` -- the SO(N) rotation generators (the SO(5)/SO(6) ones in
    `ccwz`/`symbolic.so6` are the N=5,6 instances);
  * `rodrigues_exp` -- the closed-form exp of a *single-frequency* (rank-2) antisymmetric /
    anti-Hermitian generator, polymorphic over numpy and sympy (this is the generalisation of
    `so6._rodrigues`, with N inferred from the matrix rather than a module constant);
  * `expm_num` -- a general numeric matrix exponential (scipy) for cases Rodrigues does not cover
    (e.g. an SU(N) Goldstone whose generator has two distinct frequencies);
  * `su_generators` / `sp_subalgebra` -- the SU(N) algebra and its Sp(N) symplectic subalgebra,
    for the SU(4)/Sp(4) coset.
"""
import numpy as np
import sympy as sp
from scipy.linalg import expm as _expm


# --------------------------------------------------------------------------------------- #
#  SO(N) rotation generators
# --------------------------------------------------------------------------------------- #
def so_generator(A, B, n, herm=True):
    """SO(N) generator T^{AB} in the vector (n-dim).  herm=True -> Hermitian -i convention
    (T)_{CD} = -i(d_AC d_BD - d_AD d_BC); herm=False -> the real antisymmetric rotation L^{AB}.

    `ccwz.gen_vector(A,B)` == `so_generator(A,B,5)`; `so6.gen_vector(A,B,herm)` == this with n=6."""
    T = np.zeros((n, n), dtype=complex if herm else float)
    if herm:
        T[A, B] = -1j
        T[B, A] = 1j
    else:
        T[A, B] = 1.0
        T[B, A] = -1.0
    return T


def so_generator_sym(A, B, n):
    """Exact real antisymmetric rotation generator L^{AB} as a sympy Matrix (== so6._gen_vector_sym)."""
    L = sp.zeros(n, n)
    L[A, B] = 1
    L[B, A] = -1
    return L


# --------------------------------------------------------------------------------------- #
#  Closed-form and general matrix exponentials
# --------------------------------------------------------------------------------------- #
def rodrigues_exp(G):
    """exp(G) for a single-frequency generator G (G**3 = -Theta**2 G), in closed form.

    Polymorphic over sympy.Matrix and numpy.ndarray; N is inferred from G.  This is the
    group-agnostic version of `so6._rodrigues`: valid whenever G has a single non-zero frequency
    (one +/- i*Theta eigenvalue pair), which holds for every SO(N) vector Goldstone in unitary
    gauge (a rotation supported on the <=3 indices {hat, ..., coset}).  Theta^2 = -tr(G^2)/2."""
    if isinstance(G, sp.Matrix):
        n = G.rows
        Theta2 = -(G * G).trace() / 2
        Theta = sp.sqrt(Theta2)
        a = sp.Piecewise((sp.sin(Theta) / Theta, sp.Ne(Theta, 0)), (sp.Integer(1), True))
        b = sp.Piecewise(((1 - sp.cos(Theta)) / Theta2, sp.Ne(Theta2, 0)), (sp.Rational(1, 2), True))
        return sp.eye(n) + a * G + b * (G * G)
    n = G.shape[0]
    Theta2 = max(-np.trace(G @ G) / 2, 0.0)            # clamp round-off below zero
    Theta = np.sqrt(Theta2)
    a = 1.0 if Theta < 1e-15 else np.sin(Theta) / Theta
    b = 0.5 if Theta2 < 1e-30 else (1 - np.cos(Theta)) / Theta2
    return np.eye(n) + a * G + b * (G @ G)


def rodrigues_exp_clean(G):
    """Guard-free closed-form exp(G) (sympy), Theta = sqrt(-tr(G^2)/2) assumed > 0.  Used for the
    symbolic derivations where a Piecewise blocks simplification (== so6._U6_clean_sym)."""
    n = G.rows
    Theta = sp.sqrt(-(G * G).trace() / 2)
    return sp.eye(n) + sp.sin(Theta) / Theta * G + (1 - sp.cos(Theta)) / Theta**2 * (G * G)


def expm_num(G):
    """General numeric matrix exponential (scipy), for generators Rodrigues does not cover
    (e.g. multi-frequency SU(N) Goldstones)."""
    return _expm(np.asarray(G))


# --------------------------------------------------------------------------------------- #
#  SU(N) algebra and its Sp(N) symplectic subalgebra
# --------------------------------------------------------------------------------------- #
def su_generators(n):
    """The n^2-1 Hermitian traceless generators of su(N), normalised Tr(T_a T_b) = 1/2 delta_ab
    (the Gell-Mann construction): real-symmetric off-diagonal, imaginary-antisymmetric, and the
    diagonal traceless set."""
    gens = []
    for i in range(n):
        for j in range(i + 1, n):
            S = np.zeros((n, n), dtype=complex); S[i, j] = S[j, i] = 1.0
            gens.append(S / 2.0)
            A = np.zeros((n, n), dtype=complex); A[i, j] = -1j; A[j, i] = 1j
            gens.append(A / 2.0)
    for k in range(1, n):
        d = np.zeros((n, n), dtype=complex)
        for i in range(k):
            d[i, i] = 1.0
        d[k, k] = -k
        norm = np.sqrt(2.0 * k * (k + 1))             # gives Tr(T^2)=1/2
        gens.append(d / norm)
    return gens                                       # length n^2-1


def symplectic_form(n):
    """The standard symplectic form Omega on C^n (n even): block-diagonal [[0,1],[-1,0]]."""
    assert n % 2 == 0
    Om = np.zeros((n, n), dtype=complex)
    for k in range(0, n, 2):
        Om[k, k + 1] = 1.0
        Om[k + 1, k] = -1.0
    return Om


def sp_subalgebra(n, Omega):
    """Split su(N) (N even) into the usp(N) subalgebra and the SU(N)/USp(N) coset, via the Cartan
    involution theta(X) = Omega conj(X) Omega^{-1} of the type-AII symmetric space.

    The Gell-Mann generators are not aligned with the split, so we diagonalise theta as an operator
    on the n^2-1 dimensional algebra: usp(N) (the subgroup, dim n(n+1)/2... = N(N+1)/2 for USp(N)) is
    the theta = -1 eigenspace, the coset (dim (N-1)(N+2)/2... = 5 for N=4) the theta = +1 eigenspace.
    Returns (unbroken, broken) as lists of orthonormal Hermitian generators."""
    T = su_generators(n)
    d = len(T)
    Omi = np.linalg.inv(Omega)
    coeffs = lambda M: np.array([2 * np.trace(t @ M) for t in T]).real   # M -> T-basis (real)
    Theta = np.zeros((d, d))
    for b, Tb in enumerate(T):
        Theta[:, b] = coeffs(Omega @ Tb.conj() @ Omi)
    w, V = np.linalg.eigh((Theta + Theta.T) / 2)
    pick = lambda sign: [sum(V[a, k] * T[a] for a in range(d))
                         for k in np.where(np.abs(w - sign) < 1e-6)[0]]
    return pick(-1.0), pick(+1.0)                     # (usp subalgebra, coset)
