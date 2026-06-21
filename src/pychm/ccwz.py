"""Generic CCWZ machinery for SO(5)/SO(4) composite Higgs: the representation-specific
Goldstone (Higgs) dressing of fermion mixings, computed from group theory rather than
hand-transcribed per model.

The s_h-dependence hand-coded in `mchm5`, `mchm14`, `mchm14_1_10` (the factors
cos(h/f), sin(h/f), cos^2(h/2f) for the 5; (3+5cos2h/f)/8, sqrt5 sin(2h/f)/4 for the 14)
is nothing but the matrix elements of the Goldstone matrix U(h) in the chosen SO(5)
irrep, sandwiched between the embeddings of the elementary fermions.  This module builds
U_R(h) for the phenomenologically relevant irreps -- the 5 (vector), 10 (adjoint /
antisymmetric) and 14 (symmetric traceless) -- so the dressing of *any* partner
representation follows from one construction.  This is the rep-generic core of a generic
composite-Higgs spectrum generator; the remaining per-model input is the elementary
embeddings and the composite mass spectrum.

Conventions: SO(4) acts on indices 0..3, the SO(5)/SO(4) direction is index 4.  The
Higgs vev points along one broken generator T^{(hat,4)}; U(h) = exp(i (h/f) T^{(hat,4)})
is then a rotation by theta = h/f in the (hat,4) plane of the vector.  Higher irreps are
obtained by lifting this vector rotation (S -> U S U^T on tensors).
"""
import numpy as np

# ---- SO(5) in the vector (5) ------------------------------------------------------- #
def gen_vector(A, B):
    """Antisymmetric SO(5) generator T^{AB}: (T)_{CD} = -i(d_AC d_BD - d_AD d_BC)."""
    T = np.zeros((5, 5), dtype=complex)
    T[A, B] = -1j
    T[B, A] = 1j
    return T

UNBROKEN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # SO(4)
BROKEN = [(0, 4), (1, 4), (2, 4), (3, 4)]                     # SO(5)/SO(4)


def U_vector(sh, hat=3):
    """Goldstone matrix in the 5 as a function of sh = sin(h/f) (pyCHM's convention).
    A vev along a single broken generator is a planar rotation by h/f; built in closed form
    with c = sqrt(1-sh^2), exact and matching the rest of the library bit-for-bit
    (equivalently expm(i*arcsin(sh)*T^{hat,4})).  Pass sh=sin(angle), not the angle."""
    U = np.eye(5)
    c, s = np.sqrt(max(0.0, 1.0 - sh * sh)), sh
    U[hat, hat] = c; U[4, 4] = c; U[hat, 4] = s; U[4, hat] = -s
    return U


# ---- lifts to the 10 (adjoint) and 14 (symmetric traceless) ------------------------ #
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
    """Goldstone matrix in an SO(5) irrep (sh = sin(h/f)), tensor basis.

    `rep` is '5', '10', '14' (the benchmarks), or a tensor-rep descriptor ('sym', k) /
    ('antisym', k) for an arbitrary rank-k symmetric-traceless / antisymmetric irrep -- the
    general path delegates to `symbolic.tensors` (e.g. ('sym', 3) is the 30)."""
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
    from .symbolic import tensors                 # general rank-k tensor irrep
    sym, rank = rep
    return tensors.U_rep_tensor(tensors.tensor_basis(sym, rank, 5), Uv)


# ---- embeddings (orthonormal tensors for the SO(4) sub-multiplets) ------------------ #
def embedding(rep, kind):
    """Return the orthonormal tensor for an SO(4) sub-multiplet of `rep`.
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
    """<bra| U_rep |ket> (sh = sin(h/f)): the Higgs-dressed mixing factor between embeddings.

    Pass a precomputed Uv = U_vector(sh, hat) to avoid rebuilding it across repeated
    calls at the same sh (the same vector matrix dresses reps '5', '10' and '14')."""
    if Uv is None:
        Uv = U_vector(sh, hat)
    if rep == '5':
        return complex(np.vdot(bra, Uv @ ket))
    return complex(np.sum(np.conjugate(bra) * (Uv @ ket @ Uv.T)))
