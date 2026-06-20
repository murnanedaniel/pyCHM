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
from scipy.linalg import expm

# ---- SO(5) in the vector (5) ------------------------------------------------------- #
def gen_vector(A, B):
    """Antisymmetric SO(5) generator T^{AB}: (T)_{CD} = -i(d_AC d_BD - d_AD d_BC)."""
    T = np.zeros((5, 5), dtype=complex)
    T[A, B] = -1j
    T[B, A] = 1j
    return T

UNBROKEN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # SO(4)
BROKEN = [(0, 4), (1, 4), (2, 4), (3, 4)]                     # SO(5)/SO(4)


def U_vector(theta, hat=3):
    """Goldstone matrix in the 5: rotation by theta = h/f in the (hat,4) plane."""
    return expm(1j * theta * gen_vector(hat, 4)).real


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


def U_rep(rep, theta, hat=3):
    """Goldstone matrix in irrep `rep` in {'5','10','14'}, in the orthonormal tensor basis."""
    Uv = U_vector(theta, hat)
    if rep == '5':
        return Uv
    basis = _sym_traceless_basis() if rep == '14' else _antisym_basis()
    n = len(basis)
    M = np.zeros((n, n))
    for a, Ea in enumerate(basis):
        UEU = Uv @ Ea @ Uv.T
        for b, Eb in enumerate(basis):
            M[b, a] = np.sum(Eb * UEU)
    return M


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


def overlap(rep, bra, ket, theta, hat=3):
    """<bra| U_rep(theta) |ket>: the Higgs-dressed mixing factor between two embeddings."""
    Uv = U_vector(theta, hat)
    if rep == '5':
        return float(bra @ Uv @ ket)
    return float(np.sum(bra * (Uv @ ket @ Uv.T)))   # tensor inner product after S->USU^T
