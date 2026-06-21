"""Arbitrary-rank tensor irreps of SO(n) and the Goldstone lift to them.

The 5 (rank-1), 10 (antisymmetric rank-2) and 14 (symmetric-traceless rank-2) are the first
three rungs of a tower of tensor irreps of SO(5).  This module builds an orthonormal basis of
*any* rank-k symmetric-traceless or antisymmetric irrep and lifts the vector Goldstone rotation
to it, so the CCWZ dressing follows for partner representations beyond the three benchmarks.

A tensor is held as a numpy array of shape (n,)*k.  `U_rep_tensor(basis, Uv)` reproduces
ccwz.U_rep for the 10 and 14 (same span), and extends unchanged to 30, 35, ... .
"""
import numpy as np
from itertools import permutations


def _apply_sym(T, antisym):
    """Average T over all index permutations (with sign if antisym)."""
    k = T.ndim
    out = np.zeros_like(T)
    for p in permutations(range(k)):
        sign = _perm_sign(p) if antisym else 1
        out = out + sign * np.transpose(T, p)
    return out / _factorial(k)


def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _perm_sign(p):
    p = list(p)
    sign, seen = 1, [False] * len(p)
    for i in range(len(p)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def _trace_pair(T, a, b):
    """Contract axes a, b of T with delta -> rank-(k-2) tensor."""
    return np.trace(T, axis1=a, axis2=b)


def tensor_basis(symmetry, rank, n=5, tol=1e-10, group='SO'):
    """Orthonormal basis (list of numpy arrays, shape (n,)*rank) of a tensor irrep.

    symmetry='sym'/'antisym'; group='SO' removes the delta-trace (symmetric rank-2 -> the SO(n)
    sym-traceless: 14 at rank 2 for n=5), group='SU' keeps the FULL symmetric space (irreducible
    under SU(n): 15 at rank 2 for n=5).  Antisymmetric reps are identical for SO and SU.
    """
    antisym = (symmetry == 'antisym')
    dim = n**rank
    # 1. raw (anti)symmetrized basis from canonical unit tensors
    raw = []
    for idx in np.ndindex(*([n] * rank)):
        T = np.zeros([n] * rank)
        T[idx] = 1.0
        T = _apply_sym(T, antisym)
        if np.max(np.abs(T)) > tol:
            raw.append(T.ravel())
    # orthonormal basis of the (anti)symmetric subspace
    M = np.array(raw)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    sym_basis = [Vt[i].reshape([n] * rank) for i in range(np.sum(s > tol))]
    if antisym or rank < 2 or group == 'SU':
        return [b / np.sqrt(np.vdot(b, b)) for b in sym_basis]
    # 2. remove traces: keep the subspace on which every pairwise contraction vanishes
    B = np.array([b.ravel() for b in sym_basis])              # (d_sym, n^k)
    # trace operator on the symmetric subspace (one pair suffices by symmetry):
    # a combination x of sym basis tensors is traceless iff sum_i x_i Tr(E_i) = 0.
    Tr = np.array([_trace_pair(b.reshape([n] * rank), 0, 1).ravel() for b in B])  # (d_sym, n^(k-2))
    ns = _nullspace(Tr.T, tol)                                 # columns span {x : x.Tr = 0}
    out = []
    for col in ns.T:
        v = (col @ B).reshape([n] * rank)
        nrm = np.sqrt(np.vdot(v, v))
        if nrm > tol:
            out.append(v / nrm)
    return _orthonormalize(out, tol)


def _nullspace(A, tol=1e-10):
    """Orthonormal basis of the null space of A (columns)."""
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(s > tol)
    return vh[rank:].conj().T


def _orthonormalize(vs, tol=1e-10):
    out = []
    for v in vs:
        w = v.copy()
        for u in out:
            w = w - np.vdot(u, w) * u
        nrm = np.sqrt(np.vdot(w, w))
        if nrm > tol:
            out.append(w / nrm)
    return out


def lift(Uv, T):
    """Apply the vector rotation Uv to every index of tensor T: T_{i..} -> U_{i j} .. T_{j..}."""
    out = T
    for ax in range(T.ndim):
        out = np.tensordot(Uv, out, axes=([1], [ax]))
        out = np.moveaxis(out, 0, ax)
    return out


def U_rep_tensor(basis, Uv):
    """Goldstone matrix in the irrep spanned by `basis`: M[b,a] = <E_b | lift(Uv, E_a)>."""
    n = len(basis)
    M = np.zeros((n, n))
    for a, Ea in enumerate(basis):
        UEa = lift(Uv, Ea)
        for b, Eb in enumerate(basis):
            M[b, a] = np.vdot(Eb, UEa).real
    return M
