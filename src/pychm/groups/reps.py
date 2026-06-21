"""Representation lifts for a `Coset`: build the same coset in a tensor representation of G, so the
Goldstone and the H-branching follow for fermion partners beyond the defining rep.

Reuses `symbolic.tensors` (the n-agnostic rank-k tensor-basis builder) for the basis, and lifts the
Hermitian generators of G as derivations on the tensor indices.  For SU(4) the antisymmetric rank-2
gives the 6 (= the SO(6) vector under the local isomorphism); for SO(n) it reproduces the 10/14/...
"""
import numpy as np

from . import coset as _coset
from . import tensors as _T


def _lift_generator(basis, G):
    """Lift a defining-rep generator G to the tensor irrep spanned by `basis` (a derivation on
    every index): G_rep[b,a] = <E_b | sum_axes G-on-that-axis E_a>."""
    n = len(basis)
    G = np.asarray(G, dtype=complex)
    out = np.zeros((n, n), dtype=complex)
    for a, Ea in enumerate(basis):
        Ea = np.asarray(Ea, dtype=complex)
        acc = np.zeros_like(Ea)
        for ax in range(Ea.ndim):
            t = np.tensordot(G, Ea, axes=([1], [ax]))
            acc = acc + np.moveaxis(t, 0, ax)
        for b, Eb in enumerate(basis):
            out[b, a] = np.vdot(np.asarray(Eb, dtype=complex), acc)
    return out


def tensor_rep(cos, symmetry, rank):
    """Return a new `Coset` in the rank-`k` `symmetry` ('sym'/'antisym') tensor rep of G, with the
    unbroken/broken generators lifted as derivations.  The defining-rep dimension is taken from
    `cos.dim`.  (For SO(n) this is the 10/14/... tower; for SU(4) antisym rank-2 it is the 6.)"""
    basis = _T.tensor_basis(symmetry, rank, cos.dim)
    unbroken = [_lift_generator(basis, g) for g in cos.unbroken]
    broken = [_lift_generator(basis, g) for g in cos.broken]
    out = _coset.Coset(f"{cos.name} [{symmetry}{rank}]", unbroken, broken)
    out.basis = basis
    return out
