"""The group-agnostic `Coset` abstraction.

A `Coset` is a symmetric space G/H in a chosen (Hermitian-generator) representation: it holds the
unbroken generators of H and the broken coset generators, and provides the Goldstone matrix in that
representation, U = exp(i sum_a theta_a X^a), plus the H-branching of the representation via the
unbroken Casimir.  The SO(5)/SO(4) and SO(6)/SO(5) machinery are instances; SU(4)/Sp(4) is another.

This is the unifying object behind `ccwz`/`symbolic.so6`: there, the broken generators are the
antisymmetric SO(N) rotations and the Goldstone is the planar/Rodrigues rotation; here the same
exp(i theta.X) is built for any G, including the SU(4) fundamental and the antisymmetric 6.
"""
import numpy as np

from . import lie


class Coset:
    def __init__(self, name, unbroken, broken):
        """unbroken, broken: lists of Hermitian generators (n x n) of H and of the coset G/H, in a
        chosen representation of G (the 'defining rep' of this Coset object)."""
        self.name = name
        self.unbroken = [np.asarray(t, dtype=complex) for t in unbroken]
        self.broken = [np.asarray(t, dtype=complex) for t in broken]
        self.dim = self.broken[0].shape[0]

    @property
    def n_pngb(self):
        return len(self.broken)

    @property
    def generators(self):
        return self.unbroken + self.broken

    def goldstone(self, angles):
        """U = exp(i sum_a angles[a] X^a) in this representation.  `angles` has length n_pngb
        (zeros for the pNGBs left in the vacuum); a general (multi-frequency) exponential is used,
        which is exact for any coset and rep."""
        G = sum(a * X for a, X in zip(angles, self.broken))
        return lie.expm_num(1j * np.asarray(G))

    def goldstone_dir(self, theta, direction):
        """U for a vev of magnitude theta along a single broken generator (index `direction`)."""
        angles = [0.0] * self.n_pngb
        angles[direction] = theta
        return self.goldstone(angles)

    def casimir(self):
        """The quadratic Casimir of H, sum_a (T^a_unbroken)^2, on this representation (proportional
        to the identity on each H-irrep -> its eigenvalues label the branching)."""
        return sum(t @ t for t in self.unbroken)

    def algebra_closes(self, tol=1e-9):
        """True iff the full generator set closes under commutation (a sanity check on G)."""
        B = np.array([g.ravel() for g in self.generators])
        for a in self.generators:
            for b in self.generators:
                c = (a @ b - b @ a).ravel()
                coeff, *_ = np.linalg.lstsq(B.T, c, rcond=None)
                if np.linalg.norm(B.T @ coeff - c) > tol:
                    return False
        return True

    def branch_dims(self, tol=1e-6):
        """The dimensions of the H-irreps in this representation, from the degeneracies of the
        unbroken Casimir (each distinct eigenvalue = one H-irrep, multiplicity = eigenspace dim)."""
        C = self.casimir()
        w = np.linalg.eigvalsh((C + C.conj().T) / 2).real
        out = []
        used = np.zeros(len(w), dtype=bool)
        for i in range(len(w)):
            if used[i]:
                continue
            grp = np.where(np.abs(w - w[i]) < tol)[0]
            used[grp] = True
            out.append((round(float(np.mean(w[grp])), 4), len(grp)))
        return sorted(out, key=lambda t: t[1])


# --------------------------------------------------------------------------------------- #
#  SO(N)/SO(N-1) instances (the existing MCHM / NMCHM cosets, as Coset objects)
# --------------------------------------------------------------------------------------- #
def so_coset(n, defining_rep_dim=None):
    """The SO(n)/SO(n-1) coset in the vector representation: unbroken SO(n-1) on indices 0..n-2,
    coset direction index n-1.  Reproduces `ccwz` (n=5) and `so6` (n=6)."""
    herm = lambda A, B: lie.so_generator(A, B, n, herm=True)
    unbroken = [herm(a, b) for a in range(n - 1) for b in range(a + 1, n - 1)]
    broken = [herm(a, n - 1) for a in range(n - 1)]
    return Coset(f"SO({n})/SO({n-1})", unbroken, broken)


# --------------------------------------------------------------------------------------- #
#  SU(N)/USp(N) instance in the fundamental (the SU(4)/Sp(4) coset of the NMCHM landscape)
# --------------------------------------------------------------------------------------- #
def su_sp_coset(n):
    """The SU(n)/USp(n) coset in the fundamental n, n even.  For n=4 this is the minimal pseudoreal
    Ferretti/Sannino coset, locally isomorphic to SO(6)/SO(5)."""
    Omega = lie.symplectic_form(n)
    unbroken, broken = lie.sp_subalgebra(n, Omega)
    return Coset(f"SU({n})/USp({n})", unbroken, broken)


def su_so_coset(n):
    """The SU(n)/SO(n) coset in the fundamental n (type AI).  For n=5 this is the littlest-Higgs /
    Ferretti real-rep coset: unbroken so(5) (10), coset = the 14 (symmetric-traceless of SO(5))."""
    unbroken, broken = lie.so_subalgebra(n)
    return Coset(f"SU({n})/SO({n})", unbroken, broken)
