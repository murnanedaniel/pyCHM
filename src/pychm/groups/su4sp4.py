"""SU(4)/Sp(4) coset -- a NEW coset built entirely on the group-agnostic engine, as proof that the
abstraction is not SO(N)-specific.

SU(4)/USp(4) is the minimal *pseudoreal* Ferretti/Sannino fundamental-composite-Higgs coset, and it
is locally isomorphic to SO(6)/SO(5): both have 5 broken generators (= 5 pNGBs: a Higgs doublet + a
singlet).  Under the isomorphism SU(4) ~= Spin(6), USp(4) ~= Spin(5), the SU(4) reps map to SO(6)
reps as

    SU(4) fundamental  4   =  SO(6) spinor   4
    SU(4) antisym      6   =  SO(6) vector   6
    SU(4) adjoint     15   =  SO(6) adjoint 15
    SU(4) (self-dual) 10   =  SO(6) self-dual 3-form 10

so every generator / Goldstone / branching of this coset has an exact, independent oracle in
`groups.so6` / `groups.so6_spinors`.  This module builds the coset from `groups.lie`/`coset`/`reps`
(SU(4) Gell-Mann generators, the USp(4) symplectic subalgebra, the Goldstone exp(i theta.X), the
antisymmetric-6 lift) -- no SO(N) machinery -- and `tests/test_su4sp4.py` cross-checks it against
SO(6)/SO(5).
"""
import numpy as np

from . import coset as _coset, reps as _reps


def fundamental():
    """The SU(4)/USp(4) coset in the fundamental 4 (5 broken generators, 10 unbroken usp(4))."""
    return _coset.su_sp_coset(4)


def antisym6():
    """The coset in the antisymmetric 6 = [4 (x) 4]_A (= the SO(6) vector under the isomorphism)."""
    return _reps.tensor_rep(fundamental(), 'antisym', 2)


def higgs_singlet_dirs(cos):
    """Indices of two broken generators to use as the Higgs and singlet vev directions.  Any two
    distinct coset generators work for the Goldstone (the coset is a single USp(4)-irrep); we take
    the first two for definiteness."""
    return 0, 1


def goldstone_4(theta_h, theta_s=0.0):
    """Goldstone in the fundamental 4 for a (Higgs, singlet) vev."""
    cos = fundamental()
    h, s = higgs_singlet_dirs(cos)
    ang = [0.0] * cos.n_pngb
    ang[h], ang[s] = theta_h, theta_s
    return cos.goldstone(ang)


def goldstone_6(theta_h, theta_s=0.0):
    """Goldstone in the antisymmetric 6 for a (Higgs, singlet) vev."""
    cos = antisym6()
    h, s = higgs_singlet_dirs(cos)
    ang = [0.0] * cos.n_pngb
    ang[h], ang[s] = theta_h, theta_s
    return cos.goldstone(ang)
