"""SU(4)/Sp(4): a NEW coset built on the group-agnostic engine (no SO(N) machinery), validated by
cross-check against the SO(6)/SO(5) NMCHM it is locally isomorphic to.

SU(4) ~= Spin(6), USp(4) ~= Spin(5); under the isomorphism the SU(4) fundamental 4 = SO(6) spinor 4
and the SU(4) antisymmetric 6 = SO(6) vector 6.  Every quantity below is checked against its exact
oracle in groups.so6 / groups.so6_spinors.  The only difference between the two builds is an overall
generator normalisation (the pNGB decay-constant convention), so we compare normalisation-independent
invariants: the algebra, the coset structure, the branchings, and the Goldstone's structural type.
"""
import numpy as np
import pytest

from pychm.groups import su4sp4 as X
from pychm.groups import so6, so6_spinors as sp6, reps


def _in_span(M, gens, tol=1e-9):
    B = np.array([g.ravel() for g in gens])
    c, *_ = np.linalg.lstsq(B.T, M.ravel(), rcond=None)
    return np.linalg.norm(B.T @ c - M.ravel()) < tol


# --------------------------------------------------------------------------------------- #
#  The coset: SU(4)/USp(4) has 5 pNGBs transforming as the 5 of USp(4) (= SO(6)/SO(5))
# --------------------------------------------------------------------------------------- #
def test_algebra_closes_and_five_pngbs():
    for cos in (X.fundamental(), X.antisym6()):
        assert cos.algebra_closes()
    cos = X.fundamental()
    assert len(cos.unbroken) == 10 and len(cos.broken) == 5      # usp(4)=10, coset=5
    # the coset transforms as a USp(4) irrep: [unbroken, broken] subset broken
    assert all(_in_span(u @ b - b @ u, cos.broken) for u in cos.unbroken for b in cos.broken)


# --------------------------------------------------------------------------------------- #
#  Branchings under USp(4) match SO(6) reps under SO(5) (dims; normalisation-independent)
# --------------------------------------------------------------------------------------- #
def test_fundamental_4_branches_like_so6_spinor():
    dims = sorted(d for _, d in X.fundamental().branch_dims())
    assert dims == [4]                                           # a single 4 (irreducible)
    assert sorted(d for _, d in sp6.so5_content_4()) == [4]      # the SO(6) spinor oracle


def test_antisym_6_branches_like_so6_vector():
    dims = sorted(d for _, d in X.antisym6().branch_dims())
    assert dims == [1, 5]                                        # 6 -> 1 + 5 under USp(4)
    assert sorted(d for _, d in so6.so5_content(so6.rep_basis('6'))) == [1, 5]   # SO(6) vector oracle


# --------------------------------------------------------------------------------------- #
#  The Goldstone: unitarity, identity, homomorphism, and the right structural type
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("gold,dim", [(X.goldstone_4, 4), (X.goldstone_6, 6)])
def test_goldstone_unitary_identity_homomorphism(gold, dim):
    assert np.allclose(gold(0.0, 0.0), np.eye(dim), atol=1e-12)
    for th, ts in [(0.3, 0.2), (0.7, -0.4)]:
        U = gold(th, ts)
        assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-12)
    a, b = 0.21, 0.33                                            # one-parameter subgroup (ts=0)
    assert np.allclose(gold(a, 0) @ gold(b, 0), gold(a + b, 0), atol=1e-12)


def test_fundamental_goldstone_is_half_angle_like_so6_spinor():
    """The 4 Goldstone has two distinct eigenphases, each doubly degenerate -- the half-angle
    structure of the SO(6) spinor 4 (4 = (1/2,0)+(0,1/2) under SO(4))."""
    ph = np.sort(np.angle(np.linalg.eigvals(X.goldstone_4(0.6, 0.0))))
    assert np.allclose(ph[:2], ph[0]) and np.allclose(ph[2:], ph[3])   # two doubly-degenerate phases
    assert abs(ph[0] + ph[3]) < 1e-9                                   # +/- symmetric
    # same structural type as the SO(6) spinor Goldstone
    ph6 = np.sort(np.angle(np.linalg.eigvals(sp6.U4(0.6, 0.0))))
    assert np.allclose(ph6[:2], ph6[0]) and np.allclose(ph6[2:], ph6[3])


def test_six_goldstone_rotates_one_plane_like_so6_vector():
    """The 6 Goldstone (single Higgs vev) is a rotation in one plane: exactly two non-unit
    eigenvalues, four equal to 1 -- the structure of the SO(6) vector Goldstone."""
    ev = np.linalg.eigvals(X.goldstone_6(0.6, 0.0))
    n_unit = int(np.sum(np.abs(ev - 1.0) < 1e-9))
    assert n_unit == 4                                           # SO(6) vector U6(th,0) also has 4
    ev6 = np.linalg.eigvals(so6.U6_vector(0.6, 0.0))
    assert int(np.sum(np.abs(ev6 - 1.0) < 1e-9)) == 4


# --------------------------------------------------------------------------------------- #
#  Strong cross-check: the SU(4) antisym 6 IS the SO(6) vector (an orthonormal intertwiner exists)
# --------------------------------------------------------------------------------------- #
def test_su4_antisym6_is_the_irreducible_6_of_so6():
    """The strong statement of SU(4) ~= Spin(6): the 15 generators of su(4) in the antisymmetric 6
    span the full 15-dimensional so(6) ~= su(4) Lie algebra, and act irreducibly on the 6 (commutant
    = scalars, by Schur).  So the SU(4) antisymmetric 6 IS the irreducible 6 of so(6) -- the SO(6)
    vector -- independently of any basis alignment."""
    G = X.antisym6().generators
    M = np.array([g.ravel() for g in G])
    assert len(G) == 15 and np.linalg.matrix_rank(M, tol=1e-9) == 15      # span = dim so(6)
    # commutant dimension: matrices C with [G_a, C] = 0 for all a (1 <=> irreducible)
    n = 6
    A = np.vstack([np.kron(np.eye(n), g) - np.kron(g.T, np.eye(n)) for g in G])
    commutant_dim = n * n - np.linalg.matrix_rank(A, tol=1e-9)
    assert commutant_dim == 1
    # and its USp(4) sub-branching is the SO(6) vector's SO(5) branching (1 + 5)
    assert sorted(d for _, d in X.antisym6().branch_dims()) == \
           sorted(d for _, d in so6.so5_content(so6.rep_basis('6')))
