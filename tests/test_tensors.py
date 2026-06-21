"""Stage 3: arbitrary compatible SO(5) tensor representations.

The general tensor engine reproduces the benchmark 10 and 14 and extends to new irreps (e.g.
the 30 = symmetric-traceless rank-3) with no hand-coded reference, validated by internal
consistency: dimension, unitarity, identity at sh=0, the representation homomorphism, and the
SO(4) = SU(2)_L x SU(2)_R branching.
"""
import numpy as np
import pytest

pytest.importorskip("sympy")

from pychm import ccwz
from pychm.symbolic import tensors as T, decompose as D

# (name, symmetry, rank, dim, expected SO(4) content as {(jL,jR): mult})
REPS = [
    ('5',  'sym', 1, 5,  {(0.5, 0.5): 1, (0.0, 0.0): 1}),
    ('10', 'antisym', 2, 10, {(0.5, 0.5): 1, (1.0, 0.0): 1, (0.0, 1.0): 1}),
    ('14', 'sym', 2, 14, {(1.0, 1.0): 1, (0.5, 0.5): 1, (0.0, 0.0): 1}),
    ('30', 'sym', 3, 30, {(1.5, 1.5): 1, (1.0, 1.0): 1, (0.5, 0.5): 1, (0.0, 0.0): 1}),
]


@pytest.mark.parametrize("name,sym,rank,dim,content", REPS)
def test_dimension_and_content(name, sym, rank, dim, content):
    B = T.tensor_basis(sym, rank, 5)
    assert len(B) == dim
    assert D.so4_content(B) == content
    # SO(4) multiplet dimensions sum to the irrep dimension
    assert sum(int((2 * a + 1) * (2 * b + 1)) * m for (a, b), m in content.items()) == dim


@pytest.mark.parametrize("name,sym,rank,dim,content", REPS)
def test_unitary_identity_homomorphism(name, sym, rank, dim, content):
    B = T.tensor_basis(sym, rank, 5)
    I = np.eye(dim)
    assert np.allclose(T.U_rep_tensor(B, ccwz.U_vector(0.0)), I)            # U(0) = 1
    for s in (0.2, 0.5, 0.85):
        U = T.U_rep_tensor(B, ccwz.U_vector(s))
        assert np.allclose(U @ U.T, I, atol=1e-10)                          # orthogonal
    # representation homomorphism: rotations compose additively in the angle theta
    th1, th2 = 0.31, 0.47
    U1 = T.U_rep_tensor(B, ccwz.U_vector(np.sin(th1)))
    U2 = T.U_rep_tensor(B, ccwz.U_vector(np.sin(th2)))
    U12 = T.U_rep_tensor(B, ccwz.U_vector(np.sin(th1 + th2)))
    assert np.allclose(U1 @ U2, U12, atol=1e-10)


@pytest.mark.parametrize("rep,sym,rank", [('10', 'antisym', 2), ('14', 'sym', 2)])
def test_general_matches_ccwz(rep, sym, rank):
    """The general tensor path is equivalent to ccwz's hand-built 10/14: equal as
    representations, hence equal basis-independent invariants Tr U^p at every sh."""
    B = T.tensor_basis(sym, rank, 5)
    for s in (0.13, 0.6, 0.9):
        Ug = T.U_rep_tensor(B, ccwz.U_vector(s))
        Ur = ccwz.U_rep(rep, s)
        for p in (1, 2, 3):
            assert np.isclose(np.trace(np.linalg.matrix_power(Ug, p)),
                              np.trace(np.linalg.matrix_power(Ur, p)), atol=1e-9)


def test_ccwz_U_rep_descriptor_dispatch():
    """ccwz.U_rep accepts a tensor descriptor: ('sym', 3) is the 30."""
    U = ccwz.U_rep(('sym', 3), 0.4)
    assert U.shape == (30, 30)
    assert np.allclose(U @ U.T, np.eye(30), atol=1e-10)


def test_compatibility_predicate():
    """q_L = (2,2) and t_R = (1,1) singlet embed in 5/14; the 10 hosts (2,2) and (1,3)/(3,1)."""
    B5 = T.tensor_basis('sym', 1, 5)
    B10 = T.tensor_basis('antisym', 2, 5)
    B14 = T.tensor_basis('sym', 2, 5)
    assert D.compatible(B5, 0.5, 0.5)            # q_L (2,2) in the 5
    assert D.compatible(B5, 0.0, 0.0)            # t_R singlet in the 5
    assert D.compatible(B14, 0.0, 0.0)           # t_R singlet in the 14
    assert not D.compatible(B5, 1.0, 1.0)        # 5 has no (3,3)
    assert D.compatible(B10, 1.0, 0.0)           # (3,1) lives in the 10
    assert not D.compatible(B10, 0.0, 0.0)       # 10 has no SO(4) singlet
