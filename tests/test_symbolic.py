"""Symbolic CCWZ engine: closed-form derivation matches the numeric ccwz engine, and the
hand-coded trig factors follow from group theory (symbolic equality)."""
import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from pychm import ccwz
from pychm.symbolic import core


def test_worked_example_14_singlet():
    """The 14-singlet self-overlap is exactly (3 + 5 cos 2theta)/8 -- derived, not fitted."""
    ES = core.embedding_sym('14', 'singlet')
    val = core.overlap_sym('14', ES, ES)
    target = (3 + 5 * sp.cos(2 * core.theta)) / 8
    assert sp.simplify(val - target) == 0


def test_U_vector_matches_ccwz():
    f = sp.lambdify(core.sh, core.as_sh_expr(core.U_vector_sym()), modules='numpy')
    for s in (0.0, 0.13, 0.37, 0.6, 0.92):
        assert np.allclose(np.array(f(s), dtype=float), ccwz.U_vector(s), atol=1e-13)


@pytest.mark.parametrize("rep", ['10', '14'])
def test_U_rep_matches_ccwz(rep):
    """Lambdified symbolic U_rep reproduces the numeric ccwz.U_rep bit-for-bit."""
    Usym = core.U_rep_sym(rep)
    f = sp.lambdify(core.sh, core.as_sh_expr(Usym), modules='numpy')
    for s in (0.0, 0.13, 0.37, 0.6, 0.92):
        assert np.allclose(np.array(f(s), dtype=float), ccwz.U_rep(rep, s), atol=1e-12)


@pytest.mark.parametrize("rep", ['10', '14'])
def test_U_rep_unitary_symbolically(rep):
    U = core.U_rep_sym(rep)
    I = sp.eye(U.rows)
    assert sp.simplify(U.T * U - I) == sp.zeros(U.rows, U.rows)


@pytest.mark.parametrize("rep,n", [('5', 5), ('10', 10), ('14', 14)])
def test_overlap_matches_ccwz_random(rep, n):
    """Symbolic overlap == numeric ccwz.overlap for random embeddings, several sh."""
    rng = np.random.default_rng(7)
    if rep == '5':
        bra = rng.standard_normal(5); ket = rng.standard_normal(5)
        bras, kets = sp.Matrix(bra), sp.Matrix(ket)
    else:
        basis = ccwz._sym_traceless_basis() if rep == '14' else ccwz._antisym_basis()
        cb = rng.standard_normal(n); ck = rng.standard_normal(n)
        bra = sum(x * b for x, b in zip(cb, basis))
        ket = sum(x * b for x, b in zip(ck, basis))
        bras, kets = sp.Matrix(bra), sp.Matrix(ket)
    f = core.lambdify_sh(core.overlap_sym(rep, bras, kets))
    for s in (0.05, 0.3, 0.55, 0.8):
        assert np.isclose(complex(f(s)), ccwz.overlap(rep, bra, ket, s), atol=1e-12)
