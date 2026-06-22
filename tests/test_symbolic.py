"""Symbolic CCWZ engine: closed-form derivation matches the numeric ccwz engine, and the
hand-coded trig factors follow from group theory (symbolic equality)."""
import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from pychm.groups import so5 as ccwz
from pychm import assemble
from pychm.groups import so5 as core, derive
from pychm.groups import models as M


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


# ---- Stage 2: every benchmark dressing is a genuine Goldstone matrix element (derived) ---- #
@pytest.mark.parametrize("F,rep,E", M.BLOCKS)
def test_dressings_are_matrix_elements(F, rep, E):
    """Each hand-coded closed form equals <c|U_R|E> for an EXACT composite c -- no fitting.
    `solve_composite` raises if a target is not a Goldstone matrix element, so reaching the
    equality assertion already proves derivability."""
    for k, target in F.items():
        c = derive.solve_composite(rep, E, target)        # raises if not realisable
        assert sp.simplify(core.overlap_sym(rep, c, E) - target) == 0
        if rep == '14':
            assert sp.simplify(c - c.T) == sp.zeros(5, 5)  # valid symmetric tensor
        else:
            assert sp.simplify(c + c.T) == sp.zeros(5, 5)  # valid antisymmetric tensor


def test_rep5_closed_forms_derived():
    """The 5-5-5 dressing factors cos(h/f), sin(h/f), cos^2(h/2f) follow from U_vector."""
    e = {i: core.embedding_sym('5', f'fourplet_{i}') for i in range(4)}
    e4 = core.embedding_sym('5', 'singlet')
    t = core.theta
    assert sp.simplify(core.overlap_sym('5', e[3], e[3]) - sp.cos(t)) == 0   # U[3,3]
    assert sp.simplify(core.overlap_sym('5', e[3], e4) - sp.sin(t)) == 0     # U[3,4]
    qL = (e[0] + e[3]) / sp.sqrt(2)                                          # q_L up embedding
    assert sp.simplify(core.overlap_sym('5', qL, qL) - (1 + sp.cos(t)) / 2) == 0  # cos^2(h/2f)
