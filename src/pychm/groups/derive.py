"""Exact symbolic derivation of composite states and dressing factors.

The numerical `assemble._solve_composites` fits a composite tensor to a dressing function by
least squares over 40 sh samples.  Here the same object is obtained *exactly*: a dressing
factor f(theta) is a genuine Goldstone matrix element iff it lies in the span of the
single-basis overlaps {<E_b|U_R|E>}; `solve_composite` finds the exact minimal-norm composite
tensor c with <c|U_R|E> = f, and raises if f is not realisable (proving/​disproving that a
closed form is a matrix element of U_R -- the "derive from scratch" check).
"""
import sympy as sp

from .so5 import theta, rep_basis_sym, overlap_sym

_z = sp.symbols('z')


def trig_vec(expr, K):
    """Coordinate vector of a trigonometric polynomial in theta in the exp(i k theta) basis,
    k = -K..K.  Two trig polynomials are equal iff their trig_vec(.,K) coincide for large K."""
    e = sp.expand(sp.sympify(expr).rewrite(sp.exp))
    e = e.subs(sp.exp(sp.I * theta), _z).subs(sp.exp(-sp.I * theta), 1 / _z)
    num = sp.expand(sp.together(sp.expand(e)) * _z**K)      # clear negative powers
    P = sp.Poly(num, _z)
    return sp.Matrix([P.coeff_monomial(_z**k) for k in range(0, 2 * K + 1)])


def solve_composite(rep, E, target, K=6):
    """Exact minimal-norm composite tensor c in `rep` with overlap_sym(rep, c, E) == target.

    Raises ValueError if `target` is not a Goldstone matrix element of E (not in the orbit
    span).  rep must be a tensor rep ('10' or '14')."""
    E = sp.Matrix(E)
    basis = rep_basis_sym(rep)
    G = sp.Matrix.hstack(*[trig_vec(overlap_sym(rep, b, E), K) for b in basis])
    x = G.pinv() * trig_vec(target, K)
    # overlap_sym conjugates the bra, so c = sum conj(x_b) E_b reproduces the target exactly.
    c = sum((sp.conjugate(xi) * bi for xi, bi in zip(x, basis)), sp.zeros(5, 5))
    c = sp.simplify(c)
    resid = sp.simplify(overlap_sym(rep, c, E) - sp.sympify(target))
    if resid != 0:
        raise ValueError(f"{target} is not a matrix element of U_{rep} on E (residual {resid})")
    return c


def is_matrix_element(rep, E, target, K=6):
    """True iff `target` is a Goldstone matrix element <c|U_rep|E> for some composite c."""
    try:
        solve_composite(rep, E, target, K=K)
        return True
    except ValueError:
        return False
