"""Symbolic CCWZ engine for pyCHM.

Derives the Higgs (Goldstone) dressing of fermion mixings in closed symbolic form, from
group theory alone -- no hand-coded trigonometry and no numerical curve-fitting.  Every
dressing factor is a matrix element <bra| U_R(h) |ket> of the Goldstone matrix in an SO(5)
irrep R; since U_R is built by lifting the closed-form vector rotation U_vector(h) (a planar
rotation in s = sin(h/f), c = cos(h/f)), each overlap is a closed trigonometric polynomial.

This is the symbolic counterpart of `pychm.ccwz`: `ccwz` evaluates the dressing numerically,
`symbolic` derives it in closed form and lambdifies it back to fast numpy callables.

Public API (see `core`):
    theta, sh, ch          symbols (theta = h/f; sh = sin theta; ch = cos theta)
    U_vector_sym(hat)      5x5 Goldstone matrix in the vector
    rep_basis_sym(rep)     orthonormal tensor basis of an irrep (matches ccwz ordering)
    U_rep_sym(rep, hat)    Goldstone matrix lifted to the irrep, trig-reduced
    overlap_sym(rep, ...)  <bra|U_rep|ket> in closed trig
    embedding_sym(rep,kind) orthonormal tensor for an SO(4) sub-multiplet
    as_sh_expr(expr)       rewrite a theta-expression in pyCHM's sh variable
    lambdify_sh(expr)      closed-form expr -> numpy callable f(sh)
"""
from .core import (
    theta, sh, ch,
    U_vector_sym, rep_basis_sym, U_rep_sym, overlap_sym, embedding_sym,
    to_closed_trig, as_sh_expr, lambdify_sh,
)

__all__ = [
    "theta", "sh", "ch",
    "U_vector_sym", "rep_basis_sym", "U_rep_sym", "overlap_sym", "embedding_sym",
    "to_closed_trig", "as_sh_expr", "lambdify_sh",
]
