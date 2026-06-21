"""Canonical Higgs-dressing factors of the three benchmark M4DCHM models, in closed symbolic
form (functions of theta = h/f).  These are the matrix elements of the Goldstone matrix U_R(h)
between the elementary embeddings and the composite SO(4) sub-multiplets -- not hand-fitted.

`symbolic.derive.solve_composite` proves each one is a genuine matrix element (see
tests/test_symbolic.py).  `assemble.py` lambdifies these to numpy callables for runtime, so the
mass matrices carry no curve-fitting.

Index keys are the composite column in the corresponding mass matrix.  The 14-14-10 model reuses
the same dressing *values* at shifted columns (see assemble.py).
"""
import sympy as sp

from .core import theta as _t

_c, _s = sp.cos(_t), sp.sin(_t)
_c2, _s2 = sp.cos(2 * _t), sp.sin(2 * _t)
_r5, _R2 = sp.sqrt(5), sp.sqrt(2)

# q_L (14) up-component -> composite 4-plet / mixed states
F_QL_UP = {3: _r5 * _s2 / 4,
           6: -(_c + _c2) / 2,
           8: -sp.I * (_c - _c2) / 2,
           9: (2 * _s - _s2) / 4,
           10: -_s2 / 4,
           11: (2 * _s + _s2) / 4}

# q_L (14) down-component
F_QL_DN = {4: -_c,
           5: -_s / _R2,
           6: sp.I * _s / _R2}

# b_R (10) -> composite antisymmetric states
F_BR = {3: -sp.I * _s / _R2,
        7: -(1 - _c) / 2,
        8: -(1 + _c) / 2}

# t_R as the 14 SO(4)-singlet (only present in 14-14-10)
F_TR_14 = {4: -(3 + 5 * _c2) / 8,
           7: -_r5 * _s2 / 4,
           10: -sp.I * _r5 * _s2 / 4,
           12: -_r5 * (1 - _c2) / 8,
           14: -_r5 * (1 - _c2) / 8,
           16: _r5 * (1 - _c2) / 8}

# which SO(5) irrep each dressing block lives in
REP = {'F_QL_UP': '14', 'F_QL_DN': '14', 'F_BR': '10', 'F_TR_14': '14'}


def _tensor(rep, entries):
    """Exact 5x5 (anti)symmetric tensor from {(i,j): value}; '14' symmetric, '10' antisymmetric."""
    M = sp.zeros(5, 5)
    for (i, j), v in entries.items():
        M[i, j] = v
        M[j, i] = v if rep == '14' else -v
    return M


# Exact elementary embeddings (sympy), matching assemble._E_* but with rational/sqrt entries so
# the symbolic composite solve is exact.  q_L up = sym(e3 (x) (e0+e4)); q_L down = (e0 e3);
# b_R = (e0^e1 + e0^e4); t_R = the 14 SO(4)-singlet.
E_QL_UP = _tensor('14', {(0, 3): sp.Rational(1, 2), (3, 4): sp.Rational(1, 2)})
E_QL_DN = _tensor('14', {(0, 3): 1 / sp.sqrt(2)})
E_BR = _tensor('10', {(0, 1): sp.Rational(1, 2), (0, 4): sp.Rational(1, 2)})
E_TR_14 = sp.diag(1, 1, 1, 1, -4) / sp.sqrt(20)

# (dressing dict, irrep, exact embedding) for each benchmark block
BLOCKS = [(F_QL_UP, '14', E_QL_UP), (F_QL_DN, '14', E_QL_DN),
          (F_BR, '10', E_BR), (F_TR_14, '14', E_TR_14)]
