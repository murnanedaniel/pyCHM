"""Branching of a representation under the unbroken subgroup H, via the quadratic Casimir.

Group-agnostic: each distinct eigenvalue of the H-Casimir on a rep is one H-irrep, with
multiplicity = eigenspace dimension / irrep dimension.  A per-(G,H) lookup table maps the Casimir
eigenvalue to the irrep label/dimension.  This folds `so6.so5_content*`, `so6.so4_content`, and the
Hodge self-dual split into one generic place; the SO(4) = SU(2)xSU(2) content reuses the 't Hooft
Casimirs in `symbolic.decompose`.
"""
import math

import numpy as np

from ..symbolic import decompose as _D
from ..symbolic import tensors as _T


# Casimir-eigenvalue -> (irrep name, dim) tables, in the normalisation
# C_H = sum_{generators of H} (T^a)^2 with the generators of the embedding used by each coset.
SO5_CASIMIR = {0.0: ('1', 1), 2.5: ('4', 4), 4.0: ('5', 5), 6.0: ('10', 10), 10.0: ('14', 14)}


def content_from_casimir(C, lut, tol=1e-6):
    """Branching [(name, dim), ...] from a Casimir matrix C and its eigenvalue->(name,dim) LUT.
    Degenerate eigenspaces are split into multiplicity copies (so the 3-form 20 = 10 + 10, the
    20' = 14 + 5 + 1, etc. come out right)."""
    w = np.linalg.eigvalsh((C + C.conj().T) / 2).real
    out, used = [], np.zeros(len(w), dtype=bool)
    for i in range(len(w)):
        if used[i]:
            continue
        grp = np.where(np.abs(w - w[i]) < tol)[0]
        used[grp] = True
        cval = round(float(np.mean(w[grp])), 4)
        match = min(lut, key=lambda v: abs(v - cval)) if lut else None
        if match is not None and abs(match - cval) < 1e-3:
            name, d = lut[match]
            out += [(name, d)] * (len(grp) // d)
        else:
            out.append((f'C={cval}', len(grp)))
    return sorted(out, key=lambda t: t[1])


def so4_content(generators_in_rep, tol=1e-6):
    """SO(4) = SU(2)_L x SU(2)_R multiplicity table {(jL,jR): mult} from the six SO(4) generators
    (indices 0..3) lifted to a rep.  Multiplicity-aware (folds `so6.so4_content`)."""
    table = {}
    for (jL, jR), cols in _D.so4_decompose_gen(generators_in_rep, tol):
        mult = round(cols.shape[1] / ((2 * jL + 1) * (2 * jR + 1)))
        table[(jL, jR)] = table.get((jL, jR), 0) + mult
    return table


# --------------------------------------------------------------------------------------- #
#  Hodge self-dual / anti-self-dual split of an antisymmetric p-form (folds so6._levi_civita6 etc.)
# --------------------------------------------------------------------------------------- #
def levi_civita(n):
    """Totally antisymmetric epsilon tensor in n dimensions (dense, shape (n,)*n)."""
    from itertools import permutations
    eps = np.zeros((n,) * n)
    for p in permutations(range(n)):
        eps[p] = _T._perm_sign(p)
    return eps


def selfdual_cols(basis, n, sign):
    """Columns spanning the (sign*i)-eigenspace of the Hodge dual on the rank-(n/2) form `basis`
    (n even).  For n=6, rank-3: splits the 20 into the self-dual 10 (+i) and anti-self-dual (-i)."""
    rank = n // 2
    eps = levi_civita(n)
    axes = list(range(rank, n))
    nb = len(basis)
    H = np.zeros((nb, nb), dtype=complex)
    fac = float(math.factorial(rank))
    for a, Ea in enumerate(basis):
        Da = np.tensordot(eps, Ea, axes=(axes, list(range(rank)))) / fac
        for b, Eb in enumerate(basis):
            H[b, a] = np.vdot(Eb, Da)
    w, V = np.linalg.eig(H)
    keep = np.where(np.abs(w - 1j * sign) < 1e-9)[0]
    Q, _ = np.linalg.qr(V[:, keep])
    return Q
