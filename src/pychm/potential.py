"""Electroweak symmetry breaking: fit V(s_h) and find the vacuum xi = <s_h>^2."""
import numpy as np
from . import routes

V_EW = 0.24622   # Higgs vev [TeV]


def potential_coeffs(P, route='formfactor', sh_max=0.32, n=13):
    """Fit V(s_h) = -gamma s_h^2 + beta s_h^4 + delta s_h^6.  Returns (gamma, beta, delta)."""
    shs = np.linspace(0.0, sh_max, n)
    V = routes.potential_curve(P, shs, route=route)
    x = shs**2
    A = np.vstack([-x, x**2, x**3]).T
    g, b, d = np.linalg.lstsq(A, V, rcond=None)[0]
    return g, b, d


def vacuum_xi(P, route='formfactor'):
    """xi = sin^2<h>/f at the minimum of V, or None if no viable EWSB vacuum."""
    g, b, d = potential_coeffs(P, route=route)
    # dV/dx = -g + 2b x + 3d x^2 = 0
    roots = np.roots([3*d, 2*b, -g])
    roots = [r.real for r in roots if abs(r.imag) < 1e-9 and 0.0 < r.real < 1.0]
    # keep genuine minima (V'' = 2b + 6d x > 0)
    roots = [r for r in roots if (2*b + 6*d*r) > 0]
    if not roots:
        return None
    return min(roots)
