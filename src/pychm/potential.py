"""Electroweak symmetry breaking: find the vacuum xi = <s_h>^2 by minimising V(s_h).

We minimise the full potential directly (as the original code does), rather than fitting a
polynomial and solving -- a tuned vacuum is a near-cancellation, so the polynomial-truncation
error in the gamma/beta extraction would be amplified into a large error in xi.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from . import routes

V_EW = 0.24622   # Higgs vev [TeV]


def potential_coeffs(P, route='eigenvalue', sh_max=0.32, n=13, model='5-5-5'):
    """Diagnostic fit V(s_h) = -gamma s_h^2 + beta s_h^4 + delta s_h^6 (for inspection;
    the vacuum itself is found by direct minimisation, see vacuum_xi)."""
    shs = np.linspace(0.0, sh_max, n)
    V = routes.potential_curve(P, shs, route=route, model=model)
    x = shs**2
    A = np.vstack([-x, x**2, x**3]).T
    return tuple(np.linalg.lstsq(A, V, rcond=None)[0])


def _Vfun(P, route, model='5-5-5'):
    """V(s_h) as a callable, offset to V(0)=0."""
    V0 = routes.potential_curve(P, np.array([0.0, 1e-6]), route=route, model=model)[0]
    def V(sh):
        return routes.potential_curve(P, np.array([0.0, abs(sh)]), route=route, model=model)[1]
    return V


def vacuum_xi(P, route='eigenvalue', sh_hi=0.7, model='5-5-5'):
    """xi = sin^2<h>/f at the global minimum of V in (0, sh_hi), or None if the minimum is
    at the origin (no EWSB)."""
    V = _Vfun(P, route, model=model)
    grid = np.linspace(0.0, sh_hi, 60)
    vals = np.array([V(s) for s in grid])
    k = int(np.argmin(vals))
    if k == 0 or vals[k] >= -1e-12:           # minimum at origin -> no EWSB
        return None
    lo, hi = grid[max(k-1, 0)], grid[min(k+1, len(grid)-1)]
    res = minimize_scalar(V, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-6})
    sh = abs(res.x)
    if not (0.0 < sh < 1.0) or res.fun >= -1e-12:
        return None
    return sh**2
