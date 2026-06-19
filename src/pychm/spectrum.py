"""Physical spectrum from a parameter point."""
import numpy as np
from . import mchm5, potential
from .potential import V_EW

G2, GP = 0.6483, 0.3585   # SU(2)_L, U(1)_Y couplings (for m_W, m_Z)


def spectrum(P, route='formfactor'):
    """Solve EWSB and return observables, or None if the point doesn't break EWSB.
    f is rescaled so v = 246 GeV (the LM4DCHM convention)."""
    xi = potential.vacuum_xi(P, route=route)
    if xi is None:
        return None
    f = V_EW / np.sqrt(xi)
    g, b, d = potential.potential_coeffs(P, route=route)
    mh2 = (2*b*xi + 4*d*xi**2) * 2 * xi / f**2   # m_h^2 = (2 V''(xi) xi)/f^2 ... see note
    # V(x) = -g x + b x^2 + d x^3 ; m_h^2 = (1/f^2) d^2V/d(h/f)^2 = (4 xi/f^2)(b + 3 d xi) approx
    mh2 = (4.0*xi/f**2) * (b + 3*d*xi)
    return dict(
        xi=xi, f=f,
        mt=mchm5.fermion_mass(P, xi, up=True),
        mb=mchm5.fermion_mass(P, xi, up=False),
        mh=float(np.sqrt(mh2)) if mh2 > 0 else 0.0,
        mW=G2*V_EW/2.0, mZ=np.sqrt(G2**2 + GP**2)*V_EW/2.0,
    )
