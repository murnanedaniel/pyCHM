"""Physical spectrum from a parameter point (eigenvalue route)."""
import numpy as np
from . import mchm5, mchm14, mchm14_1_10, potential
from .potential import V_EW

G2, GP = 0.6483, 0.3585   # SU(2)_L, U(1)_Y couplings (for m_W, m_Z)

_MODELS = {'5-5-5': mchm5, '14-14-10': mchm14, '14-1-10': mchm14_1_10}


def _higgs_mass2(P, xi, route, f, model='5-5-5'):
    """m_h^2 = (1-xi) V''(s_h) / f^2 at the minimum (V'' = d^2 V/d s_h^2)."""
    sh = np.sqrt(xi)
    h = 1e-3
    V = lambda s: potential.routes.potential_curve(P, np.array([0.0, s]), route=route, model=model)[1]
    Vpp = (V(sh + h) - 2*V(sh) + V(sh - h))/h**2
    return (1 - xi)*Vpp/f**2


def spectrum(P, route='eigenvalue', model='5-5-5'):
    """Solve EWSB and return observables, or None if no EWSB. f rescaled so v = 246 GeV."""
    mod = _MODELS[model]
    xi = potential.vacuum_xi(P, route=route, model=model)
    if xi is None:
        return None
    f = V_EW/np.sqrt(xi)
    sh = np.sqrt(xi)
    mU = np.sort(np.linalg.svd(mod.mass_U(P, sh), compute_uv=False))   # u,c,t,partners
    mD = np.sort(np.linalg.svd(mod.mass_D(P, sh), compute_uv=False))   # d,s,b,partners
    mh2 = _higgs_mass2(P, xi, route, f, model=model)
    return dict(
        xi=xi, f=f,
        mt=float(mU[2]), mb=float(mD[2]),               # 3rd lightest = SM top / bottom
        mh=float(np.sqrt(mh2)) if mh2 > 0 else 0.0,
        mW=G2*V_EW/2.0, mZ=np.sqrt(G2**2 + GP**2)*V_EW/2.0,
        mtop_partner=float(mU[3]),                       # lightest top partner
    )
