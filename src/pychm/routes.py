"""The two evaluation routes for the Coleman-Weinberg potential.

Both compute the same one-loop V(s_h); they differ in how the trace-log is taken:
  - 'eigenvalue' : diagonalise the mass matrix M(s_h), sum the closed-form CW over m_i.
  - 'formfactor' : integrate out the heavy partners (form factors), one momentum integral.

The gauge sector is computed by the eigenvalue route in BOTH cases (its mass matrices are
simple; the gauge form factor has resonance subtleties). The route choice therefore selects
the FERMION method. The two fermion methods are mathematically identical and must agree;
`tests/test_routes_equivalence.py` enforces this.

NOTE (roadmap): the eigenvalue FERMION method needs the 11x11 fermion mass matrices, which
are being ported (Phase 1). Until then `route='eigenvalue'` raises NotImplementedError and
`route='formfactor'` is the working route.
"""
import numpy as np
from scipy import integrate
from . import mchm5

# momentum grid for the form-factor / log-det integrals
_PE = np.linspace(1e-4, 60.0, 6000)
_PE2 = _PE**2
_PE3 = _PE**3


# --- gauge (eigenvalue route; shared) ---------------------------------------- #
def _veff_V(m_TeV):
    m2 = np.maximum(m_TeV**2, 0.0)
    return np.where(m2 > 1e-30, (3/64./np.pi**2) * m2**2 * np.log(np.maximum(m2, 1e-30)), 0.0)

def gauge_cw(P, shs):
    """Gauge CW V(s_h) over s_h values (eigenvalue route: W x2, Z skip photon)."""
    out = np.zeros(len(shs))
    for j, sh in enumerate(shs):
        eW = np.linalg.eigvalsh(mchm5.mass2_W(P, sh))
        eZ = np.sort(np.linalg.eigvalsh(mchm5.mass2_Z(P, sh)))
        mW = np.sqrt(np.abs(eW)); mZ = np.sqrt(np.abs(eZ[1:]))   # skip photon (lightest=0)
        out[j] = 2*np.sum(_veff_V(mW)) + np.sum(_veff_V(mZ))
    return out


# --- fermion: form-factor route ---------------------------------------------- #
def fermion_cw_formfactor(P, shs):
    """Fermion CW V(s_h) via the momentum integral of the log-determinant of form factors."""
    out = np.zeros(len(shs))
    for up in (True, False):
        L0, Ls, R0, Rs, M2c = mchm5.formfactor_pieces(P, _PE2, up=up)
        for j, sh in enumerate(shs):
            s = sh**2
            arg = _PE2*(L0 + s*Ls)*(R0 + s*Rs) + s*(1 - s)*M2c
            arg = np.where(arg > 1e-300, arg, 1e-300)
            out[j] += -2.0*mchm5.Nc*(1/(8*np.pi**2))*integrate.simpson(_PE3*np.log(arg), _PE)
    return out


# --- fermion: eigenvalue route (Phase 1) ------------------------------------- #
def fermion_cw_eigenvalue(P, shs):
    raise NotImplementedError(
        "eigenvalue fermion route needs the fermion mass matrices (Phase 1 port). "
        "Use route='formfactor' for now; see two_routes_equivalence demonstration.")


_FERMION = {'formfactor': fermion_cw_formfactor, 'eigenvalue': fermion_cw_eigenvalue}


def potential_curve(P, shs, route='formfactor'):
    """Total V(s_h) = fermion (chosen route) + gauge (eigenvalue), offset to V(0)=0."""
    V = _FERMION[route](P, shs) + gauge_cw(P, shs)
    return V - V[0]
