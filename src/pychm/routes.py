"""The two evaluation routes for the Coleman-Weinberg potential.

Both diagonalise the mass matrices M(s_h); they differ only in the per-eigenvalue kernel:

  route='eigenvalue' : closed-form CW,   K(m^2) = m^4 log(m^2)
  route='momentum'   : Euclidean momentum integral of the *renormalised* one-loop density

This is the same one-loop potential (int log det = sum int log(eigenvalue)); the two routes
are an internal consistency check (tests/test_routes_equivalence.py enforces agreement on
random points).

The momentum kernel is the manifestly convergent, divergence-subtracted integrand

  K_mom(m^2) = 4 int_0^inf pE^3 [ log(pE^2+m^2) - log(pE^2+1)
                                  - (m^2-1)/(pE^2+1) + (m^2-1)^2/(2(pE^2+1)^2) ] dpE
             = m^4 log(m^2) - 3/2 m^4 + 2 m^2 - 1/2 .

The two subtractions cancel the quadratic and logarithmic UV divergences at the integrand
level (so there is no catastrophic cancellation of a ~Lambda^4 piece against a ~1e-3 signal,
the failure mode of the naive cutoff integral).  The extra -3/2 m^4 + 2 m^2 - 1/2 is a
polynomial of degree <=2 in m^2; because Str 1, Str m^2 and Str m^4 are all s_h-independent
in this model, it drops out of V(s_h)-V(0) -- exactly the closed-form route.

Sectors and dof weights (per eigenvalue): fermions U, D carry -3/(16 pi^2) (Dirac x3 colour);
gauge W +2*3/(64 pi^2), Z +3/(64 pi^2) skipping the photon. The s_h-independent exotic (Q4,Q5)
and lepton sectors drop out of V(s_h)-V(0) and are omitted.
"""
import numpy as np
from scipy import integrate
from . import mchm5, mchm14, mchm14_1_10
from . import assemble

_MODELS = {'5-5-5': mchm5, '14-14-10': mchm14, '14-1-10': mchm14_1_10, '5-5-5-assembled': assemble.model_555, '14-1-10-assembled': assemble.model_14_1_10}

_PI2 = np.pi**2
# uniform Euclidean grid; the subtracted integrand decays as 1/pE^3 so a moderate cutoff
# suffices.  This resolution gives the kernel to ~1e-5 relative -- needed because a tuned
# vacuum V(s_h)-V(0) is a near-cancellation that amplifies kernel error.
_PE = np.linspace(1e-5, 600.0, 200000)
_PE2 = _PE**2
_PE3 = _PE**3
_DEN = _PE2 + 1.0
_LOGDEN = np.log(_DEN)
_CF = -1.0/(16*_PI2)        # one colour, Dirac fermion (x3 colour applied below)
_CV = 3.0/(64*_PI2)         # one massive vector (3 polarisations)


def _K_closed(m2):
    m2 = np.maximum(np.atleast_1d(m2), 0.0)
    return np.where(m2 > 1e-30, m2**2 * np.log(np.maximum(m2, 1e-30)), 0.0)

def _K_mom(m2):
    """Convergent divergence-subtracted momentum integral, vectorised over eigenvalues."""
    m2 = np.atleast_1d(m2)[:, None]                  # (n_eig, 1)
    a = m2 - 1.0
    integ = _PE3*(np.log(_PE2 + m2) - _LOGDEN - a/_DEN + a*a/(2*_DEN*_DEN))
    return 4.0*integrate.simpson(integ, x=_PE, axis=1)

_KERNEL = {'eigenvalue': _K_closed, 'momentum': _K_mom}


def _sector_masses(P, sh, model='5-5-5'):
    """All s_h-dependent mass-squared eigenvalues with their CW coefficients."""
    mod = _MODELS[model]
    out = []
    for M in (mod.mass_U(P, sh), mod.mass_D(P, sh)):
        sv = np.linalg.svd(M, compute_uv=False)
        out.append((sv**2, 3.0*_CF))
    out.append((np.abs(np.linalg.eigvalsh(mod.mass2_W(P, sh))), 2.0*_CV))
    eZ = np.sort(np.abs(np.linalg.eigvalsh(mod.mass2_Z(P, sh))))
    out.append((eZ[1:], _CV))
    return out


def potential_curve(P, shs, route='eigenvalue', model='5-5-5'):
    """Total V(s_h) over the given s_h values, offset to V(0)=0."""
    K = _KERNEL[route]
    V = np.zeros(len(shs))
    for j, sh in enumerate(shs):
        for m2, c in _sector_masses(P, max(sh, 1e-9), model=model):
            V[j] += c * np.sum(K(m2))
    return V - V[0]
