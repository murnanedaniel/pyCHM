"""Fine-tuning measures from the electroweak vacuum.

For the single tuned observable m_Z (proportional to v = f sqrt(xi); at fixed v the tuning
lives in f), the sensitivity vector is J_i = d ln f / d ln x_i over the microscopic
parameters x_i.  The 1x1 Fisher matrix is F = |J|^2, giving:
  Delta_BG = max_i |J_i|          (Barbieri-Giudice)
  HOT      = |J|                  (L2 norm)
  I        = 1/2 log(1 + |J|^2)   (information = 1/2 log det(I+F), nats)
  KL       = I                    (prior->posterior KL, Gaussian limit)
"""
import numpy as np
from . import spectrum

# microscopic parameters differentiated for the tuning
TUNED = ['mU', 'mUt', 'mD', 'mDt', 'mYu', 'Yu', 'mYd', 'Yd', 'Lq', 'Lt', 'Lb', 'grho']


def sensitivity_vector(P, route='formfactor', pars=TUNED):
    s0 = spectrum.spectrum(P, route=route)
    if s0 is None:
        return None
    J = []
    for k in pars:
        x = P.get(k, 0.0)
        if abs(x) < 1e-9:
            J.append(0.0); continue
        h = 0.01*abs(x)
        sp = spectrum.spectrum({**P, k: x+h}, route=route)
        sm = spectrum.spectrum({**P, k: x-h}, route=route)
        if sp is None or sm is None:
            J.append(0.0); continue
        J.append((np.log(sp['f']) - np.log(sm['f']))/(2*h)*x)
    return np.array(J)


def tuning(P, route='formfactor', pars=TUNED):
    """Return {BG, HOT, I, KL} (and the sensitivity vector J), or None if no EWSB."""
    J = sensitivity_vector(P, route=route, pars=pars)
    if J is None:
        return None
    g2 = float(J @ J)
    I = 0.5*np.log1p(g2)
    return dict(BG=float(np.max(np.abs(J))), HOT=float(np.sqrt(g2)), I=I, KL=I, J=J)
