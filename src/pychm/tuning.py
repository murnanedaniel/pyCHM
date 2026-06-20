"""Fine-tuning measures from the electroweak vacuum.

The tuned observable is m_Z (proportional to v = f sqrt(xi); at fixed v the tuning lives
in f), and the sensitivity vector is J_i = d ln f / d ln x_i over the *fundamental*
Lagrangian parameters x_i.

Basis matters.  Barbieri-Giudice is a max over a chosen parameter basis, so the basis must
be the fundamental mass parameters of the Lagrangian.  In the two-site M4DCHM these are the
partner masses {mU, mUt, mD, mDt}, the two Y-sector masses {mY, mSY = mY + Y} per quark
type, and the elementary-composite mixings {Delta_uL, Delta_uR, Delta_dL, Delta_dR}.  We
store the Y sector as (mY, Y); differentiating in the (mY, mSY) basis -- holding the *other*
mass fixed -- reproduces the standard BG number (and matches the validated pypngb engine to
<0.5% on the reference point).

The 1x1 Fisher matrix is F = |J|^2, giving:
  Delta_BG = max_i |J_i|          (Barbieri-Giudice)
  HOT      = |J|                  (L2 norm)
  I        = 1/2 log(1 + |J|^2)   (information = 1/2 log det(I+F), nats)
  KL       = I                    (prior->posterior KL, Gaussian limit)
"""
import math
import numpy as np
from . import spectrum

# fundamental log-parameters differentiated for the tuning.  'mSYu'/'mSYd' are the
# singlet-Y masses (mY + Y), held independent of mY in the BG basis.
TUNED = ['mU', 'mUt', 'mYu', 'mSYu', 'mD', 'mDt', 'mYd', 'mSYd',
         'Delta_uL', 'Delta_uR', 'Delta_dL', 'Delta_dR']

# 14-14-10 fundamental massive params.  The two singlet-Y combinations are
#   mSYu  = mYu + Yu/2    (the (6,7)/(9,8) matrix entries)
#   mSYtu = mYu + 4(Yu+Ytu)/5    (the (3,4) entry, top-partner Y)
# both held independent of mYu in the BG basis; differentiating Yu/Ytu directly
# would mix these into mYu, not the Lagrangian-mass basis pypngb defines tuning over.
TUNED_14 = ['mQ', 'mU', 'mD', 'mYu', 'mSYu', 'mSYtu', 'Yd',
            'Delta_q', 'Delta_u', 'Delta_d']

# 14-1-10 fundamental massive params.  The t_R partner is an SO(4) singlet, so the
# up Y-sector collapses to a single (mU, Yu): the matrix-entry params ARE the
# Lagrangian-mass params (functions_14_1_10_log maps each x = exp(log_x) directly),
# so every parameter is differentiated by plain log-scaling -- no (mY, mSY) basis.
TUNED_14_1_10 = ['mQ', 'mU', 'mD', 'Yu', 'Yd',
                 'Delta_q', 'Delta_u', 'Delta_d']

_TUNED = {'5-5-5': TUNED, '14-14-10': TUNED_14, '14-1-10': TUNED_14_1_10,
          '5-5-5-assembled': TUNED}


def _perturb(P, key, fac, model='5-5-5'):
    """Scale fundamental parameter `key` by `fac`, in the (mY, mSY) basis."""
    q = dict(P)
    if model == '14-14-10':
        # 14-14-10 (mYu, mSYu, mSYtu) basis: change Yu/Ytu so that whichever
        # combination is *not* being varied stays fixed.
        mYu, mSYu, mSYtu = P['mYu'], P['mYu'] + P['Yu']/2.0, P['mYu'] + 4.0*(P['Yu'] + P['Ytu'])/5.0
        if key == 'mSYu':
            mSYu *= fac
        elif key == 'mSYtu':
            mSYtu *= fac
        elif key == 'mYu':
            mYu *= fac
        else:
            q[key] = P[key]*fac
            return q
        q['mYu'] = mYu
        q['Yu'] = 2.0*(mSYu - mYu)
        q['Ytu'] = 5.0*(mSYtu - mYu)/4.0 - q['Yu']
        return q
    if key == 'mSYu':
        q['Yu'] = (P['mYu'] + P['Yu'])*fac - P['mYu']
    elif key == 'mYu':
        q['mYu'] = P['mYu']*fac; q['Yu'] = (P['mYu'] + P['Yu']) - q['mYu']
    elif key == 'mSYd':
        q['Yd'] = (P['mYd'] + P['Yd'])*fac - P['mYd']
    elif key == 'mYd':
        q['mYd'] = P['mYd']*fac; q['Yd'] = (P['mYd'] + P['Yd']) - q['mYd']
    else:
        q[key] = P[key]*fac
    return q


def sensitivity_vector(P, route='eigenvalue', pars=None, h=1e-3, model='5-5-5'):
    """J_i = d ln f / d ln x_i by central log-difference, or None if no EWSB at the point."""
    if pars is None:
        pars = _TUNED[model]
    if spectrum.spectrum(P, route=route, model=model) is None:
        return None
    J = []
    for k in pars:
        sp = spectrum.spectrum(_perturb(P, k, math.exp(h), model=model), route=route, model=model)
        sm = spectrum.spectrum(_perturb(P, k, math.exp(-h), model=model), route=route, model=model)
        if sp is None or sm is None:
            J.append(0.0); continue
        J.append((math.log(sp['f']) - math.log(sm['f']))/(2*h))
    return np.array(J)


def tuning(P, route='eigenvalue', pars=None, model='5-5-5'):
    """Return {BG, HOT, I, KL} (and the sensitivity vector J), or None if no EWSB."""
    J = sensitivity_vector(P, route=route, pars=pars, model=model)
    if J is None:
        return None
    g2 = float(J @ J)
    I = 0.5*np.log1p(g2)
    return dict(BG=float(np.max(np.abs(J))), HOT=float(np.sqrt(g2)), I=I, KL=I, J=J)
