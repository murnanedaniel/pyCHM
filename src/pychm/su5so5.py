"""SU(5)/SO(5) composite Higgs -- the littlest-Higgs / Ferretti real-rep coset as a runnable model.

Two-site model with the fermion partners dressed by the SU(5)/SO(5) Goldstone (`groups.su5so5`).
The coset carries 14 pNGBs: a Higgs doublet `h`, a complex triplet `phi`, and a singlet `eta`.
EWSB is along the Higgs; the triplet and singlet get calculable masses from the Coleman-Weinberg
potential -- the new observables relative to the MCHM/NMCHM.

Two registered variants:
  '5-5-su5'   : q_L, t_R partners in the SU(5) fundamental 5 (= the (2,2)+(1,1) of SO(4), i.e. the
                5-5-5 partner content placed in the SU(5)/SO(5) coset).  At theta_eta=theta_phi=0 the
                Higgs dressing's magnitude reduces to the MCHM 5-plet (top/Higgs match the 5-5-5).
  '15-15-su5' : partners in the symmetric 15 of SU(5) (branches 14+1 of SO(5)); the fuller top sector.

There is no public engine for SU(5)/SO(5); the model is validated by internal consistency (algebra,
branchings, Goldstone unitarity, EWSB to a finite spectrum, positive stencil-robust pNGB masses) and
by the reduction of the Higgs dressing to the validated MCHM on the Higgs axis.

Parameters: the 5-5-5 dict (mU,mUt,mD,mDt,mYu,Yu,mYd,Yd,Delta_uL/uR/dL/dR, f,f1,fX,g,gp,grho,gX);
optional `te0`,`tp0` (frozen singlet/triplet angles for the 1-field pipeline; default 0).
"""
import numpy as np

from . import mchm5
from .groups import su5so5 as _g

Nc = 3.0
_R2 = np.sqrt(2.0)
_MU, _MC = 0.544573 * 0.00216, 0.480885 * 1.275
_MD, _MS = 0.544573 * 0.00468, 0.544573 * 0.095
_e5 = np.eye(5, dtype=complex)
_e15 = None     # symmetric-15 basis, built lazily


def _ov(c, U, E):
    return np.vdot(c, U @ E)


# ----- the 5-5-5 partner content placed in the SU(5)/SO(5) fundamental 5 ------------------------- #
def spec5(P, up=True):
    if up:
        m4, m1, mY, Yk, dL, dR = P['mU'], P['mUt'], P['mYu'], P['Yu'], P['Delta_uL'], P['Delta_uR']
        o4, o1, oYo, odL = P['mD'], P['mDt'], P['mYd'], P['Delta_dL']
        light = [_MU * 1e-3, _MC * 1e-3]
    else:
        m4, m1, mY, Yk, dL, dR = P['mD'], P['mDt'], P['mYd'], P['Yd'], P['Delta_dL'], P['Delta_dR']
        o4, o1, oYo, odL = P['mU'], P['mUt'], P['mYu'], P['Delta_uL']
        light = [_MD * 1e-3, _MS * 1e-3]
    EqL = (_e5[0] + _e5[3]) / _R2          # q_L up-component: the (2,2) bidoublet
    EtR = _e5[4]                           # t_R: the custodial SO(5)-singlet (index 4)
    return dict(light=light, E_qL=EqL, E_tR=EtR, multiplets=[
        dict(m4=m4, m1=m1, mY=mY,      c4=-(_e5[0] + _e5[3]) / _R2, dL=dL,  c1=1j * _e5[3] / _R2, dR=dR),
        dict(m4=m4, m1=m1, mY=mY,      c4=(_e5[0] - _e5[3]) / _R2,  dL=dL,  c1=1j * _e5[3] / _R2, dR=dR),
        dict(m4=o4, m1=o1, mY=oYo,     c4=-_R2 * _e5[0],            dL=odL, c1=0 * _e5[0],         dR=0.0),
        dict(m4=m4, m1=m1, mY=mY + Yk, c4=1j * _e5[4],             dL=dL,  c1=-_e5[4],            dR=dR),
    ])


def _assemble(sector, U):
    light, EqL, EtR, mults = sector['light'], sector['E_qL'], sector['E_tR'], sector['multiplets']
    n = len(light) + 1 + 2 * len(mults)
    M = np.zeros((n, n), dtype=complex)
    for i, m in enumerate(light):
        M[i, i] = m
    el, base = len(light), len(light) + 1
    for k, mp in enumerate(mults):
        i4, i1 = base + 2 * k, base + 2 * k + 1
        M[i4, i4], M[i1, i1], M[i4, i1] = mp['m4'], mp['m1'], mp['mY']
        if np.any(mp['dL']):
            M[el, i4] = mp['dL'] * _ov(mp['c4'], U, EqL)
        if mp.get('dR'):
            M[i1, el] = mp['dR'] * _ov(mp['c1'], U, EtR)
    return M


# ----- three-angle mass matrices (Higgs, singlet eta, triplet phi) ------------------------------ #
def mass_U3(P, th, te, tp):
    return _assemble(spec5(P, up=True), _g.goldstone_5(th, te, tp))


def mass_D3(P, th, te, tp):
    return _assemble(spec5(P, up=False), _g.goldstone_5(th, te, tp))


# ----- single-angle adapters for the existing pipeline (eta,phi frozen at te0,tp0) -------------- #
def mass_U(P, sh):
    return mass_U3(P, np.arcsin(np.clip(sh, -1.0, 1.0)), P.get('te0', 0.0), P.get('tp0', 0.0))


def mass_D(P, sh):
    return mass_D3(P, np.arcsin(np.clip(sh, -1.0, 1.0)), P.get('te0', 0.0), P.get('tp0', 0.0))


def mass2_W(P, sh):
    return mchm5.mass2_W(P, sh)


def mass2_Z(P, sh):
    return mchm5.mass2_Z(P, sh)


# ----- the three-field CW potential and the pNGB masses ----------------------------------------- #
_CF = -1.0 / (16 * np.pi**2)
_CV = 3.0 / (64 * np.pi**2)


def _K(m2):
    m2 = np.maximum(np.atleast_1d(m2), 0.0)
    return np.where(m2 > 1e-30, m2**2 * np.log(np.maximum(m2, 1e-30)), 0.0)


def potential(P, th, te, tp):
    """V(Higgs, eta, triplet) (eigenvalue route), offset to V(0,0,0)=0."""
    def Vraw(a, b, c):
        v = 0.0
        for Mf in (mass_U3(P, a, b, c), mass_D3(P, a, b, c)):
            v += 3.0 * _CF * np.sum(_K(np.linalg.svd(Mf, compute_uv=False)**2))
        sh = np.sin(a)
        v += 2.0 * _CV * np.sum(_K(np.abs(np.linalg.eigvalsh(mchm5.mass2_W(P, sh)))))
        eZ = np.sort(np.abs(np.linalg.eigvalsh(mchm5.mass2_Z(P, sh))))
        v += _CV * np.sum(_K(eZ[1:]))
        return v
    return Vraw(th, te, tp) - Vraw(1e-9, 1e-9, 1e-9)


# =============================================================================================== #
#  '15-15-su5' variant: composite partners in the symmetric 15 of SU(5) (branches 14+1 of SO(5))
# =============================================================================================== #
_SYM15 = None


def _sym15_basis():
    global _SYM15
    if _SYM15 is None:
        from .groups import tensors as _T
        _SYM15 = _T.tensor_basis('sym', 2, 5, group='SU')
    return _SYM15


def _to15(Tmat):
    """Coordinates of a symmetric 5x5 tensor in the orthonormal sym-15 basis."""
    return np.array([np.vdot(b, Tmat) for b in _sym15_basis()])


def _symprod(v, w):
    v, w = np.asarray(v, dtype=complex), np.asarray(w, dtype=complex)
    M = np.outer(v, w) + np.outer(w, v)
    return M / np.sqrt(np.vdot(M, M))


def _lift5to15(v):
    """Embed a 5-vector partner/elementary direction into the symmetric 15 as sym(v (x) e4) (one
    leg along the custodial singlet), giving a (1/2,1/2)-or-(0,0) 15 state."""
    return _to15(_symprod(np.asarray(v, dtype=complex), _e5[4]))


def spec15(P, up=True):
    """The 5-5-5 partner content lifted into the symmetric 15 of SU(5): q_L in the (1/2,1/2), t_R in
    the (0,0) singlet, four composite 15-plet levels (11x11 matrix, so the light top is the 3rd
    singular value as the pipeline expects)."""
    s5 = spec5(P, up=up)                                # reuse the 5-5-5 structure, then lift to 15
    return dict(light=s5['light'], E_qL=_lift5to15(s5['E_qL']), E_tR=_lift5to15(s5['E_tR']),
                multiplets=[dict(m4=mp['m4'], m1=mp['m1'], mY=mp['mY'],
                                 c4=_lift5to15(mp['c4']), dL=mp['dL'],
                                 c1=_lift5to15(mp['c1']) if np.any(mp['c1']) else _to15(np.zeros((5, 5))),
                                 dR=mp['dR']) for mp in s5['multiplets']])


def _mass15(sector, U15):
    return _assemble(sector, U15)


def mass_U15(P, th, te=0.0, tp=0.0):
    U = _g.goldstone_rep(_sym15_basis(), th, te, tp)
    return _mass15(spec15(P, up=True), U)


def mass_D15(P, th, te=0.0, tp=0.0):
    U = _g.goldstone_rep(_sym15_basis(), th, te, tp)
    return _mass15(spec15(P, up=False), U)


class _Model15:
    """Drop-in model object with partners in the symmetric 15 (gauge sector reused from mchm5)."""
    def mass_U(self, P, sh):
        return mass_U15(P, np.arcsin(np.clip(sh, -1.0, 1.0)), P.get('te0', 0.0), P.get('tp0', 0.0))

    def mass_D(self, P, sh):
        return mass_D15(P, np.arcsin(np.clip(sh, -1.0, 1.0)), P.get('te0', 0.0), P.get('tp0', 0.0))

    def __getattr__(self, name):       # mass2_W, mass2_Z
        return getattr(mchm5, name)


model_15 = _Model15()


def singlet_mass2(P, th_vac, f=1.0, h=1e-3):
    """m_eta^2 = (1/f^2) d^2V/d(theta_eta)^2 at the vacuum (the SO(5)-singlet pNGB)."""
    V = lambda b: potential(P, th_vac, b, 0.0)
    return (V(h) - 2 * V(0.0) + V(-h)) / h**2 / f**2


def triplet_mass2(P, th_vac, f=1.0, h=1e-3):
    """m_phi^2 = (1/f^2) d^2V/d(theta_phi)^2 at the vacuum (the complex-triplet pNGB; one custodial
    mass)."""
    V = lambda c: potential(P, th_vac, 0.0, c)
    return (V(h) - 2 * V(0.0) + V(-h)) / h**2 / f**2
