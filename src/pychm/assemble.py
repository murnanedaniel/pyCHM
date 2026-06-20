"""Generic fermion mass-matrix assembler for SO(5)/SO(4) composite Higgs.

A model is specified declaratively -- the partner representation, the elementary-fermion
embeddings, and the composite states (their masses and SO(5) content) -- and the Higgs
(s_h) dependence of every elementary-composite mixing is computed by `ccwz`, not typed in.
The same assembler therefore builds the mass matrix for *any* partner representation; the
representation enters only through `ccwz.overlap(rep, ...)`.

A `Sector` lists:
  light    : [m_light, ...]                       decoupled light-generation masses
  E_qL,E_tR: complex vectors in the rep            elementary embeddings (q_L up-comp, t_R)
  mults    : list of composite multiplets, each
               m4, m1   : 4-plet and singlet Dirac masses
               mY       : proto-Yukawa (4-plet <-> singlet) mixing
               c4, dL   : composite 4-plet state (vector in rep) and its q_L mixing strength
               c1, dR   : composite singlet state (vector in rep) and its t_R mixing strength

The assembler builds, in the basis [light..., elementary, (c4,c1) per multiplet]:
  M[el, c4_k] = dL_k * <c4_k| U(s_h) |E_qL>      (q_L -> 4-plet, Higgs-dressed)
  M[c1_k, el] = dR_k * <c1_k| U(s_h) |E_tR>      (t_R -> singlet, Higgs-dressed)
  M[c4_k,c4_k]=m4_k,  M[c1_k,c1_k]=m1_k,  M[c4_k,c1_k]=mY_k
"""
import numpy as np
from . import ccwz


def assemble(sector, sh):
    rep = sector['rep']
    EqL, EtR = sector['E_qL'], sector['E_tR']
    light = sector.get('light', [])
    mults = sector['multiplets']
    n = len(light) + 1 + 2 * len(mults)
    M = np.zeros((n, n), dtype=complex)
    for i, m in enumerate(light):
        M[i, i] = m
    el = len(light)                       # elementary q_L/t_R index
    base = el + 1
    U = ccwz.U_rep(rep, sh) if rep != '5' else ccwz.U_vector(sh)
    for k, mp in enumerate(mults):
        i4, i1 = base + 2 * k, base + 2 * k + 1
        M[i4, i4] = mp['m4']
        M[i1, i1] = mp['m1']
        M[i4, i1] = mp['mY']
        if mp.get('dL'):
            M[el, i4] = mp['dL'] * _ov(rep, mp['c4'], U, EqL)
        if mp.get('dR'):
            M[i1, el] = mp['dR'] * _ov(rep, mp['c1'], U, EtR)
    return M


def _ov(rep, c, U, E):
    """<c| U |E> for vector (5) or tensor (10/14) reps."""
    if rep == '5':
        return np.vdot(c, U @ E)
    return np.sum(np.conjugate(c) * (U @ E @ U.T))


# --------------------------------------------------------------------------------------
# Worked model: the 5-5-5 fermion sector, assembled from embeddings + ccwz dressing.
# Validated to reproduce mchm5.mass_U / mass_D entry-for-entry (tests/test_assemble.py).
# --------------------------------------------------------------------------------------
_E = np.eye(5, dtype=complex)
_R2 = np.sqrt(2.0)
_MU, _MC = 0.544573 * 0.00216, 0.480885 * 1.275       # SM u,c [GeV] (as in mchm5)
_MD, _MS = 0.544573 * 0.00468, 0.544573 * 0.095        # SM d,s [GeV]


def spec_555(P, up=True):
    """Declarative 5-5-5 spec for the up (or down) sector: embeddings + composite content.
    The q_L up-component embeds in the (2,2) as (e0+e3)/sqrt2; t_R in the SO(5) singlet e4.
    The composite states are the (2,2)/(1,1) combinations whose ccwz overlaps give the
    Higgs dressing (cos h/f, sin h/f, cos^2 h/2f, ...) hand-coded in mchm5."""
    if up:
        m4, m1, mY, Yk, dL, dR = P['mU'], P['mUt'], P['mYu'], P['Yu'], P['Delta_uL'], P['Delta_uR']
        o4, o1, oYo, odL = P['mD'], P['mDt'], P['mYd'], P['Delta_dL']        # opposite (down) 4-plet
        light = [_MU * 1e-3, _MC * 1e-3]
    else:
        m4, m1, mY, Yk, dL, dR = P['mD'], P['mDt'], P['mYd'], P['Yd'], P['Delta_dL'], P['Delta_dR']
        o4, o1, oYo, odL = P['mU'], P['mUt'], P['mYu'], P['Delta_uL']
        light = [_MD * 1e-3, _MS * 1e-3]
    return dict(rep='5', light=light, E_qL=(_E[0] + _E[3]) / _R2, E_tR=_E[4], multiplets=[
        dict(m4=m4, m1=m1, mY=mY,      c4=-(_E[0] + _E[3]) / _R2, dL=dL,  c1=1j * _E[3] / _R2, dR=dR),
        dict(m4=m4, m1=m1, mY=mY,      c4=(_E[0] - _E[3]) / _R2,  dL=dL,  c1=1j * _E[3] / _R2, dR=dR),
        dict(m4=o4, m1=o1, mY=oYo,     c4=-_R2 * _E[0],           dL=odL, c1=0 * _E[0],         dR=0.0),
        dict(m4=m4, m1=m1, mY=mY + Yk, c4=1j * _E[4],             dL=dL,  c1=-_E[4],            dR=dR),
    ])


class _Assembled555:
    """A drop-in model object (mass_U/mass_D/mass2_W/mass2_Z) built by the generic assembler;
    the gauge sector is representation-independent, so it is reused from mchm5."""
    def mass_U(self, P, sh):
        return assemble(spec_555(P, up=True), sh)

    def mass_D(self, P, sh):
        return assemble(spec_555(P, up=False), sh)

    def __getattr__(self, name):       # mass2_W, mass2_Z, form factors: reuse mchm5
        from . import mchm5
        return getattr(mchm5, name)


model_555 = _Assembled555()

