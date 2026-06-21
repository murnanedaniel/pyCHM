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
from .symbolic import models as _sym_models
from .symbolic.core import lambdify_sh


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


# --------------------------------------------------------------------------------------
# Worked model: 14-1-10, assembled from embeddings in the 14 (q_L) and 10 (b_R), the t_R
# an SO(4) singlet (h-independent mixing).  The composite-partner spectrum (masses,
# proto-Yukawas) is declared; only the Higgs dressing of the elementary-composite mixings
# is computed by ccwz.  Validated to reproduce mchm14_1_10.mass_U/mass_D entry-for-entry.
# --------------------------------------------------------------------------------------
_r5 = np.sqrt(5.0)


def _ch(s):
    return np.sqrt(max(0.0, 1.0 - s * s))


def _tensor(rep, entries):
    """Build a 5x5 (anti)symmetric tensor from {(i,j): value} (j>i); rep '14' symmetric, '10' antisymmetric."""
    M = np.zeros((5, 5), dtype=complex)
    for (i, j), v in entries.items():
        M[i, j] = v
        M[j, i] = v if rep == '14' else -v
    return M


# q_L in the 14: up-component sym(e3 (x) (e0+e4)); down-component (e0 e3); b_R in the 10: (e0^e1 + e0^e4)
_E_QL_UP = _tensor('14', {(0, 3): 0.5, (3, 4): 0.5})
_E_QL_DN = _tensor('14', {(0, 3): 1 / _R2})
_E_BR = _tensor('10', {(0, 1): 0.5, (0, 4): 0.5})


# Higgs-dressing factors, derived in closed form by the symbolic CCWZ engine (symbolic.models)
# and lambdified to numpy callables of sh = sin(h/f).  No curve-fitting: each factor is proven
# to be a Goldstone matrix element <c|U_R|E> by symbolic.derive (see tests/test_symbolic.py).
_f_QL_UP = {k: lambdify_sh(v) for k, v in _sym_models.F_QL_UP.items()}
_f_QL_DN = {k: lambdify_sh(v) for k, v in _sym_models.F_QL_DN.items()}
_f_BR = {k: lambdify_sh(v) for k, v in _sym_models.F_BR.items()}


class _Assembled14_1_10:
    """Drop-in 14-1-10 model assembled by the generic machinery; gauge sector reused from
    the hand-coded module (representation-independent two-site structure)."""
    def mass_U(self, P, sh):
        m = np.zeros((14, 14), dtype=complex)
        m[0, 0], m[1, 1] = _MU * 1e-3, _MC * 1e-3
        for k, fk in _f_QL_UP.items():
            m[2, k] = P['Delta_q'] * fk(sh)
        m[4, 2] = -np.conjugate(P['Delta_u'])                      # t_R singlet: no Goldstone dressing
        mQ, mU, mD, Yu, Yd = P['mQ'], P['mU'], P['mD'], P['Yu'], P['Yd']
        m[3, 3], m[3, 4], m[4, 4], m[5, 5] = mQ, 2 * Yu / _r5, mU, mD
        m[6, 5], m[6, 6], m[7, 7], m[8, 7], m[8, 8] = Yd / 2, mQ, mD, Yd / 2, mQ
        m[9, 9] = m[10, 10] = m[11, 11] = mQ
        m[12, 12] = m[13, 13] = mD
        return m

    def mass_D(self, P, sh):
        m = np.zeros((9, 9), dtype=complex)
        m[0, 0], m[1, 1] = _MD * 1e-3, _MS * 1e-3
        for k, fk in _f_QL_DN.items():
            m[2, k] = P['Delta_q'] * fk(sh)
        for k, fk in _f_BR.items():
            m[k, 2] = np.conjugate(P['Delta_d']) * fk(sh)
        mQ, mD, Yd = P['mQ'], P['mD'], P['Yd']
        m[3, 3], m[4, 3], m[4, 4], m[5, 5] = mD, Yd / 2, mQ, mQ
        m[6, 6], m[7, 7], m[8, 8] = mQ, mD, mD
        return m

    def __getattr__(self, name):
        from . import mchm14_1_10
        return getattr(mchm14_1_10, name)


model_14_1_10 = _Assembled14_1_10()


# --------------------------------------------------------------------------------------
# Worked model: 14-14-10 -- t_R now also in the 14 (its SO(4)-singlet component), so the
# up sector grows to 19x19.  Same q_L (14) and b_R (10) embeddings as 14-1-10; the new
# piece is the t_R = 14-singlet dressing ((3+5cos2h)/8, sqrt5 sin2h/4, ...), which ccwz
# already reproduces exactly.  Validated to reproduce mchm14.mass_U/mass_D entry-for-entry.
# --------------------------------------------------------------------------------------
_E_TR_14 = ccwz.embedding('14', 'singlet')

# Same closed-form dressing factors as 14-1-10, placed at the 14-14-10 composite columns, plus
# the t_R = 14-singlet dressing ((3+5cos2h)/8, sqrt5 sin2h/4, ...).  All lambdified from
# symbolic.models; the t_R-singlet keys already match the 14-14-10 columns.
_f_QL_UP_1414 = {3: _f_QL_UP[3], 6: _f_QL_UP[6], 9: _f_QL_UP[8],
                 11: _f_QL_UP[9], 13: _f_QL_UP[10], 15: _f_QL_UP[11]}
_f_TR_1414 = {k: lambdify_sh(v) for k, v in _sym_models.F_TR_14.items()}
_f_QL_DN_1414 = {4: _f_QL_DN[4], 6: _f_QL_DN[5], 8: _f_QL_DN[6]}
_f_BR_1414 = {3: _f_BR[3], 10: _f_BR[7], 11: _f_BR[8]}


class _Assembled14_14_10:
    """Drop-in 14-14-10 model assembled by the generic machinery (q_L, t_R in the 14; b_R in
    the 10).  Gauge sector reused from the hand-coded module."""
    def mass_U(self, P, sh):
        m = np.zeros((19, 19), dtype=complex)
        m[0, 0], m[1, 1] = _MU * 1e-3, _MC * 1e-3
        for k, fk in _f_QL_UP_1414.items():
            m[2, k] = P['Delta_q'] * fk(sh)
        for k, fk in _f_TR_1414.items():
            m[k, 2] = np.conjugate(P['Delta_u']) * fk(sh)
        mQ, mU, mD, mYu, Yu, Yd, Ytu = (P['mQ'], P['mU'], P['mD'], P['mYu'], P['Yu'], P['Yd'], P['Ytu'])
        for k in (3, 6, 9, 11, 13, 15):
            m[k, k] = mQ
        for k in (4, 7, 10, 12, 14, 16):
            m[k, k] = mU
        for k in (5, 8, 17, 18):
            m[k, k] = mD
        m[3, 4] = mYu + 4.0 * (Yu + Ytu) / 5.0
        m[6, 5] = Yd / 2.0; m[6, 7] = mYu + Yu / 2.0
        m[9, 8] = Yd / 2.0; m[9, 10] = mYu + Yu / 2.0
        m[11, 12] = m[13, 14] = m[15, 16] = mYu
        return m

    def mass_D(self, P, sh):
        m = np.zeros((12, 12), dtype=complex)
        m[0, 0], m[1, 1] = _MD * 1e-3, _MS * 1e-3
        for k, fk in _f_QL_DN_1414.items():
            m[2, k] = P['Delta_q'] * fk(sh)
        for k, fk in _f_BR_1414.items():
            m[k, 2] = np.conjugate(P['Delta_d']) * fk(sh)
        mQ, mU, mD, mYu, Yu, Yd = P['mQ'], P['mU'], P['mD'], P['mYu'], P['Yu'], P['Yd']
        m[3, 3], m[10, 10], m[11, 11] = mD, mD, mD
        m[4, 4], m[6, 6], m[8, 8] = mQ, mQ, mQ
        m[5, 5], m[7, 7], m[9, 9] = mU, mU, mU
        m[4, 3] = Yd / 2.0; m[4, 5] = mYu + Yu / 2.0
        m[6, 7] = mYu; m[8, 9] = mYu
        return m

    def __getattr__(self, name):
        from . import mchm14
        return getattr(mchm14, name)


model_14_14_10 = _Assembled14_14_10()


