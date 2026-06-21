"""NM4DCHM6: the Next-to-Minimal 4D Composite Higgs Model with the quark partners in the 6 of
SO(6) (thesis arXiv:2606.18364, Ch.8 -- the NM4DCHM_6).

The coset SO(6)/SO(5) carries five pNGBs: the Higgs doublet (h1..h4) and a real SO(5)-singlet
`s`.  The model is the SO(6) lift of the validated MCHM 5-5-5: indices 0..4 are the unbroken
SO(5), the coset direction is index 5, and the fermion mass matrices are the 5-5-5 assembler
(`pychm.assemble.spec_555`) with the singlet leg moved from the MCHM coset index 4 to the SO(6)
coset index 5, dressed by the SO(6) Goldstone U6(th, ts) (`symbolic.so6`) instead of the SO(5)
U_vector.  Two consequences, both physical and both checked in tests/test_nmchm6.py:

  * **At ts = 0 the model reduces to the MCHM 5-5-5 exactly** (U6(th,0) is the planar (3,5)
    rotation = U_vector(sin th) with 4->5), so EWSB, m_t, m_h and the pypngb anchor are inherited.
  * With the t_R embedding `E_tR = e5` (beta = 0) no composite or elementary state carries an
    e4 component, so every Goldstone overlap is **even in ts** -> the potential is even in s,
    the vacuum sits at <s> = 0, and the singlet's mass is the s-curvature of V there.  Because
    the q_L bidoublet dressing carries explicit Higgs-direction (th) dependence -- distinct from
    the radial Theta = sqrt(th^2 + ts^2) the t_R feels -- V(th, s) is genuinely two-dimensional
    and the singlet acquires a finite, calculable mass from the fermion loop (the SO(5)-singlet
    pNGB).  Turning on `beta` (t_R = cos(beta) e5 + sin(beta) e4) adds e4 support, an odd-in-ts
    overlap and hence an s tadpole: <s> moves off zero (a misaligned vacuum), recovering the
    U(1)_s / electroweak-axion limit the thesis discusses as the embedding is varied.

Scope / honesty (docs/THESIS_VALIDATION.md): the SO(6) representation machinery (generators,
Goldstone, the 6 and its branchings, the dressing overlaps) is *derived* from group theory; the
model *structure* (the 5-5-5 partner content + the t_R embedding angle beta) is input.  There is
no public NMCHM engine to anchor the numbers to, so the NMCHM-specific results (the singlet
sector) are validated by internal consistency and by the exact reduction to the anchored MCHM.

Parameters: the 5-5-5 parameter dict (mU,mUt,mD,mDt,mYu,Yu,mYd,Yd,Delta_uL/uR/dL/dR, f,f1,fX,
g,gp,grho,gX) plus optional `beta` (t_R singlet-mixing angle; default 0) and `ts0` (frozen
singlet vev for the 1-field pipeline; default 0).
"""
import numpy as np

from . import mchm5
from .symbolic import so6

Nc = 3.0
_R2 = np.sqrt(2.0)
_MU, _MC = 0.544573 * 0.00216, 0.480885 * 1.275       # SM u,c [GeV]  (as in mchm5)
_MD, _MS = 0.544573 * 0.00468, 0.544573 * 0.095        # SM d,s [GeV]

_e = np.eye(6, dtype=complex)


def _ov(c, U, E):
    """<c| U6 |E> in the vector 6 (the (h,s)-dressed elementary-composite overlap)."""
    return np.vdot(c, U @ E)


def _E_tR(beta):
    """t_R embedding in the 6: cos(beta) e5 (SO(5)-singlet) + sin(beta) e4 (SO(5)-vector
    singlet).  beta = 0 is the minimal (U(1)_s-symmetric) embedding."""
    return np.cos(beta) * _e[5] + np.sin(beta) * _e[4]


def spec(P, up=True):
    """Declarative NM4DCHM6 spec: the 5-5-5 content lifted to the 6 (coset index 4 -> 5).

    Identical to `assemble.spec_555` with every e4 replaced by e5 and t_R = _E_tR(beta); at
    beta = 0, ts = 0 it is the 5-5-5 model verbatim."""
    beta = P.get('beta', 0.0)
    if up:
        m4, m1, mY, Yk, dL, dR = P['mU'], P['mUt'], P['mYu'], P['Yu'], P['Delta_uL'], P['Delta_uR']
        o4, o1, oYo, odL = P['mD'], P['mDt'], P['mYd'], P['Delta_dL']
        light = [_MU * 1e-3, _MC * 1e-3]
    else:
        m4, m1, mY, Yk, dL, dR = P['mD'], P['mDt'], P['mYd'], P['Yd'], P['Delta_dL'], P['Delta_dR']
        o4, o1, oYo, odL = P['mU'], P['mUt'], P['mYu'], P['Delta_uL']
        light = [_MD * 1e-3, _MS * 1e-3]
    EqL = (_e[0] + _e[3]) / _R2
    EtR = _E_tR(beta)
    return dict(light=light, E_qL=EqL, E_tR=EtR, multiplets=[
        dict(m4=m4, m1=m1, mY=mY,      c4=-(_e[0] + _e[3]) / _R2, dL=dL,  c1=1j * _e[3] / _R2, dR=dR),
        dict(m4=m4, m1=m1, mY=mY,      c4=(_e[0] - _e[3]) / _R2,  dL=dL,  c1=1j * _e[3] / _R2, dR=dR),
        dict(m4=o4, m1=o1, mY=oYo,     c4=-_R2 * _e[0],           dL=odL, c1=0 * _e[0],         dR=0.0),
        dict(m4=m4, m1=m1, mY=mY + Yk, c4=1j * _e[5],             dL=dL,  c1=-_e[5],            dR=dR),
    ])


def assemble(sector, th, ts):
    """Build the (h,s)-dressed fermion mass matrix from `sector`, exactly as
    `pychm.assemble.assemble` but with the SO(6) vector Goldstone U6(th, ts)."""
    light, EqL, EtR, mults = sector['light'], sector['E_qL'], sector['E_tR'], sector['multiplets']
    n = len(light) + 1 + 2 * len(mults)
    M = np.zeros((n, n), dtype=complex)
    for i, m in enumerate(light):
        M[i, i] = m
    el = len(light)
    base = el + 1
    U = so6.U6_vector(th, ts)
    for k, mp in enumerate(mults):
        i4, i1 = base + 2 * k, base + 2 * k + 1
        M[i4, i4] = mp['m4']
        M[i1, i1] = mp['m1']
        M[i4, i1] = mp['mY']
        if np.any(mp['dL']):
            M[el, i4] = mp['dL'] * _ov(mp['c4'], U, EqL)
        if mp.get('dR'):
            M[i1, el] = mp['dR'] * _ov(mp['c1'], U, EtR)
    return M


# ----- two-angle mass matrices (the genuine NMCHM interface) -------------------------------- #
def mass_U2(P, th, ts):
    return assemble(spec(P, up=True), th, ts)


def mass_D2(P, th, ts):
    return assemble(spec(P, up=False), th, ts)


# ----- single-angle interface for the existing pipeline ------------------------------------- #
# With the minimal embedding (beta = 0) the model is s -> -s symmetric, <s> = 0, and the EWSB
# direction is ts = 0: routes/spectrum/tuning (which scan sh = sin(h/f)) then compute xi, f,
# m_t, m_h, Delta_BG exactly -- and identically to the MCHM 5-5-5 (see test_nmchm6).
def mass_U(P, sh):
    return mass_U2(P, np.arcsin(np.clip(sh, -1.0, 1.0)), P.get('ts0', 0.0))


def mass_D(P, sh):
    return mass_D2(P, np.arcsin(np.clip(sh, -1.0, 1.0)), P.get('ts0', 0.0))


def mass2_W(P, sh):
    """Gauge sector: the singlet is a gauge singlet, so W/Z see only the Higgs -> reuse the
    two-site gauge matrices of mchm5 (representation independent)."""
    return mchm5.mass2_W(P, sh)


def mass2_Z(P, sh):
    return mchm5.mass2_Z(P, sh)


# ----- the two-field potential, vacuum and the singlet mass --------------------------------- #
_CF = -1.0 / (16 * np.pi**2)
_CV = 3.0 / (64 * np.pi**2)


def _K(m2):
    m2 = np.maximum(np.atleast_1d(m2), 0.0)
    return np.where(m2 > 1e-30, m2**2 * np.log(np.maximum(m2, 1e-30)), 0.0)


def potential(P, th, ts):
    """Two-field Coleman-Weinberg potential V(th, ts) (eigenvalue route), offset to V(0,0)=0."""
    def Vraw(a, b):
        v = 0.0
        for Mf in (mass_U2(P, a, b), mass_D2(P, a, b)):
            sv = np.linalg.svd(Mf, compute_uv=False)
            v += 3.0 * _CF * np.sum(_K(sv**2))
        sh = np.sin(a)
        v += 2.0 * _CV * np.sum(_K(np.abs(np.linalg.eigvalsh(mchm5.mass2_W(P, sh)))))
        eZ = np.sort(np.abs(np.linalg.eigvalsh(mchm5.mass2_Z(P, sh))))
        v += _CV * np.sum(_K(eZ[1:]))
        return v
    return Vraw(th, ts) - Vraw(1e-9, 1e-9)


def vacuum2(P, th_hi=0.7, ts_hi=0.7, n=41):
    """The two-field vacuum (<th>, <ts>) = argmin V(th, ts) on a grid + local refinement, or
    None if the minimum is at the origin (no EWSB)."""
    from scipy.optimize import minimize
    ths = np.linspace(0.0, th_hi, n)
    tss = np.linspace(-ts_hi, ts_hi, n)
    best, bv = (0.0, 0.0), 0.0
    for a in ths:
        for b in tss:
            v = potential(P, a, b)
            if v < bv:
                bv, best = v, (a, b)
    if best == (0.0, 0.0):
        return None
    res = minimize(lambda x: potential(P, abs(x[0]), x[1]), x0=best, method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-12})
    return (abs(res.x[0]), res.x[1]) if res.fun < -1e-12 else None


def singlet_mass2(P, th_vac, ts_vac=0.0, f=1.0, h=1e-3):
    """m_s^2 = (1/f^2) d^2V/dts^2 at the vacuum (the SO(5)-singlet pNGB mass; s = f*ts).  f in
    TeV.  Zero for the minimal embedding (beta = 0, U(1)_s axion); positive for beta != 0."""
    V = lambda b: potential(P, th_vac, b)
    Vpp = (V(ts_vac + h) - 2 * V(ts_vac) + V(ts_vac - h)) / h**2
    return Vpp / f**2
