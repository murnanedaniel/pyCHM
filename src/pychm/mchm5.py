"""MCHM 5-5-5: the two-site Minimal 4D Composite Higgs Model with all partners in the
fundamental (5) of SO(5).

Provides everything both computational routes need:
  * fermion form factors  Pi_L, Pi_R, |M|^2  (App. A7 of arXiv:2606.18364)  -> form-factor route
  * gauge mass-squared matrices  mass2_W, mass2_Z  (functions of s_h)        -> eigenvalue route

Units: masses in TeV.  Euclidean momentum p_E (Wick rotation p^2 -> -p_E^2).
Parameters (dict, TeV / dimensionless):
  mU, mUt   : up-type 4-plet and singlet partner masses
  mD, mDt   : down-type 4-plet and singlet partner masses
  mYu, Yu, mYd, Yd : Y-partner masses and Yukawa shifts (m_SY = m_Y + Y)
  Lq, Lt, Lb       : q_L, t_R, b_R elementary-composite mixings
  f, f1, fX        : decay constants;  g, gp, grho, gX : couplings (physical)
"""
import numpy as np

Nc = 3.0


# --- form-factor building blocks (App. A7), p2 = -pE2 ------------------------ #
def _AL(m1, m2, m3, m4, Lam, p2):
    return Lam**2 * (m1**2*m2**2 + m1**2*m4**2 + m2**2*m3**2
                     - p2*(m1**2+m2**2+m3**2+m4**2) + p2**2)

def _AR(m1, m2, m3, m4, Lam, p2):
    return Lam**2 * (m1**2*m2**2 + m2**2*m3**2
                     - p2*(m1**2+m2**2+m3**2+m4**2) + p2**2)

def _AM(m1, m2, m3, m4, L1, L2, p2):
    return L1*L2*m1*m2*m4*(m3**2 - p2)

def _B(m1, m2, m3, m4, m5, p2):
    return (m1**2*m2**2*m3**2
            - p2*(m1**2*m2**2 + m1**2*m3**2 + m2**2*m3**2 + m2**2*m5**2 + m3**2*m4**2)
            + p2**2*(m1**2+m2**2+m3**2+m4**2+m5**2) - p2**3)


def formfactor_pieces(P, pE2, up=True):
    """s_h-independent pieces: Pi_L = L0 + s*Ls, Pi_R = R0 + s*Rs, |M|^2 = s(1-s)*M2c,
    with s = s_h^2.  Separate up/down 4-plets (the q_L self-energy mixes with both)."""
    p2 = -pE2
    mU, mUt, mD, mDt = P['mU'], P['mUt'], P['mD'], P['mDt']
    if up:
        mQ, mS, mY, Yk, LR = mU, mUt, P['mYu'], P['Yu'], P['Lt']
        mQo, mSo, mYo = mD, mDt, P['mYd']
    else:
        mQ, mS, mY, Yk, LR = mD, mDt, P['mYd'], P['Yd'], P['Lb']
        mQo, mSo, mYo = mU, mUt, P['mYu']
    Lq = P['Lq']
    hq = lambda m3: _AL(mS, 0., m3, 0., Lq, p2) / _B(mQ, mS, 0., m3, 0., p2)
    hqo = lambda m3: _AL(mSo, 0., m3, 0., Lq, p2) / _B(mQo, mSo, 0., m3, 0., p2)
    hR = lambda m3: _AR(mQ, 0., m3, 0., LR, p2) / _B(mQ, mS, 0., m3, 0., p2)
    hM = lambda m3: _AM(mQ, mS, 0., m3, Lq, LR, p2) / _B(mQ, mS, 0., m3, 0., p2)
    L0 = 1. + hq(mY) + hqo(mYo)
    Ls = 0.5 * (hq(mY + Yk) - hq(mY))
    R0 = 1. + hR(mY + Yk)
    Rs = -(hR(mY + Yk) - hR(mY))
    M2c = 0.5 * (hM(mY + Yk) - hM(mY))**2
    return L0, Ls, R0, Rs, M2c


def fermion_mass(P, sh2, up=True):
    """SM fermion pole mass m = |M|/sqrt(Pi_L Pi_R) at p=0 (eq. 518)."""
    L0, Ls, R0, Rs, M2c = formfactor_pieces(P, np.array([1e-12]), up=up)
    L = L0 + sh2*Ls; R = R0 + sh2*Rs; M2 = sh2*(1-sh2)*M2c
    return float(np.sqrt(M2[0] / (L[0]*R[0])))


# --- gauge sector (mass-squared matrices; eigenvalue route) ------------------ #
def _ch(sh):    return np.sqrt(1 - sh**2)
def _s2h2(sh):  return 0.5*(1 - _ch(sh))
def _c2h2(sh):  return 0.5*(1 + _ch(sh))

def bare_couplings(P):
    """elementary g10 (U(1)), g20 (SU(2)) from the physical couplings."""
    g, gp, grho, gX = P['g'], P['gp'], P['grho'], P['gX']
    g20 = g*grho/np.sqrt(grho**2 - g**2)
    g10 = 1./np.sqrt(1./gp**2 - 1./grho**2 - 1./gX**2)
    return g10, g20

def mass2_W(P, sh):
    g10, g20 = bare_couplings(P)
    grho, f, f1 = P['grho'], P['f'], P['f1']
    r2 = np.sqrt(2)
    return np.array([
      [(f1**2*g20**2)/2., -(f1**2*g20*grho*_c2h2(sh))/2., -(f1**2*g20*grho*_s2h2(sh))/2., -(f1**2*g20*grho*sh)/(2*r2)],
      [-(f1**2*g20*grho*_c2h2(sh))/2., (f1**2*grho**2)/2., 0, 0],
      [-(f1**2*g20*grho*_s2h2(sh))/2., 0, (f1**2*grho**2)/2., 0],
      [-(f1**2*g20*grho*sh)/(2*r2), 0, 0, (f1**4*grho**2)/(2*(-f**2+f1**2))]])

def mass2_Z(P, sh):
    g10, g20 = bare_couplings(P)
    grho, gX, f, f1, fX = P['grho'], P['gX'], P['f'], P['f1'], P['fX']
    r2 = np.sqrt(2)
    return np.array([
      [(f1**2*g20**2)/2., 0, -(f1**2*g20*grho*(1+_ch(sh)))/4., -(f1**2*g20*grho*_s2h2(sh))/2., -(f1**2*g20*grho*sh)/(2*r2), 0, 0],
      [0, ((f1**2+fX**2)*g10**2)/2., -(f1**2*g10*grho*_s2h2(sh))/2., -(f1**2*g10*grho*(1+_ch(sh)))/4., (f1**2*g10*grho*sh)/(2*r2), -(fX**2*g10*gX)/2., 0],
      [-(f1**2*g20*grho*(1+_ch(sh)))/4., -(f1**2*g10*grho*_s2h2(sh))/2., (f1**2*grho**2)/2., 0, 0, 0, 0],
      [-(f1**2*g20*grho*_s2h2(sh))/2., -(f1**2*g10*grho*(1+_ch(sh)))/4., 0, (f1**2*grho**2)/2., 0, 0, 0],
      [-(f1**2*g20*grho*sh)/(2*r2), (f1**2*g10*grho*sh)/(2*r2), 0, 0, (f1**4*grho**2)/(2*(-f**2+f1**2)), 0, 0],
      [0, -(fX**2*g10*gX)/2., 0, 0, 0, (fX**2*gX**2)/2., 0],
      [0, 0, 0, 0, 0, 0, (f1**4*grho**2)/(2*(-f**2+f1**2))]])


# --- fermion mass matrices (eigenvalue route; ported from arXiv:2606.18364 /
#     the two-site M4DCHM_3G).  s_h-dependent up/down 11x11 matrices; the exotic-charge
#     and lepton blocks are s_h-independent and drop out of gamma, beta, xi. -------------- #
import math as _math
_MU, _MC = 0.544573*0.00216, 0.480885*1.275      # SM u,c MSbar masses [GeV]
_MD, _MS = 0.544573*0.00468, 0.544573*0.095      # SM d,s [GeV]


def mass_U(P, sh):
    """Up-type 11x11 mass matrix (lightest 3 eigenvalues = u,c,t). Masses in TeV."""
    mU, mUt, mYu = P['mU'], P['mUt'], P['mYu']
    mD, mDt, mYd = P['mD'], P['mDt'], P['mYd']
    Yu = P['Yu']
    duL, duR, ddL = P['Delta_uL'], P['Delta_uR'], P['Delta_dL']
    r2 = _math.sqrt(2)
    m = np.zeros((11, 11), dtype=complex)
    m[0, 0] = _MU*1e-3; m[1, 1] = _MC*1e-3
    m[2, 3] = -(duL*_c2h2(sh)); m[2, 5] = duL*_s2h2(sh); m[2, 7] = -ddL
    m[2, 9] = (1j*sh*duL)/r2
    m[4, 2] = (-1j*sh*duR)/r2; m[6, 2] = (-1j*sh*duR)/r2; m[10, 2] = -(_ch(sh)*duR)
    m[3, 3] = mU; m[4, 4] = mUt; m[5, 5] = mU; m[6, 6] = mUt
    m[7, 7] = mD; m[8, 8] = mDt; m[9, 9] = mU; m[10, 10] = mUt
    m[3, 4] = mYu; m[5, 6] = mYu; m[7, 8] = mYd; m[9, 10] = mYu + Yu
    return m


def mass_D(P, sh):
    """Down-type 11x11 mass matrix (lightest 3 = d,s,b). Masses in TeV."""
    mD, mDt, mYd = P['mD'], P['mDt'], P['mYd']
    mU, mUt, mYu = P['mU'], P['mUt'], P['mYu']
    Yd = P['Yd']
    ddL, ddR, duL = P['Delta_dL'], P['Delta_dR'], P['Delta_uL']
    r2 = _math.sqrt(2)
    m = np.zeros((11, 11), dtype=complex)
    m[0, 0] = _MD*1e-3; m[1, 1] = _MS*1e-3
    m[2, 3] = -(ddL*_c2h2(sh)); m[2, 5] = ddL*_s2h2(sh); m[2, 7] = -duL
    m[2, 9] = (1j*sh*ddL)/r2
    m[4, 2] = (-1j*sh*ddR)/r2; m[6, 2] = (-1j*sh*ddR)/r2; m[10, 2] = -(_ch(sh)*ddR)
    m[3, 3] = mD; m[4, 4] = mDt; m[5, 5] = mD; m[6, 6] = mDt
    m[7, 7] = mU; m[8, 8] = mUt; m[9, 9] = mD; m[10, 10] = mDt
    m[3, 4] = mYd; m[5, 6] = mYd; m[7, 8] = mYu; m[9, 10] = mYd + Yd
    return m
