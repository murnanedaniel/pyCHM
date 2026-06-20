"""MCHM 14-1-10: the two-site Minimal 4D Composite Higgs Model with the q_L partner in the
symmetric (14) of SO(5), the t_R partner an SO(4) *singlet* (the "1"), and the b_R partner
in the (10).

This is the simpler cousin of 14-14-10: because the right-handed top partner is an SO(4)
singlet, the up-type sector has a single composite singlet mass mU and a single Yukawa Yu
(no mYu / Ytu / mSYu / mSYtu Y-sector).  The matrices are 14x14 (up) and 9x9 (down).

Provides the s_h-dependent mass matrices needed by the eigenvalue route:
  * up-type fermion matrix    mass_U(P, sh)   (14x14; lightest 3 eigenvalues = u,c,t)
  * down-type fermion matrix  mass_D(P, sh)   (9x9;   lightest 3 eigenvalues = d,s,b)
  * gauge mass-squared        mass2_W, mass2_Z  (identical to the 5-5-5 / 14-14-10 two-site)

This is a transcription of the validated independent engine (pypngb M4DCHM_14_1_10),
mirroring mchm14.py's style and (P, sh) signature.  Masses in TeV.

Parameters (dict, TeV / dimensionless):
  mQ          : q_L 4-plet partner mass (single SO(5) symmetric -> one mQ)
  mU          : t_R singlet partner mass
  mD          : b_R / down (10) partner mass
  Yu, Yd      : up / down Yukawa shifts entering the composite mass matrix
  Delta_q, Delta_u, Delta_d : elementary-composite mixings (q_L, t_R, b_R)
  f, f1, fX   : decay constants;  g, gp, grho, gX : couplings (physical)
"""
import numpy as np
import math as _math

Nc = 3.0

# SM light-quark MSbar masses [GeV] (same anchors pypngb / mchm5 use)
_MU, _MC = 0.544573*0.00216, 0.480885*1.275      # u, c
_MD, _MS = 0.544573*0.00468, 0.544573*0.095      # d, s


# --- trigonometric helpers of s_h (cos(h/f), sin^2(h/2f), ...) --------------- #
def _ch(sh):        return np.sqrt(1 - sh**2)
def _s2h2(sh):      return 0.5*(1 - _ch(sh))
def _c2h2(sh):      return 0.5*(1 + _ch(sh))
def _sindouble(sh): return 2*sh*_ch(sh)            # sin(2h/f)
def _cosdouble(sh): return _ch(sh)**2 - sh**2      # cos(2h/f)


# --- gauge sector (identical structure to the two-site 5-5-5 / 14-14-10 model) -- #
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


# --- fermion mass matrices (eigenvalue route; transcribed from the independent
#     two-site M4DCHM_14_1_10 engine).  s_h-dependent up/down matrices. ------------ #
def mass_U(P, sh):
    """Up-type 14x14 mass matrix (lightest 3 eigenvalues = u,c,t). Masses in TeV."""
    mQ, mU, mD = P['mQ'], P['mU'], P['mD']
    Yu, Yd = P['Yu'], P['Yd']
    Delta_q = P['Delta_q']
    Delta_u = P['Delta_u']
    Delta_u_dag = np.conjugate(Delta_u)
    r5 = _math.sqrt(5)

    m = np.zeros((14, 14), dtype=complex)
    m[0, 0] = _MU*1e-3
    m[1, 1] = _MC*1e-3
    m[2, 3]  = r5*_sindouble(sh)*Delta_q/4.0
    m[2, 6]  = -(_ch(sh) + _cosdouble(sh))*Delta_q/2.0
    m[2, 8]  = -1j*(_ch(sh) - _cosdouble(sh))*Delta_q/2.0
    m[2, 9]  = (2*sh - _sindouble(sh))*Delta_q/4.0
    m[2, 10] = -_sindouble(sh)*Delta_q/4.0
    m[2, 11] = (2*sh + _sindouble(sh))*Delta_q/4.0
    m[4, 2]  = -Delta_u_dag
    m[3, 3]   = mQ
    m[3, 4]   = 2*Yu/r5
    m[4, 4]   = mU
    m[5, 5]   = mD
    m[6, 5]   = Yd/2.0
    m[6, 6]   = mQ
    m[7, 7]   = mD
    m[8, 7]   = Yd/2.0
    m[8, 8]   = mQ
    m[9, 9]   = mQ
    m[10, 10] = mQ
    m[11, 11] = mQ
    m[12, 12] = mD
    m[13, 13] = mD
    return m


def mass_D(P, sh):
    """Down-type 9x9 mass matrix (lightest 3 eigenvalues = d,s,b). Masses in TeV."""
    mQ, mD = P['mQ'], P['mD']
    Yd = P['Yd']
    Delta_q = P['Delta_q']
    Delta_d = P['Delta_d']
    Delta_d_dag = np.conjugate(Delta_d)
    r2 = _math.sqrt(2)

    m = np.zeros((9, 9), dtype=complex)
    m[0, 0] = _MD*1e-3
    m[1, 1] = _MS*1e-3
    m[2, 4]  = -_ch(sh)*Delta_q
    m[2, 5]  = -sh*Delta_q/r2
    m[2, 6]  = 1j*sh*Delta_q/r2
    m[3, 2]  = -1j*sh*Delta_d_dag/r2
    m[7, 2]  = -(1 - _ch(sh))*Delta_d_dag/2.0
    m[8, 2]  = -(1 + _ch(sh))*Delta_d_dag/2.0
    m[3, 3]   = mD
    m[4, 3]   = Yd/2.0
    m[4, 4]   = mQ
    m[5, 5]   = mQ
    m[6, 6]   = mQ
    m[7, 7]   = mD
    m[8, 8]   = mD
    return m
