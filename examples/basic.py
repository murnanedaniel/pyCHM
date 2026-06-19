"""Minimal pyCHM example: potential, spectrum, and the four fine-tuning measures.

The point below is the validated benchmark (xi ~ 0.070, m_t ~ 0.163 TeV, Delta_BG ~ 127).
Masses are in TeV; the Y sector is stored as (mY, Y) with the singlet-Y mass mSY = mY + Y.
"""
import pychm

m = pychm.Model('5-5-5')
point = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)

s = m.spectrum(point)
print("spectrum:", {k: round(v, 4) for k, v in s.items()} if s else "no EWSB")
t = m.tuning(point)
if t:
    print("tuning:  Delta_BG=%.1f  HOT=%.1f  I=%.2f nats  KL=%.2f nats"
          % (t['BG'], t['HOT'], t['I'], t['KL']))
