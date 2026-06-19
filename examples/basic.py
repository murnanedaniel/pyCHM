"""Minimal pyCHM example: potential, spectrum, and the four fine-tuning measures."""
import pychm

m = pychm.Model('5-5-5')
point = dict(mU=2.0, mUt=1.2, mD=1.8, mDt=1.0, mYu=0.9, Yu=1.2, mYd=0.7, Yd=0.4,
             Lq=1.1, Lt=1.3, Lb=0.3, f=0.9, f1=1.5, fX=3.0,
             g=0.671, gp=0.358, grho=5.0, gX=3.0)

s = m.spectrum(point)
print("spectrum:", {k: round(v, 4) for k, v in s.items()} if s else "no EWSB")
t = m.tuning(point)
if t:
    print("tuning:  Delta_BG=%.1f  HOT=%.1f  I=%.2f nats  KL=%.2f nats"
          % (t['BG'], t['HOT'], t['I'], t['KL']))
