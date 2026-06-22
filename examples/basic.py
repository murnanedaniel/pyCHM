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


# The 14-14-10 representation (q_L, t_R in the symmetric 14 of SO(5); b_R in the 10)
# uses the same eigenvalue-route pipeline.  Its Y sector carries an extra top-partner
# Yukawa shift Ytu, and the mixings are the SO(5) (Delta_q, Delta_u, Delta_d).
m14 = pychm.Model('14-14-10')
point14 = dict(
    mQ=3.965100934, mU=2.397190093, mD=1.482462237,
    mYu=0.1478991938, Yu=0.4989142322, Ytu=2.538912347, Yd=0.5293788277,
    Delta_q=2.850102384, Delta_u=1.915943759, Delta_d=0.2222079908,
    f=1.4396717743254574, f1=1.8934618718580186, fX=2.2076564878795057,
    g=0.6709494105248374, gp=0.3581380846874656, grho=5.525258191589697, gX=4.696070132753916)

s14 = m14.spectrum(point14)
print("\n14-14-10 spectrum:", {k: round(v, 4) for k, v in s14.items()} if s14 else "no EWSB")
t14 = m14.tuning(point14)
if t14:
    print("14-14-10 tuning:  Delta_BG=%.1f  HOT=%.1f  I=%.2f nats"
          % (t14['BG'], t14['HOT'], t14['I']))


# The 14-1-10 representation (q_L in the symmetric 14 of SO(5); t_R an SO(4) singlet;
# b_R in the 10).  With a singlet t_R partner the up Y-sector collapses to a single
# (mU, Yu) -- no mYu/Ytu -- so the fundamental basis is {mQ, mU, mD, Yu, Yd, Delta_*}.
m1410 = pychm.Model('14-1-10')
point1410 = dict(
    mQ=1.5774575432442343, mU=0.053295037861913734, mD=3.9639579843374463,
    Yu=3.213522953043455, Yd=1.5033041681268544,
    Delta_q=0.8229454568382461, Delta_u=3.843401100178204, Delta_d=0.19968869968780683,
    f=1.1329813572955627, f1=1.5049448725076554, fX=1.4519742141888066,
    g=0.6709494105248374, gp=0.3581380846874656, grho=3.3254933316652062, gX=9.7650801141742978)

s1410 = m1410.spectrum(point1410)
print("\n14-1-10 spectrum:", {k: round(v, 4) for k, v in s1410.items()} if s1410 else "no EWSB")
t1410 = m1410.tuning(point1410)
if t1410:
    print("14-1-10 tuning:  Delta_BG=%.1f  HOT=%.1f  I=%.2f nats"
          % (t1410['BG'], t1410['HOT'], t1410['I']))
