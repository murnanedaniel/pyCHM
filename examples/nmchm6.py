"""NM4DCHM6 example: the Next-to-Minimal Composite Higgs (SO(6)/SO(5), partners in the 6).

Beyond the MCHM, the coset carries a second pNGB -- a real SO(5) singlet `s`.  At the
electroweak vacuum the model reproduces the MCHM 5-5-5 Higgs sector (it reduces to it exactly
at <s> = 0, inheriting the pypngb-anchored xi/m_t/m_h/Delta_BG) and, in addition, gives the
SO(5) singlet a finite calculable mass from the fermion loop.

It also shows the SO(6) representation tower (6, 15, 20', the self-dual 10, the spinor 4) and
their SO(5) branchings -- the group theory the model is built from.
"""
import numpy as np
import pychm
from pychm import nmchm6 as N
from pychm.symbolic import so6, so6_spinors

# the validated 5-5-5 benchmark, reused (NMCHM6 shares the Lagrangian-mass basis at <s>=0)
point = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)

m = pychm.Model('6-6-6')
s = m.spectrum(point)
t = m.tuning(point)
print("NM4DCHM6 spectrum:", {k: round(v, 4) for k, v in s.items() if k != 'J'})
print("NM4DCHM6 tuning:  Delta_BG=%.1f  HOT=%.1f  I=%.2f nats" % (t['BG'], t['HOT'], t['I']))

# the new NMCHM observable: the SO(5)-singlet pNGB mass (the s-curvature of V at the vacuum)
thv = np.arcsin(np.sqrt(s['xi']))
ms = np.sqrt(N.singlet_mass2(point, thv, 0.0, f=s['f']))
print("singlet pNGB mass m_s = %.0f GeV  (massless in the MCHM -- a genuine NMCHM state)"
      % (ms * 1000))

# the SO(6) representation tower and the SO(6) -> SO(5) branchings
print("\nSO(6) irreps and their SO(5) content (NMCHM partner candidates):")
for rep in ('6', '15', "20'"):
    print("  %-4s -> %s" % (rep, ' + '.join(n for n, _ in so6.so5_content(so6.rep_basis(rep)))))
print("  10   -> 10            (self-dual 3-form; SU(4) 10)")
print("  4    -> %s            (Weyl spinor; no bidoublet -> cannot host q_L)"
      % so6_spinors.so5_content_4()[0][0])
