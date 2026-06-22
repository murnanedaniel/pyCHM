"""SU(5)/SO(5) example: the littlest-Higgs / Ferretti real-rep composite Higgs.

The coset carries 14 pNGBs -- a Higgs doublet, a complex triplet, and a singlet -- so beyond EWSB,
m_t and m_h the model predicts the **triplet and singlet pNGB masses** (curvatures of the
Coleman-Weinberg potential).  Two fermion variants: partners in the fundamental 5, and in the
symmetric 15.  Also shows the coset-landscape enumerator.
"""
import numpy as np
import pychm
from pychm import su5so5 as M
from pychm.groups import su5so5 as G, landscape

point = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)

# the 14 pNGBs under custodial SO(4): triplet (3,3) + Higgs (2,2) + singlet (1,1)
print("SU(5)/SO(5) coset: 14 pNGBs ->", G.coset_so4_content(),
      "= triplet (1,1) + Higgs (1/2,1/2) + singlet (0,0)")

m = pychm.Model('5-5-su5')
s = m.spectrum(point)
print("\n5-5-su5 spectrum:", {k: round(v, 4) for k, v in s.items() if k != 'J'})
print("5-5-su5 tuning: Delta_BG = %.1f" % m.tuning(point)['BG'])

thv = np.arcsin(np.sqrt(s['xi']))
m_eta = np.sqrt(max(M.singlet_mass2(point, thv, f=s['f']), 0)) * 1000
m_phi = np.sqrt(max(M.triplet_mass2(point, thv, f=s['f']), 0)) * 1000
print("the NEW pNGB observables:  m_singlet(eta) = %.0f GeV,  m_triplet(phi) = %.0f GeV" % (m_eta, m_phi))

# the coset landscape: pyCHM enumerates the viable cosets (those with a (2,2) Higgs)
print("\nCoset landscape (cosets with a custodial (2,2) Higgs):")
landscape.print_scan(max_ngb=14)
