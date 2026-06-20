"""The rep-generic Higgs dressing, and a model assembled from it.  Convention: sh = sin(h/f).
Run: python examples/generic_ccwz.py
"""
import numpy as np, pychm, pychm.ccwz as ccwz

sh = 0.48
ch = np.sqrt(1 - sh**2)
print("Higgs-dressed mixing factors at sh=sin(h/f)=%.2f, from group theory:\n" % sh)
U5 = ccwz.U_vector(sh)
print("  rep 5 : <4|U|4>=%.4f (=cos h/f=%.4f) 'ch';  <4|U|5>=%.4f (=sin h/f) 'sh'"
      % (U5[3, 3], ch, U5[3, 4]))
print("          c2h2=cos^2(h/2f)=%.4f ; s2h2=%.4f" % ((1 + ch) / 2, (1 - ch) / 2))
ES = ccwz.embedding('14', 'singlet')
print("  rep 14: <(1,1)|U14|(1,1)>=%.6f  (=(3+5cos2h/f)/8=%.6f)  exact mchm14 m[4,2]"
      % (ccwz.overlap('14', ES, ES, sh).real, (3 + 5 * (1 - 2 * sh**2)) / 8))

print("\nA model assembled purely from embeddings + ccwz reproduces the validated 5-5-5:")
REF = dict(mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391, mYu=0.00754655, Yu=7.95104,
           mYd=0.03112, Yd=0.60624, Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056,
           Delta_dR=0.106068, f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138,
           grho=5.44665, gX=2.96355)
for tag in ('5-5-5', '5-5-5-assembled'):
    s = pychm.Model(tag).spectrum(REF)
    print("  %-16s xi=%.6f  m_t=%.5f  m_h=%.5f TeV" % (tag, s['xi'], s['mt'], s['mh']))
