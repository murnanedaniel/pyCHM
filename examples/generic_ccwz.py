"""The rep-generic Higgs dressing: the same machinery yields each representation's factors.

Run: python examples/generic_ccwz.py
"""
import numpy as np
import pychm.ccwz as ccwz

th = 0.5  # h/f
print("Higgs-dressed mixing factors at h/f = %.2f, from group theory:\n" % th)

print("  rep 5  (vector):")
U5 = ccwz.U_vector(th)
print("    <4|U|4> = %.4f  (= cos h/f = %.4f)   hand-coded 'ch'"   % (U5[3, 3], np.cos(th)))
print("    <4|U|5> = %.4f  (= sin h/f = %.4f)   hand-coded 'sh'"   % (U5[3, 4], np.sin(th)))
print("    cos^2(h/2f) = %.4f   hand-coded 'c2h2';  sin^2 = %.4f  's2h2'"
      % ((1 + np.cos(th)) / 2, (1 - np.cos(th)) / 2))

print("\n  rep 14 (symmetric traceless):")
ES = ccwz.embedding('14', 'singlet')
print("    <(1,1)|U14|(1,1)> = %.6f   (= (3+5cos 2h/f)/8 = %.6f)   hand-coded mchm14 m[4,2]"
      % (ccwz.overlap('14', ES, ES, th), (3 + 5 * np.cos(2 * th)) / 8))

print("\n  The 5's factors and the 14's (3+5cos2h)/8 -- different reps, one construction.")
print("  Choosing the partner representation is now a parameter, not a re-derivation.")
