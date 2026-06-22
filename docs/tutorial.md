# Tutorial

A copy-pasteable walk from install to a viability verdict. Every block runs against the validated
benchmark — no setup beyond the install.

## 1. Install

```bash
pip install -e ".[test]"
```

```python
import numpy as np
import pychm
```

## 2. A model and a point

A `Model` is chosen by its fermion-representation string; the parameter `point` is a dict of
Lagrangian masses and mixings (TeV).

```python
m = pychm.Model('5-5-5')          # the two-site MCHM, SO(5)/SO(4)

point = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)
```

## 3. The spectrum

`spectrum()` solves electroweak symmetry breaking (returns `None` if the point does **not** break
EW), rescales \(f\) so \(v = 246\) GeV, and reports the observables.

```python
s = m.spectrum(point)
print({k: round(v, 4) for k, v in s.items()})
# {'xi': 0.07, 'f': 0.9307, 'mt': 0.1625, 'mb': 0.0024,
#  'mh': 0.1118, 'mW': 0.0798, 'mZ': 0.0912, 'mtop_partner': 0.9693}
```

`xi = (v/f)^2` is the compositeness scale; `mtop_partner` is the lightest fermionic top partner.

## 4. The fine-tuning

Four measures, computed by differentiating the vacuum with respect to the fundamental Lagrangian
masses.

```python
t = m.tuning(point)
print({k: round(v, 1) for k, v in t.items()})   # {'BG': ..., 'HOT': ..., 'I': ..., 'KL': ..., 'J': ...}
```

`BG` is the Barbieri–Giudice measure \(\max_i |\partial \log v^2 / \partial \log p_i|\); the others
are the higher-order-tuning, information, and KL measures (see [Derivations](DERIVATIONS.md)).

## 5. Is the point allowed?

`constraints` maps the spectrum onto current LHC/EWPT/SMEFT bounds.

```python
print(pychm.constraints.report(s))
# observable      value      bound                status   ref
# xi                0.0700   < 0.1                PASS    2207.01465
# kappa_V           0.9644   > 0.95               PASS    2207.01465
# mtop_partner      0.9693   > 1.5                FAIL    2212.05263
# kappa_lambda      0.8918   in (-0.71, 6.1)      PASS    2602.23991
# EXCLUDED
```

This benchmark passes compositeness but has a **light top partner** (0.97 TeV), so it is excluded by
direct VLQ searches — exactly the tension naturalness predicts. `m.constraints(point)` returns the
same result as a dict for programmatic use.

## 6. Beyond the minimal model

Larger cosets predict **extra pNGBs** whose masses pyCHM computes as curvatures of the CW potential.

NMCHM — SO(6)/SO(5) — adds a singlet:

```python
nm = pychm.Model('6-6-6')
print(nm.spectrum(point)['xi'])   # same Higgs sector, plus a CP-odd singlet eta
```

SU(5)/SO(5) — the Ferretti/littlest-Higgs real-rep coset — carries **14 pNGBs**: a Higgs, a complex
triplet, and a singlet.

```python
from pychm import su5so5 as M
from pychm.groups import su5so5 as G

print(G.coset_so4_content())      # {(0,0):1, (0.5,0.5):1, (1.0,1.0):1} = singlet + Higgs + triplet

pt = dict(point, f1=1.50568, fX=2.99808, gX=2.96355)
s5 = pychm.Model('5-5-su5').spectrum(pt)
thv = np.arcsin(np.sqrt(s5['xi']))
m_eta = np.sqrt(max(M.singlet_mass2(pt, thv, f=s5['f']), 0)) * 1000
m_phi = np.sqrt(max(M.triplet_mass2(pt, thv, f=s5['f']), 0)) * 1000
print("m_singlet = %.0f GeV,  m_triplet = %.0f GeV" % (m_eta, m_phi))
```

## 7. The coset landscape

Which cosets even *give* a natural Higgs? `landscape` enumerates the symmetric cosets whose pNGBs
contain a custodial \((2,2)\) doublet.

```python
from pychm.groups import landscape
landscape.print_scan(max_ngb=14)
# SO(5)/SO(4):  (2,2)                          <- MCHM
# SO(6)/SO(5):  (1,1) + (2,2)                  <- NMCHM
# SU(5)/SO(5):  (1,1) + (2,2) + (3,3)          <- littlest Higgs
# ...
```

See [The coset landscape](COSET_LANDSCAPE.md) for the organizing principle and
[Extending pyCHM](EXTENDING_BSM.md) to add your own coset, representation, or bound.
