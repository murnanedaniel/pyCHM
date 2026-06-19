# pyCHM

**Open, pure-`numpy`/`scipy` Composite Higgs Model calculator** — the radiatively-generated
Coleman–Weinberg Higgs potential, the electroweak vacuum, the spectrum, and **four fine-tuning
measures** (Barbieri–Giudice, HOT, the information measure `I = ½ log det(I+F)`, and the
prior→posterior KL tuning), with a **dual computational route** (form-factor and mass-eigenvalue).

No private dependencies. The physics is from the published two-site M4DCHM (SO(5)→SO(4)) of
*Murnane, The Landscape of Composite Higgs Models*, arXiv:2606.18364.

> There is no other public composite-Higgs potential-and-fine-tuning calculator — the analogue of
> SoftSUSY/FeynHiggs for SUSY. pyCHM fills that gap.

## Install

```bash
pip install -e ".[test]"
```

## Use

```python
import pychm
m = pychm.Model('5-5-5')
point = dict(mU=2.0, mUt=1.2, mD=1.8, mDt=1.0, mYu=0.9, Yu=1.2, mYd=0.7, Yd=0.4,
             Lq=1.1, Lt=1.3, Lb=0.3, f=0.9, f1=1.5, fX=3.0, g=0.671, gp=0.358, grho=5.0, gX=3.0)
m.spectrum(point)   # -> {xi, f, mt, mb, mh, mW, mZ}  or None (no EWSB)
m.tuning(point)     # -> {BG, HOT, I, KL}
```

## The two routes

Both evaluate the *same* one-loop CW potential — the trace-log of `p² + M²(h)` — and must agree:
- **eigenvalue**: diagonalise the mass matrix `M(s_h)`, sum the closed-form CW over eigenvalues.
- **form-factor**: integrate out the heavy partners (form factors), one momentum integral.

`route='formfactor'` (default) or `route='eigenvalue'` selects the fermion method; the gauge
sector uses the eigenvalue route in both. The equivalence is the central validation: on the
original model's mass matrices the two routes agree to **<1%** (`validation/two_routes_equivalence.py`),
and CI enforces it across random points once the eigenvalue fermion route lands.

## Status

| | state |
|---|---|
| 5-5-5 form factors + gauge mass matrices | ✅ |
| form-factor route (potential, spectrum, tuning) | ✅ |
| four fine-tuning measures (BG/HOT/I/KL) | ✅ |
| eigenvalue **fermion** route (port the 11×11 matrices) | 🔜 Phase 1 |
| CI route-equivalence on random points × models | 🔜 Phase 2 |
| 14-14-10, 14-1-10 representations | 🔜 |

Validated against the published anchors (`m_Z = 91.2` GeV exact; observed `m_t`, `m_h` reachable;
`Δ_BG ≈ 33` at `f ∼ 1` TeV for the M4DCHM, matching the thesis). See `validation/`.



## Current limitation (v0.1) — read this

v0.1 computes the fermion sector by the **form-factor** route and the gauge sector by the
**eigenvalue** route. These two routes carry a constant relative-normalisation difference of a
few percent. For *untuned* quantities that is harmless, but a *tuned* electroweak vacuum is a
near-cancellation between the top (≈ +0.11) and gauge (≈ −0.11) contributions, and a few-percent
route-mismatch flips the small residual — so **v0.1 does not yet give correct tuned vacua or
tuned-point fine-tuning numbers.** This is the route-mixing pitfall the project exists to handle,
and the fix is **Phase 1**: implement the eigenvalue fermion route (port the fermion mass matrices),
so the whole potential is computed in one consistent scheme. The route equivalence is already proven
on the original model's matrices (`validation/two_routes_equivalence.py`, agree to <1%); Phase 1
brings that into the package and turns on the CI equivalence test.

Until then, use v0.1 for the potential machinery, the gauge sector, the form factors, and the
mechanism — not for trustworthy tuned-point tuning.

## Licence
MIT.
