# pyCHM

**Open, pure-`numpy`/`scipy` Composite Higgs Model calculator** — the radiatively-generated
Coleman–Weinberg Higgs potential, the electroweak vacuum, the spectrum, **four fine-tuning
measures**, and the **experimental-viability check** against current LHC/EWPT/SMEFT bounds.

No private dependencies. The physics is the published two-site M4DCHM (SO(5)→SO(4)) of
*Murnane, The Landscape of Composite Higgs Models* (arXiv:2606.18364), generalised onto a single
`Coset(G,H)` engine that also carries NMCHM (SO(6)/SO(5)), SU(4)/Sp(4), and SU(5)/SO(5).

!!! quote
    There is no other public composite-Higgs potential-and-fine-tuning calculator — the analogue of
    SoftSUSY/FeynHiggs for SUSY. pyCHM fills that gap.

## Install

```bash
pip install -e ".[test]"      # runtime + tests
pip install -e ".[docs]"      # to build this site
```

## Use

```python
import pychm
m = pychm.Model('5-5-5')

# validated benchmark (xi ~ 0.070, m_t ~ 0.163 TeV, Delta_BG ~ 127). masses in TeV.
point = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)

m.spectrum(point)              # -> {xi, f, mt, mb, mh, mW, mZ, mtop_partner}  or None (no EWSB)
m.tuning(point)                # -> {BG, HOT, I, KL, J}
m.constraints(point)           # -> pass/fail against the experimental bounds
```

New to the library? Start with the **[Tutorial](tutorial.md)**.

## The two routes

Both evaluate the *same* one-loop CW potential — \(\tfrac12 \,\mathrm{Tr}\log(p^2 + M^2(s_h))\) — by
diagonalising the mass matrices \(M(s_h)\); they differ only in the per-eigenvalue kernel:

- **`eigenvalue`** (default, **the precision route**): closed-form CW, \(K(m^2) = m^4 \log m^2\).
  Exact, manifestly finite. This is the route used for the spectrum and fine-tuning.
- **`momentum`**: the divergence-subtracted Euclidean momentum integral. The two subtractions cancel
  the quadratic and log UV divergences at the integrand level; the leftover polynomial drops out of
  \(V(s_h)-V(0)\).

## What's here

| Section | Contents |
|---|---|
| [Tutorial](tutorial.md) | End-to-end, copy-pasteable walkthrough |
| [Models](models.md) | MCHM, NMCHM, SU(4)/Sp(4), SU(5)/SO(5) |
| [Coset landscape](COSET_LANDSCAPE.md) | The finite classification + enumerator |
| [The Coset(G,H) engine](engine.md) | How any coset is built from \(G,H\) |
| [Derivations](DERIVATIONS.md) | First-principles math (also as a [PDF](DERIVATIONS.pdf)) |
| [Validation](THESIS_VALIDATION.md) | Anchors, audit, and what is checked against what |
| [Extending pyCHM](EXTENDING_BSM.md) | Adding a coset, a rep, or a bound |
| [API reference](api/model.md) | Auto-generated from the source |
