# pyCHM

**Open, pure-`numpy`/`scipy` Composite Higgs Model calculator** — the radiatively-generated
Coleman–Weinberg Higgs potential, the electroweak vacuum, the spectrum, and **four fine-tuning
measures** (Barbieri–Giudice, HOT, the information measure `I = ½ log det(I+F)`, and the
prior→posterior KL tuning), with **two evaluation routes** for the one-loop potential.

No private dependencies. The physics is the published two-site M4DCHM (SO(5)→SO(4)) of
*Murnane, The Landscape of Composite Higgs Models* (arXiv:2606.18364).

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

# validated benchmark (xi ~ 0.070, m_t ~ 0.163 TeV, Delta_BG ~ 127). masses in TeV;
# the Y sector is stored as (mY, Y) with the singlet-Y mass mSY = mY + Y.
point = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)

m.spectrum(point)   # -> {xi, f, mt, mb, mh, mW, mZ, mtop_partner}  or None (no EWSB)
m.tuning(point)     # -> {BG, HOT, I, KL, J}
```

## The two routes

Both evaluate the *same* one-loop CW potential — `½ Tr log(p² + M²(s_h))` — by diagonalising the
mass matrices `M(s_h)`; they differ only in the per-eigenvalue kernel:

- **`eigenvalue`** (default, **the precision route**): closed-form CW, `K(m²) = m⁴ log m²`. Exact,
  manifestly finite. This is the route used for the spectrum and fine-tuning.
- **`momentum`**: the divergence-subtracted Euclidean momentum integral,
  `K(m²) = 4∫₀^∞ pₑ³ [log(pₑ²+m²) − log(pₑ²+1) − (m²−1)/(pₑ²+1) + (m²−1)²/2(pₑ²+1)²] dpₑ`.
  The two subtractions cancel the quadratic and log UV divergences *at the integrand level*; the
  leftover polynomial in `m²` drops out of `V(s_h)−V(0)` because the supertraces `Str 1, m², m⁴`
  are all `s_h`-independent.

The two routes agree on the potential **curve** `V(s_h)` to ~1% of its depth (CI enforces it on
random points, `tests/test_routes_equivalence.py`). **Caveat:** a *tuned* electroweak vacuum is a
near-cancellation, which amplifies the momentum route's residual quadrature error into a
several-percent error on `xi` (and a large error on `m_h''`). That is exactly why fine-tuning is
computed with the closed-form `eigenvalue` route — and is the central lesson of the project.

## Validation

On the benchmark point above, pyCHM reproduces an **independent mass-eigenvalue engine** (the one
behind the published global fits of arXiv:2101.00428; not redistributed here) **with no shared
code**:

| quantity | pyCHM | independent engine |
|---|---|---|
| `xi` (vacuum) | 0.07006 | 0.07005 |
| `Delta_BG` | 126.9 | 126.7 |
| HOT (`|J|`) | 195.2 | 194.9 |
| `I` (nats) | 5.27 | 5.27 |

i.e. **0.03% on `xi` and <0.5% on `Delta_BG`**. The fine-tuning is differentiated in the
fundamental Lagrangian-mass basis `{mU, mUt, mY, mSY, mD, mDt, …, Δ}`, which is the basis the BG
number is defined over.

### 14-14-10

The **14-14-10** representation (`q_L` and `t_R` partners in the symmetric **14** of SO(5),
`b_R` in the **10**) is wired into the same eigenvalue-route pipeline:

```python
m = pychm.Model('14-14-10')
point = dict(
    mQ=3.965100934, mU=2.397190093, mD=1.482462237,
    mYu=0.1478991938, Yu=0.4989142322, Ytu=2.538912347, Yd=0.5293788277,
    Delta_q=2.850102384, Delta_u=1.915943759, Delta_d=0.2222079908,
    f=1.4396717743254574, f1=1.8934618718580186, fX=2.2076564878795057,
    g=0.6709494105248374, gp=0.3581380846874656, grho=5.525258191589697, gX=4.696070132753916)
m.spectrum(point)   # xi ~ 0.0268, m_t ~ 0.128 TeV
m.tuning(point)     # Delta_BG ~ 16.2
```

On this benchmark pyCHM reproduces the independent engine to **0.0002% on `sh`, <0.01% on `m_t`,
0.06% on `m_h`** and **~0.1% on `Delta_BG`** (`tests/test_mchm14.py`).  The BG measure is
differentiated in the 14-14-10 Lagrangian-mass basis `{mQ, mU, mD, mYu, mSYu, mSYtu, Yd, Δq, Δu, Δd}`,
with the two singlet-Y combinations `mSYu = mYu + Yu/2` and `mSYtu = mYu + 4(Yu+Ytu)/5` held
independent of `mYu`.

### 14-1-10

The **14-1-10** representation (`q_L` partner in the symmetric **14** of SO(5), `t_R` partner an
SO(4) **singlet**, `b_R` in the **10**) is the simpler cousin: with a singlet `t_R` partner the
up Y-sector collapses to a single `(mU, Yu)` — no `mYu`/`Ytu` — so the fundamental massive basis is
just `{mQ, mU, mD, Yu, Yd, Δq, Δu, Δd}`, each mapping directly onto a mass-matrix entry.

```python
m = pychm.Model('14-1-10')
point = dict(
    mQ=1.5774575432442343, mU=0.053295037861913734, mD=3.9639579843374463,
    Yu=3.213522953043455, Yd=1.5033041681268544,
    Delta_q=0.8229454568382461, Delta_u=3.843401100178204, Delta_d=0.19968869968780683,
    f=1.1329813572955627, f1=1.5049448725076554, fX=1.4519742141888066,
    g=0.6709494105248374, gp=0.3581380846874656, grho=3.3254933316652062, gX=9.7650801141742978)
m.spectrum(point)   # xi ~ 0.0485, m_t ~ 0.154 TeV
m.tuning(point)     # Delta_BG ~ 7.3 (argmax Delta_q)
```

On this benchmark pyCHM reproduces the independent engine to **0.035% on `sh`, 0.034% on `m_t`,
<0.01% on `m_h`** (matched scheme at the oracle `f`) and **0.038% on `Delta_BG`**
(`tests/test_mchm14_1_10.py`).

## Toward a generic spectrum generator (`ccwz.py`)

The three models above hand-code their fermion mass matrices. Their *only* representation-specific
content is the Higgs (Goldstone) dressing — the s_h factors `cos(h/f), sin(h/f), cos²(h/2f)` for the
**5**; `(3+5cos2h/f)/8, √5 sin(2h/f)/4` for the **14** — which are just the matrix elements of the
Goldstone matrix `U(h)` in the chosen SO(5) irrep. `pychm.ccwz` builds `U_R(h)` for the 5, 10 and 14
from group theory, so this dressing follows from one construction for any partner representation:

```python
import numpy as np, pychm.ccwz as ccwz
ES = ccwz.embedding('14', 'singlet')
ccwz.overlap('14', ES, ES, h_over_f)        # == (3 + 5 cos(2 h/f)) / 8, exactly
```

`tests/test_ccwz.py` checks these reproduce the hand-coded factors (the 14 singlet overlap to machine
precision). Everything downstream — the Coleman–Weinberg potential, the vacuum, the spectrum and the
four tuning measures — is already representation-agnostic and dispatches on a `model=` string.

**The assembler (`assemble.py`) closes the loop.** Given a declarative spec — the partner
representation, the elementary embeddings, and the composite states (masses + SO(5) content) — it
emits `mass_U`/`mass_D` with the Higgs dressing supplied by `ccwz`, for *any* representation. It is
validated to reproduce the hand-coded **5-5-5** mass matrices entry-for-entry (to machine precision),
and the assembled model — registered as `pychm.Model('5-5-5-assembled')` — reproduces the full
validated 5-5-5 spectrum and tuning end-to-end (to the precision the tuned vacuum permits):

```python
import pychm
pychm.Model('5-5-5-assembled').spectrum(point)   # built from embeddings + ccwz, not hand-coded
```

So a composite-Higgs model is now specifiable by group-theoretic data. Extending the bundled specs
to the 14-plet models (the embeddings are larger but the machinery is identical) is the remaining
bookkeeping; `tests/test_assemble.py` is the regression harness.

## Status

| | state |
|---|---|
| 5-5-5 fermion + gauge mass matrices, both routes | ✅ |
| electroweak vacuum, spectrum (`xi, f, mt, mb, mh, mW, mZ`) | ✅ |
| four fine-tuning measures (BG/HOT/I/KL), validated to <0.5% | ✅ |
| CI route-equivalence on random points | ✅ |
| 14-14-10 representation (eigenvalue route), validated to <0.1% | ✅ |
| 14-1-10 representation (eigenvalue route), validated to <0.1% | ✅ |
| generic CCWZ Goldstone dressing (`ccwz.py`, reps 5/10/14), rep factors validated | ✅ |
| generic mass-matrix assembler (`assemble.py`), reproduces 5-5-5 end-to-end | ✅ |
| bundled assembler specs for the 14-plet models | 🔜 |
| CI: all models + route-equivalence tested (24 tests, 3.9/3.11/3.12) | ✅ |

## Licence
MIT.
