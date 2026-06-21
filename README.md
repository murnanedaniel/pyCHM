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

### 6-6-6 (NMCHM, SO(6)/SO(5))

The **Next-to-Minimal** model promotes the coset to **SO(6)/SO(5)**: the five pNGBs are the Higgs
doublet *plus a real SO(5) singlet* `s`, and the quark partners sit in the **6** (the NM4DCHM6).
The full SO(6) representation tower — the **6**, the adjoint **15**, the symmetric-traceless
**20'**, the self-dual **10**/`10bar` and the Weyl spinors **4**/`4bar` — is built from group
theory in `pychm.symbolic.so6` (closed-form Goldstone via Rodrigues, lifts via the same tensor /
Clifford machinery as SO(5)), and validated against the thesis in closed form (the Goldstone `Φ`
eq. 474, the broken generators eq. 632, the `6 = 4+1+1` embedding eq. 633) and by the SO(6)→SO(5)
branchings.

```python
m = pychm.Model('6-6-6')
m.spectrum(point)   # same xi/m_t/m_h as 5-5-5 at <s>=0 (inherits the pypngb anchor)
m.tuning(point)     # same Delta_BG
from pychm import nmchm6
nmchm6.singlet_mass2(point, thv, f=f)   # the new SO(5)-singlet pNGB mass (massless in the MCHM)
```

At `<s>=0` the SO(6) mass matrices reduce to the validated 5-5-5 **entry-for-entry** (so EWSB and
the anchor carry over), and the model adds the genuine NMCHM observable: a finite, calculable
singlet pNGB mass from the fermion loop (`tests/test_so6.py`, `test_so6_spinors.py`,
`test_nmchm6.py`; see `docs/THESIS_VALIDATION.md` for the derived-vs-input-vs-anchored boundary).

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
validated to reproduce **all three** hand-coded models — **5-5-5**, **14-1-10** and **14-14-10** —
entry-for-entry to machine precision, exercising the **5**, **10** and **14** of SO(5). Each assembled
model is registered (`pychm.Model('5-5-5-assembled')`, `'14-1-10-assembled'`, `'14-14-10-assembled'`)
and reproduces the full validated spectrum and tuning end-to-end, to the precision the tuned vacuum
permits:

```python
import pychm
pychm.Model('14-14-10-assembled').spectrum(point)  # 19x19 up sector from 14/10 embeddings + ccwz
```

So a composite-Higgs model is now specifiable purely by group-theoretic data: choose the partner
representations, write down where the elementary fermions embed, and `ccwz` + `assemble` build the
Higgs-dependent mass matrices — no per-model transcription. `tests/test_assemble.py` is the regression
harness (every model checked entry-for-entry and end-to-end against its hand-coded oracle).

## Symbolic derivation from scratch, and arbitrary representations (`pychm.symbolic`)

The dressing factors are not only computed numerically — they are **derived in closed form**. The
symbolic engine builds the Goldstone matrix `U_R(θ)` (θ = h/f) with sympy and reduces each overlap to
a closed trigonometric expression, so the benchmark factors follow from group theory with no
curve-fitting:

```python
from pychm.symbolic import core, derive
ES = core.embedding_sym('14', 'singlet')
core.overlap_sym('14', ES, ES)                 # -> (5*cos(2*θ) + 3)/8, derived symbolically
```

The previous assembler reverse-fit its composite states by least squares over `s_h` samples; that is
gone. Every benchmark dressing is now **proven** to be a genuine Goldstone matrix element `⟨c|U_R|E⟩`
by an exact symbolic solve (`symbolic.derive.solve_composite`), and the assembled models lambdify these
closed forms to numpy callables (no sympy on the hot path).

The construction extends to **any compatible representation**:

- **Arbitrary tensor irreps** — `symbolic.tensors` builds an orthonormal basis of any rank-`k`
  symmetric-traceless / antisymmetric SO(5) irrep and lifts `U` to it; `ccwz.U_rep(('sym', 3), s_h)`
  is the **30**, etc.
- **SO(4) decomposition** — `symbolic.decompose` finds the `(j_L, j_R)` sub-multiplets via the two
  SU(2) Casimirs (5 = (2,2)+(1,1); 10 = (2,2)+(3,1)+(1,3); 14 = (3,3)+(2,2)+(1,1)), with a
  `compatible(rep, jL, jR)` predicate for placing the elementary fermions.
- **Spinorial reps** — `symbolic.spinors` gives the SO(5)≅Sp(4) gamma matrices and the Goldstone
  matrix in the **4** (the MCHM4 partner, 4 = (2,1)+(1,2)) and the **16**.
- **NMCHM SO(6)/SO(5)** — `symbolic.so6` / `symbolic.so6_spinors` give the SO(6) coset, the
  closed-form 5-pNGB Goldstone, and the full irrep tower **6 / 15 / 20' / 10 / 4** with their
  SO(6)→SO(5)→SO(4) branchings; `nmchm6` is the worked NM4DCHM6 model.

New representations have no hand-coded oracle, so they are validated by internal consistency
(unitarity, the representation homomorphism, the SO(4) branching) in `tests/test_tensors.py`,
`tests/test_spinors.py`, `tests/test_so6.py` and `tests/test_so6_spinors.py`.

## Validation against the thesis

`tests/test_thesis_equations.py` cross-checks pyCHM against the published equations of
*Murnane, The Landscape of Composite Higgs Models* (arXiv:2606.18364) in closed form: the
Goldstone matrix and SO(5) generators, the SO(4) bases and branchings, the App. A7 form-factor
building blocks (verbatim), the per-representation Higgs dressing, the Coleman–Weinberg kernel
and pole mass, the vacuum / Higgs-mass / gauge relations, and the Barbieri–Giudice tuning. The
14-rep prefactors are **derived** as exact SO(4) Clebsch weights (`symbolic.decompose.
channel_weights_sym`); the overall `4/5` normalization is itself derived from the 14-singlet
embedding (`|S₄₄|²=4/5`), so the prefactors are group theory end to end, not a thesis read-off.

The **first-principles derivations** behind every "derived" claim are written up in
[`docs/DERIVATIONS.md`](docs/DERIVATIONS.md) (and `docs/DERIVATIONS.pdf`), each step tied to the
function and the test that closes it.

**Scope is stated precisely** in [`docs/THESIS_VALIDATION.md`](docs/THESIS_VALIDATION.md): the
Goldstone dressing is derived from group theory, the model structure is thesis input, and the
validated equation classes are the ones the library implements. The NMCHM **SO(6)/SO(5)**
representations (Ch.7/8) are now implemented and validated in closed form (Goldstone `Φ` eq. 474,
broken generators eq. 632, the `6` embedding eq. 633, the full rep tower and its branchings);
higher-order tuning, Bayesian evidence, large-N and the scanning machinery remain out of scope
(not implemented). The thesis has no per-point numeric tables: the 14 models are anchored to the
independent pypngb engine to <0.1%, and the NM4DCHM6 inherits that anchor by reducing to the
5-5-5 at `<s>=0`. This is a precise validation of the implemented physics, not a sweep of all 218
pages.

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
| generic mass-matrix assembler (`assemble.py`), reproduces **all 3 models** (reps 5/10/14) | ✅ |
| symbolic CCWZ engine (`symbolic/`): dressing derived in closed form, curve-fitting removed | ✅ |
| arbitrary tensor irreps + SO(4) decomposition (5/10/14/30/…); spinor reps **4**, **16** | ✅ |
| NMCHM **SO(6)/SO(5)**: reps **6/15/20'/10/4**, closed-form Goldstone + branchings; NM4DCHM6 model + singlet pNGB | ✅ |
| CI: all models (hand-coded + assembled + symbolic + NMCHM) + route-equivalence (113 tests, 3.9/3.11/3.12) | ✅ |

## Licence
MIT.
