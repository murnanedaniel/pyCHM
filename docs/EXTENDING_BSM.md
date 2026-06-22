# Extending pyCHM to the post-2019 BSM landscape

A research-grounded roadmap for growing pyCHM beyond the Minimal (SO(5)/SO(4)) and
Next-to-Minimal (SO(6)/SO(5)) composite-Higgs cosets it implements today. Each section states
the model class, the relevant literature (arXiv ids verified against the abstract pages), the
coset `G/H` and fermion content, and an honest **pyCHM integration sketch** — what the existing
CCWZ / tensor / potential / tuning machinery already covers versus what new infrastructure is
needed.

## What pyCHM already provides (the reuse surface)

- **CCWZ Goldstone engine** (`groups/so5.py`, `groups/so5.py`, `groups/so6.py`): the Goldstone
  matrix `U` for a coset, currently built as an `SO(N)` rotation (a planar rotation for one vev
  direction; a closed-form Rodrigues rotation for the SO(6) two-field case).
- **Representation tower** (`groups/tensors.py`): an `n`-agnostic builder of symmetric-traceless
  and antisymmetric `SO(n)` tensor irreps of any rank, plus Clifford spinors (`groups/spinors.py`,
  `groups/so6_spinors.py`). The Goldstone lift `U_rep` works for any of these.
- **Branching machinery** (`groups/decompose.py`, `groups/so6.py`): `SO(N)→SO(N-1)→SO(4)`
  decompositions via quadratic Casimirs / SU(2)×SU(2) content.
- **Model assembler** (`assemble.py`, `nmchm6.py`): partial-compositeness fermion mass matrices
  from declarative embeddings + the Goldstone dressing.
- **Physics pipeline** (`routes.py`, `potential.py`, `spectrum.py`, `tuning.py`): the
  Coleman-Weinberg potential (eigenvalue + momentum routes), the EW vacuum, the spectrum, and the
  Barbieri-Giudice / information fine-tuning. **This pipeline is coset-agnostic** — it only needs a
  model exposing `mass_U/mass_D/mass2_W/mass2_Z` (and, for multi-pNGB cosets, a multi-field
  potential as in `nmchm6.potential`).

**The single biggest lever:** the tensor/CCWZ core is hard-wired to **`SO(N)` cosets**. The
modern composite-Higgs frontier (Ferretti–Karateev, Sannino, little Higgs) lives on **`SU(N)`
cosets** (`SU(N)/SO(N)`, `SU(N)/Sp(N)`, `SU(N)×SU(N)/SU(N)`). Generalising the Goldstone
construction and the irrep builder from `SO(N)` to general symmetric spaces / `SU(N)` Young
tableaux is the infrastructure investment that unlocks most of the list below.

---

## 1. Fundamental composite dynamics & the Ferretti–Karateev classification (highest priority)

The dominant post-2019 program: UV-complete the composite Higgs with a confining gauge theory of
"hyperfermions," classified by Ferretti & Karateev.

- Ferretti, Karateev, *Fermionic UV completions of composite Higgs models* — **arXiv:1312.5330**
  (JHEP 03 (2014) 077).
- Ferretti, *Gauge theories of partial compositeness: scenarios for Run-II* — **arXiv:1604.06467**.
- Belyaev et al., *Di-boson signatures as Standard Candles…* — **arXiv:1610.06591** (defines the
  **M1–M12** model labels).
- Cacciapaglia, Ferretti, Flacke, Serôdio, *Light scalars in composite Higgs models* —
  **arXiv:1902.06890** (Front. Phys. 7 (2019) 22; the clean M1–M12 table).
- Cacciapaglia, Pica, Sannino, *Fundamental Composite Dynamics: A Review* — **arXiv:2002.04914**
  (Phys. Rept. 877 (2020) 1).
- Witzel, *Review on Composite Higgs Models* (lattice) — **arXiv:1901.08216**; e.g. Sp(4) lattice
  PRD 109 (2024) 094512 — **arXiv:2401.05637**.

**Three EW Higgs cosets** (set by whether the EW hyperfermion rep is real / pseudoreal / complex):

| EW coset | rep type | # Goldstones | pNGB Higgs content |
|---|---|---|---|
| **SU(4)/Sp(4)** | pseudoreal | 5 | Higgs doublet **+ 1 singlet** |
| **SU(5)/SO(5)** | real | 14 | Higgs doublet + extra triplet/singlets |
| **SU(4)×SU(4)/SU(4)** | complex | 15 | two Higgs doublets + … |

Each model has **two hyperfermion species** (ψ building the EW coset, χ carrying QCD colour); the
**top partner is a "chimera baryon"** ψψχ or ψχχ. The 12 minimal models M1–M12 differ by the
hypercolour group (Sp(4), SU(4), SO(7), …) and the ψ/χ irreps; the most lattice-studied are the
**Sp(4)** model with fundamental+antisymmetric hyperfermions (M5-type) and the **SU(4)** models.

### pyCHM integration sketch

- **SU(4)/Sp(4) — already implemented.** `SU(4)/Sp(4)` is locally isomorphic to **`SO(6)/SO(5)`**:
  its 5 Goldstones are exactly the Higgs-doublet-plus-singlet of pyCHM's NMCHM (`groups/so6.py`,
  `nmchm6.py`). **The minimal pseudoreal Ferretti/Sannino EW coset is the NMCHM pyCHM already
  has.** Near-term win: (i) document this equivalence; (ii) add the chimera-baryon top-partner
  embeddings (the partner sits in an Sp(4) irrep — the 5/10/14 of SO(5), already built in
  `symbolic`), so the existing assembler + potential pipeline yields the spectrum and tuning of the
  minimal fundamental composite Higgs. **Effort: low** (embeddings + a model file; coset is done).
- **SU(5)/SO(5) — the key new coset** (covers both the real-rep Ferretti models *and* the littlest
  Higgs, §3). Needs: (a) the symmetric-space Goldstone `Σ = U Σ₀ Uᵀ` with `U=exp(iΠ/f)` over the
  14 broken (symmetric) `SU(5)` generators — a generalisation of the so6 Rodrigues construction to
  a non-abelian multi-field coset; (b) `SU(5)` irreps for the partners (Young tableaux, i.e. an
  `SU(N)` analogue of `tensor_basis`); (c) the `SU(5)→SO(5)→SU(2)×SU(2)` branching. **Effort:
  medium-high** (the symmetric-space CCWZ + SU(N) rep builder is the reusable infrastructure).
- **SU(4)×SU(4)/SU(4)** (complex, two-Higgs-doublet): left-right product coset; needs the product-
  group CCWZ. **Effort: high.**
- Reusable as-is once the coset/embeddings exist: the **entire potential/spectrum/tuning pipeline**
  (multi-field potential already demonstrated in `nmchm6`), and the light-scalar phenomenology
  (extra singlets/octet) is just extra pNGB directions in the same `V(π)`.

**Recommended first step:** promote the SO(6)/SO(5) NMCHM to an explicit **SU(4)/Sp(4) fundamental
composite Higgs** with chimera-baryon partners (low effort, high scientific value — it connects
pyCHM directly to the lattice program), *then* build the general `SU(N)/SO(N)` symmetric-space
engine for SU(5)/SO(5).

---

## 2. Little Higgs (the littlest Higgs and collective symmetry breaking)

The pre-composite-Higgs pNGB program; shares the `SU(5)/SO(5)` coset, so it rides on the same
infrastructure as §1.

- Arkani-Hamed, Cohen, Katz, Nelson, *The Littlest Higgs* — **arXiv:hep-ph/0206021**
  (`SU(5)/SO(5)`, f ~ TeV, cutoff Λ ≲ 4πf ~ 10 TeV).
- Arkani-Hamed, Cohen, Georgi, *(De)Constructing Dimensions* — **arXiv:hep-th/0104005** (the
  moose/deconstruction machinery).
- Cheng, Low, *Little hierarchy, little Higgses, and a little symmetry* — **arXiv:hep-ph/0405243**
  (T-parity); Low, *T-parity and the littlest Higgs* — **arXiv:hep-ph/0409025**.
- Reuter et al., *The fate of the littlest Higgs with T-parity under 13 TeV LHC data* —
  **arXiv:1811.02268** (the most complete recent recast; the subfield has since gone quiet,
  absorbed into generic vector-like-quark searches).

**Collective symmetry breaking:** the Higgs mass is protected because *no single coupling* breaks
all the global symmetry — at least two must act together, pushing the quadratic divergence to two
loops (Λ ~ 10 TeV).

### pyCHM integration sketch

- The coset is `SU(5)/SO(5)` → **shares the §1 SU(5)/SO(5) engine.** Once that exists, the littlest
  Higgs is a *different embedding + gauge structure* on the same coset: gauge `[SU(2)×U(1)]²`
  (collective), and the heavy gauge/top partners that cancel the SM loops.
- **New piece — collective gauging / the moose:** pyCHM's gauge sector (`mchm5.mass2_W/Z`) is a
  two-site `SU(2)×U(1)`. Little Higgs needs the *doubled* gauge group and the deconstructed
  multi-site moose. This is a generalisation of the existing two-site gauge matrices to an
  N-site quiver. **Effort: medium** on top of the SU(5)/SO(5) coset.
- **T-parity** is a discrete symmetry on the moose — a bookkeeping/projection layer, not new CCWZ.
- The Coleman-Weinberg/vacuum/tuning pipeline is reusable; the fine-tuning module is directly
  relevant (little Higgs is *about* the little hierarchy, so `tuning.py`'s BG/information measures
  apply).

---

## 3. Twin Higgs & neutral naturalness

Naturalness with **colourless** top partners — the partners cancelling the top loop live in a
mirror/hidden sector, evading LHC coloured-partner searches.

- Chacko, Goh, Harnik, *The Twin Higgs: natural EWSB from mirror symmetry* —
  **arXiv:hep-ph/0506256** (approximate global `SU(4)`, realised as `SU(4)/SU(3)`).
- Craig, Katz, Strassler, Sundrum, *Naturalness in the dark at the LHC* (fraternal twin Higgs) —
  **arXiv:1501.05310**.
- Burdman, Chacko, Goh, Harnik, *Folded supersymmetry…* — **arXiv:hep-ph/0609152**;
  Craig, Knapen, Longhi, *Neutral naturalness from the orbifold Higgs* — **arXiv:1410.6808**;
  Cohen, Craig, Giudice, McCullough, *The Hyperbolic Higgs* — **arXiv:1803.03647**.
- Batell, Low, Neil, Verhaaren, *Review of Neutral Naturalness* — **arXiv:2203.05531**;
  Bansal et al., *Mirror Twin Higgs cosmology (H0, S8 tensions)* — **arXiv:2110.04317**.

**Coset:** global `SU(4)→SU(3)` gives `2·(4²−1)−… = 7` broken generators; one `SU(2)` doublet (4
real Goldstones) is the Higgs, protected by the `Z2` mirror symmetry.

### pyCHM integration sketch

- Coset `SU(4)/SU(3)` → needs the `SU(N)/SU(N-1)` symmetric-space CCWZ (a third coset family
  beyond `SU(N)/SO(N)` and `SU(N)/Sp(N)`). Once the general `SU(N)` Goldstone engine exists, this
  is a new `H` choice. **Effort: medium** for the coset.
- **Genuinely new infrastructure — the twin sector:** the defining physics is a *mirror copy of
  the SM* with a `Z2` exchange symmetry, and the cancellation happens between visible and twin
  loops. pyCHM's potential currently sums one sector's mass spectrum; twin Higgs needs the visible
  **+ twin** spectra with the `Z2` relation enforced. This is a structural extension of
  `routes._sector_masses` (add a mirror sector) plus the `Z2`-symmetric potential. **Effort:
  high.** Cosmology (dark radiation, H0/S8) is out of scope for a spectrum/tuning library.
- Folded SUSY / hyperbolic / orbifold Higgs are different UV realisations of the same IR
  protection; the orbifold-Higgs framework (1410.6808) is the natural general target if pursued.

---

## 4. Holographic / 5D / warped composite Higgs (pyCHM's own lineage)

The original dual picture: a pNGB Higgs from a 5D gauge field in a warped (Randall–Sundrum)
background; the 4D CCWZ models pyCHM implements are the "two-site" deconstruction of these.

- Contino, Nomura, Pomarol, *Higgs as a holographic pseudo-Goldstone boson* —
  **arXiv:hep-ph/0306259** (the AdS/CFT origin).
- Agashe, Contino, Pomarol, *The Minimal Composite Higgs Model* — **arXiv:hep-ph/0412089** (the
  `SO(5)/SO(4)` MCHM in 5D AdS — the coset pyCHM's `mchm5` implements).
- Contino, Da Rold, Pomarol, *Light custodians…* — **arXiv:hep-ph/0612048** (the custodial
  `X5/3`/top-partner structure that the partner embeddings encode).
- Panico, Wulzer, *The Discrete Composite Higgs Model* — **arXiv:1106.2719**: the 4D multi-site
  deconstruction; a **two-site truncation suffices for a finite, calculable Higgs potential**.
  **This is exactly the two-site M4DCHM pyCHM implements** — the form factors `Π(p)`
  (`mchm5.formfactor_pieces`) are the truncated 5D propagators.
- Carragher, Murnane, Stangl, Su, White, Williams, *Minimal 4D Composite Higgs Models Under Current
  LHC Constraints* — **arXiv:2007.11943**: global fits of the M4DCHM^{5-5-5}, ^{14-14-10},
  ^{14-1-10} variants — the **direct pyCHM lineage** (shares an author with this repo). A natural
  external cross-check / validation target for the spectrum + tuning pyCHM computes.
- Blasi, Bollig, Goertz, *Holographic Composite Higgs Model Building: …Maximal Symmetry…* —
  **arXiv:2212.11007** (recent: maximal symmetry decouples the light Higgs from light top partners).
  Review: Goertz, *Composite Higgs theory* — **arXiv:1812.07362**.

### pyCHM integration sketch
- **N-site deconstruction** (more KK levels) → a tower of partners converging to the continuum 5D
  result: a generalisation of the assembler to N composite sites, reusing the CCWZ dressing per
  site, same coset. **Effort: medium.** This is the natural way to study the maximal-symmetry /
  light-top-partner mechanism (2212.11007) within pyCHM's existing framework.
- A genuine 5D solver (bulk profiles, holographic two-point functions) is different numerical
  technology and **out of scope** for the current eigenvalue/momentum-route design.

## 5. Current experimental constraints (validation/phenomenology targets)

Not new cosets, but the numbers any extension must respect — and concrete external anchors pyCHM
could be checked against (all arXiv ids verified):

- **Compositeness scale:** custodial + Higgs-coupling (`g_hVV ∝ √(1−ξ)`) and S-parameter data give
  the standard bound **ξ = v²/f² ≲ 0.1, i.e. f ≳ 0.8–1 TeV** (relaxable to ξ ~ 0.2–0.4 with extra
  light states). EWPT analysis: Frandsen, Rosenlyst, **arXiv:2207.01465**. *(pyCHM's REF point sits
  at ξ ≈ 0.07 — inside this bound.)*
- **Top partners (VLQ):** pair production excludes `m ≳ 1.5 TeV` (up to ~1.6 TeV degenerate/doublet)
  — ATLAS **arXiv:2212.05263**; single production is coupling-dependent, reaching 2+ TeV at large
  mixing — ATLAS **arXiv:2305.03401**; review Benbrik et al. **arXiv:2412.01761**. *(pyCHM's
  `spectrum()` returns `mtop_partner`; this is a direct cut.)*
- **Higgs self-coupling (di-Higgs):** ATLAS+CMS combination **−0.71 < κ_λ < 6.1** (95% CL,
  **arXiv:2602.23991**); the pNGB shift `δκ_λ ∝ ξ` is still allowed at O(1), so di-Higgs is not yet
  the leading ξ constraint.
- **Global SMEFT fits:** `fitmaker` (Ellis et al., **arXiv:2012.02779**) and `SMEFiT` (Giani,
  Magni, Rojo, **arXiv:2302.06660**) bound the dim-6 Higgs/EW operators onto which ξ maps — all
  consistent with ξ ≲ 0.1.

**Library opportunity:** a thin `constraints` module mapping pyCHM's spectrum (`ξ`, `mtop_partner`,
`κ_λ` shift) onto these published bounds would turn the calculator into a phenomenology tool —
small effort, high utility, and (via 2007.11943) a natural external validation of the spectrum.

---

## 6. Clockwork / relaxion / exotic naturalness (longer term)

- **Clockwork / linear dilaton** (Choi–Im; Giudice–McCullough, *A Clockwork Theory* —
  arXiv:1610.07962): a chain of fields with hierarchical couplings generating exponentially small
  effective couplings. Could be modelled as an N-site moose (overlaps with §2/§4 deconstruction
  infrastructure), but the Higgs-as-clockwork-pNGB constructions are niche. **Effort: high, value:
  speculative.**
- **Relaxion** (Graham, Kaplan, Rajendran, arXiv:1504.07551): solves the hierarchy cosmologically
  via a scanning axion-like field. This is **cosmological dynamics, not a spectrum/tuning
  calculation** — out of scope for pyCHM's design, though the relaxion *potential* could be a
  module if ever wanted.

---

## Prioritised roadmap

1. **SU(4)/Sp(4) fundamental composite Higgs (now).** Re-cast the NMCHM as the minimal pseudoreal
   Ferretti/Sannino model with chimera-baryon top partners. Coset already done; only embeddings +
   a model file. Connects pyCHM to the active Sp(4) lattice program. **Low effort, high value.**
2. **General `SU(N)/SO(N)` symmetric-space CCWZ + an `SU(N)` irrep builder (the infrastructure).**
   The reusable core that unlocks SU(5)/SO(5). Generalise the so6 Rodrigues/exp-map to a multi-field
   non-abelian coset; add an `SU(N)` Young-tableau analogue of `tensor_basis`. **Medium-high effort,
   unlocks §1 real-rep models and §2 little Higgs.**
3. **Littlest Higgs `SU(5)/SO(5)` + collective (moose) gauging.** Rides on (2); add the doubled
   gauge group and T-parity. **Medium effort.**
4. **`SU(N)/Sp(N)` and `SU(N)/SU(N-1)` coset families** → the rest of the Ferretti M1–M12 table and
   the twin-Higgs coset. **Medium effort given (2).**
5. **Twin sector / neutral naturalness** (mirror spectrum + Z2 potential) and **N-site holographic
   deconstruction.** Larger structural extensions. **High effort.**

The throughline: investment (2) — a general symmetric-space CCWZ on `SU(N)` cosets with an `SU(N)`
representation builder — is what converts pyCHM from "the two minimal `SO(N)` models" into a
general composite-pNGB-Higgs calculator covering the modern (Ferretti/Sannino) landscape. The
`SU(4)/Sp(4) ≅ SO(6)/SO(5)` coincidence means pyCHM is already one concrete model into that
landscape today.

*Citations above were verified against arXiv abstract pages by the research agents; integration
effort estimates are this library's assessment, not from the literature.*
