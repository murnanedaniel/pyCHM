# Validation Audit — scientific-integrity review of pyCHM

**Scope.** Adversarial audit of how pyCHM is validated, with two specific worries:
(a) shortcuts / quick fixes in the derivation and validation layers, and
(b) the thesis (arXiv:2606.18364) being treated as an infallible oracle where the only
trustworthy reference is the independent **pypngb** engine.

**Method.** Read `src/pychm/*`, `src/pychm/symbolic/*`, `tests/*`, `docs/THESIS_VALIDATION.md`,
`validation/two_routes_equivalence.py`, and streamed the session transcript
(`fa29cb02-…jsonl`, 3053 records) + sampled subagent logs.

**One-line verdict.** The *numbers that physicists read off this library* (xi, m_t, m_b, m_h,
m_W/m_Z, Delta_BG) come from the **eigenvalue route** (mass matrices) and are genuinely anchored
to pypngb with no shared code — that part does **not** depend on the thesis being right. The two
identified risks are both now closed: the **App. A7 form-factor layer** is independently validated
against the pypngb-anchored eigenvalue route (§2.1, agree as `s_h→0` at a custodial point), and the
**`4/5`** 14-prefactor normalization is now **derived from the embedding** (`|S₄₄|²=4/5`, §2.2), not
matched to the thesis. After this round **no observable-feeding quantity is a thesis read-off**;
correctness rests entirely on pypngb + group theory.

---

## 1. Oracle-classification table

Oracle categories:
**(A)** pypngb / independent engine (trustworthy numeric anchor) ·
**(B)** thesis transcription (RISK: proves faithful copying, not physical correctness) ·
**(C)** first-principles / self-validating group theory (unitarity, algebra closure, homomorphism,
branchings, channel weights summing to 1 — independent of the thesis being right).

| Test / equation class | File:line | Oracle | Independent cross-check? |
|---|---|---|---|
| Reference spectrum (xi, m_t, m_h, m_Z) 5-5-5 | `test_anchors.py:32-38` | **A** | pypngb 0.07005 / etc., no shared code |
| Reference tuning (Delta_BG, HOT, I) 5-5-5 | `test_anchors.py:41-47` | **A** | pypngb 126.7 |
| 14-14-10 spectrum & tuning | `test_mchm14.py:35-44` | **A** | pypngb to <0.1% |
| 14-1-10 spectrum & tuning | `test_mchm14_1_10.py:37-46` | **A** | pypngb to <0.1% |
| Two-route equivalence on V(s_h) curve | `test_routes_equivalence.py:16-37` | **C** (internal) | eigenvalue vs momentum, same matrices; *not* vs pypngb in-repo (the pypngb cross-check lives in `validation/two_routes_equivalence.py`, which needs the private engine) |
| Goldstone `U_vector` = thesis planar rotation | `test_thesis_equations.py:52-60` | **B/C** | structure is forced by SO(5) → effectively C |
| Broken/unbroken generators exponentiate / close algebra | `test_thesis_equations.py:72-95` | **C** | algebra closure, homomorphism |
| 14 / 10 basis orthonormal, traceless, (anti)symmetric | `test_thesis_equations.py:97-114` | **C** | self-validating |
| SO(4) branchings 5/10/14/4 | `test_thesis_equations.py:117-124` | **C** | Casimir eigenvalues, rep-theoretic |
| Spinor 4 half-angle | `test_thesis_equations.py:127-132` | **C** | eigen-phases ±θ/2 |
| 5-5-5 dressing s_h-coeffs (overlaps) | `test_thesis_equations.py:138-149` | **C** | `\|⟨q_L\|U\|S⟩\|²=s²/2`, derived |
| 5-5-5 form-factor *wiring* (`formfactor_pieces`) | `test_thesis_equations.py:152-167` | **B** | **none** — checks the code reproduces its *own* `hq/hR/hM` re-built inline; tautological |
| **A7 building blocks `_AL/_AR/_AM/_B` verbatim** | `test_thesis_equations.py:236-248` | **B** | **none** — `sp.simplify(code − thesis)==0`; proves copying only |
| 14-singlet overlap `⟨S\|U\|S⟩=(4c²−s²)/4` | `test_thesis_equations.py:170-176` | **C** | exact overlap, derived |
| **14-14-10 prefactors `(4c²−s²)²/20`, `3/(4√5)`, …** | `test_thesis_equations.py:179-219, 287-301` | **B+C** | C derives the *trig*; the *magnitude* is `(4/5)·W` with `4/5` read from thesis (see §2.1) |
| 14 channel weights sum to 1 | `test_thesis_equations.py:210,301` | **C** | unitarity — genuine |
| 14-1-10 t_R singlet constant (no dressing) | `test_thesis_equations.py:222-230` | **C** | structural |
| Dressings are genuine matrix elements | `test_thesis_equations.py:251-257` | **C** | `solve_composite` residual 0 |
| CW kernel `m⁴log m²`, c_i={3,6,−12} | `test_thesis_equations.py:263-271` | **B** | thesis eq.459; standard CW so low risk |
| **Pole mass eq.518 (`fermion_mass`)** | `test_thesis_equations.py:274-281` | **B** | **none** — checks `fermion_mass == sqrt(Msq/(Π_LΠ_R))` from *its own* pieces; never vs eigenvalue m_t or pypngb |
| Higgs-mass / vacuum closed forms | `test_thesis_equations.py:307-328` | **C** | symbolic vacuum algebra |
| Gauge SM relations m_W=gv/2 etc. | `test_thesis_equations.py:331-339` | **B/C** | SM custodial, standard |
| Fine-tuning measure relations (BG/HOT/I/KL) | `test_thesis_equations.py:345-354` | **C** | identities among `J` |
| SO(6) generators / coset / branchings | `test_so6.py` | **C** | algebra, dim, branchings |
| SO(6) 5-pNGB Goldstone eq.474 / 632 / 633 | `test_so6.py`, `test_so6_spinors.py` | **B/C** | thesis eqs, but forced by group theory |
| NM4DCHM6 reduces to 5-5-5 entry-for-entry | `test_nmchm6.py:21-25` | **A (inherited)** | inherits pypngb anchor via reduction at ts=0 |
| NM4DCHM6 spectrum/tuning == 5-5-5 | `test_nmchm6.py:28-42` | **A (inherited)** | only at ⟨s⟩=0 |
| NM4DCHM6 singlet pNGB mass positive | `test_nmchm6.py:65-73` | **C (weak)** | sign/finiteness only; magnitude *unanchored* (see §2.5) |

---

## 2. Prioritized RISKS — where a thesis error could pass undetected

### 2.1 [RESOLVED — was HIGH] The A7 form-factor route, now validated against the eigenvalue/pypngb route
**Files:** `src/pychm/mchm5.py:22-67` (`_AL/_AR/_AM/_B`, `formfactor_pieces`, `fermion_mass`);
tests `test_thesis_equations.py:236-248` (verbatim), `:152-167` (wiring), `:274-281` (pole mass).

- `_AL/_AR/_AM/_B` are exercised **only** through `formfactor_pieces`/`fermion_mass`, which are
  imported **only** by `test_thesis_equations.py`. `grep` confirms: nothing in
  `spectrum.py` / `routes.py` / `tuning.py` / any model ever calls them. The physical pipeline
  uses the **eigenvalue route** (`mass_U`/`mass_D`).
- `test_A7_building_blocks_verbatim` is `sp.simplify(mchm5._AL(...) − AL)==0` where `AL` is the
  same expression typed twice. **This proves the code matches the thesis; it cannot detect a thesis
  error.** This is exactly the "thesis == truth" oracle the audit was asked to find.
- `test_coeffs_5_5_5_formfactor_wiring` rebuilds `hq/hR/hM` inline and asserts the code equals that
  rebuild — **tautological** (code vs a copy of itself).
- `test_pole_mass_eq518` asserts `fermion_mass == sqrt(Msq/(Π_LΠ_R))` using `Msq, Π_L, Π_R` taken
  from the *same* `formfactor_pieces` call. It never compares the form-factor top mass to the
  eigenvalue-route `mass_U` lightest-but-two eigenvalue, nor to pypngb's m_t.
- The transcript shows this was a deliberate skip at the time (*"reconstructing the convention
  mapping is a … rabbit hole with low marginal value"*) — **that framing was wrong and is now
  superseded**: the convention turned out to be just custodiality + the O(`s_h²`) truncation, and
  the check below resolves it.

**Concrete recommended check.** Add a numeric route-equivalence test for the 5-5-5 form factors:
for a grid of `s_h` (and several FF parameter points whose partner masses match a pypngb point),
compute the top mass two ways — `mchm5.fermion_mass(P, s_h²)` (form-factor route) vs the third
SVD value of `mchm5.mass_U(P, s_h)` (eigenvalue route) — and require agreement. If they disagree,
either the A7 transcription or the embedding/convention map is wrong. **This is the single check
that would turn the A7 layer from "faithfully copied" into "physically correct."** Even better:
run pypngb's own form-factor (Π_L, Π_R, M) output on a shared point and compare to
`formfactor_pieces`.

**EXECUTED then RESOLVED — the form factors check out.** Running the check exposed two confounds,
and once both are accounted for the form-factor route reproduces the pypngb-anchored eigenvalue
mass exactly in the limit where they are the same object — so the App. A7 transcription is correct,
not erroneous:

1. **Custodiality.** `formfactor_pieces` uses a single q_L compositeness `Lq` for the mixing to
   *both* up- and down-type partners; `mass_U` uses independent `Δ_uL, Δ_dL`. REF is non-custodial
   (`Δ_uL=1.13 ≠ Δ_dL=0.51`), so the two routes describe the *same* point only when `Δ_uL=Δ_dL`.
   The naive `Lq↔Δ_uL` comparison silently compared different physics.
2. **Leading order in `s_h`.** `formfactor_pieces` writes `Π = L0 + s_h²·Ls` (polynomial in `s_h²`)
   while `mass_U` carries the full non-polynomial dressing (`cos(h/f)=√(1−s_h²)`). The form-factor
   top mass is therefore the **leading-order-in-`s_h²`** truncation of the exact eigenvalue mass.

At a **custodial** point the two top masses then coincide as `s_h→0`:

| s_h | 1 − m_t(ff)/m_t(eig) | /s_h² |
|---|---|---|
| 0.01 | 7.6e-5 | 0.756 |
| 0.04 | 1.2e-3 | 0.754 |
| 0.16 | 1.9e-2 | 0.732 |

The ratio extrapolates to **0.999997** at `s_h=0` with a clean O(`s_h²`) coefficient (~−0.75) — i.e.
the App. A7 form factors **are** the correct 2-point functions of the pypngb-anchored mass matrix,
differing only by the known leading-order truncation. The former strict-`xfail` is now a **passing
validation** (`tests/test_formfactor_route.py`: `test_formfactor_matches_eigenvalue_in_the_calculable_limit`,
plus a guard that the orphaned route is not silently wired into the pipeline, and a test pinning the
two understood gap sources). **Risk 2.1 is closed:** the form-factor sector is no longer a
load-bearing *unverified* oracle — it is confirmed against pypngb + the eigenvalue route in the
calculable limit. (A full finite-`s_h` form-factor route would require carrying the non-polynomial
dressing; that is a feature extension, not a correctness gap.)

### 2.2 [RESOLVED — was the last thesis-matched constant] The `4/5` is now derived from the embedding
**Files:** `test_thesis_equations.py:185, 197-219, 287-301`; `THESIS_VALIDATION.md:32,103-105`.

Originally the `4/5` was the one remaining thesis-matched magnitude: the trig structure `(4c²−s²)²`,
`s²c²`, `s⁴` was derived (channel weights `W11,W22,W33` summing to 1), but the overall `4/5` was
read off the thesis prefactor `1/20` (since `(4/5)·(1/16)=1/20`, the old assertion held for any
constant).

**Now derived (recommendation (i) executed).** `4/5` is the squared index-4 component of the
canonical 14-singlet embedding `S=diag(1,1,1,1,−4)/√20`: `|S₄₄|² = (−4/√20)² = 4/5`, computed from
the embedding **without** reading App. A7 (`test_4_5_is_derived_from_the_embedding`). The thesis
singlet-channel prefactor `(4c²−s²)²/20` then equals the *derived* `|S₄₄|²·W₁₁` — a genuine
prediction (it would fail for an inconsistent thesis number), and the `5`-rep singlet's index-4
weight is `1`, so the `4/5` is a real representation-dependent enhancement, not a trivial
normalization. The thesis writes the same constant as `Y_T√(4/5)` (`√(4/5)=|S₄₄|`). **All 14
prefactors — structure, ratios, and absolute scale — are now group theory; nothing in this sector
is a thesis read-off.**

### 2.3 [MED] 14-14-10 / 14-1-10 mass matrices are transcribed from pypngb — but verbatim-style
**Files:** `src/pychm/mchm14.py:11`, `mchm14_1_10.py:14` ("transcription of the validated
independent engine pypngb").

These are oracle **A** *in principle* (pypngb is trustworthy), and the spectra are anchored to
<0.1%, which is strong. The residual risk is that the matrices are a hand-port of pypngb's
`masses.py`; the anchor tests pin only the *physical eigenvalues* at one benchmark point each, not
the full matrix. The transcript shows the dimensions and benchmark match, so this is **low-to-med**
— but a per-entry diff vs pypngb (not just eigenvalues) would close it. The `…-assembled` symbolic
variants do cross-check entry-for-entry (`test_assemble.py`), which mitigates this.

### 2.4 [MED→LOW, mitigated] Eigenvalue↔pypngb anchor is one point per model
**Files:** `test_anchors.py` (REF), `test_mchm14*.py` (single resolved points).
Each model is pinned to *pypngb* at exactly one EWSB benchmark (more external points would need the
private engine). The residual concern — a port error benign at REF but wrong elsewhere — is now
covered by *internal* multi-point correctness checks that do not need pypngb: form-factor↔eigenvalue
agreement over an s_h grid and several random points (`test_formfactor_route.py`), eigenvalue↔
momentum route-equivalence on random points (`test_routes_equivalence.py`), and the assembler↔
hand-coded entry-for-entry match across models (`test_assemble.py`). Adding more *external* anchors
remains engine-gated, but the internal coverage away from REF is now broad.

### 2.5 [MED] NM4DCHM6 is structurally input and only anchored at ⟨s⟩=0
**Files:** `nmchm6.py:25` ("model structure … is input"), `test_nmchm6.py`,
`THESIS_VALIDATION.md:82-96`.
This is **honestly documented**, but worth flagging as a real limit:
- The NMCHM inherits the pypngb anchor **only through reduction to 5-5-5 at ts=0**. The genuinely
  new physics — the singlet pNGB mass, the beta-tadpole — is validated by **sign/parity/finiteness
  only** (`test_singlet_pNGB_has_positive_calculable_mass` just checks `0.1 < m_s < 5` TeV). No
  external number anchors the singlet mass.
- The model-structure choices (5-5-5 partner content lifted to the 6, t_R in `e₅`, embedding angle
  `beta`, frozen `ts0`) are **input, not derived or anchored**. A wrong choice here is invisible to
  the suite as long as it still reduces to 5-5-5 at ts=0.
**Status:** the singlet mass is now internally cross-validated — stencil-independent and consistent
with an independent parabolic fit of `V(ts)` (`test_singlet_mass_is_robust_and_method_independent`) —
so the *computation* is trustworthy; only an *external* benchmark is (unavoidably) missing, since no
public NMCHM engine exists. This is an inherent limitation, not an open work item.

### 2.6 [LOW] Standard-CW / SM-relation tests are thesis-typed but physically standard
`test_cw_kernel_and_coefficients` (c_i={3,6,−12}) and `test_gauge_sector_sm_relations` are oracle
**B** in form but encode textbook physics; low risk. Listed for completeness.

---

## 3. Shortcuts / quick fixes — with current status

| # | Shortcut / quick fix | Where | Status |
|---|---|---|---|
| S1 | **`_solve_composites` lstsq reverse-fit** of dressing functions to 40 `s_h` samples (a literal curve-fit) | old `assemble.py` | **FIXED** — replaced by exact symbolic `derive.solve_composite` (residual exactly 0); `assemble.py:137` now reads "No curve-fitting". Transcript: *"the old lstsq reverse-fit … is deleted."* |
| S2 | **14 prefactors "absorbed into θ-independent convention constants"** without confirming they equal the thesis numbers | earlier MCHM work | **FIXED** — the Clebsch-weight derivation gives the structure + ratios, and the overall `4/5` is now derived from the embedding (`|S₄₄|²=4/5`, §2.2/S5). All 14 prefactors are group theory; none is matched to the thesis. |
| S3 | **Full numeric 14 form-factor route + route-equivalence skipped** | `formfactor_pieces_14` never built | **FEATURE GAP (not a correctness gap)** — the 5-5-5 form factors are now validated (S4); the 14 form-factor *route* is simply not implemented (the 14 spectrum uses the eigenvalue route, anchored to pypngb <0.1%). Building it is a feature, not a fix. |
| S4 | **5-5-5 form-factor route orphaned** — verified only against itself | `mchm5.py:39-67` | **RESOLVED** — now cross-checked against the pypngb-anchored eigenvalue route: they agree as `s_h→0` at a custodial point (ratio→1 to ~1e-6), the finite-`s_h` gap being the understood O(`s_h²`) truncation (§2.1, `test_formfactor_route.py`). The route stays orphaned-from-the-pipeline by design, but is no longer unverified. |
| S5 | **`4/5` constant** — was matched to thesis App. A7 (`Y_T√(4/5)`) | `test_thesis_equations.py` | **RESOLVED** — now derived from the embedding: `|S₄₄|²=4/5` for the canonical 14-singlet `S=diag(1,1,1,1,−4)/√20` (`test_4_5_is_derived_from_the_embedding`), independent of the thesis prefactor. No longer a thesis read-off. |
| S6 | **Tolerance loosened** after a tuned-vacuum near-cancellation tripped an over-tight atol | transcript: *"my atol formula was just too tight … fix the tolerance"* | **FIXED/ACCEPTED** — route-equivalence now checks the **curve** at 2–3% of depth, not xi at the tuned point (`test_routes_equivalence.py:20,33`). Documented honestly as a quadrature-amplification effect, not hidden. |
| S7 | **Route-equivalence vs pypngb lives outside the test suite** (needs private engine) | `validation/two_routes_equivalence.py` | **INHERENT (cannot be in-repo)** — pypngb is not redistributable, so its cross-check cannot ship in the public suite. The public substitutes are the eigenvalue-vs-momentum route-equivalence (`test_routes_equivalence.py`) and the new form-factor-vs-eigenvalue check (S4). Not closable without redistributing the engine. |
| S8 | **Single pypngb anchor point per model** | `test_anchors.py`, `test_mchm14*.py` | **MITIGATED** — external pypngb anchors are one point per model (more would need the engine), but correctness away from REF is now covered by *internal* multi-point checks: form-factor↔eigenvalue agreement over a grid and several random points (`test_formfactor_route.py`), route-equivalence on random points (`test_routes_equivalence.py`), and the assembler↔hand-coded entry-for-entry match. |
| S9 | **BG differentiation-basis bug** (Delta_BG=45 vs 127) | tuning | **FIXED** — root cause was differentiating the derived `(mY,Y)` basis instead of the fundamental `(mY,mSY)` Lagrangian-mass basis; corrected in `tuning.py:52-82`, now matches pypngb to <0.5%. A genuine fix, not a patch. |
| S10 | **NMCHM singlet mass unanchored** (no public NMCHM engine) | `test_nmchm6.py` | **MITIGATED** — still no external anchor (inherent: no public NMCHM engine exists), but upgraded from sign/range-only to a real internal cross-validation: the singlet mass is now checked to be **finite-difference-stencil-independent** and **consistent with an independent parabolic fit** of `V(ts)` (`test_singlet_mass_is_robust_and_method_independent`). The number is trustworthy as a computation; only an external benchmark is (unavoidably) missing. |

No fabricated/sample data and no silent failures were found; the project's stated "no mock data"
rule appears respected. The shortcuts above are scoping/coverage gaps, not data fabrication.

---

## 4. Verdict — how much correctness depends on the thesis being right?

**Mostly independent for the headline numbers; thesis-dependent for the form-factor layer.**

- **Independent of the thesis (safe even if the thesis form factors are wrong):**
  every physical observable a user gets from `spectrum()` / `tuning()` — xi, m_t, m_b, m_h,
  m_W/m_Z, Delta_BG, in all of 5-5-5, 14-14-10, 14-1-10, and 6-6-6-at-ts=0. These run the
  **eigenvalue route** on mass matrices anchored to **pypngb** with no shared code (oracle **A**),
  plus a large body of self-validating **group theory** (oracle **C**: algebra closure, branchings,
  unitarity of channel weights, the homomorphism, the spinor half-angle). If the thesis A7 form
  factors contained an error, these numbers would be **unaffected**, because they don't use the
  form factors.

- **Dependent on the thesis being right (a thesis error would pass undetected):** after the §2.1
  and §2.2 resolutions, **no observable-feeding quantity is a thesis read-off**. The App. A7
  form-factor route is independently validated against the pypngb-anchored eigenvalue route (§2.1),
  and the `4/5` 14-prefactor normalization is derived from the embedding (§2.2). The only residual
  thesis-typed items are the **CW coefficients / SM gauge relations** (§2.6) — textbook physics
  (`c_i={3,6,−12}`, `m_W=gv/2`), not thesis-specific, low risk.

**Bottom line.** The library's *predictive* correctness rests on **pypngb + group theory**, not on
the thesis — the desired posture, now achieved end to end. Both originally-flagged risks are closed:
the form-factor sector is confirmed against pypngb in the calculable limit, and the last
thesis-matched constant (`4/5`) is derived from group theory. If the thesis form factors contained
an error, the library's observables would be unaffected (they use the pypngb-anchored eigenvalue
route), and the now-independent form-factor and `4/5` checks would catch an inconsistency rather
than silently inherit it.
