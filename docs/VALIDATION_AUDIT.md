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
to pypngb with no shared code — that part does **not** depend on the thesis being right. But the
**App. A7 form-factor layer** (`_AL/_AR/_AM/_B`, `formfactor_pieces`, `fermion_mass`) and the
**14 form-factor prefactors** are validated *only* by transcription-from-thesis and by a
group-theory identity that is rescaled by a constant **read off the thesis (`4/5`)**. If those
thesis equations contain an error, **nothing in the suite would catch it**, because the
form-factor route is never run against pypngb or against the eigenvalue masses.

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

### 2.1 [HIGH] The A7 form-factor route is validated ONLY by transcription, never numerically
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
- The transcript confirms this was a deliberate skip:
  *"I did not build the full numeric form-factor route (`formfactor_pieces_14`) and prove
  route-equivalence … reconstructing the convention mapping is a genuine rabbit hole with low
  marginal value."*

**Concrete recommended check.** Add a numeric route-equivalence test for the 5-5-5 form factors:
for a grid of `s_h` (and several FF parameter points whose partner masses match a pypngb point),
compute the top mass two ways — `mchm5.fermion_mass(P, s_h²)` (form-factor route) vs the third
SVD value of `mchm5.mass_U(P, s_h)` (eigenvalue route) — and require agreement. If they disagree,
either the A7 transcription or the embedding/convention map is wrong. **This is the single check
that would turn the A7 layer from "faithfully copied" into "physically correct."** Even better:
run pypngb's own form-factor (Π_L, Π_R, M) output on a shared point and compare to
`formfactor_pieces`.

**EXECUTED (this audit round) — result: NOT reconciled.** The check above was run on REF with the
natural map `Lq,Lt,Lb ↔ Δ_uL,Δ_uR,Δ_dL` (`tests/test_formfactor_route.py`):

| s_h | m_t eigenvalue (pypngb-anchored) | m_t form-factor (A7) | ratio |
|---|---|---|---|
| 0.05 | 0.03103 | 0.02877 | 0.927 |
| 0.265 | 0.16277 | 0.14380 | 0.884 |
| 0.60 | 0.33957 | 0.25918 | 0.763 |

The ratio is **s_h-dependent** (0.93→0.76), and a 3-parameter fit of `Lq,Lt,Lb` cannot remove it
(rms residual ~6%, the optimiser drives `Lt,Lb → ~1e7` — degenerate). So **no constant parameter
mapping reconciles the two routes**; their `s_h`-dependence genuinely differs. Three explanations
remain entangled and cannot be separated without the full convention derivation: (i) the routes
compute *different objects* — the form-factor route is the `p=0` **pole** mass
`M/√(Π_LΠ_R)` (with wavefunction renormalisation), the eigenvalue route a **tree-level** singular
value — a real, `s_h`-dependent physics difference, **not** necessarily an error; (ii) an unresolved
parameter/convention mapping; (iii) an actual App. A7 transcription / thesis issue. Verdict
unchanged and made concrete: the form-factor route is **faithfully transcribed but not independently
validated**. This is now recorded honestly as a **strict-`xfail`** test
(`test_formfactor_route_matches_eigenvalue`) plus a guard that the orphaned route is not silently
wired into the pipeline (`test_formfactor_route_is_orphaned_from_the_pipeline`), rather than hidden
behind tautological self-checks. Fully resolving it (deriving the pole-vs-tree relation and the
convention map, or cross-checking against pypngb's own `Π_L,Π_R,M`) is the open item — and is the
right place for the thesis-owner to check whether an App. A7 form factor is itself wrong.

### 2.2 [HIGH] The `4/5` normalization is matched to the thesis, not independently derived
**Files:** `test_thesis_equations.py:185, 197-219, 287-301`; `THESIS_VALIDATION.md:32,103-105`.

The claim "the 14 prefactors are *derived* up to a single constant `4/5` that the thesis writes
explicitly" is half true and worth stating precisely:
- The **trig structure** `(4c²−s²)²`, `s²c²`, `s⁴` *is* derived (group-theoretic channel weights
  `W11,W22,W33`, `test_…:205-210`, summing to 1 — genuine **C**).
- The **magnitude** is not. `test_…:212` asserts `(4c²−s²)²/20 == (4/5)·W11` with
  `W11=(4c²−s²)²/16`. Since `(4/5)·(1/16)=1/20`, this is an **algebraic identity for any constant**
  — it tests that `4/5 = (1/20)/(1/16)`, i.e. it *reads the ratio off the thesis prefactor* `1/20`.
  `_ratio_is_constant` only certifies the ratio is θ-independent; it does **not** derive its value.
- So if the thesis's `1/20` (its `4/5` coupling/d-factor) were wrong, every one of these tests
  would still pass, because the thesis number appears on *both* sides of the comparison.

**Recommended check.** Pin `4/5` from an independent source: either (i) derive the t_R-in-14
coupling/d-factor from the embedding normalization in the symbolic engine and show it equals `4/5`
*without* reading App. A7, or (ii) anchor a 14-14-10 form-factor observable (e.g. a Π ratio at two
s_h values) to pypngb. Until then, `4/5` is an **input matched to the thesis**, and
`THESIS_VALIDATION.md` should not call the prefactors "derived."

### 2.3 [MED] 14-14-10 / 14-1-10 mass matrices are transcribed from pypngb — but verbatim-style
**Files:** `src/pychm/mchm14.py:11`, `mchm14_1_10.py:14` ("transcription of the validated
independent engine pypngb").

These are oracle **A** *in principle* (pypngb is trustworthy), and the spectra are anchored to
<0.1%, which is strong. The residual risk is that the matrices are a hand-port of pypngb's
`masses.py`; the anchor tests pin only the *physical eigenvalues* at one benchmark point each, not
the full matrix. The transcript shows the dimensions and benchmark match, so this is **low-to-med**
— but a per-entry diff vs pypngb (not just eigenvalues) would close it. The `…-assembled` symbolic
variants do cross-check entry-for-entry (`test_assemble.py`), which mitigates this.

### 2.4 [MED] Eigenvalue↔pypngb anchor is one point per model, not a grid
**Files:** `test_anchors.py` (REF), `test_mchm14*.py` (single resolved points).
Each model is pinned at exactly one EWSB benchmark. `test_ewsb_fires_for_some_points` and the
random-point loops check finiteness/keys, not correctness, away from REF. A thesis/port error that
happens to be benign at REF but wrong elsewhere would survive. **Recommend** a small grid of
pypngb-cross-checked points per model.

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
**Recommend** stating in the README that NMCHM singlet observables are *unanchored* (the doc already
says so; the README's "inherits the anchor" phrasing oversells it for the new sector).

### 2.6 [LOW] Standard-CW / SM-relation tests are thesis-typed but physically standard
`test_cw_kernel_and_coefficients` (c_i={3,6,−12}) and `test_gauge_sector_sm_relations` are oracle
**B** in form but encode textbook physics; low risk. Listed for completeness.

---

## 3. Shortcuts / quick fixes — with current status

| # | Shortcut / quick fix | Where | Status |
|---|---|---|---|
| S1 | **`_solve_composites` lstsq reverse-fit** of dressing functions to 40 `s_h` samples (a literal curve-fit) | old `assemble.py` | **FIXED** — replaced by exact symbolic `derive.solve_composite` (residual exactly 0); `assemble.py:137` now reads "No curve-fitting". Transcript: *"the old lstsq reverse-fit … is deleted."* |
| S2 | **14 prefactors "absorbed into θ-independent convention constants"** without confirming they equal the thesis numbers | earlier MCHM work | **PARTIALLY FIXED** — upgraded to the Clebsch-weight derivation, but the residual `4/5` is still *matched to the thesis*, not independently derived (§2.2). The honest admission is in the transcript: *"I did not confirm those prefactors equal your thesis's numbers."* |
| S3 | **Full numeric 14 form-factor route + route-equivalence skipped** | `formfactor_pieces_14` never built | **STILL PRESENT** — deliberately not done ("rabbit hole, low marginal value"). This is the §2.1/§2.2 gap for the 14. |
| S4 | **5-5-5 form-factor route orphaned** — `fermion_mass`/`formfactor_pieces` not wired into spectrum; verified only against themselves | `mchm5.py:39-67` | **STILL PRESENT** — no numeric cross-check to eigenvalue route or pypngb (§2.1). |
| S5 | **`4/5` constant** — asserted via `_ratio_is_constant` (θ-independence only), value taken from thesis App. A7 (`Y_T√(4/5)`) | `test_thesis_equations.py:185` | **STILL PRESENT** as an input, not a derivation (§2.2). |
| S6 | **Tolerance loosened** after a tuned-vacuum near-cancellation tripped an over-tight atol | transcript: *"my atol formula was just too tight … fix the tolerance"* | **FIXED/ACCEPTED** — route-equivalence now checks the **curve** at 2–3% of depth, not xi at the tuned point (`test_routes_equivalence.py:20,33`). Documented honestly as a quadrature-amplification effect, not hidden. |
| S7 | **Route-equivalence vs pypngb lives outside the test suite** (needs private engine) | `validation/two_routes_equivalence.py` | **STILL PRESENT** — in-repo `test_routes_equivalence.py` only does eigenvalue-vs-momentum (internal), not vs pypngb. |
| S8 | **Single anchor point per model** | `test_anchors.py`, `test_mchm14*.py` | **STILL PRESENT** (§2.4). |
| S9 | **BG differentiation-basis bug** (Delta_BG=45 vs 127) | tuning | **FIXED** — root cause was differentiating the derived `(mY,Y)` basis instead of the fundamental `(mY,mSY)` Lagrangian-mass basis; corrected in `tuning.py:52-82`, now matches pypngb to <0.5%. A genuine fix, not a patch. |
| S10 | **NMCHM singlet mass unanchored** (positivity/range only) | `test_nmchm6.py:65-73` | **STILL PRESENT** by necessity (no public NMCHM engine); documented (§2.5). |

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

- **Dependent on the thesis being right (a thesis error would pass undetected):**
  1. the **App. A7 building blocks** `_AL/_AR/_AM/_B` and the whole 5-5-5 **form-factor route**
     (`formfactor_pieces`, `fermion_mass`) — validated by transcription + self-consistency only
     (§2.1, S4);
  2. the **14-14-10 form-factor prefactors'** absolute normalization via the `4/5` constant
     (§2.2, S5) — the trig is derived, the magnitude is matched to the thesis;
  3. the **CW coefficients / SM gauge relations** typed from thesis equations (low risk, standard
     physics, §2.6).

**Bottom line.** The library's *predictive* correctness rests on **pypngb + group theory**, not on
the thesis — which is the desired posture. The thesis is still treated as an oracle in exactly one
load-bearing-but-orphaned place: the **form-factor sector**, whose numbers are never confronted with
pypngb or with the library's own eigenvalue masses. The highest-value remediation is a single
numeric **form-factor ↔ eigenvalue (and ↔ pypngb) route-equivalence** test (§2.1) plus an
independent derivation of `4/5` (§2.2). Until then, "the form factors are validated" should read
"the form factors are faithfully transcribed."
