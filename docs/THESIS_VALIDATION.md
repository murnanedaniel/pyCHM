# Thesis validation coverage map

This documents, chapter by chapter, how pyCHM is validated against Murnane's thesis
*The Higgs as a Composite Goldstone Boson* (arXiv:2606.18364) — what is checked in closed form,
and what is **out of scope** because the library does not implement it. It exists to make the
validation claim precise rather than sweeping.

Honest summary up front:
- The **Goldstone/CCWZ dressing** (the `s_h`-dependence of every fermion mixing) is *derived from
  group theory*, not transcribed — see `pychm.symbolic`. The model *structure* (embeddings,
  partner spectrum, mass placements) is input taken from the thesis, not derived.
- The fermion-sector and convention equations are validated **exactly** in closed form
  (`tests/test_thesis_equations.py`). The higher-rep form-factor prefactors are derived as exact
  SO(4) Clebsch weights, with the overall `4/5` itself derived from the 14-singlet embedding
  (`|S₄₄|²=4/5`) — group theory end to end. (The earlier text said this constant was matched to the
  thesis; it is now derived — see `VALIDATION_AUDIT.md §2.2`.) The thesis writes the same constant
  explicitly (`Y_T√(4/5)`, App. A7 `eq:broken14-14-10`).
- The thesis contains **no per-point numerical benchmark tables**; it tabulates parameter domains
  and scan-level results. "Numerical validation" therefore means the closed-form physical
  relations, plus the existing anchor tests (`tests/test_anchors.py`, `test_mchm14*.py`) that pin
  the library to the independent pypngb engine to <0.1%.

Legend: ✅ validated in closed form (test named) · 🔵 validated numerically/by anchor ·
◐ partially validated (structure derived, overall magnitude matched to the thesis) ·
⚪ out of scope (not implemented in pyCHM).

## Chapter / appendix coverage

| Source | Equation class | pyCHM | Status |
|---|---|---|---|
| Ch.2, A0, A1 | CCWZ Goldstone `U`, SO(5) generators `T_L,T_R,X`, lifts to 10/14, spinor `U_4` | `ccwz`, `symbolic/core`, `symbolic/spinors` | ✅ `test_U_vector_matches_thesis_goldstone`, `test_so5_broken_generator_exponentiates_to_U_vector`, `test_so4_unbroken_generators_close_the_algebra`, `test_spinor_4_half_angle` |
| A1 | 14/10 basis tensors `T̂^0,T̂^{ab},X̂^a`; SO(4) branchings 5/10/14/4 | `symbolic/core`, `symbolic/decompose` | ✅ `test_14_basis_matches_thesis`, `test_10_basis_antisymmetric_orthonormal`, `test_branchings_match_thesis` |
| A7 (`eq:formulas`) | form-factor building blocks `A_L,A_R,A_M,B`; the 5-5-5 form-factor route | `mchm5._AL/_AR/_AM/_B`, `formfactor_pieces`, `fermion_mass` | ✅ `test_A7_building_blocks_verbatim` (transcription) **+ now independently validated**: at a custodial point the form-factor top mass equals the pypngb-anchored eigenvalue top mass as `s_h→0` (ratio→1 to ~1e-6), confirming the A7 form factors are the correct leading-order 2-point functions — not just transcribed (`test_formfactor_route.py`, `VALIDATION_AUDIT.md §2.1`). The finite-`s_h` gap is the understood custodial + O(`s_h²`) truncation, not a thesis error. |
| Ch.6 / A7 (`eq:broken5-5-5`) | 5-5-5 `s_h`-coefficients (`s_h²/2`, `c_h²`, `½s_h²c_h²`) | `mchm5.formfactor_pieces` | ✅ `test_coeffs_5_5_5`, `test_coeffs_5_5_5_formfactor_wiring` |
| Ch.6 / A7 (`eq:broken14-14-10`) | 14 form-factor `s_h`-prefactors (`(4c²−s²)²/20`, `4c²/5+s²/20`, `3/(4√5)`, `1/(2√5)`) | `symbolic/decompose.channel_weights_sym`, `core.embedding_sym` | ✅ **fully derived**: the trig structure + channel ratios are exact SO(4) Clebsch weights (`W₁₁+W₂₂+W₃₃=1`), and the overall `4/5` is **derived from the embedding** — `\|S₄₄\|²=4/5` for `S=diag(1,1,1,1,−4)/√20` (`test_4_5_is_derived_from_the_embedding`), so `(4c²−s²)²/20 = \|S₄₄\|²·W₁₁` is a genuine prediction, not a thesis read-off (`VALIDATION_AUDIT.md §2.2`). |
| Ch.6 / A7 (`eq:broken14-1-10`) | 14-1-10 t_R-singlet: no up-right dressing, constant `M_u` | `assemble`, `mchm14_1_10` | ✅ `test_coeffs_14_1_10_tR_singlet_constant` |
| Ch.5 (`5-M4DCHM.tex:459`) | CW kernel `V=Σ c_i/(64π²) m_i⁴ log m_i²`, `c_i={3,6,−12}` | `routes._K_closed`, `_CF/_CV` | ✅ `test_cw_kernel_and_coefficients` |
| Ch.5 (`5-M4DCHM.tex:518`) | pole mass `m=M(0,v)/√(Π_LΠ_R)` | `mchm5.fermion_mass` | ✅ `test_pole_mass_eq518` |
| Ch.5 | `V=−γs_h²+βs_h⁴`, vacuum `ξ=γ/2β`, `m_h²=(8β/f²)ξ(1−ξ)`, `v_EW=f√ξ` | `potential`, `spectrum` | ✅ `test_higgs_mass_scaling_closed_form`, `test_vev_and_vacuum_relations` |
| Ch.5 | gauge sector SM relations `m_W=gv/2`, `m_Z=√(g²+g'²)v/2`, `m_W/m_Z=cosθ_W` | `spectrum` | ✅ `test_gauge_sector_sm_relations` |
| Ch.3 (`eq:BG`) | Barbieri–Giudice `Δ_BG=max|J_i|`; first-order `|J|`; information `½log(1+|J|²)=KL` | `tuning` | ✅ `test_fine_tuning_measures` |
| Ch.5/6 | full 14/14-1-10 fermion mass matrices (eigenvalue route), physical `m_t,m_b,m_h,ξ,Δ_BG` | `mchm14`, `mchm14_1_10`, `spectrum`, `tuning` | 🔵 anchored to pypngb <0.1% (`test_anchors`, `test_mchm14`, `test_mchm14_1_10`); symbolic engine reproduces the matrices entry-for-entry (`test_assemble`, `test_symbolic`) |
| Ch.5 (`eq.366/370`) | gauge form factor `Π_W(p,s_h)`, `γ_g` (the momentum-dependent two-point function) | `mchm5.mass2_W/mass2_Z` (eigenvalue masses only) | 🔵 the eigenvalue-route gauge masses are used and anchored; the full momentum-dependent `Π_W` is not separately validated |
| Ch.3 / A2 (`eq.217–263`) | higher-order HOT `Δ_2,…,Δ_N` (Gram-determinant volumes), pseudo-determinant | — | ⚪ not implemented (library has BG + first-order `|J|` only) |
| Ch.3 / A2 (`eq.27,130`) | Bayesian evidence `Z`, Occam factor, Athron volume ratios | — | ⚪ not implemented |
| Ch.4 | nested sampling / MultiNest, likelihood scans | — | ⚪ not implemented (external) |
| Ch.7/8 | NMCHM `SO(6)/SO(5)`, the 5-component Goldstone `Φ` (eq. 474), broken generators (eq. 632), the 6-embedding (eq. 633), the singlet pNGB | `symbolic/so6`, `symbolic/so6_spinors`, `nmchm6` | ✅ `test_so6.py`, `test_so6_spinors.py`, `test_nmchm6.py` (closed-form Goldstone + reps + branchings; model reduces to the anchored 5-5-5 and adds the singlet mass) |
| A6 | large-N scaling, meson sum rule | — | ⚪ not implemented (implicit in the form-factor structure only) |
| Ch.1, A0 (defs) | SM review, Lie-group definitions | — | ⚪ review material, nothing to validate against code |

## NMCHM / SO(6) representation coverage (Ch.7/8)

The Next-to-Minimal coset `SO(6)/SO(5)` and **all** its small irreps are implemented in
`pychm.symbolic.so6` / `so6_spinors` and validated to the same bar as the SO(5) layer — closed
form against the thesis where the thesis is explicit, and by internal group-theory consistency
otherwise. The Goldstone dressing is *derived* (Rodrigues closed form for the 5-pNGB vector
Goldstone, then lifted to every rep by the rank-`k` tensor / Clifford construction), not
transcribed.

| SO(6) irrep | built as | dim | SO(5) branching | test |
|---|---|---|---|---|
| `6` vector | `tensor_basis('sym',1,6)` | 6 | `5 + 1` | ✅ `test_rep_dimension_and_branchings` |
| `15` adjoint | `tensor_basis('antisym',2,6)` | 15 | `10 + 5` | ✅ `test_rep_dimension_and_branchings` |
| `20'` sym-traceless | `tensor_basis('sym',2,6)` | 20 | `14 + 5 + 1` | ✅ `test_rep_dimension_and_branchings` |
| `10` / `10bar` self-dual 3-form | `tensor_basis('antisym',3,6)` ± Hodge dual | 10 | `10` (SO(5) adjoint) | ✅ `test_three_form_splits_into_10_and_10bar` |
| `4` / `4bar` Weyl spinor | 8-dim Clifford, chirality split | 4 | `4` (no bidoublet) | ✅ `test_spinor_branchings_match_thesis` |

Closed-form thesis identities (`sp.simplify(...) == 0` or exact numeric):
- **eq. 474** — the Goldstone field `Φ = U6·e₅ = (1/φ)sin(φ/f)(h₁..h₄, s, φ cot(φ/f))`
  (`test_goldstone_vacuum_matches_thesis_eq474`).
- **eq. 632** — the broken generators (bidoublet `X_B^a` + singlet `X_S`) exponentiate to the
  vector Goldstone (`test_broken_generators_exponentiate_to_goldstone`); the 15 SO(6) generators
  close the algebra (`test_so6_generators_close_the_algebra`).
- **eq. 633** — the fundamental `6 = 4 + 1 + 1` embedding basis
  (`test_fundamental_embedding_6_is_4_plus_1_plus_1`).
- the thesis statement that the **4 has no `SU(2)_L×SU(2)_R` bidoublet** (so it cannot host the SM
  `q_L`, the reason the NM4DCHM uses the 6) — `test_spinor_branchings_match_thesis`.
- the **(h,s) Higgs/singlet dressing is derived in closed form**, not fitted: `so6.channel_weights6_sym`
  gives the exact closed-trig SO(4)-channel weights of the dressed `q_L` in the 6, which sum to 1
  (unitarity) and reduce to the MCHM5 `sin²(θ_h)/2` vector weight at `s=0` — the SO(6) analogue of
  the SO(5) `decompose.channel_weights_sym`, and the symbolic Goldstone lambdifies to the numeric
  one bit-for-bit (`test_closed_form_channel_weights_derive_unity_and_mchm_limit`,
  `test_symbolic_goldstone_matches_numeric_bit_for_bit`).

What is *derived* vs *input* vs *anchored* — the honest boundary, identical in spirit to the
MCHM accounting above:
- **Derived (group theory):** the SO(6) generators and coset split, the closed-form 5-pNGB
  Goldstone `U6(h,s)`, every irrep and its `U6` dressing, and the SO(6)→SO(5)→SO(4) branchings.
- **Input (model structure):** the `nmchm6` NM4DCHM6 partner content — the `5-5-5` spectrum lifted
  to the 6 with `t_R` in the SO(5)-singlet `e₅`. This is the standard partial-compositeness
  choice, not a transcription of a thesis table.
- **Anchored:** there is **no public NMCHM engine** (the thesis NM4DCHM6 scan is not reproduced
  here), so the NMCHM model is not anchored to an external benchmark the way the MCHM is. Instead
  it is validated by (i) **exact reduction to the pypngb-anchored 5-5-5 at `⟨s⟩=0`**
  (`test_reduces_to_555_assembler_entry_for_entry`, `test_spectrum_matches_555`,
  `test_tuning_matches_555`) and (ii) the new singlet observable being a finite, positive,
  calculable fermion-loop mass with the vacuum at `⟨s⟩=0` (`test_potential_even_in_singlet…`,
  `test_singlet_pNGB_has_positive_calculable_mass`). This reduction-plus-consistency is the
  honest substitute for an external anchor, and is stated as such — an inherent limitation (no
  public NMCHM engine exists), not an open work item.

## What "validated" means here

- **Closed-form (✅):** a symbolic identity `sp.simplify(pyCHM_expr − thesis_expr) == 0`, or an
  exact numeric identity between two pyCHM quantities that encode a thesis equation.
- **Derivation, not transcription (now complete).** The 14 form-factor prefactors are derived end
  to end: the *trig structure* + channel *ratios* are exact Clebsch weights from `channel_weights_sym`
  (summing to 1), and the overall `4/5` is the **index-4 component of the canonical 14-singlet
  embedding** `S=diag(1,1,1,1,−4)/√20`, `|S₄₄|²=4/5` (`test_4_5_is_derived_from_the_embedding`).
  The thesis prefactor `(4c²−s²)²/20` then equals the *derived* `|S₄₄|²·W₁₁` — a prediction that would
  fail if the thesis number were inconsistent, no longer a read-off of the thesis `1/20`. The thesis
  writes the same constant as `Y_T√(4/5)` (App. A7); it is no longer an unverified input.
- **Oracle honesty — the form-factor route (resolved).** The App. A7 building blocks `A_L,A_R,A_M,B`
  and the 5-5-5 form-factor route (`mchm5.formfactor_pieces`, `fermion_mass`) are **transcribed** from
  the thesis and used by *no* pipeline code (spectrum/tuning/routes run the eigenvalue route). They
  were previously checked only against themselves; they are now **independently validated** against
  the pypngb-anchored eigenvalue route: at a custodial point (`Δ_uL=Δ_dL`, which the single-`Lq`
  form factor assumes) the form-factor top mass equals the eigenvalue top mass as `s_h→0`
  (ratio→1 to ~1e-6), with the finite-`s_h` difference a clean O(`s_h²`) leading-order truncation —
  *not* a thesis error (`VALIDATION_AUDIT.md §2.1`, `test_formfactor_route.py`). The form factors are
  thus confirmed correct in the regime where both routes compute the same object.
- **Anchored (🔵):** validated numerically against the independent pypngb engine (the same engine
  behind arXiv:2101.00428), not re-derived from the thesis equations.
- **Out of scope (⚪):** the library does not implement this physics, so there is no code to check
  the thesis equation against. This is the honest boundary of the validation.
