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
  SO(4) Clebsch weights up to a **single** overall coupling constant (`4/5`), which the thesis
  itself writes explicitly (`Y_T√(4/5)`, App. A7 `eq:broken14-14-10`).
- The thesis contains **no per-point numerical benchmark tables**; it tabulates parameter domains
  and scan-level results. "Numerical validation" therefore means the closed-form physical
  relations, plus the existing anchor tests (`tests/test_anchors.py`, `test_mchm14*.py`) that pin
  the library to the independent pypngb engine to <0.1%.

Legend: ✅ validated in closed form (test named) · 🔵 validated numerically/by anchor ·
⚪ out of scope (not implemented in pyCHM).

## Chapter / appendix coverage

| Source | Equation class | pyCHM | Status |
|---|---|---|---|
| Ch.2, A0, A1 | CCWZ Goldstone `U`, SO(5) generators `T_L,T_R,X`, lifts to 10/14, spinor `U_4` | `ccwz`, `symbolic/core`, `symbolic/spinors` | ✅ `test_U_vector_matches_thesis_goldstone`, `test_so5_broken_generator_exponentiates_to_U_vector`, `test_so4_unbroken_generators_close_the_algebra`, `test_spinor_4_half_angle` |
| A1 | 14/10 basis tensors `T̂^0,T̂^{ab},X̂^a`; SO(4) branchings 5/10/14/4 | `symbolic/core`, `symbolic/decompose` | ✅ `test_14_basis_matches_thesis`, `test_10_basis_antisymmetric_orthonormal`, `test_branchings_match_thesis` |
| A7 (`eq:formulas`) | form-factor building blocks `A_L,A_R,A_M,B` | `mchm5._AL/_AR/_AM/_B` | ✅ `test_A7_building_blocks_verbatim` (verbatim, `simplify==0`) |
| Ch.6 / A7 (`eq:broken5-5-5`) | 5-5-5 `s_h`-coefficients (`s_h²/2`, `c_h²`, `½s_h²c_h²`) | `mchm5.formfactor_pieces` | ✅ `test_coeffs_5_5_5`, `test_coeffs_5_5_5_formfactor_wiring` |
| Ch.6 / A7 (`eq:broken14-14-10`) | 14 form-factor `s_h`-prefactors (`(4c²−s²)²/20`, `4c²/5+s²/20`, `3/(4√5)`, `1/(2√5)`) | `symbolic/decompose.channel_weights_sym` | ✅ `test_14_channel_weights_derive_thesis_prefactors`, `test_14_formfactor_structure_ties_to_channel_weights` (exact Clebsch weights × the explicit thesis `4/5`) |
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
| Ch.7 | NMCHM `SO(6)/SO(5)`, the singlet pNGB, 5-component `Φ` | — | ⚪ not implemented |
| A6 | large-N scaling, meson sum rule | — | ⚪ not implemented (implicit in the form-factor structure only) |
| Ch.1, A0 (defs) | SM review, Lie-group definitions | — | ⚪ review material, nothing to validate against code |

## What "validated" means here

- **Closed-form (✅):** a symbolic identity `sp.simplify(pyCHM_expr − thesis_expr) == 0`, or an
  exact numeric identity between two pyCHM quantities that encode a thesis equation.
- **Derivation, not transcription:** the 14 form-factor prefactors are produced by
  `channel_weights_sym` (SO(4) projection of the Goldstone-dressed embedding), so they are exact
  Clebsch factors; the single `4/5` is the t_R coupling/d-factor normalization, which the thesis
  writes explicitly in App. A7 (`Y_T√(4/5)`).
- **Anchored (🔵):** validated numerically against the independent pypngb engine (the same engine
  behind arXiv:2101.00428), not re-derived from the thesis equations.
- **Out of scope (⚪):** the library does not implement this physics, so there is no code to check
  the thesis equation against. This is the honest boundary of the validation.
