# pyCHM — working guide for Claude

pyCHM is an open, pure-`numpy`/`scipy`/`sympy` Composite Higgs Model calculator: the
Coleman–Weinberg Higgs potential, the electroweak vacuum, the spectrum, and four fine-tuning
measures, for the two-site M4DCHM (SO(5)→SO(4)) and its NMCHM extension (SO(6)→SO(5)). The
reference physics is Murnane's thesis *The Landscape of Composite Higgs Models* (**arXiv:2606.18364**).
There is no other public composite-Higgs potential/fine-tuning calculator, so correctness is judged
against group theory and an independent engine, not against a peer tool.

This file records the **debugging and validation methodology** that actually worked here. Follow it.

## Project layout (what lives where)
- `src/pychm/` — numeric engine: `mchm5`, `mchm14`, `mchm14_1_10`, `nmchm6` (the models);
  `potential`, `spectrum`, `tuning`, `routes` (representation-agnostic downstream); `ccwz` (numeric
  Goldstone dressing); `assemble` (generic mass-matrix assembler).
- `src/pychm/symbolic/` — the **sympy mirror** of `ccwz`: `core`, `derive` (exact composite solver),
  `models`, `tensors` (arbitrary rank-k irreps), `decompose` (SO(4) Casimir decomposition),
  `spinors` (4, 16), `so6` / `so6_spinors` (NMCHM tower). The symbolic engine must reproduce the
  numeric one *bit-for-bit* after `lambdify`.
- `tests/` — `test_thesis_equations.py` (closed-form thesis checks), `test_anchors.py` /
  `test_mchm14*.py` (pypngb numeric anchors), `test_routes_equivalence.py`, plus per-module
  group-theory consistency tests. ~93 tests; keep them all green on Python 3.9/3.11/3.12.
- `docs/THESIS_VALIDATION.md` — the chapter-by-chapter coverage map and the derived/input/anchored
  boundary. **Update it whenever you change what is validated.**

## Validation philosophy (the bar)
A result is only "validated" if it clears one of these bars. Do not use the word otherwise.

1. **Closed-form symbolic identity** (✅): `sp.simplify(pyCHM_expr - thesis_expr) == 0` against a
   *named* thesis equation, or an exact numeric identity between two pyCHM quantities encoding a
   thesis equation. Examples: the Goldstone `U`, the NMCHM Goldstone `Φ = U6·e₅` (eq. 474), the
   broken generators (eq. 632), the `6 = 4+1+1` embedding (eq. 633). **Caveat:** a
   `simplify(code−thesis)==0` against a *transcribed* expression (e.g. the App. A7 blocks
   `A_L,A_R,A_M,B`) only proves faithful copying, **not** physical correctness — that is bar (B)
   "transcription", not validation. Reserve "validated" for derived/anchored/group-theoretic checks.
2. **Anchored to the independent engine** (🔵): reproduce the private **pypngb** engine (behind
   arXiv:2101.00428; *not* redistributed) to **<0.1%** on `ξ`, `m_t`, `m_h`, `Δ_BG`. This is how the
   full eigenvalue-route mass matrices are validated — the thesis has *no per-point numeric tables*.
3. **Internal group-theory consistency** for reps with no oracle: unitarity, the representation
   **homomorphism** (`U_R(θ₁)U_R(θ₂)=U_R(θ₁+θ₂)`), identity at the origin, correct dimension, and the
   correct SO(4)/SO(5) branching.
4. **Route-equivalence**: the `eigenvalue` and `momentum` routes must agree on the potential *curve*
   `V(s_h)` to ~1% of its depth on random points (CI-enforced in `test_routes_equivalence.py`).

Derive, don't fit. Every dressing factor is *proven* to be a Goldstone matrix element `⟨c|U_R|E⟩`
by exact symbolic solve (`symbolic.derive.solve_composite`, residual exactly 0) — the old
least-squares reverse-fit is gone and must not return. Channel weights are **exact Clebsch factors**
(`decompose.channel_weights_sym`, `W₁₁+W₂₂+W₃₃=1`), not tuned numbers.

## The derived / input / anchored taxonomy
State which bucket each claim is in. This is the honest boundary that the user demanded; respect it.
- **Derived (group theory):** the Goldstone/CCWZ dressing — the `s_h`-dependence of every fermion
  mixing — the SO(5)/SO(6) generators and coset split, every irrep and its `U_R` dressing, the
  branchings. For the 14 prefactors the *trig structure/ratios* are derived (Clebsch weights summing
  to 1); the single overall `4/5` is **matched to the thesis** (`Y_T√(4/5)`, App. A7), **not**
  independently derived — see `docs/VALIDATION_AUDIT.md §2.2`. Don't call it "derived".
- **Thesis-as-oracle (the one risk surface):** the App. A7 form-factor route
  (`mchm5.formfactor_pieces`, `fermion_mass`, `_AL/_AR/_AM/_B`) is *transcribed* and used by no
  pipeline code; it is **not** independently validated (form-factor vs eigenvalue/pypngb top mass
  does not yet reconcile — strict-`xfail` in `tests/test_formfactor_route.py`). Headline observables
  do not depend on it. If you touch this layer, resolve the xfail before claiming validation.
- **Input (model structure):** the partner content (embeddings, which SO(4) multiplets exist,
  partner masses `mQ,mU,mY,…`, mass-matrix placements). Taken from the thesis / pypngb, not derived.
- **Anchored:** the full mass matrices and physical observables, validated numerically against
  pypngb. The NMCHM has **no public engine**, so it is anchored *indirectly* by reducing to the
  pypngb-anchored 5-5-5 **entry-for-entry at ⟨s⟩=0**, plus consistency of the new singlet observable.

## Coding & convention rules
- **SO(5)/SO(4):** SO(4) acts on indices **0..3**; the SO(5)/SO(4) coset direction is **index 4**.
  Broken generators `T^{(â,4)}`, unbroken `T^{(a,b)}`, `a<b∈0..3` (see `ccwz.UNBROKEN/BROKEN`).
- **SO(6)/SO(5):** SO(5) acts on **0..4**; the coset direction is **index 5** (`so6.COSET = 5`).
  The thesis's 6th index is our index 5.
- **Goldstone normalization:** `U(h) = exp(i (h/f) T^{(â,4)})` (rotation angle = `h/f`, generator
  `i√2 hᵃTᵃ/f` form). Getting the `√2` / `i/2` factor wrong is the classic first bug — verify by the
  *normalization-independent* singlet overlap `⟨1,1|U₁₄|1,1⟩ = (3+5cos2θ)/8` (machine precision).
- **The `sh` (sine) convention is library-wide:** pass `sh = sin(h/f)`, with `c = sqrt(1-sh²)` — **no
  `arcsin` round-trip** (it reintroduces machine-epsilon that the tuned vacuum amplifies). All of
  `ccwz`, `symbolic`, and the model files use this; keep new code consistent.
- **Model interface:** every model object exposes `mass_U`, `mass_D`, `mass2_W`, `mass2_Z`. The
  downstream layer (`potential`/`spectrum`/`tuning`) dispatches on a `model=` string via `_MODELS`
  and never knows its representation — keep it that way; do not leak rep-specific logic downstream.
- **The assembler pattern:** a model is specified declaratively (partner rep, elementary embeddings,
  composite states); `assemble` emits `mass_U`/`mass_D` with the Higgs dressing supplied by `ccwz`.
  New models should go through it, validated entry-for-entry against a hand-coded oracle when one
  exists. Register assembled variants as `'<name>-assembled'`.
- **`symbolic/` mirrors `ccwz` exactly** (same basis ordering, same conventions) so lambdify
  reproduces the numeric engine bit-for-bit. Lambdify once at import to cached numpy callables; keep
  sympy off the hot path (~4000 builds/tuning must stay fast).
- Use **exact sympy integers/rationals** in symbolic construction (`sp.Rational`, exact `sqrt`), not
  Python floats — floats defeat `simplify(...)==0`.

## Testing patterns
- **Tolerances reflect the physics, not laziness.** Anchor tests pin pypngb values *tight*
  (`ξ` <0.1%, `m_t` <1%, `Δ_BG` ~0.1%). But comparing a *re-minimised tuned vacuum* of two
  bit-identical matrix builders is environment-sensitive: identical matrices to ~2e-16 amplify into
  ~1e-6 on `ξ`, ~6e-5 on `m_h` (different float op-order across numpy/scipy builds). That amplified
  spread *is the fine-tuning*. So: compare the **matrices/potential directly** (atol ~1e-10) and only
  loosely compare the downstream `ξ`/`m_h`. CI on 3.11 has repeatedly caught too-tight tolerances —
  set them from the observed numeric scale, not from hope.
- New reps with no oracle → test by **unitarity + homomorphism + branching + dimension** (see
  `test_tensors.py`, `test_spinors.py`, `test_so6*.py`).
- Closed-form thesis checks live in `test_thesis_equations.py` and must assert against *named*
  equation numbers verified from the source PDF.
- Keep the suite green before every commit. CI runs bare `pytest` (so `pythonpath=["."]` and the
  `tests/__init__.py` package marker matter — don't break collection).

## Debugging playbook (concrete gotchas seen here)
- **SO(4) multiplicity counting (`so4_content` / `so5_content`).** The original counted *distinct*
  `(jₗ,jᵣ)` blocks, which is fine when all multiplicities are 1 (MCHM) but **wrong** for the 20'
  (2×(½,½), 3×singlet) and the antisym-3 **20** (= 2×**10**, a degenerate Casimir eigenspace).
  Identify irreps by **Casimir value with multiplicity** (`C₅ ∈ {0→1, 4→5, 6→10, 10→14}`,
  SO(5) 4 at 2.5), not by listing blocks.
- **Casimir / SU(2) construction signs.** Wrong SU(2) generators give non-canonical Casimirs
  (`0.375` instead of `j(j+1)`). The self-dual term needs `Lᵢ = ε_{ijk}T^{jk}` summed over *both*
  orderings (the full generator, not half). Always check the algebra closes and Casimirs are exactly
  `j(j+1)` before trusting a branching.
- **Structure-constant / broken-generator signs.** Pin these by direct algebra-closure checks, not
  by inspection; a sign error there silently corrupts the coset split.
- **Exact-sympy-vs-float.** `simplify` leaves junk like `conjugate(1/√…)` when a `Piecewise`
  positivity guard hides the sign. Give the symbolic Goldstone a clean **guard-free** form so
  `Σweights = 1` simplifies exactly.
- **Rodrigues / `Piecewise` lambdify pitfalls.** The numeric Rodrigues rotation can hit tiny
  *negative* round-off under a sqrt — clamp it. Lambdifying a `Piecewise` warns `0/0` at the origin;
  drop the `(0,0)` grid point (the identity there is covered elsewhere) rather than fight it.
- **Catastrophic cancellation in the momentum route.** The naive momentum integral has `~Λ⁴`
  divergences dwarfing the `~10⁻³` signal. Because `Str 1, m², m⁴` are all `s_h`-independent, subtract
  a degree-≤2 polynomial **inside the integrand** so it is convergent at integrand level. Even then,
  a *tuned* vacuum amplifies residual quadrature error into several-percent error on `ξ` — which is
  exactly why fine-tuning uses the closed-form `eigenvalue` route. (Central lesson of the project.)
- **Fine-tuning basis.** `Δ_BG` is defined over the **fundamental Lagrangian-mass basis**
  (`{mU,mUt,mY,mSY,mD,…,Δ}`); differentiate there, not in derived quantities. The 14-14-10 holds
  `mSYu=mYu+Yu/2` and `mSYtu=mYu+4(Yu+Ytu)/5` independent of `mYu`. A sign flip in `d ln f` is
  expected (`ln f = −½ ln ξ`). When pypngb's own `Δ_BG` routine won't run (a scipy incompat returns
  ~0), validate `Δ_BG` *by definition* — reproduce the known-good 5-5-5 value with the
  `d ln f/d ln x` convention, then apply the same convention.
- **Verify equation numbers from the source PDF, not from memory or from sub-agent reports.** When
  agents disagreed on numbering, the resolution was always to read `/tmp/thesis` (the PDF) directly.

## Working principles
- **No curve-fitting.** Every dressing factor must be a derived Goldstone matrix element with an
  exact symbolic residual of 0. If you find yourself least-squares fitting `s_h` samples, stop.
- **No oracle/result assumptions.** "Validated" means cleared a bar above. Do not assume an external
  number; reproduce it or scope it as out-of-scope.
- **Verify from source.** Equation numbers, basis tensors (e.g. `T̂⁰ = diag(1,1,1,1,−4)/√20`),
  branchings — confirm against the thesis PDF, not memory.
- **Be honest about scope.** The user pushed back hard on "validated all the expressions in my
  thesis" — it was an overstatement (only a targeted subset; higher-rep absolute normalizations were
  initially unverified). The fix was the precise `docs/THESIS_VALIDATION.md` coverage map with the
  ✅/🔵/⚪ legend and the derived/input/anchored taxonomy. Keep that precision: claim exactly what the
  tests prove, mark out-of-scope physics (HOT Δ₂…, Bayesian evidence, large-N, nested-sampling scans)
  as **⚪ not implemented**, and never sweep "all 218 pages."
- **Commit per logical change; keep the tree green and pushed.** End commit messages with the
  Co-Authored-By / Claude-Session trailers. The signing key in this environment is a 0-byte file, so
  commits cannot be GPG-signed (the "Unverified" badge is unavoidable); just ensure the author is
  `Claude <noreply@anthropic.com>`.
