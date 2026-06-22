# Models

pyCHM ships four cosets, all built on the same [`Coset(G,H)` engine](engine.md) and exposed through
`pychm.Model(<string>)`. Each is a global symmetry breaking \(G \to H\) whose pseudo-Nambu–Goldstone
bosons contain the Higgs doublet; they differ in the size of \(G/H\) (hence the extra pNGBs) and in
the fermion representations that generate the potential.

| String | Coset | pNGBs | Extra states |
|---|---|---|---|
| `5-5-5`, `14-14-10`, `14-1-10` | SO(5)/SO(4) | 4 = Higgs | — |
| `6-6-6` | SO(6)/SO(5) | 5 = Higgs + singlet | CP-odd singlet \(\eta\) |
| `5-5-su5`, `15-15-su5` | SU(5)/SO(5) | 14 = Higgs + triplet + singlet | complex triplet \(\phi\), singlet \(\eta\) |
| (engine cross-check) | SU(4)/Sp(4) | 5 = Higgs + singlet | (≅ NMCHM by invariants) |

## MCHM — SO(5)/SO(4)

The minimal composite Higgs. The coset has exactly **four** pNGBs forming the custodial \((2,2)\)
doublet — nothing else. The fermion representation sets how the top sector feeds the Coleman–Weinberg
potential:

- **`5-5-5`** — the two-site M4DCHM with partners in the fundamental **5**. This is the validated
  reference (arXiv:2606.18364 / 2101.00428); the benchmark reproduces an independent mass-eigenvalue
  engine to 0.03% on \(\xi\).
- **`14-14-10`**, **`14-1-10`** — partners in the symmetric **14** and the antisymmetric **10**;
  alternative top embeddings with different tuning.

The `*-assembled` variants build the same mass matrices block-by-block from the CCWZ form factors as
a cross-check of the direct construction.

## NMCHM — SO(6)/SO(5)

The next-to-minimal coset adds **one** pNGB: a gauge-singlet (CP-odd) \(\eta\), a dark-matter / extra
scalar candidate. The Higgs sector is unchanged; the model predicts \(m_\eta\) as a curvature of the
potential. `pychm.Model('6-6-6')` shares the 5-5-5 Lagrangian-mass basis (the singlet alignment angle
\(t_s = 0\) at the vacuum).

## SU(5)/SO(5) — littlest Higgs / Ferretti

A real-representation coset with **14** pNGBs decomposing under custodial \(SO(4)\) as

\[
\mathbf{14} \;=\; (1,1)\ \oplus\ (2,2)\ \oplus\ (3,3)
\;=\; \text{singlet} \ \oplus\ \text{Higgs} \ \oplus\ \text{complex triplet}.
\]

Beyond \(\xi\), \(m_t\), \(m_h\) the model therefore predicts the **triplet and singlet pNGB masses**
(`su5so5.triplet_mass2`, `su5so5.singlet_mass2`). Two fermion variants: partners in the fundamental
**5** (`5-5-su5`) and in the symmetric **15** (`15-15-su5`).

## SU(4)/Sp(4)

Carried by the engine as a proof that the `Coset(G,H)` abstraction is group-agnostic. It is
physically equivalent to NMCHM (5 pNGBs = Higgs + singlet); pyCHM cross-checks the two cosets by
their invariants — branchings under custodial \(SO(4)\), the span dimension (15) and irreducibility of
the broken algebra — rather than by a spectrum match.

See [the coset landscape](COSET_LANDSCAPE.md) for where these sit in the full finite classification,
and the [API reference](api/model.md) for the per-model functions.
