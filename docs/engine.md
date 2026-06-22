# The Coset(G,H) engine

Every model in pyCHM is an instance of one abstraction: a symmetric coset \(G/H\) built from its Lie
algebra. The group-specific code (MCHM, NMCHM, SU(4)/Sp(4), SU(5)/SO(5)) is *data* fed to a common
engine in `pychm.groups`, not duplicated machinery.

## What a coset needs

A composite-Higgs coset is fixed by three ingredients:

1. **The algebra of \(G\)** — a basis of generators (Hermitian, trace-normalised \(\mathrm{Tr}\,T^aT^b=\tfrac12\delta^{ab}\)).
2. **A Cartan involution \(\theta\)** — the automorphism whose \(\theta=-1\) eigenspace is the
   unbroken subalgebra \(\mathfrak h\) and whose \(\theta=+1\) eigenspace is the broken (coset)
   generators \(X\). For the symmetric spaces here:
     - **type-AII** (SU(N)/Sp(N)): \(\theta(X) = \Omega \bar X \Omega^{-1}\) with the symplectic form \(\Omega\);
     - **type-AI** (SU(N)/SO(N)): \(\theta(X) = \bar X\) (complex conjugation);
     - SO(N)/SO(N−1): the broken generators are the last row/column.
3. **A representation** carrying the fermions, in which the Goldstone matrix dresses the masses.

## The pieces

`pychm.groups` provides these as composable functions:

- **`lie`** — generators and matrix exponentials. `so_generator`, `su_generators` (Gell-Mann basis),
  `symplectic_form`, `cartan_split(n, involution)` (diagonalises \(\theta\) as an operator on the
  algebra and returns the unbroken/broken split), plus closed-form (`rodrigues_exp`) and numerical
  (`expm_num`) exponentials.
- **`coset`** — the `Coset` class. Given the generators it builds the Goldstone matrix
  \(U(\pi) = \exp(i\,\pi\!\cdot\!X/f)\) via `goldstone(angles)`, and exposes invariants:
  `casimir`, `algebra_closes`, `branch_dims`. Factories `so_coset(n)`, `su_sp_coset(n)`,
  `su_so_coset(n)`.
- **`tensors` / `reps`** — tensor irreps (`tensor_basis(symmetry, rank, n, group=...)`) and the
  Goldstone action on them. The `group='SO'|'SU'` switch toggles \(\delta\)-trace removal so that,
  e.g., the SU(5) symmetric-2 is the **15** while the SO(5) symmetric-2 is the **14**.
- **`branch`** — decomposition of a rep under a subgroup (custodial \(SO(4)\)), multiplicity-aware.
- **`landscape`** — the [enumerator](COSET_LANDSCAPE.md): scans SO(N)/SO(N−1) and SU(N)/SO(N),
  branches the coset under custodial \(SO(4)\), and flags those containing a \((2,2)\) Higgs.

## Adding a coset

```python
from pychm.groups.coset import su_so_coset
cos = su_so_coset(5)          # SU(5)/SO(5): 24 - 10 = 14 broken generators
cos.algebra_closes()          # sanity: [h, h] in h, [h, X] in X
U = cos.goldstone([0.1]*14)   # the Goldstone matrix for a pNGB direction
```

A new model is then a choice of representation for the fermions plus the mass spectrum that the
dressed currents generate — see [Extending pyCHM](EXTENDING_BSM.md). The full per-symbol reference is
in the [API: the group engine](api/groups.md).
