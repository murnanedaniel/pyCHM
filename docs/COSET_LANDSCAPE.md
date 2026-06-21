# The landscape of composite-Higgs cosets

*Is there a general way to define ALL the model space that yields a natural Higgs from spontaneous
symmetry breaking?* Short answer: **yes, up to a finite number of Nambu–Goldstone bosons it is a
complete, finite classification** — and pyCHM's `Coset(G,H)` abstraction can enumerate a slice of it
(`pychm.groups.landscape`). This note surveys the literature and states the organizing principle.

## The organizing principle
A composite Higgs is the set of pseudo-Nambu–Goldstone bosons (pNGBs) of a global symmetry breaking
`G → H`. For the Higgs to be a *natural*, custodially-protected electroweak doublet, three conditions
must hold:

1. **Custodial `SO(4) ≅ SU(2)_L × SU(2)_R ⊂ H`** — so the electroweak `T` parameter is protected at
   tree level. This is the central physical requirement.
2. **The Higgs is a `(2,2)` of that custodial `SO(4)`** — i.e. the coset `G/H` (the pNGBs) must
   *contain a `(2,2)`*. This is the criterion the enumerator filters on.
3. **The SM gauge group (and hypercharge) embeds in `H`**, with no colored or fractionally-charged
   pNGBs.

Because `G` and `H` are compact (semi-)simple Lie groups and the broken generators form a symmetric
space, **the set of `(G,H)` with a fixed number of pNGBs is finite** and exhaustively enumerable by
scanning the Lie-subgroup lattice.

## The landmark: a complete enumeration (up to 13 NGBs)
- **Chala & Fonseca, "The Landscape of Composite Higgs Models", [arXiv:2309.10635](https://arxiv.org/abs/2309.10635)
  (JHEP 03(2024)017)** — the definitive recent classification. Scanning all compact semi-simple
  `G/H` with **≤ 13 NGBs** and applying the criteria above yields **642 distinct composite Higgs
  models** (after factoring out spurious `U(1)`s). *This is complete within the 13-NGB bound; beyond
  13 NGBs the classification is open.* (Coincidentally the same title as Murnane's 2019 thesis —
  they are different works.)

So the answer to "is the model space completely defined" is: **finite and fully enumerated up to 13
NGBs (642 models); open beyond.** The organizing axes are *compact symmetric spaces* + the
*custodial-`(2,2)`* requirement + *finite subgroup enumeration*.

## The systematic tables (earlier, narrower classifications)
- **Ferretti & Karateev, [arXiv:1312.5330](https://arxiv.org/abs/1312.5330)** (+ Cacciapaglia et al.,
  [arXiv:1902.06890](https://arxiv.org/abs/1902.06890)) — the cosets that admit a **fermionic (gauge)
  UV completion** with top partial compositeness. Exactly **three** EW cosets qualify, by the reality
  of the hyperfermion representation:

  | EW coset | hyperfermion rep | #pNGB | Higgs content |
  |---|---|---|---|
  | **SU(4)/Sp(4)** | pseudoreal | 5 | doublet + singlet |
  | **SU(5)/SO(5)** | real | 14 | doublet + triplet + singlet |
  | **SU(4)×SU(4)/SU(4)** | complex | — | two doublets + … |

- **Mrazek, Pomarol, Rattazzi, Redi, Serra, Wulzer, "The Other Natural Two-Higgs-Doublet Model",
  [arXiv:1105.5403](https://arxiv.org/abs/1105.5403)** — `SO(6)/SO(4)×SO(2)` and the cosets giving
  *two* Higgs doublets.
- **Bellazzini, Csáki, Serra, "Composite Higgses", [arXiv:1401.2457](https://arxiv.org/abs/1401.2457)**
  — the foundational taxonomy (bona-fide composite / little Higgs / holographic / twin / dilatonic).
- **Cacciapaglia, Pica, Sannino, "Fundamental Composite Dynamics: A Review",
  [arXiv:2002.04914](https://arxiv.org/abs/2002.04914)** (Phys. Rept.) — the UV/lattice program,
  focused on SU(4)/Sp(4).
- **Arkani-Hamed, Cohen, Katz, Nelson, "The Littlest Higgs", [hep-ph/0206021](https://arxiv.org/abs/hep-ph/0206021)**
  — `SU(5)/SO(5)`, the 14 pNGBs (doublet + complex triplet + singlet), collective symmetry breaking.

## Where pyCHM sits, and the enumerator
pyCHM implements four cosets, each a `Coset(G,H)` instance on one shared engine:

| pyCHM model | coset | #pNGB | content (custodial SO(4)) |
|---|---|---|---|
| `5-5-5`, … | SO(5)/SO(4) | 4 | `(2,2)` |
| `6-6-6` | SO(6)/SO(5) | 5 | `(1,1)+(2,2)` |
| (cross-check) | SU(4)/Sp(4) | 5 | `(1,1)+(2,2)` *(= SO(6)/SO(5))* |
| `5-5-su5`, `15-15-su5` | SU(5)/SO(5) | 14 | `(1,1)+(2,2)+(3,3)` |

`pychm.groups.landscape.scan()` reproduces a slice of the Chala–Fonseca classification *from pyCHM's
own abstraction*: for each `(G,H)` in the families the engine supports (`SO(N)/SO(N-1)`,
`SU(N)/SO(N)`), it builds the coset, branches it under the custodial `SO(4)`, and flags the `(2,2)`:

```
>>> from pychm.groups import landscape; landscape.print_scan(max_ngb=14)
coset          #pNGB  has (2,2)? content
SO(5)/SO(4)        4  YES        (2,2)
SO(6)/SO(5)        5  YES        (1,1) + (2,2)
SO(7)/SO(6)        6  YES        2x(1,1) + (2,2)
SO(8)/SO(7)        7  YES        3x(1,1) + (2,2)
SU(5)/SO(5)       14  YES        (1,1) + (2,2) + (3,3)
```

This is the capstone of the abstraction: the library that *builds* any coset can also *enumerate* the
viable ones. The natural extensions are the `SU(N)/Sp(N)` family (SU(4)/Sp(4) is reached today via
its SO(6)/SO(5) isomorphism — the NMCHM row) and the product cosets (`SU(N)×SU(N)/SU(N)`), which would
extend the scan toward the full 642-model table.

*All arXiv ids above were verified against their abstract pages.*
