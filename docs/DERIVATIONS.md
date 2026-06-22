# pyCHM derivations — first-principles group theory of the (N)MCHM

This note derives, from group theory alone, every quantity that `pyCHM` treats as **derived** (as
opposed to model input, or numerically anchored to the independent `pypngb` engine). Each step is
tied to the function that implements it and the test that verifies it in closed form, so the
document and the code are one object: **if a derivation here is wrong, a named test fails.** The
reference physics is Murnane, *The Landscape of Composite Higgs Models* (arXiv:2606.18364);
conventions match the library exactly. Nothing here assumes the thesis is correct — the closed
forms are recomputed from the generators, and where the thesis is the only source (the App. A7 form
factors) it is cross-checked against the `pypngb`-anchored eigenvalue route ([§5](#5-the-form-factor-route-and-the-oracle-boundary)).

> This is the human-readable companion to `docs/DERIVATIONS.tex` / `DERIVATIONS.pdf`. The two are
> kept in sync (see the CLAUDE.md rule: *all new derivations go in both markdown and LaTeX*).

Notation: each section ends with **▶ Implemented in / Verified by**, naming the code and the test.

---

## 1. The CCWZ framework

A composite Higgs is a set of pseudo-Nambu–Goldstone bosons (pNGBs) of a spontaneously broken
global symmetry $G\to H$. Split the generators of $\mathfrak g=\mathrm{Lie}(G)$ into unbroken
$T^a\in\mathfrak h$ and broken $X^{\hat a}\in\mathfrak g/\mathfrak h$. The pNGBs $\pi^{\hat a}(x)$
parameterise the coset through the **Goldstone matrix**

$$U(\pi)=\exp\!\Big(\tfrac{i}{f}\,\pi^{\hat a}X^{\hat a}\Big),$$

with $f$ the decay constant. Fermions are embedded in (generally incomplete) representations $R$ of
$G$; the Higgs dependence of every elementary–composite mixing is the matrix element of $U$ in $R$,

$$\big\langle c\,\big|\,U_R(\pi)\,\big|\,E\big\rangle,$$

between the embedding $E$ of the elementary field and a composite state $c$. The central claim of
the library: **these matrix elements are the only source of the $s_h$-dependence, and they are
computable in closed form from the generators — never fitted.** The rest of this document carries
that out for $\mathrm{SO}(5)/\mathrm{SO}(4)$ (MCHM) and $\mathrm{SO}(6)/\mathrm{SO}(5)$ (NMCHM).

---

## 2. Minimal Composite Higgs: SO(5)/SO(4)

### 2.1 Generators and conventions

$\mathrm{SO}(5)$ acts on indices $0,\dots,4$. The unbroken $\mathrm{SO}(4)$ acts on $0,\dots,3$; the
coset direction is index $4$. In the vector (the **5**) the Hermitian generators are

$$\big(T^{AB}\big)_{CD}=-i\big(\delta_{AC}\delta_{BD}-\delta_{AD}\delta_{BC}\big),\qquad 0\le A<B\le4.$$

The six $T^{ab}$ ($a<b\le3$) generate $\mathrm{SO}(4)$; the four broken generators are
$X^{\hat a}=T^{\hat a 4}$. With the Higgs vev along $\hat a=3$, only the physical Higgs $h$ survives
in unitary gauge and $U$ becomes a single planar rotation.

**▶ Implemented in:** `groups.so5.gen_vector`, `groups/so5.py` — **Verified by:** `test_so4_unbroken_generators_close_the_algebra`

### 2.2 The vector Goldstone matrix

Exponentiating $X^3=T^{34}$ with angle $\theta=h/f$ gives a rotation in the $(3,4)$ plane,

$$U_{\mathbf 5}(\theta)=\exp\!\big(i\theta\,T^{34}\big)=\begin{pmatrix}\mathbb 1_3&&\\&c_h&s_h\\&-s_h&c_h\end{pmatrix},\qquad s_h\equiv\sin\theta,\ \ c_h\equiv\cos\theta=\sqrt{1-s_h^2}.$$

The library stores everything in $s_h=\sin(h/f)$ with $c_h=\sqrt{1-s_h^2}$, avoiding an $\arcsin$
round-trip (machine precision). This is the thesis "explicit Goldstone matrix."

**▶ Implemented in:** `groups.groups.so5.U_vector`, `groups.so5.U_vector_sym` — **Verified by:** `test_U_vector_matches_thesis_goldstone`

### 2.3 Lifting to tensor representations

A rank-$k$ $\mathrm{SO}(5)$ tensor irrep is the symmetric-traceless or antisymmetric part of
$\mathbf 5^{\otimes k}$. The Goldstone acts on each index,

$$\big(U_R\big)_{ba}=\big\langle E_b\,\big|\,U_{\mathbf 5}^{\otimes k}E_a\big\rangle=\mathrm{Tr}\!\big(E_b^{\dagger}\,U E_a U^{\top}\big)\quad(\text{rank }2),$$

in an orthonormal basis $\{E_a\}$. The first three rungs are the **5** (rank 1), the **10**
(antisymmetric rank 2), the **14** (symmetric-traceless rank 2). The construction is
dimension-agnostic, so the same code gives the **30** (rank-3 symmetric) and, with $n=6$, the
$\mathrm{SO}(6)$ tower of [§3](#3-next-to-minimal-composite-higgs-so6so5).

**▶ Implemented in:** `tensors.tensor_basis`, `tensors.U_rep_tensor`, `groups.so5.U_rep` — **Verified by:** `test_tensors.py` (dimension, unitarity, homomorphism)

### 2.4 SO(4) = SU(2)_L × SU(2)_R branching

The unbroken $\mathrm{SO}(4)$ on $0,\dots,3$ is $\mathrm{SU}(2)_L\times\mathrm{SU}(2)_R$ via the
't Hooft combinations

$$J^{L/R}_i=\tfrac12\Big(\tfrac12\,\epsilon_{ijk}T^{jk}\pm T^{0i}\Big),\qquad i=1,2,3.$$

Diagonalising the commuting Casimirs $C_L=\sum_i(J^L_i)^2$, $C_R=\sum_i(J^R_i)^2$ labels each
sub-multiplet by $(j_L,j_R)$ with $C=j(j+1)$:

$$\mathbf 5=(\tfrac12,\tfrac12)\oplus(0,0),\quad \mathbf{10}=(\tfrac12,\tfrac12)\oplus(1,0)\oplus(0,1),\quad \mathbf{14}=(1,1)\oplus(\tfrac12,\tfrac12)\oplus(0,0).$$

**▶ Implemented in:** `decompose.so4_decompose`, `decompose._su2_casimirs` — **Verified by:** `test_branchings_match_thesis`

### 2.5 Dressings are matrix elements (derive, don't fit)

A claimed closed form $f(\theta)$ is admissible only if it lies in the span of the single-basis
overlaps $\{\langle E_b|U_R|E\rangle\}$. The library solves for the exact minimal-norm composite
tensor $c$ with $\langle c|U_R|E\rangle=f$ and **raises** if $f$ is not realisable. This is the
from-scratch guarantee: every dressing factor is *proven* to be a Goldstone matrix element, with
symbolic residual exactly $0$.

**▶ Implemented in:** `groups/derive.py::solve_composite` — **Verified by:** `test_dressings_are_matrix_elements`

### 2.6 The 5-5-5 coefficients

Embed $E_{q_L}=(e_0+e_3)/\sqrt2$ (bidoublet) and $E_{t_R}=e_4$ (SO(4) singlet). From the rotation,
$Ue_4=s_h e_3+c_h e_4$, $Ue_3=c_h e_3-s_h e_4$, $Ue_0=e_0$, so

$$\langle q_L|U_{\mathbf 5}|t_R\rangle=\frac{s_h}{\sqrt2},\qquad \big|\langle q_L|U|t_R\rangle\big|^2=\frac{s_h^2}{2},\qquad \langle q_L|U_{\mathbf 5}|q_L\rangle=\tfrac12(1+c_h)=\cos^2\!\tfrac{h}{2f}.$$

These are exactly the $\Pi_L\sim s_h^2/2$ and $\cos^2(h/2f)$ structures of the thesis 5-5-5 form
factors — here **derived** from the overlaps.

**▶ Implemented in:** `groups.so5.overlap_sym`, `mchm5.formfactor_pieces` — **Verified by:** `test_coeffs_5_5_5`

### 2.7 The 14 channel weights and the 4/5 enhancement

The $t_R$ embeds in the SO(4) singlet of the **14**, the unit-normalised symmetric-traceless tensor

$$S=\frac{1}{\sqrt{20}}\,\mathrm{diag}(1,1,1,1,-4),\qquad \mathrm{Tr}\,S^2=\frac{4+16}{20}=1.$$

Dressing $S\mapsto USU^{\top}$ and projecting onto the SO(4) channels of the **14** gives the closed
trigonometric **channel weights**

$$W_{11}=\frac{(4c_h^2-s_h^2)^2}{16},\qquad W_{22}=\frac52 s_h^2 c_h^2,\qquad W_{33}=\frac{15}{16}s_h^4,$$

which satisfy unitarity exactly:

$$W_{11}+W_{22}+W_{33}=\Big(1-\tfrac52 s_h^2+\tfrac{25}{16}s_h^4\Big)+\tfrac52 s_h^2(1-s_h^2)+\tfrac{15}{16}s_h^4=1.$$

The thesis writes the $t_R$ self-energy with an overall coupling $Y_T\sqrt{4/5}$. That normalisation
is **not** an external input — it is the squared index-4 component of the embedding itself:

$$\boxed{\ \big|S_{44}\big|^2=\Big(\tfrac{-4}{\sqrt{20}}\Big)^2=\frac{16}{20}=\frac45\ }\qquad\Longleftrightarrow\qquad \sqrt{\tfrac45}=|S_{44}|.$$

Index 4 is the coset-singlet direction through which the top mass is generated; the **5**-singlet
has weight $1$ there (the unenhanced reference), so $4/5$ is a genuine representation-dependent
enhancement. The thesis singlet-channel prefactor then follows as a **prediction**,

$$\frac{(4c_h^2-s_h^2)^2}{20}=|S_{44}|^2\,W_{11}=\frac45\cdot\frac{(4c_h^2-s_h^2)^2}{16},$$

which would fail for any inconsistent thesis number — no longer a tautology that reads $4/5$ off the
prefactor $1/20$.

**▶ Implemented in:** `groups.so5.channel_weights_sym`, `groups.so5.embedding_sym` — **Verified by:** `test_14_channel_weights_derive_thesis_prefactors`, `test_4_5_is_derived_from_the_embedding`

---

## 3. Next-to-Minimal Composite Higgs: SO(6)/SO(5)

### 3.1 Coset and broken generators

$\mathrm{SO}(6)$ acts on $0,\dots,5$. The unbroken $\mathrm{SO}(5)$ acts on $0,\dots,4$; the coset
direction is index $5$. The five broken generators split, under $\mathrm{SO}(4)\subset\mathrm{SO}(5)$,
into the Higgs bidoublet and a singlet:

$$X_B^{a}=T^{a5}\ (a=0,1,2,3)\ \ (\text{the doublet } h_1,\dots,h_4),\qquad X_S=T^{45}\ \ (\text{the singlet } s).$$

This is the thesis decomposition (its 6th index is our index 5). The 15 generators close
$\mathfrak{so}(6)$; the ten $T^{ab}$ with $a<b\le4$ are the unbroken $\mathrm{SO}(5)$.

**▶ Implemented in:** `so6.gen_vector`, `so6.UNBROKEN/BROKEN` — **Verified by:** `test_so6_generators_close_the_algebra`, `test_coset_split_10_plus_5`

### 3.2 The five-pNGB Goldstone matrix (closed form)

In unitary gauge the Higgs points along $\hat a=3$ and the singlet along $X_S$, so the vector
Goldstone is generated by $G=\theta_h L^{(3,5)}+\theta_s L^{(4,5)}$ with $\theta_h=h/f$,
$\theta_s=s/f$. $G$ is supported on $\{3,4,5\}$ and has rank 2, so $G^3=-\Theta^2 G$ with
$\Theta=\sqrt{\theta_h^2+\theta_s^2}$. **Rodrigues' formula** gives the closed form

$$U_6(\theta_h,\theta_s)=\mathbb 1+\frac{\sin\Theta}{\Theta}\,G+\frac{1-\cos\Theta}{\Theta^2}\,G^2.$$

Acting on the vacuum $e_5$ reproduces the thesis Goldstone field (its eq. 474):

$$\Phi=U_6 e_5=\Big(0,0,0,\ \theta_h\tfrac{\sin\Theta}{\Theta},\ \theta_s\tfrac{\sin\Theta}{\Theta},\ \cos\Theta\Big)^{\top}=\frac{\sin\Theta}{\Theta}\big(h_1,\dots,h_4,s,\ \Theta\cot\Theta\big),$$

with $\Theta=\varphi/f$, $\varphi=\sqrt{\sum h_i^2+s^2}$. Setting $\theta_s=0$ returns the MCHM
planar rotation (coset index $4\to5$).

**▶ Implemented in:** `so6.U6_vector`, `so6.U6_vector_sym`, `so6.goldstone_vacuum` — **Verified by:** `test_goldstone_vacuum_matches_thesis_eq474`, `test_U6_vector_orthogonal_identity_homomorphism`

### 3.3 The representation tower

The fermion-partner irreps of $\mathrm{SO}(6)\cong\mathrm{SU}(4)$ are built and dressed by the same
machinery — tensors via $n=6$, spinors via the Clifford algebra:

| irrep | construction | dim | SO(5) branching |
|---|---|---|---|
| **6** | vector | 6 | $\mathbf 5\oplus\mathbf 1$ |
| **15** | antisym. rank 2 (adjoint) | 15 | $\mathbf{10}\oplus\mathbf 5$ |
| **20'** | sym.-traceless rank 2 | 20 | $\mathbf{14}\oplus\mathbf 5\oplus\mathbf 1$ |
| **10**, **10̄** | (anti)self-dual 3-form | 10 | $\mathbf{10}$ (each) |
| **4**, **4̄** | Weyl spinor | 4 | $\mathbf 4$ |

The Casimir calibration $C_5\in\{0,\tfrac52,4,6,10\}\leftrightarrow\{\mathbf 1,\mathbf 4,\mathbf 5,\mathbf{10},\mathbf{14}\}$
identifies each sub-irrep with its multiplicity. The antisymmetric 3-form is 20-dimensional and
splits, by the $\pm i$ eigenspaces of the Hodge dual
$(\star T)_{ijk}=\tfrac{1}{3!}\epsilon_{ijklmn}T_{lmn}$, into the self-dual **10** and anti-self-dual
**10̄** (the $\mathrm{SU}(4)$ $\mathbf{10}+\overline{\mathbf{10}}$). The spinors come from six
$8\times8$ Euclidean gammas $\Gamma^A=(\sigma_1\otimes\gamma^a_{\mathrm{SO}(5)},\,\sigma_2\otimes\mathbb 1)$
split by chirality $\chi=\sigma_3\otimes\mathbb 1$; the **4** carries no $(\tfrac12,\tfrac12)$
bidoublet — exactly the thesis reason the NM4DCHM uses the **6** (not the **4**) for $q_L$.

**▶ Implemented in:** `so6.rep_basis`, `so6.U6_rep`, `so6.so5_content`, `so6_spinors.py` — **Verified by:** `test_so6.py`, `test_so6_spinors.py`

### 3.4 Two-field dressing and the singlet pNGB

The **6** Goldstone dresses overlaps with *both* angles. With $E_{q_L}=(e_0+e_3)/\sqrt2$ and
$E_{t_R}=e_5$, no embedding has support on index 4, so every overlap is even in $\theta_s$: the
potential is even in $s$, the vacuum sits at $\langle s\rangle=0$, and the singlet mass is the
$s$-curvature of the Coleman–Weinberg potential there. Because the $q_L$ bidoublet dressing carries
explicit $\theta_h$-dependence distinct from the radial $\Theta$ the $t_R$ feels, $V(h,s)$ is
genuinely two-dimensional and the singlet acquires a finite, calculable mass — the defining NMCHM
observable, absent in the MCHM. At $\theta_s=0$ the mass matrices reduce **entry-for-entry** to the
`pypngb`-anchored 5-5-5 model.

**▶ Implemented in:** `nmchm6.py` (`spec`, `assemble`, `potential`, `singlet_mass2`) — **Verified by:** `test_nmchm6.py` (reduction, even-in-$s$, singlet mass)

---

## 4. The Coleman–Weinberg potential and observables

The one-loop pNGB potential is the supertraced kernel

$$V(s_h)=\sum_i\frac{c_i}{64\pi^2}\,m_i^2(s_h)^2\log m_i^2(s_h),\qquad c_i=\{+3,+6,-12\}$$

for {neutral vector, charged vector, coloured Dirac}, with $m_i(s_h)$ the eigenvalues of the
$s_h$-dependent mass matrices. Writing $V=-\gamma s_h^2+\beta s_h^4$,

$$\xi=\sin^2\frac{\langle h\rangle}{f}=\frac{\gamma}{2\beta},\qquad m_h^2=\frac{8\beta}{f^2}\xi(1-\xi),\qquad v_{\mathrm{EW}}=f\sqrt\xi,$$

and the Barbieri–Giudice tuning is $\Delta_{\mathrm{BG}}=\max_i|\partial\ln f/\partial\ln x_i|$ over
the fundamental Lagrangian masses. These relations are validated symbolically; the full mass
matrices and the resulting $\xi,m_t,m_h,\Delta_{\mathrm{BG}}$ are anchored to `pypngb` to $<0.1\%$.

**▶ Implemented in:** `routes.py`, `potential.py`, `spectrum.py`, `tuning.py` — **Verified by:** `test_cw_kernel_and_coefficients`, `test_higgs_mass_scaling_closed_form`, `test_anchors.py`

---

## 5. The form-factor route and the oracle boundary

The thesis App. A7 form factors $A_L,A_R,A_M,B$ are *transcribed* in `mchm5`; transcription proves
faithful copying, not physical correctness. They are confronted with the `pypngb`-anchored
eigenvalue route by comparing the top mass two ways. With the **custodial** parametrisation the
form-factor route requires (a single $q_L$ coupling, i.e. $\Delta_{uL}=\Delta_{dL}$), the two
coincide as $s_h\to0$,

$$\frac{m_t^{\mathrm{ff}}(s_h)}{m_t^{\mathrm{eig}}(s_h)}=1-0.755\,s_h^2+\mathcal O(s_h^4)\ \xrightarrow{\ s_h\to0\ }\ 0.999997,$$

the finite-$s_h$ gap being the form-factor route's leading-order-in-$s_h^2$ truncation of the exact
(non-polynomial) eigenvalue mass. The App. A7 form factors are therefore the correct leading-order
two-point functions of the `pypngb`-anchored mass matrix — **validated, not merely transcribed.**

**▶ Implemented in:** `mchm5.formfactor_pieces`, `mchm5.fermion_mass` — **Verified by:** `test_formfactor_route.py`

---

## 6. Beyond SO(N): the coset abstraction and SU(4)/Sp(4)

Everything above is built from one group-agnostic engine (`pychm.groups`): a coset $G/H$ is a list of
unbroken generators of $H$ and broken generators $X^a$ in some representation, and the Goldstone is
$U(\pi)=\exp(i\pi^a X^a)$ — in unitary gauge a rank-≤2 rotation, so the closed Rodrigues form serves
every case. MCHM and NMCHM are the instances SO(5)/SO(4) and SO(6)/SO(5); the **same code** builds
SU(N)/Sp(N) cosets, with the SU(N) Gell-Mann generators and the USp(N) subalgebra from the Cartan
involution $\theta(X)=\Omega\bar X\Omega^{-1}$ (usp = the $\theta=-1$ eigenspace, coset = $\theta=+1$).

**SU(4)/Sp(4) ≅ SO(6)/SO(5).** The minimal pseudoreal Ferretti–Sannino coset has $15-10=5$ broken
generators — the same five pNGBs as SO(6)/SO(5). Under SU(4)≅Spin(6), Sp(4)≅Spin(5) the
fundamental $\mathbf 4$ = SO(6) spinor, the antisymmetric $\mathbf 6=[\mathbf 4\otimes\mathbf 4]_A$ =
SO(6) vector, and $\mathbf{15}$ = SO(6) adjoint. The library builds SU(4)/Sp(4) **independently** and
cross-checks it against the SO(6) code (an exact oracle): the $\mathbf 4$ branches to a single
$\mathbf 4$ of Sp(4), the $\mathbf 6$ to $\mathbf 1\oplus\mathbf 5$, matching the SO(6) spinor and
vector; and — the strong statement of the isomorphism — the $\mathbf 6$ generators span the full
15-dim so(6)≅su(4) algebra and act irreducibly (commutant = scalars, by Schur), so the SU(4)
antisymmetric $\mathbf 6$ *is* the irreducible SO(6) vector. The new coset reproduces the NMCHM group
theory while sharing all of its tooling.

**▶ Implemented in:** `groups/lie.py` (`su_generators`, `sp_subalgebra`), `groups/coset.py`, `groups/su4sp4.py` — **Verified by:** `test_su4sp4.py`

---

## 7. SU(5)/SO(5) (littlest Higgs) and the coset landscape

SU(5)/SO(5) is a type-AI symmetric space: the Cartan involution `θ(X)=conj(X)` splits su(5) into the
unbroken so(5) (10 imaginary-antisymmetric generators, θ=−1) and the coset (14 real-symmetric, θ=+1)
— the same engine (`lie.cartan_split`) that built SU(4)/Sp(4). The 14 pNGBs are the
symmetric-traceless **14** of SO(5), branching under custodial SO(4) as

$$\mathbf{14}=(3,3)\oplus(2,2)\oplus(1,1)=\text{complex triplet }\phi\ \oplus\ \text{Higgs doublet }h\ \oplus\ \text{singlet }\eta,$$

the littlest-Higgs content. The Goldstone in the **5** is `U=exp(i θᵃXᵃ)` (a complex unitary, unlike
the real SO(N) rotation); along the Higgs axis the `(2,2)` overlap magnitude reduces to the MCHM
`|⟨q_L|U|t_R⟩|²=sin²(h/f)/2`, so the top/Higgs match the 5-5-5 there. The symmetric **15** of SU(5)
(the fuller partner rep) is built with the SU(N) tensor basis (no δ-trace removal,
`tensor_basis(...,group='SU')`) and branches `15 = 14 ⊕ 1` under SO(5). The new observables are the
singlet and triplet pNGB masses — curvatures of the multi-field CW potential at the vacuum (even in
η,φ, so ⟨η⟩=⟨φ⟩=0).

**▶ Implemented in:** `groups/lie.py` (`cartan_split`), `groups/su5so5.py`, `su5so5.py` — **Verified by:** `test_su5so5.py`

**The landscape.** A composite Higgs needs a custodial `SO(4)=SU(2)_L×SU(2)_R ⊂ H` and a `(2,2)`
Higgs in `G/H`. Chala & Fonseca (arXiv:2309.10635) enumerate all compact `G/H` with ≤13 pNGBs
satisfying this — **642 models, complete up to 13 NGBs**. `groups.landscape` reproduces a slice from
pyCHM's own `Coset`: branch each coset under the custodial `SO(4)`, flag the `(2,2)`, recovering MCHM
`(2,2)`, NMCHM `(1,1)+(2,2)`, littlest-Higgs `(1,1)+(2,2)+(3,3)`, and discovering SO(7)/SO(6),
SO(8)/SO(7) (see [`COSET_LANDSCAPE.md`](COSET_LANDSCAPE.md)).

**▶ Implemented in:** `groups/landscape.py` — **Verified by:** `test_landscape.py`

---

## Provenance: derived vs. input vs. anchored

- **Derived (this document):** the Goldstone matrices $U,U_6$; the rep lifts; the SO(4)/SO(5)
  branchings; the 5-5-5 coefficients; the **14** channel weights and the $4/5=|S_{44}|^2$
  enhancement; the full SO(6) tower. Each closes symbolically (a named test asserts
  `simplify(...) == 0`).
- **Input (model structure):** the partner content — which composite multiplets exist, the
  embeddings, the partner masses. Taken from the thesis / `pypngb`, not derived.
- **Anchored:** the full mass matrices and the physical observables, validated against `pypngb`
  ($<0.1\%$). The NMCHM has no public engine and is anchored indirectly by its exact reduction to
  the 5-5-5 at $\langle s\rangle=0$, with the new singlet observable cross-checked internally
  (stencil independence + parabolic fit).

After §2.7 and §5, **no quantity that feeds an observable is a thesis read-off**: correctness rests
on `pypngb` + group theory.
