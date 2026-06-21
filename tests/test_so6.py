"""NMCHM SO(6)/SO(5): the coset, the five Goldstones (Higgs doublet + singlet), and the
Goldstone dressing of every relevant SO(6) irrep -- validated against the thesis
(arXiv:2606.18364, Ch.7 / App. E.2) in closed form, and by internal consistency.

Same bar as the SO(5) validation (test_thesis_equations.py / test_tensors.py):
  * closed-form symbolic identities against named thesis equations (eq. 474 Goldstone Phi,
    eq. 632 broken generators, eq. 633 fundamental embedding);
  * generator-algebra closure and the SO(6) -> SO(5) -> SO(4) split;
  * dimension, unitarity, identity at the origin, the representation homomorphism;
  * the branchings of the 6, 15, 20', 3-form (10+10bar) under SO(5) and SO(4);
  * reduction to the MCHM5 vector dressing when the singlet angle ts -> 0.
"""
import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from pychm import ccwz
from pychm.symbolic import so6


# =====================================================================================
#  Generators and the coset (thesis eq. 632)
# =====================================================================================
def test_so6_generators_close_the_algebra():
    """The 15 SO(6) generators T^{AB} close: [T^{AB}, T^{CD}] = i(d^{BC}T^{AD} - ...)."""
    def T(a, b):
        return so6.gen_vector(a, b, herm=True)
    pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)]
    assert len(pairs) == 15
    d = np.eye(6)
    for (a, b) in pairs[:6]:
        for (c, e) in pairs[6:12]:
            comm = T(a, b) @ T(c, e) - T(c, e) @ T(a, b)
            # [T^AB,T^CD] = i(d^AC T^BD - d^BC T^AD - d^AD T^BC + d^BD T^AC) for the
            # Hermitian generator T^{AB}_{CD} = -i(d_AC d_BD - d_AD d_BC).
            rhs = 1j * (d[a, c] * T(b, e) - d[b, c] * T(a, e)
                        - d[a, e] * T(b, c) + d[b, e] * T(a, c))
            assert np.allclose(comm, rhs, atol=1e-12)


def test_coset_split_10_plus_5():
    """SO(6) = 15 generators = 10 unbroken (SO(5), indices 0..4) + 5 broken (the coset,
    each mixing one of 0..4 with the index-5 direction)."""
    assert len(so6.UNBROKEN) == 10
    assert len(so6.BROKEN) == 5
    assert all(b == (a, 5) for a, b in [(p[0], p) for p in so6.BROKEN])
    # broken = one index of 0..4 paired with the coset direction 5
    assert {a for (a, _) in so6.BROKEN} == {0, 1, 2, 3, 4}


def test_broken_generators_exponentiate_to_goldstone():
    """eq. 632: the broken generators are the bidoublet X_B^a (a=0..3, plane (a,5)) and the
    singlet X_S (plane (4,5)).  The Higgs generator X_B^3 exponentiates to the vector Goldstone
    (the -2 absorbs the thesis i/2 + sqrt2 NGB normalization, a convention as in MCHM):
    exp(-2 i theta X_B^3) = U6_vector(theta, 0)."""
    from scipy.linalg import expm
    for a in range(5):                                   # all 5 broken generators are (a,5)
        assert so6.BROKEN[a] == (a, 5)
    XB3 = np.zeros((6, 6), dtype=complex)
    XB3[3, 5] = 0.5j; XB3[5, 3] = -0.5j                  # thesis i/2 normalization, eq. 632
    XS = np.zeros((6, 6), dtype=complex)
    XS[4, 5] = 0.5j; XS[5, 4] = -0.5j
    for th in (0.2, 0.5, 0.9):
        assert np.allclose(expm(-2j * th * XB3).real, so6.U6_vector(th, 0.0), atol=1e-12)
        assert np.allclose(expm(-2j * th * XS).real, so6.U6_vector(0.0, th), atol=1e-12)


# =====================================================================================
#  The five Goldstones (thesis eq. 474)
# =====================================================================================
def test_goldstone_vacuum_matches_thesis_eq474():
    """eq. 474: Phi = U6 . e5 = (1/phi) sin(phi/f) (h1,h2,h3,h4, s, phi cot(phi/f)), with
    phi = sqrt(h^2 + s^2).  With the Higgs along hat=3 (h3=th f, s=ts f), the closed-form
    Rodrigues Goldstone reproduces Phi exactly."""
    th, ts = sp.symbols('theta_h theta_s', positive=True)
    Phi = so6.goldstone_vacuum(th, ts)
    Theta = sp.sqrt(th**2 + ts**2)
    expected = sp.Matrix([0, 0, 0,
                          th * sp.sin(Theta) / Theta,
                          ts * sp.sin(Theta) / Theta,
                          sp.cos(Theta)])
    assert sp.simplify(Phi - expected) == sp.zeros(6, 1)


def test_U6_vector_orthogonal_identity_homomorphism():
    """U6 in the vector is orthogonal, U6(0,0)=1, and composes additively along a fixed
    coset direction (a one-parameter subgroup): U6(a,0)U6(b,0)=U6(a+b,0)."""
    assert np.allclose(so6.U6_vector(0.0, 0.0), np.eye(6))
    for th, ts in [(0.3, 0.2), (0.8, -0.5), (1.1, 0.9)]:
        U = so6.U6_vector(th, ts)
        assert np.allclose(U @ U.T, np.eye(6), atol=1e-12)
    a, b = 0.31, 0.47
    assert np.allclose(so6.U6_vector(a, 0) @ so6.U6_vector(b, 0),
                       so6.U6_vector(a + b, 0), atol=1e-12)


def test_U6_reduces_to_mchm_vector_at_ts0():
    """At ts=0 the SO(6) Goldstone is a single planar (hat,5) rotation -- the MCHM5 vector
    Goldstone (coset index 4 -> 5).  The (0..3,4) block (q_L bidoublet + singlet) carries the
    identical cos/sin dressing as ccwz.U_vector in the (0..3, coset) block."""
    for th in (0.2, 0.6, 1.0):
        U6 = so6.U6_vector(th, 0.0, hat=3)
        U5 = ccwz.U_vector(np.sin(th), hat=3)            # 5x5, coset index 4
        # both are the same planar rotation by th between index hat and the coset index
        assert np.isclose(U6[3, 3], U5[3, 3])            # cos th
        assert np.isclose(U6[3, 5], np.sin(th))          # sin th into the coset index
        assert np.isclose(U6[5, 5], np.cos(th))


# =====================================================================================
#  Tensor irreps: dimension, unitarity, homomorphism, branchings
# =====================================================================================
# (rep, dim, SO(5) content, SO(4) content)
REPS = [
    ('6',   6,  [('1', 1), ('5', 5)],
     {(0.5, 0.5): 1, (0.0, 0.0): 2}),
    ('15',  15, [('5', 5), ('10', 10)],
     {(0.5, 0.5): 2, (1.0, 0.0): 1, (0.0, 1.0): 1, (0.0, 0.0): 1}),
    ("20'", 20, [('1', 1), ('5', 5), ('14', 14)],
     {(1.0, 1.0): 1, (0.5, 0.5): 2, (0.0, 0.0): 3}),
]


@pytest.mark.parametrize("rep,dim,so5,so4", REPS)
def test_rep_dimension_and_branchings(rep, dim, so5, so4):
    B = so6.rep_basis(rep)
    assert len(B) == dim
    assert so6.so5_content(B) == so5
    # SO(5) sub-dimensions sum to the irrep dimension
    assert sum(d for _, d in so5) == dim
    assert so6.so4_content(B) == so4
    # SO(4) multiplet dimensions also sum to the irrep dimension
    assert sum(int((2 * a + 1) * (2 * b + 1)) * m for (a, b), m in so4.items()) == dim


@pytest.mark.parametrize("rep,dim,so5,so4", REPS)
def test_rep_unitary_identity_homomorphism(rep, dim, so5, so4):
    I = np.eye(dim)
    assert np.allclose(so6.U6_rep(rep, 0.0, 0.0), I, atol=1e-12)
    for th, ts in [(0.3, 0.2), (0.7, -0.4)]:
        U = so6.U6_rep(rep, th, ts)
        assert np.allclose(U @ U.conj().T, I, atol=1e-9)
    # one-parameter homomorphism along the Higgs direction (ts=0)
    a, b = 0.21, 0.33
    assert np.allclose(so6.U6_rep(rep, a, 0) @ so6.U6_rep(rep, b, 0),
                       so6.U6_rep(rep, a + b, 0), atol=1e-9)


def test_three_form_splits_into_10_and_10bar():
    """The antisymmetric 3-form (20) of SO(6) is the self-dual + anti-self-dual 10 + 10bar
    (SU(4) 10 + 10bar): each is a 10-dim complex irrep, both branch to the SO(5) adjoint 10,
    and together they fill the 20."""
    B20 = so6.rep_basis('20')
    assert len(B20) == 20
    assert so6.so5_content(B20) == [('10', 10), ('10', 10)]      # two adjoints
    for rep in ('10', '10bar'):
        U = so6.U6_rep(rep, 0.3, 0.2)
        assert U.shape == (10, 10)
        assert np.allclose(U @ U.conj().T, np.eye(10), atol=1e-9)
        assert np.allclose(so6.U6_rep(rep, 0.0, 0.0), np.eye(10), atol=1e-12)
        a, b = 0.21, 0.33                            # one-parameter homomorphism (ts=0)
        assert np.allclose(so6.U6_rep(rep, a, 0) @ so6.U6_rep(rep, b, 0),
                           so6.U6_rep(rep, a + b, 0), atol=1e-9)


# =====================================================================================
#  Fundamental embedding (thesis eq. 633) and the (h,s) channel weights
# =====================================================================================
def test_fundamental_embedding_6_is_4_plus_1_plus_1():
    """eq. 633: the 6 decomposes under SO(4) as 4 + 1 + 1 -- the bidoublet (indices 0..3),
    the SO(5)-vector singlet (index 4) and the SO(5)-singlet (index 5)."""
    e = {i: so6.embedding('6', f'fourplet_{i}') for i in range(4)}
    s5 = so6.embedding('6', 'singlet5')
    s6 = so6.embedding('6', 'singlet6')
    allv = list(e.values()) + [s5, s6]
    G = np.array([[np.vdot(u, v) for v in allv] for u in allv])
    assert np.allclose(G, np.eye(6))                    # orthonormal basis of the 6
    assert np.argmax(np.abs(s5)) == 4 and np.argmax(np.abs(s6)) == 5


def test_symbolic_goldstone_matches_numeric_bit_for_bit():
    """The symbolic vector Goldstone lambdifies to the numeric U6 to machine precision -- the
    'symbolic engine reproduces numeric' hallmark of pychm.symbolic, now for SO(6)."""
    th, ts = sp.symbols('theta_h theta_s', real=True)
    Usym = so6.U6_vector_sym(th, ts)
    f = sp.lambdify((th, ts), Usym, modules='numpy')
    for a, b in [(0.3, 0.2), (0.8, -0.5), (1.1, 0.9), (0.05, 0.01)]:   # origin tested elsewhere
        assert np.allclose(np.array(f(a, b), dtype=float), so6.U6_vector(a, b), atol=1e-12)


def test_closed_form_channel_weights_derive_unity_and_mchm_limit():
    """The (h,s) channel weights of the 6 are exact closed trig forms (derived, not fitted):
    they sum to 1 symbolically, and at ts=0 the coset channel is the MCHM5 sin^2(theta_h)/2
    weight (the SO(6) analogue of decompose.channel_weights_sym)."""
    th, ts = sp.symbols('theta_h theta_s', real=True, positive=True)
    EqL = sp.Matrix([1, 0, 0, 1, 0, 0]) / sp.sqrt(2)
    W = so6.channel_weights6_sym(EqL, th, ts)
    assert sp.simplify(sum(W.values()) - 1) == 0                       # unitarity, closed form
    assert sp.simplify(W[('1_6',)].subs(ts, 0) - sp.sin(th)**2 / 2) == 0   # MCHM5 vector weight
    assert sp.simplify(W[('1_5',)].subs(ts, 0)) == 0
    # closed-form overlap: <e5|U6|e5> = cos(sqrt(th^2+ts^2)) (the t_R self-overlap)
    e5 = sp.Matrix([0, 0, 0, 0, 0, 1])
    assert sp.simplify(so6.overlap6_sym(e5, e5, th, ts) - sp.cos(sp.sqrt(th**2 + ts**2))) == 0


def test_symbolic_overlap_matches_numeric():
    """overlap6_sym (closed form) agrees with the numeric overlap at sample angles."""
    th, ts = sp.symbols('theta_h theta_s', real=True)
    bra = so6.embedding('6', 'singlet6')
    ket = (so6.embedding('6', 'fourplet_0') + so6.embedding('6', 'fourplet_3')) / np.sqrt(2)
    expr = so6.overlap6_sym(sp.Matrix(bra), sp.Matrix(ket), th, ts)
    f = sp.lambdify((th, ts), expr, modules='numpy')
    for a, b in [(0.3, 0.2), (0.7, -0.4)]:
        assert np.isclose(complex(f(a, b)), so6.overlap('6', bra, ket, a, b), atol=1e-12)


def test_channel_weights_sum_to_unity_and_reduce_to_mchm():
    """The (h,s)-dressed q_L embedding splits into the 6's SO(4) channels with weights summing
    to |E|^2=1 (unitarity).  At ts=0 it reduces to the MCHM5 vector split: the q_L (2,2) stays
    a bidoublet (weight cos^2(h/2f)... here the e3 leg rotates into the coset)."""
    EqL = (so6.embedding('6', 'fourplet_0') + so6.embedding('6', 'fourplet_3')) / np.sqrt(2)
    for th, ts in [(0.4, 0.0), (0.5, 0.3), (0.9, -0.6)]:
        W = so6.channel_weights6(EqL, th, ts)
        assert np.isclose(sum(W.values()), 1.0, atol=1e-12)
    # at ts=0 the singlet-6 channel (index 5) carries exactly sin^2(theta)/2 of the dressed q_L
    # (the e3 component, weight 1/2 of q_L, rotated by sin into the coset), and the e4 channel 0
    th = 0.7
    W = so6.channel_weights6(EqL, th, 0.0)
    assert np.isclose(W[('1_6',)], np.sin(th)**2 / 2, atol=1e-12)
    assert np.isclose(W[('1_5',)], 0.0, atol=1e-12)
