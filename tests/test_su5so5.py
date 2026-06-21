"""SU(5)/SO(5) (littlest-Higgs / Ferretti real-rep): a NEW coset built on the engine, with two
runnable models.  No external engine exists, so validation is internal consistency + the reduction
of the Higgs dressing to the validated MCHM on the Higgs axis (following test_su4sp4 / test_nmchm6).
"""
import numpy as np
import pytest

pytest.importorskip("sympy")

import pychm
from pychm.groups import lie, coset, tensors
from pychm.groups import su5so5 as G
from pychm import su5so5 as M
from tests.test_anchors import REF

# a demonstration EWSB point for the 15-partner model (found by scan; unphysical m_h, but EWSB fires)
REF15 = dict(REF, mU=2.20774, mUt=0.46932, mD=3.18264, mDt=0.48547, mYu=0.02592, Yu=13.28897,
             Delta_uL=0.95773, Delta_uR=1.82963, Delta_dL=0.58788)


# =====================================================================================
#  Group theory: the coset, the SU(N) split, the 14 pNGBs and their custodial content
# =====================================================================================
def test_su_so_split_and_coset():
    unb, brk = lie.so_subalgebra(5)
    assert len(unb) == 10 and len(brk) == 14                       # so(5) + the 14 coset
    for g in unb:                                                  # so(5) = imaginary antisymmetric
        assert np.allclose(g, g.conj().T) and np.allclose(g + g.T, 0, atol=1e-9)
    c = G.fundamental()
    assert c.algebra_closes() and c.n_pngb == 14


def test_14_pngbs_are_higgs_triplet_singlet():
    """The 14 coset -> custodial SO(4) is (1,1)+(1/2,1/2)+(0,0) = complex triplet + Higgs doublet +
    singlet (the littlest-Higgs pNGB content)."""
    content = G.coset_so4_content()
    assert content == {(0.0, 0.0): 1, (0.5, 0.5): 1, (1.0, 1.0): 1}
    assert sum(int((2 * a + 1) * (2 * b + 1)) * m for (a, b), m in content.items()) == 14


def test_su_vs_so_tensor_basis():
    assert len(tensors.tensor_basis('sym', 2, 5, group='SO')) == 14    # SO(5) sym-traceless
    assert len(tensors.tensor_basis('sym', 2, 5, group='SU')) == 15    # SU(5) symmetric (irreducible)
    assert len(tensors.tensor_basis('antisym', 2, 5)) == 10            # unaffected


def test_sym15_branches_14_plus_1():
    s15 = G.sym15()
    assert s15.dim == 15 and s15.algebra_closes()
    C = sum(g @ g for g in s15.unbroken)
    w = np.linalg.eigvalsh((C + C.conj().T) / 2).real
    _, cnts = np.unique(np.round(w, 4), return_counts=True)
    assert sorted(cnts.tolist()) == [1, 14]                        # 15 -> 14 + 1 of SO(5)


@pytest.mark.parametrize("gold,dim", [(lambda *a: G.goldstone_5(*a), 5),
                                      (lambda *a: G.goldstone_rep(G.sym15().basis, *a), 15)])
def test_goldstone_unitary_identity_homomorphism(gold, dim):
    assert np.allclose(gold(0.0, 0.0, 0.0), np.eye(dim), atol=1e-12)
    for th, te, tp in [(0.3, 0.2, 0.1), (0.7, -0.4, 0.2)]:
        U = gold(th, te, tp)
        assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10)
    a, b = 0.21, 0.33                                              # one-parameter subgroup (Higgs)
    assert np.allclose(gold(a, 0, 0) @ gold(b, 0, 0), gold(a + b, 0, 0), atol=1e-10)


# =====================================================================================
#  The '5-5-su5' model: Higgs reduction, EWSB, the new triplet/singlet pNGB masses
# =====================================================================================
def test_higgs_dressing_reduces_to_mchm_on_higgs_axis():
    """At theta_eta=theta_phi=0 the (2,2) Higgs overlap magnitude is the MCHM5 value sin^2(h/f)/2."""
    EqL = (np.eye(5)[0] + np.eye(5)[3]) / np.sqrt(2)
    EtR = np.eye(5)[4]
    for th in (0.2, 0.5, 0.9):
        U = G.goldstone_5(th, 0.0, 0.0, hat=3)
        assert np.isclose(abs(np.vdot(EqL, U @ EtR))**2, np.sin(th)**2 / 2, atol=1e-12)


def test_model_runs_to_finite_ewsb_spectrum():
    s = pychm.Model('5-5-su5').spectrum(REF)
    assert s is not None and 0 < s['xi'] < 1
    assert 0.1 < s['mt'] < 0.25 and 0 < s['mh'] < 0.3            # top/Higgs in physical ballpark
    assert all(np.isfinite(v) for k, v in s.items() if k != 'J')
    t = pychm.Model('5-5-su5').tuning(REF)
    assert t is not None and t['BG'] > 0


def test_potential_even_in_eta_and_phi():
    for th in (0.15, 0.265):
        for a in (0.1, 0.2):
            assert abs(M.potential(REF, th, a, 0.0) - M.potential(REF, th, -a, 0.0)) < 1e-12
            assert abs(M.potential(REF, th, 0.0, a) - M.potential(REF, th, 0.0, -a)) < 1e-12


def test_triplet_and_singlet_pNGB_masses_positive_and_robust():
    s = pychm.Model('5-5-su5').spectrum(REF)
    thv = np.arcsin(np.sqrt(s['xi']))
    f = s['f']
    for name, fn in (('eta', M.singlet_mass2), ('phi', M.triplet_mass2)):
        m2 = fn(REF, thv, f=f)
        assert m2 > 0, name
        assert 0.01 < np.sqrt(m2) < 5.0, name                     # finite, sub-cutoff
        # stencil robustness (5x step range)
        vals = [fn(REF, thv, f=f, h=h) for h in (4e-3, 2e-3, 1e-3)]
        assert max(vals) / min(vals) - 1 < 5e-3, name


def test_gauge_singlet_does_not_dress_W_Z():
    from pychm import mchm5
    for sh in (0.1, 0.3):
        assert np.allclose(M.mass2_W(REF, sh), mchm5.mass2_W(REF, sh))
        assert np.allclose(M.mass2_Z(REF, sh), mchm5.mass2_Z(REF, sh))


# =====================================================================================
#  The '15-15-su5' model (partners in the symmetric 15): runs end-to-end
# =====================================================================================
def test_15_partner_model_builds_and_grows_top():
    """The symmetric-15 dressing builds an 11x11 mass matrix whose light top grows with the Higgs
    vev (the SU(N) tensor-irrep machinery working in a model)."""
    assert M.mass_U15(REF, 0.265).shape == (11, 11)
    tops = [np.sort(np.linalg.svd(M.mass_U15(REF, np.arcsin(s)), compute_uv=False))[2]
            for s in (0.1, 0.3, 0.5)]
    assert tops[0] < tops[1] < tops[2] and tops[0] > 0            # monotone, nonzero


def test_15_partner_model_fires_ewsb():
    s = pychm.Model('15-15-su5').spectrum(REF15)
    assert s is not None and 0 < s['xi'] < 1 and 0 < s['mt'] < 0.5
