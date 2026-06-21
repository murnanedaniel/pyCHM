"""NM4DCHM6 (SO(6)/SO(5), partners in the 6): the worked NMCHM model.

Validated by internal consistency and by exact reduction to the anchored MCHM 5-5-5 (there is no
public NMCHM engine to anchor numbers to, unlike the MCHM):
  * the SO(6) mass matrices reduce to the validated 5-5-5 assembler entry-for-entry at ts = 0;
  * the full pipeline (spectrum, tuning) gives the same EWSB, m_t, m_h, Delta_BG as the 5-5-5
    (so the NMCHM inherits the pypngb anchor through that reduction);
  * the potential is even in the singlet s, the vacuum sits at <s> = 0, and the SO(5)-singlet
    pNGB acquires a finite calculable mass from the fermion loop (the genuine NMCHM observable).
"""
import numpy as np
import pytest

pytest.importorskip("sympy")

import pychm
from pychm import nmchm6 as N, assemble, mchm5
from tests.test_anchors import REF


def test_reduces_to_555_assembler_entry_for_entry():
    """At beta = 0, ts = 0 the SO(6) mass matrices ARE the validated 5-5-5 (coset 4 -> 5)."""
    for sh in (0.05, 0.1, 0.265, 0.5, 0.8):
        assert np.allclose(N.mass_U(REF, sh), assemble.model_555.mass_U(REF, sh))
        assert np.allclose(N.mass_D(REF, sh), assemble.model_555.mass_D(REF, sh))


def test_spectrum_matches_555():
    """Through the full pipeline the 6-6-6 spectrum equals the 5-5-5 spectrum (EWSB, m_t, m_h,
    gauge masses) -- the NMCHM inherits the MCHM anchor at ts = 0."""
    s6 = pychm.Model('6-6-6').spectrum(REF)
    s5 = pychm.Model('5-5-5').spectrum(REF)
    for k in ('xi', 'f', 'mt', 'mb', 'mh', 'mW', 'mZ', 'mtop_partner'):
        assert np.isclose(s6[k], s5[k], rtol=1e-4), k


def test_tuning_matches_555():
    """Barbieri-Giudice tuning is the same as the 5-5-5 (same Lagrangian-mass basis at ts = 0)."""
    t6 = pychm.Model('6-6-6').tuning(REF)
    t5 = pychm.Model('5-5-5').tuning(REF)
    assert np.isclose(t6['BG'], t5['BG'], rtol=1e-3)
    assert 100.0 < t6['BG'] < 160.0                       # the validated 5-5-5 reference regime


def test_gauge_singlet_does_not_dress_W_Z():
    """The pNGB singlet is a gauge singlet: the W/Z mass matrices depend only on the Higgs angle,
    so 6-6-6 reuses the mchm5 gauge sector unchanged."""
    for sh in (0.1, 0.3):
        assert np.allclose(N.mass2_W(REF, sh), mchm5.mass2_W(REF, sh))
        assert np.allclose(N.mass2_Z(REF, sh), mchm5.mass2_Z(REF, sh))


def test_potential_even_in_singlet_and_vacuum_at_zero():
    """With beta = 0 no state carries e4 support, so every overlap is even in ts: V(th, s) =
    V(th, -s) exactly, and the singlet vacuum is <s> = 0 (the EWSB direction is ts = 0)."""
    for th in (0.1, 0.265, 0.4):
        for ts in (0.1, 0.2, 0.3):
            assert abs(N.potential(REF, th, ts) - N.potential(REF, th, -ts)) < 1e-12
    thv = np.arcsin(np.sqrt(0.07))
    tss = np.linspace(-0.3, 0.3, 13)
    k = int(np.argmin([N.potential(REF, thv, b) for b in tss]))
    assert abs(tss[k]) < 1e-6                              # minimum in s at the origin


def test_singlet_pNGB_has_positive_calculable_mass():
    """The second pNGB (the SO(5) singlet) gets a finite positive mass from the fermion loop --
    the defining NMCHM observable absent in the MCHM."""
    s6 = pychm.Model('6-6-6').spectrum(REF)
    thv = np.arcsin(np.sqrt(s6['xi']))
    ms2 = N.singlet_mass2(REF, thv, 0.0, f=s6['f'])
    assert ms2 > 0
    ms = np.sqrt(ms2)                                      # TeV
    assert 0.1 < ms < 5.0                                  # a heavy-ish but sub-cutoff pNGB


def test_singlet_mass_is_robust_and_method_independent():
    """The singlet mass is not a stencil artefact: it is (a) converged in the finite-difference
    step, and (b) consistent with an independent parabolic fit of V(ts) near the vacuum.  With no
    public NMCHM engine to anchor to, this internal cross-validation is what makes the number
    trustworthy (the curvature of the same eigenvalue-route potential, computed two ways)."""
    s6 = pychm.Model('6-6-6').spectrum(REF)
    thv = np.arcsin(np.sqrt(s6['xi']))
    f = s6['f']
    # (a) stencil convergence: m_s^2 stable across a 5x range of step sizes
    vals = [N.singlet_mass2(REF, thv, 0.0, f=f, h=h) for h in (4e-3, 2e-3, 1e-3)]
    assert max(vals) / min(vals) - 1 < 1e-3
    # (b) independent method: fit V(ts) = V0 + 1/2 ms^2 f^2 ts^2 over a grid (even potential)
    tss = np.linspace(-0.05, 0.05, 11)
    V = np.array([N.potential(REF, thv, b) for b in tss])
    a2 = np.polyfit(tss, V, 2)[0]                          # coefficient of ts^2
    ms2_fit = 2.0 * a2 / f**2
    ms2_fd = N.singlet_mass2(REF, thv, 0.0, f=f)
    assert np.isclose(ms2_fit, ms2_fd, rtol=2e-2)         # two independent computations agree


def test_beta_generates_singlet_tadpole():
    """Turning on beta (t_R picks up an e4 component) breaks the s -> -s symmetry: V acquires an
    odd-in-ts piece (a singlet tadpole), so <s> is driven off zero."""
    Rb = dict(REF); Rb['beta'] = 0.2
    th = 0.265
    odd = max(abs(N.potential(Rb, th, ts) - N.potential(Rb, th, -ts)) for ts in (0.1, 0.2))
    assert odd > 1e-9                                      # genuine tadpole (cf. exact 0 at beta=0)
