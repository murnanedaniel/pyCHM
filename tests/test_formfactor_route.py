"""Honest status of the App. A7 form-factor route (the one place the thesis is a load-bearing
oracle -- see docs/VALIDATION_AUDIT.md).

The 5-5-5 form-factor functions `mchm5.formfactor_pieces` / `mchm5.fermion_mass` are transcribed
from thesis App. A7 and are NOT used by the physics pipeline (spectrum/tuning/routes all run the
eigenvalue route).  They were previously "validated" only against themselves
(`test_A7_building_blocks_verbatim` proves faithful copying; `test_pole_mass_eq518` re-derives a
quantity from its own pieces) -- which cannot detect a thesis error.

This module runs the genuine independent check the audit recommends: compare the top mass from the
form-factor route to the top mass from the pypngb-anchored eigenvalue route.  It currently does NOT
reconcile: with the natural parameter identification (Lq,Lt,Lb <-> Delta_uL,uR,dL) the two routes
disagree by ~8-24% with an s_h-dependent ratio, and no constant parameter rescaling removes it
(scipy fit leaves a ~6% residual and runs Lt,Lb off to ~1e7).  The discrepancy is consistent with
(i) the routes computing different objects -- the form-factor route is the p=0 *pole* mass
m = M/sqrt(Pi_L Pi_R) including wavefunction renormalisation, the eigenvalue route is a tree-level
singular value -- and/or (ii) an unresolved convention/parameter mapping, and/or (iii) an actual
App. A7 transcription/thesis issue.  We cannot separate these without the full convention
derivation, so the equivalence is marked xfail rather than asserted: the form-factor route is
faithfully transcribed but NOT independently validated.  Do not present it as validated until this
xfail is resolved (then turn it into a real assertion).
"""
import numpy as np
import pytest

from pychm import mchm5
from tests.test_anchors import REF


def _ff_params():
    """REF with the form-factor mixings named (the natural Lq,Lt,Lb <-> Delta_* identification)."""
    P = dict(REF)
    P['Lq'], P['Lt'], P['Lb'] = REF['Delta_uL'], REF['Delta_uR'], REF['Delta_dL']
    return P


def _mt_eigenvalue(sh):
    """Top mass from the eigenvalue route (pypngb-anchored): 3rd-lightest up singular value."""
    return float(np.sort(np.linalg.svd(mchm5.mass_U(REF, sh), compute_uv=False))[2])


def _mt_formfactor(sh):
    """Top mass from the App. A7 form-factor route (thesis-transcribed)."""
    return float(mchm5.fermion_mass(_ff_params(), sh**2, up=True))


def test_formfactor_route_is_orphaned_from_the_pipeline():
    """Guard the audit finding: the form-factor functions are NOT used by the physics pipeline
    (which runs the eigenvalue route).  If this ever changes, the route must first be validated."""
    import inspect
    from pychm import spectrum, routes, tuning
    for mod in (spectrum, routes, tuning):
        src = inspect.getsource(mod)
        assert 'formfactor_pieces' not in src and 'fermion_mass' not in src, (
            f"{mod.__name__} now uses the form-factor route -- validate it first (see "
            "test_formfactor_route_matches_eigenvalue / VALIDATION_AUDIT.md 2.1)")


def test_formfactor_and_eigenvalue_top_masses_are_same_order():
    """A weak sanity bound that IS true: the two routes agree to within 30% (same ballpark, same
    monotonic s_h growth) -- enough to know the A7 transcription is not wildly wrong, not enough to
    call it validated."""
    for sh in (0.05, 0.1, 0.265, 0.45):
        r = _mt_formfactor(sh) / _mt_eigenvalue(sh)
        assert 0.7 < r < 1.3


@pytest.mark.xfail(reason="form-factor route not reconciled with the pypngb-anchored eigenvalue "
                          "route: s_h-dependent ~8-24% discrepancy, no constant parameter map "
                          "removes it; pole-vs-tree-mass and convention mapping unresolved. "
                          "See docs/VALIDATION_AUDIT.md 2.1. Resolve before claiming validation.",
                   strict=True)
def test_formfactor_route_matches_eigenvalue():
    """The check that WOULD turn the A7 form-factor layer from 'faithfully transcribed' into
    'physically correct': the form-factor top mass must equal the eigenvalue top mass over an s_h
    grid.  Currently xfails (see module docstring)."""
    for sh in (0.05, 0.1, 0.15, 0.2, 0.265, 0.35, 0.45, 0.6):
        assert np.isclose(_mt_formfactor(sh), _mt_eigenvalue(sh), rtol=1e-2)
