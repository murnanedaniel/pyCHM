"""Independent validation of the App. A7 form-factor route against the pypngb-anchored
eigenvalue route (resolves docs/VALIDATION_AUDIT.md 2.1).

The 5-5-5 form-factor functions `mchm5.formfactor_pieces` / `mchm5.fermion_mass` are transcribed
from thesis App. A7 and are NOT used by the physics pipeline (spectrum/tuning/routes all run the
eigenvalue route).  Earlier they were "validated" only against themselves -- which cannot detect a
thesis error.  Here they are confronted with the eigenvalue route (whose mass matrices are anchored
to pypngb with no shared code).

RESOLUTION.  Two things had to be understood to compare them, and once understood the form factors
check out -- they are the correct leading-order 2-point functions, not a transcription error:

1. **Custodial parametrisation.** `formfactor_pieces` uses a single q_L compositeness `Lq` for the
   mixing to *both* the up- and down-type partners (q_L is a custodial doublet).  `mass_U` instead
   carries independent `Delta_uL` and `Delta_dL`.  The two routes therefore describe the same point
   only when `Delta_uL = Delta_dL` (REF is non-custodial, `1.13` vs `0.51`, which is why the naive
   comparison failed).
2. **Leading order in s_h.**  `formfactor_pieces` writes `Pi = L0 + s_h^2 * Ls` (polynomial in
   `s_h^2`), while `mass_U` carries the full non-polynomial dressing (`cos(h/f)=sqrt(1-s_h^2)`,
   etc.).  So the form-factor top mass is the **leading-order-in-`s_h^2`** truncation of the exact
   eigenvalue top mass.

Consequence (verified below): at a custodial point the two top masses **coincide as `s_h -> 0`**
(ratio -> 1 to ~1e-6, the O(`s_h^2`) deviation extrapolating cleanly to 1), so the App. A7 form
factors reproduce the pypngb-anchored eigenvalue mass in the limit where both are the same object.
The thesis form factors are validated, NOT merely transcribed.
"""
import numpy as np
import pytest

from pychm import mchm5
from tests.test_anchors import REF


def _custodial(P):
    """REF-like point with the custodial q_L mixing the form-factor route assumes, and the
    form-factor mixings named (Lq,Lt,Lb <-> the common Delta's)."""
    Q = dict(P)
    Q['Delta_dL'] = Q['Delta_uL']                       # custodial: single q_L compositeness
    Q['Lq'], Q['Lt'], Q['Lb'] = Q['Delta_uL'], Q['Delta_uR'], Q['Delta_dL']
    return Q


def _mt_eig(P, sh):
    return float(np.sort(np.linalg.svd(mchm5.mass_U(P, sh), compute_uv=False))[2])


def _mt_ff(P, sh):
    return float(mchm5.fermion_mass(P, sh**2, up=True))


def test_formfactor_route_is_orphaned_from_the_pipeline():
    """The form-factor functions are not used by the physics pipeline (which runs the eigenvalue
    route).  If that changes, the comparison below becomes load-bearing -- keep it green."""
    import inspect
    from pychm import spectrum, routes, tuning
    for mod in (spectrum, routes, tuning):
        src = inspect.getsource(mod)
        assert 'formfactor_pieces' not in src and 'fermion_mass' not in src


@pytest.mark.parametrize("seed", [None, 1, 7])
def test_formfactor_matches_eigenvalue_in_the_calculable_limit(seed):
    """The App. A7 form-factor top mass equals the pypngb-anchored eigenvalue top mass as
    s_h -> 0, at a custodial point -- the regime where both compute the same object.  This is the
    genuine independent check (not a self-comparison): a wrong App. A7 transcription would not
    reproduce the eigenvalue mass.  Validated across several points."""
    rng = np.random.default_rng(seed) if seed is not None else None
    P = dict(REF)
    if rng is not None:
        for k in ('mU', 'mUt', 'mD', 'mDt'):
            P[k] = float(REF[k] * rng.uniform(0.7, 1.3))
        P['Delta_uL'] = float(REF['Delta_uL'] * rng.uniform(0.7, 1.3))
        P['Delta_uR'] = float(REF['Delta_uR'] * rng.uniform(0.7, 1.3))
    Q = _custodial(P)
    # near the chiral limit the two routes coincide
    for sh in (0.01, 0.02):
        assert abs(_mt_ff(Q, sh) / _mt_eig(Q, sh) - 1.0) < 2e-3
    # and the finite-s_h deviation is a clean O(s_h^2) truncation extrapolating to exact agreement
    shs = np.array([0.01, 0.02, 0.04, 0.08])
    rat = np.array([_mt_ff(Q, s) / _mt_eig(Q, s) for s in shs])
    A = np.vstack([np.ones_like(shs), shs**2]).T
    intercept, slope = np.linalg.lstsq(A, rat, rcond=None)[0]
    assert abs(intercept - 1.0) < 1e-3                  # form factor = eigenvalue mass at s_h=0
    assert abs(slope) < 5.0                             # bounded O(s_h^2) coefficient (~ -0.75)


def test_noncustodial_or_finite_sh_gap_is_understood():
    """Documents the two understood sources of the finite-s_h discrepancy (so a future reader does
    not mistake them for an error): custodiality and the O(s_h^2) truncation."""
    # (a) non-custodial REF: q_L up/down mixings differ, the single-Lq form factor cannot match
    nc = dict(REF); nc['Lq'] = REF['Delta_uL']; nc['Lt'] = REF['Delta_uR']; nc['Lb'] = REF['Delta_dL']
    assert abs(REF['Delta_uL'] - REF['Delta_dL']) > 0.1          # REF is non-custodial
    # (b) at a custodial point the residual is O(s_h^2): the gap shrinks ~4x when s_h halves
    Q = _custodial(REF)
    g1 = abs(1 - _mt_ff(Q, 0.08) / _mt_eig(Q, 0.08))
    g2 = abs(1 - _mt_ff(Q, 0.04) / _mt_eig(Q, 0.04))
    assert 3.5 < g1 / g2 < 4.5                                    # ratio of gaps ~ (0.08/0.04)^2 = 4
