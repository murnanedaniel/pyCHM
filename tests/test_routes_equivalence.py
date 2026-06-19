"""Route equivalence: the form-factor and eigenvalue routes must give the same potential.

The cross-route fermion equivalence (the heart of pyCHM's validation) activates once the
eigenvalue fermion route lands (Phase 1). The assertion below is written and ready; it is
skipped until then. The full equivalence is already demonstrated on the original model's
mass matrices in validation/two_routes_equivalence.py (agree to <1%)."""
import numpy as np
import pytest
import pychm
from tests.test_anchors import _rand_point

ROUTES_READY = True
try:
    pychm.Model().potential_coeffs(
        dict(mU=2.,mUt=1.2,mD=1.8,mDt=1.,mYu=.9,Yu=1.2,mYd=.7,Yd=.4,Lq=1.1,Lt=1.3,Lb=.3,
             f=.9,f1=1.5,fX=3.,g=.671,gp=.358,grho=5.,gX=3.), route='eigenvalue')
except NotImplementedError:
    ROUTES_READY = False


@pytest.mark.skipif(not ROUTES_READY, reason="eigenvalue fermion route: Phase 1")
def test_routes_agree_on_xi():
    rng = np.random.default_rng(11); m = pychm.Model(); checked = 0
    for _ in range(200):
        P = _rand_point(rng)
        gf, bf, df = m.potential_coeffs(P, route='formfactor')
        ge, be, de = m.potential_coeffs(P, route='eigenvalue')
        if bf <= 0 or be <= 0:
            continue
        xi_f, xi_e = gf/(2*bf), ge/(2*be)
        if not (0.01 < xi_f < 0.3):
            continue
        assert abs(xi_f - xi_e)/xi_f < 0.02      # routes agree to 2%
        checked += 1
        if checked >= 20:
            break
    assert checked >= 5


def test_gauge_route_deterministic():
    """Until Phase 1, at least guard the gauge eigenvalue CW is stable."""
    from pychm import routes
    P = dict(mU=2.,mUt=1.2,mD=1.8,mDt=1.,mYu=.9,Yu=1.2,mYd=.7,Yd=.4,Lq=1.1,Lt=1.3,Lb=.3,
             f=.9,f1=1.5,fX=3.,g=.671,gp=.358,grho=5.,gX=3.)
    shs = np.linspace(0, 0.3, 5)
    assert np.allclose(routes.gauge_cw(P, shs), routes.gauge_cw(P, shs))
