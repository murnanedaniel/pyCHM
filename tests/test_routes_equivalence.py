"""Route equivalence: the eigenvalue (closed-form CW) and momentum (divergence-subtracted
integral) routes are the same one-loop potential.

They agree on the potential CURVE V(s_h), measured relative to the potential depth max|V|.
Note: a *tuned* vacuum is a near-cancellation that amplifies the momentum route's residual
quadrature error into a several-percent error on xi (and a large error on m_h''); this is why
the eigenvalue closed form is the precision route used for the spectrum and fine-tuning.  The
robust, always-valid check is therefore on the curve shape, not on xi at a tuned point.
"""
import numpy as np
import pychm
from pychm import routes
from tests.test_anchors import REF, _rand_point


def test_routes_agree_on_potential_curve():
    shs = np.linspace(0.0, 0.4, 9)
    Ve = routes.potential_curve(REF, shs, route='eigenvalue')
    Vm = routes.potential_curve(REF, shs, route='momentum')
    assert np.max(np.abs(Ve - Vm))/np.max(np.abs(Ve)) < 0.02   # agree to 2% of the depth


def test_routes_agree_on_curve_random():
    rng = np.random.default_rng(3); checked = 0
    for _ in range(40):
        P = _rand_point(rng)
        shs = np.linspace(0.0, 0.4, 9)
        Ve = routes.potential_curve(P, shs, route='eigenvalue')
        Vm = routes.potential_curve(P, shs, route='momentum')
        depth = np.max(np.abs(Ve))
        if depth < 1e-6:
            continue
        assert np.max(np.abs(Ve - Vm))/depth < 0.03
        checked += 1
        if checked >= 5:
            break
    assert checked >= 3


def test_eigenvalue_route_deterministic():
    shs = np.linspace(0, 0.3, 5)
    assert np.allclose(routes.potential_curve(REF, shs), routes.potential_curve(REF, shs))
