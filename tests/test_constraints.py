"""Experimental-viability checker: the bounds map a spectrum to pass/fail correctly."""
import math
import pychm
from pychm import constraints

# canonical EWSB benchmark (xi ~ 0.07); shared with the other anchors
REF = dict(
    mU=2.82899, mUt=1.5395, mD=2.30783, mDt=0.969391,
    mYu=0.00754655, Yu=7.95104, mYd=0.03112, Yd=0.60624,
    Delta_uL=1.12956, Delta_uR=1.11262, Delta_dL=0.5056, Delta_dR=0.106068,
    f=0.871937, f1=1.50568, fX=2.99808, g=0.67095, gp=0.358138, grho=5.44665, gX=2.96355)


def _by_name(res):
    return {c.observable: c for c in res["checks"]}


def test_ref_passes_compositeness_but_flags_light_partner():
    s = pychm.Model('5-5-5').spectrum(REF)
    res = constraints.check(s)
    c = _by_name(res)
    assert c["xi"].passes                       # xi ~ 0.07 < 0.1
    assert c["kappa_V"].passes                   # kappa_V ~ 0.96 > 0.95
    assert not c["mtop_partner"].passes          # REF partner ~ 0.97 TeV < 1.5 TeV
    assert not res["passes"]                     # so the point as a whole is excluded


def test_high_xi_fails_compositeness():
    s = {"xi": 0.3, "mtop_partner": 5.0}
    c = _by_name(constraints.check(s))
    assert not c["xi"].passes
    assert not c["kappa_V"].passes               # kappa_V = sqrt(0.7) ~ 0.837 < 0.95


def test_heavy_low_xi_point_is_viable():
    s = {"xi": 0.05, "mtop_partner": 2.5}
    res = constraints.check(s)
    assert res["passes"]
    for c in res["checks"]:
        assert c.passes


def test_light_partner_fails_vlq():
    s = {"xi": 0.05, "mtop_partner": 1.0}
    c = _by_name(constraints.check(s))
    assert not c["mtop_partner"].passes


def test_kappa_V_monotonic_in_xi():
    ks = [constraints.derived({"xi": x, "mtop_partner": 2.0})["kappa_V"]
          for x in (0.02, 0.05, 0.1, 0.2)]
    assert all(a > b for a, b in zip(ks, ks[1:]))   # strictly decreasing
    assert math.isclose(ks[0], math.sqrt(0.98))


def test_kappa_lambda_band():
    # SM-like (xi -> 0) gives kappa_lambda -> 1, inside the band
    c = _by_name(constraints.check({"xi": 1e-6, "mtop_partner": 2.0}))
    assert c["kappa_lambda"].passes
    assert abs(c["kappa_lambda"].value - 1.0) < 1e-3


def test_no_ewsb_is_not_viable():
    res = constraints.check(None)
    assert not res["passes"]
    assert res["checks"] == []


def test_report_nonempty_and_cited():
    s = pychm.Model('5-5-5').spectrum(REF)
    txt = constraints.report(s)
    assert isinstance(txt, str) and txt.strip()
    assert "2207.01465" in txt and "2212.05263" in txt
    assert "PASS" in txt and "FAIL" in txt


def test_model_constraints_convenience():
    res = pychm.Model('5-5-5').constraints(REF)
    assert "checks" in res and "passes" in res
