"""Experimental viability: map a model's spectrum onto current LHC / EWPT / SMEFT bounds.

`spectrum()` answers "what does this point predict"; this module answers "is that point still
allowed".  It takes a spectrum dict and runs each prediction past a published limit, returning a
per-bound record `(observable, value, limit, passes, ref)`.  The bounds live in one editable table
(`BOUNDS`) with their arXiv ids, so updating an experimental result is a one-line change.

    >>> import pychm
    >>> s = pychm.Model('5-5-5').spectrum(point)
    >>> r = pychm.constraints.check(s)
    >>> r['passes']                       # True iff every bound is satisfied
    >>> print(pychm.constraints.report(s)) # pretty pass/fail table with citations

The four observables and their derived combinations:

* **Compositeness** xi = (v/f)^2.  EWPT + Higgs-coupling global fits bound xi <~ 0.1 (relaxable
  to ~0.2-0.4 in models with light states that partially cancel the S-parameter).  2207.01465.
* **Higgs gauge coupling** kappa_V = sqrt(1 - xi), the universal (2,2)-pNGB modifier; the Higgs
  signal-strength combination requires kappa_V >~ 0.95.  2207.01465.
* **Top partner (VLQ)** the lightest top-partner mass; pair production excludes m <~ 1.5 TeV
  (single production reaches ~2 TeV but is coupling-dependent).  2212.05263.
* **Di-Higgs trilinear** the MCHM estimate kappa_lambda = (1 - 2 xi)/sqrt(1 - xi); the ATLAS+CMS
  combination allows -0.71 < kappa_lambda < 6.1 (model-dependent, flagged as such).  2602.23991.
"""
import math
from collections import namedtuple

Check = namedtuple("Check", "observable value limit relation passes ref note")

# the editable bound table: name -> (limit, relation, arXiv, note)
BOUNDS = {
    "xi":            (0.10, "<", "2207.01465", "EWPT+SMEFT global fit; relaxable to ~0.2-0.4 with light states"),
    "kappa_V":       (0.95, ">", "2207.01465", "universal (2,2)-pNGB Higgs-coupling modifier sqrt(1-xi)"),
    "mtop_partner":  (1.5,  ">", "2212.05263", "VLQ pair production (TeV); single production ~2 TeV"),
    "kappa_lambda":  (None, "in", "2602.23991", "MCHM estimate (1-2xi)/sqrt(1-xi); model-dependent"),
}
KAPPA_LAMBDA_BAND = (-0.71, 6.1)


def _check(name, value):
    limit, rel, ref, note = BOUNDS[name]
    if rel == "<":
        passes = value < limit
        lim = limit
    elif rel == ">":
        passes = value > limit
        lim = limit
    elif rel == "in":
        lo, hi = KAPPA_LAMBDA_BAND
        passes = lo < value < hi
        lim = KAPPA_LAMBDA_BAND
    else:
        raise ValueError(rel)
    return Check(name, float(value), lim, rel, bool(passes), ref, note)


def derived(spec):
    """The experimental observables derived from a spectrum dict (xi, kappa_V, kappa_lambda, ...)."""
    xi = spec["xi"]
    return dict(
        xi=xi,
        kappa_V=math.sqrt(max(1.0 - xi, 0.0)),
        kappa_lambda=(1.0 - 2.0 * xi) / math.sqrt(max(1.0 - xi, 1e-12)),
        mtop_partner=spec["mtop_partner"],
    )


def check(spec):
    """Run a spectrum dict past every bound.  Returns {checks: [Check...], passes: bool}.

    `spec` is the dict from `Model.spectrum(point)` (or None -> no EWSB, vacuously failing).
    """
    if spec is None:
        return {"checks": [], "passes": False, "note": "no electroweak symmetry breaking"}
    d = derived(spec)
    checks = [
        _check("xi", d["xi"]),
        _check("kappa_V", d["kappa_V"]),
        _check("mtop_partner", d["mtop_partner"]),
        _check("kappa_lambda", d["kappa_lambda"]),
    ]
    return {"checks": checks, "passes": all(c.passes for c in checks)}


def report(spec):
    """A human-readable pass/fail table with citations.  Returns a (multi-line) string."""
    res = check(spec)
    if not res["checks"]:
        return "no electroweak symmetry breaking -- point is not viable"
    lines = [
        "observable      value      bound                status   ref",
        "-" * 72,
    ]
    for c in res["checks"]:
        if c.relation == "in":
            lo, hi = c.limit
            bound = f"in ({lo:g}, {hi:g})"
        else:
            bound = f"{c.relation} {c.limit:g}"
        status = "PASS" if c.passes else "FAIL"
        lines.append(f"{c.observable:<15} {c.value:>8.4f}   {bound:<20} {status:<6}  {c.ref}")
    lines.append("-" * 72)
    lines.append("VIABLE" if res["passes"] else "EXCLUDED")
    return "\n".join(lines)
