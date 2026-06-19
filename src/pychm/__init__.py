"""pyCHM -- open Composite Higgs Model potential, spectrum and fine-tuning.

Dual route (form-factor / mass-eigenvalue), pure numpy+scipy, no private dependencies.

    >>> import pychm
    >>> m = pychm.Model('5-5-5')
    >>> s = m.spectrum(point)              # {xi, f, mt, mb, mh, mW, mZ} or None
    >>> t = m.tuning(point)                # {BG, HOT, I, KL}
    >>> g, b, d = m.potential_coeffs(point)

`route='formfactor'` (default) or `route='eigenvalue'` selects the fermion method; the gauge
sector uses the eigenvalue route in both.  See README for the validation status and roadmap.
"""
from . import mchm5, routes, potential, spectrum, tuning

__version__ = "0.1.0"
__all__ = ["Model", "mchm5", "routes", "potential", "spectrum", "tuning"]

_MODELS = {"5-5-5": mchm5}


class Model:
    def __init__(self, representation="5-5-5"):
        if representation not in _MODELS:
            raise NotImplementedError(
                f"representation {representation!r} not implemented; available: {list(_MODELS)}")
        self.representation = representation

    def spectrum(self, point, route="formfactor"):
        return spectrum.spectrum(point, route=route)

    def tuning(self, point, route="formfactor"):
        return tuning.tuning(point, route=route)

    def potential_coeffs(self, point, route="formfactor"):
        return potential.potential_coeffs(point, route=route)
