"""The model registry is a single source of truth: every pipeline module resolves model strings
through the same `registry.MODELS` / `registry.TUNED`, so a new model is wired in one place."""
import pychm
from pychm import registry, routes, spectrum, tuning


def test_all_pipeline_modules_share_one_registry():
    assert pychm._MODELS is registry.MODELS
    assert routes._MODELS is registry.MODELS
    assert spectrum._MODELS is registry.MODELS
    assert tuning._TUNED is registry.TUNED


def test_registry_keys_consistent():
    # every model has a tuning basis, and every model object exposes the contract
    for name, mod in registry.MODELS.items():
        assert name in registry.TUNED, f"{name} missing a TUNED basis"
        for fn in ('mass_U', 'mass_D', 'mass2_W', 'mass2_Z'):
            assert hasattr(mod, fn), f"{name} model missing {fn}"
