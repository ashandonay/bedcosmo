"""bedcosmo: Bayesian Experimental Design for Cosmology"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# Lazy exports avoid importing plotting/IPython (and heavy deps) on every
# `import bedcosmo` or `from bedcosmo.util import ...`.

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Bijector": ("bedcosmo.transform", "Bijector"),
    "auto_seed": ("bedcosmo.util", "auto_seed"),
    "init_experiment": ("bedcosmo.util", "init_experiment"),
    "init_nf": ("bedcosmo.util", "init_nf"),
    "load_model": ("bedcosmo.util", "load_model"),
    "get_experiments_dir": ("bedcosmo.util", "get_experiments_dir"),
    "get_experiment_config_path": ("bedcosmo.util", "get_experiment_config_path"),
    "BaseExperiment": ("bedcosmo.base", "BaseExperiment"),
    "CosmologyMixin": ("bedcosmo.cosmology", "CosmologyMixin"),
    "NumTracers": ("bedcosmo.num_tracers", "NumTracers"),
    "VariableRedshift": ("bedcosmo.variable_redshift", "VariableRedshift"),
    "NumVisits": ("bedcosmo.num_visits", "NumVisits"),
    "GridCalculation": ("bedcosmo.grid_calc", "GridCalculation"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
