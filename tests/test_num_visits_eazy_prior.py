"""Tests for empirical KDE SED prior and NumVisits integration."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bedcosmo.custom_dist import EmpiricalPrior
from bedcosmo.transform import Bijector
from bedcosmo.util import get_experiment_config_path
from bedcosmo.num_visits.empirical.sed_prior import (
    build_gpu_prior_pool,
    sample_prior_batch,
)
from bedcosmo.num_visits.empirical.simplex import (
    PARAMETERIZATION_CLR,
    PARAMETERIZATION_ILR,
    prior_clr_feature_names,
    prior_ilr_feature_names,
    split_feature_matrix,
)
from bedcosmo.num_visits.empirical.fit_sed_prior_kde import get_parameterization, load_sed_prior_kde
from bedcosmo.num_visits.empirical.templates import build_common_rest_grid, load_eazy_template_bank

N_TEMPLATES = 12


def _expected_feature_names(parameterization: str) -> list[str]:
    if parameterization == PARAMETERIZATION_ILR:
        return prior_ilr_feature_names(N_TEMPLATES)  # f1..f11, log_c_scale, z
    return prior_clr_feature_names(N_TEMPLATES)  # f1..f12, log_c_scale, z (legacy)

KDE_PATH = Path(
    os.environ.get(
        "BEDCOSMO_TEST_KDE_PATH",
        Path.home()
        / "scratch/bedcosmo/num_visits/empirical_prior/sed_prior_kde_native.joblib",
    )
).expanduser()

pytestmark = pytest.mark.skipif(
    not KDE_PATH.is_file(),
    reason=f"KDE artifact not found at {KDE_PATH}",
)


def test_build_common_rest_grid_optical_coverage():
    """Log-spaced bank grid should have many points in the LSST optical window."""
    native = [np.array([91.0, 1e8])]
    grid = build_common_rest_grid(native, n_points=4000)
    optical = (grid >= 1000.0) & (grid <= 12000.0)
    assert optical.sum() >= 800
    assert grid[0] >= 499.0
    assert grid[-1] <= 50000.0


@pytest.fixture(scope="module")
def kde_artifact():
    artifact = load_sed_prior_kde(KDE_PATH)
    if get_parameterization(artifact) not in (PARAMETERIZATION_ILR, PARAMETERIZATION_CLR):
        pytest.skip(
            "KDE artifact is not ILR/CLR parameterization; rebuild with fit_sed_prior_kde.py"
        )
    return artifact


def test_build_gpu_prior_pool_shape_and_simplex(kde_artifact):
    param = get_parameterization(kde_artifact)
    feature_names = _expected_feature_names(param)
    pool = build_gpu_prior_pool(kde_artifact, 2000, seed=0, device="cpu")
    assert pool.pool.shape == (2000, len(feature_names))
    assert pool.n_templates == N_TEMPLATES
    assert pool.feature_names == feature_names
    a, _, _ = split_feature_matrix(
        pool.pool.cpu().numpy(),
        N_TEMPLATES,
        parameterization=param,
    )
    assert np.all(a >= -1e-9)
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-5)


def test_sample_prior_batch_on_device(kde_artifact):
    n_features = len(_expected_feature_names(get_parameterization(kde_artifact)))
    pool = build_gpu_prior_pool(kde_artifact, 500, seed=1, device="cpu")
    rows = sample_prior_batch(pool, 128)
    assert rows.shape == (128, n_features)
    assert rows.device == pool.pool.device


def test_empirical_prior_bijector_bracket():
    low, high = Bijector._bin_bracket(EmpiricalPrior(0.0, 1.5, device="cpu"), "z")
    assert float(low) == 0.0
    assert float(high) == 1.5


@pytest.mark.slow
def test_load_eazy_template_bank():
    wave, stack, paths = load_eazy_template_bank()
    assert wave.ndim == 1
    assert stack.shape[0] == len(paths)
    assert stack.shape[1] == wave.shape[0]
    optical = (wave >= 1000.0) & (wave <= 12000.0)
    assert optical.sum() >= 800
    assert wave[0] >= 500.0
    assert wave[-1] <= 50000.0
    t0 = stack[0, optical]
    assert not np.all(np.diff(t0) >= -1e-20)


def test_empirical_transform_input_loads_build_prior_bijector(kde_artifact):
    """Empirical NumVisits should load NF bijector from KDE artifact, not resample CDFs."""
    if kde_artifact.get("gaussianizer_state") is None:
        pytest.skip("KDE artifact has no gaussianizer_state")

    from bedcosmo.num_visits import NumVisits
    from bedcosmo.util import get_experiment_config_path

    design_path = get_experiment_config_path("num_visits", "design_args.yaml")
    with open(design_path) as f:
        design_args = yaml.safe_load(f)
    design_args["input_type"] = "nominal"

    prior_args = {
        "prior_dir": str(KDE_PATH.parent.resolve()),
        "prior_pool_size": 512,
        "prior_pool_seed": 0,
        "parameters": {},
    }
    models_path = get_experiment_config_path("num_visits", "models.yaml")
    with open(models_path) as f:
        models = yaml.safe_load(f)
    for name in models["empirical"]["parameters"]:
        prior_args["parameters"][name] = {
            "distribution": {"type": "empirical"},
            "plot": {"lower": -8.0, "upper": 8.0},
        }

    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="empirical",
        device="cpu",
        transform_input=True,
        input_transform_type="joint",
        verbose=False,
    )
    assert exp.param_bijector is not None
    assert exp.param_bijector.uses_joint_gaussianizer()
    rows = exp.sed_prior.sample_batch(32)
    y = exp.params_to_unconstrained(rows)
    assert y.shape == (32, len(exp.cosmo_params))
    assert torch.isfinite(y).all()


@pytest.mark.slow
def test_numvisits_eazy_init_and_magnitudes():
    from bedcosmo.num_visits import NumVisits

    design_path = get_experiment_config_path("num_visits", "design_args.yaml")
    with open(design_path) as f:
        design_args = yaml.safe_load(f)
    design_args["input_type"] = "nominal"

    prior_args = {
        "prior_dir": str(KDE_PATH.parent.resolve()),
        "prior_pool_size": 512,
        "prior_pool_seed": 0,
        "parameters": {},
    }
    models_path = get_experiment_config_path("num_visits", "models.yaml")
    with open(models_path) as f:
        models = yaml.safe_load(f)
    for name in models["empirical"]["parameters"]:
        prior_args["parameters"][name] = {
            "distribution": {"type": "empirical"},
            "plot": {"lower": -8.0, "upper": 8.0},
        }
    prior_args["parameters"]["log_c_scale"]["plot"] = {"lower": 4.0, "upper": 10.5}
    prior_args["parameters"]["z"]["plot"] = {"lower": 0.0, "upper": 1.75}

    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="empirical",
        device="cpu",
        cdf_samples=2000,
        transform_input=False,
        verbose=False,
    )
    expected_param = get_parameterization(load_sed_prior_kde(KDE_PATH))
    assert exp.cosmo_model == "empirical"
    assert exp._n_eazy_templates == N_TEMPLATES
    assert len(exp.cosmo_params) == len(_expected_feature_names(expected_param))
    assert exp._prior_parameterization == expected_param
    z = torch.tensor([0.5, 0.9], dtype=torch.float64)
    a = torch.ones(2, N_TEMPLATES, dtype=torch.float64) / N_TEMPLATES
    log_s = torch.tensor([7.0, 7.5], dtype=torch.float64)
    flux_aa = exp._observed_spectral_flux(z, a=a, log_s=log_s)
    mags = exp._calculate_magnitudes(flux_aa)
    assert mags.shape == (2, 6)
    assert torch.all(torch.isfinite(mags))
