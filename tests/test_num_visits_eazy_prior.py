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
from bedcosmo.num_visits.sed_prior.prior_sampler import (
    build_gpu_prior_pool,
    load_empirical_prior,
    sample_prior_batch,
)
from bedcosmo.num_visits.sed_prior.simplex import (
    PARAMETERIZATION_CLR,
    prior_clr_feature_names,
    split_feature_matrix,
)
from bedcosmo.num_visits.sed_prior.templates import build_common_rest_grid, load_eazy_template_bank

N_TEMPLATES = 12
N_FEATURES = N_TEMPLATES + 2  # f1..f12, log_c_scale, z

KDE_PATH = Path(
    os.environ.get(
        "BEDCOSMO_TEST_KDE_PATH",
        Path.home()
        / "scratch/bedcosmo/num_visits/empirical_prior/sed_prior_kde.joblib",
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
    artifact = load_empirical_prior(KDE_PATH)
    if artifact.get("parameterization", "weights") != PARAMETERIZATION_CLR:
        pytest.skip(
            "KDE artifact is not CLR parameterization; rebuild with "
            "build_empirical_sed_prior_kde.py (default --parameterization clr)"
        )
    return artifact


def test_build_gpu_prior_pool_shape_and_simplex(kde_artifact):
    pool = build_gpu_prior_pool(kde_artifact, 2000, seed=0, device="cpu")
    assert pool.pool.shape == (2000, N_FEATURES)
    assert pool.n_templates == N_TEMPLATES
    assert pool.feature_names == prior_clr_feature_names(N_TEMPLATES)
    a, _, _ = split_feature_matrix(
        pool.pool.cpu().numpy(),
        N_TEMPLATES,
        parameterization=PARAMETERIZATION_CLR,
    )
    assert np.all(a >= -1e-9)
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-5)


def test_sample_prior_batch_on_device(kde_artifact):
    pool = build_gpu_prior_pool(kde_artifact, 500, seed=1, device="cpu")
    rows = sample_prior_batch(pool, 128)
    assert rows.shape == (128, N_FEATURES)
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


@pytest.mark.slow
def test_numvisits_eazy_init_and_magnitudes():
    from bedcosmo.num_visits import NumVisits

    design_path = get_experiment_config_path("num_visits", "design_args.yaml")
    with open(design_path) as f:
        design_args = yaml.safe_load(f)
    design_args["input_type"] = "nominal"

    prior_args = {
        "prior_kde_path": str(KDE_PATH.resolve()),
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
    assert exp.use_eazy_sed
    assert len(exp.cosmo_params) == N_FEATURES
    assert exp._prior_parameterization == PARAMETERIZATION_CLR
    z = torch.tensor([0.5, 0.9], dtype=torch.float64)
    a = torch.ones(2, N_TEMPLATES, dtype=torch.float64) / N_TEMPLATES
    log_s = torch.tensor([7.0, 7.5], dtype=torch.float64)
    flux_aa = exp._observed_spectral_flux(z, a=a, log_s=log_s)
    mags = exp._calculate_magnitudes(flux_aa)
    assert mags.shape == (2, 6)
    assert torch.all(torch.isfinite(mags))
