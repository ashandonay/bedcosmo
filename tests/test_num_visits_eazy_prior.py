"""Tests for empirical KDE SED prior and NumVisits eazy_kde integration."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bedcosmo.util import EmpiricalPrior, get_experiment_config_path
from bedcosmo.num_visits.sed_prior.prior_sampler import (
    build_gpu_prior_pool,
    load_empirical_prior,
    sample_prior_batch,
)
from bedcosmo.num_visits.sed_prior.simplex import (
    PARAMETERIZATION_LOGITS,
    split_feature_matrix,
)
from bedcosmo.num_visits.sed_prior.templates import load_eazy_template_bank


KDE_PATH = Path(
    os.environ.get(
        "BEDCOSMO_TEST_KDE_PATH",
        Path.home() / "scratch/bedcosmo/desi_eazy_empirical_prior_nnls/sed_prior_kde.joblib",
    )
).expanduser()

pytestmark = pytest.mark.skipif(
    not KDE_PATH.is_file(),
    reason=f"KDE artifact not found at {KDE_PATH}",
)


@pytest.fixture(scope="module")
def kde_artifact():
    artifact = load_empirical_prior(KDE_PATH)
    if artifact.get("parameterization", "weights") != PARAMETERIZATION_LOGITS:
        pytest.skip(
            "KDE artifact uses legacy weight features; rebuild with "
            "build_empirical_sed_prior_kde.py --parameterization logits"
        )
    return artifact


def test_build_gpu_prior_pool_shape_and_simplex(kde_artifact):
    pool = build_gpu_prior_pool(kde_artifact, 2000, seed=0, device="cpu")
    assert pool.pool.shape == (2000, 13)
    assert pool.n_templates == 12
    a, _, _ = split_feature_matrix(
        pool.pool.cpu().numpy(), 12, parameterization=PARAMETERIZATION_LOGITS
    )
    assert np.all(a >= -1e-9)
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-5)


def test_sample_prior_batch_on_device(kde_artifact):
    pool = build_gpu_prior_pool(kde_artifact, 500, seed=1, device="cpu")
    rows = sample_prior_batch(pool, 128)
    assert rows.shape == (128, 13)
    assert rows.device == pool.pool.device


def test_empirical_prior_bijector_bracket():
    from bedcosmo.util import Bijector

    low, high = Bijector._bin_bracket(EmpiricalPrior(0.0, 1.5, device="cpu"), "z")
    assert float(low) == 0.0
    assert float(high) == 1.5


@pytest.mark.slow
def test_load_eazy_template_bank():
    wave, stack, paths = load_eazy_template_bank()
    assert wave.ndim == 1
    assert stack.shape[0] == len(paths)
    assert stack.shape[1] == wave.shape[0]


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
    for name in models["eazy_kde"]["parameters"]:
        prior_args["parameters"][name] = {
            "distribution": {"type": "empirical"},
            "plot": {"lower": -8.0, "upper": 8.0},
        }
    prior_args["parameters"]["log_c_scale"]["plot"] = {"lower": 4.0, "upper": 10.5}
    prior_args["parameters"]["z"]["plot"] = {"lower": 0.0, "upper": 1.75}

    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="eazy_kde",
        device="cpu",
        cdf_samples=2000,
        transform_input=False,
        verbose=False,
    )
    assert exp.use_eazy_sed
    assert len(exp.cosmo_params) == 13
    z = torch.tensor([0.5, 0.9], dtype=torch.float64)
    a = torch.ones(2, 12, dtype=torch.float64) / 12.0
    log_s = torch.tensor([7.0, 7.5], dtype=torch.float64)
    mags = exp._calculate_magnitudes_eazy(z, a, log_s)
    assert mags.shape == (2, 6)
    assert torch.all(torch.isfinite(mags))
