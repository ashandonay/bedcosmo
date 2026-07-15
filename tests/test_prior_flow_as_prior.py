"""Tests for the native PriorFlow acting as the empirical prior sampler + density.

Covers Path-B "flow as prior" wiring: PriorFlow.sample() (native only),
EmpiricalSedPrior.enable_flow_prior / build_pool_from_flow / uses_flow_prior,
and log_prob_gaussianized_from_native (production transform_input entropy path).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bedcosmo.num_visits.empirical.prior_flow import (
    SPACE_GAUSSIANIZED,
    SPACE_NATIVE,
    train_prior_flow,
)
from bedcosmo.num_visits.empirical.sed_prior import EmpiricalPriorPool, EmpiricalSedPrior

FEATURES = ["f1", "f2", "log_c_scale", "z"]


def _tiny_flow(space: str, seed: int = 0):
    """A small, quickly-trained flow over 4 synthetic features (non-trivial mean/std)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, len(FEATURES))) * np.array([1.0, 2.0, 0.5, 1.5]) + 3.0
    return train_prior_flow(
        x,
        space=space,
        feature_names=FEATURES,
        epochs=6,
        hidden_features=(16, 16),
        transforms=2,
        bins=6,
        verbose=False,
    )


def _prior_with_pool(n: int = 200, seed: int = 1) -> EmpiricalSedPrior:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    pool = EmpiricalPriorPool(
        pool=torch.tensor(x, dtype=torch.float64),
        feature_names=list(FEATURES),
        n_templates=2,
        bounds_min=torch.tensor(x.min(axis=0), dtype=torch.float64),
        bounds_max=torch.tensor(x.max(axis=0), dtype=torch.float64),
    )
    return EmpiricalSedPrior({"feature_names": list(FEATURES), "n_templates": 2}, pool)


def test_sample_shape_and_finite():
    pf = _tiny_flow(SPACE_NATIVE)
    s = pf.sample(500, seed=3)
    assert s.shape == (500, len(FEATURES))
    assert np.isfinite(s).all()


def test_sample_requires_native_space():
    gf = _tiny_flow(SPACE_GAUSSIANIZED)
    with pytest.raises(ValueError, match="native"):
        gf.sample(10)


def test_sample_then_log_prob_is_finite():
    pf = _tiny_flow(SPACE_NATIVE)
    s = pf.sample(256, seed=5)
    lp = pf.log_prob(s)
    assert lp.shape == (256,)
    assert np.isfinite(lp).all()


def test_sample_reproducible_with_seed():
    pf = _tiny_flow(SPACE_NATIVE)
    a = pf.sample(64, seed=7)
    b = pf.sample(64, seed=7)
    assert np.array_equal(a, b)


def test_build_pool_from_flow_flips_source_and_samples():
    pf = _tiny_flow(SPACE_NATIVE)
    prior = _prior_with_pool()
    assert not prior.uses_flow_prior()
    prior.attach_flow(pf)
    assert not prior.uses_flow_prior()  # attached but source still "kde"
    prior.build_pool_from_flow(300, seed=2)
    assert prior.uses_flow_prior()
    assert tuple(prior.pool.pool.shape) == (300, len(FEATURES))
    assert prior.pool.feature_names == FEATURES
    assert tuple(prior.sample_batch(32).shape) == (32, len(FEATURES))


def test_build_pool_from_flow_requires_native_flow():
    prior = _prior_with_pool()
    with pytest.raises(RuntimeError, match="native flow"):
        prior.build_pool_from_flow(100)


def test_enable_flow_prior_loads_native_and_gaussianized(tmp_path):
    """enable_flow_prior attaches both flows and rebuilds the pool from native."""
    from bedcosmo.num_visits.empirical.prior_flow import SED_PRIOR_FLOW_FILENAMES

    flow_dir = tmp_path / "flows"
    flow_dir.mkdir()
    native = _tiny_flow(SPACE_NATIVE, seed=0)
    gauss = _tiny_flow(SPACE_GAUSSIANIZED, seed=1)
    native.save(flow_dir / SED_PRIOR_FLOW_FILENAMES[SPACE_NATIVE])
    gauss.save(flow_dir / SED_PRIOR_FLOW_FILENAMES[SPACE_GAUSSIANIZED])

    prior = _prior_with_pool()
    # Fake a KDE path so enable_flow_prior can default flow_dir to its parent.
    prior.path = str(flow_dir / "sed_prior_kde_native.joblib")
    loaded = prior.enable_flow_prior(250, seed=4)
    assert SPACE_NATIVE in loaded
    assert SPACE_GAUSSIANIZED in loaded
    assert prior.uses_flow_prior()
    assert prior.has_native_flow()
    assert prior.has_gaussianized_flow()
    assert tuple(prior.pool.pool.shape) == (250, len(FEATURES))


def test_enable_flow_prior_requires_native_file(tmp_path):
    prior = _prior_with_pool()
    with pytest.raises(FileNotFoundError, match="native flow"):
        prior.enable_flow_prior(100, flow_dir=tmp_path)


def test_log_prob_gaussianized_from_native_scores_at_y():
    """Production entropy path: score gaussianized flow at y = transform_fn(theta)."""
    native = _tiny_flow(SPACE_NATIVE, seed=2)
    gauss = _tiny_flow(SPACE_GAUSSIANIZED, seed=3)
    prior = _prior_with_pool()
    prior.attach_flow(native)
    prior.attach_flow(gauss)
    prior.prior_source = "flow"

    rng = np.random.default_rng(13)
    x = torch.tensor(rng.normal(size=(64, len(FEATURES))), dtype=torch.float64)

    def transform_fn(flat: torch.Tensor) -> torch.Tensor:
        # Affine map into the gaussianized flow's coordinate frame.
        return 0.5 * flat + 0.1

    lp = prior.log_prob_gaussianized_from_native(x, transform_fn, param_names=FEATURES)
    y = transform_fn(x)
    expected = torch.as_tensor(gauss.log_prob(y.numpy()), dtype=torch.float64)
    assert lp.shape == (64,)
    assert torch.allclose(lp, expected, atol=1e-9)


def test_log_prob_gaussianized_from_native_requires_gaussianized_flow():
    prior = _prior_with_pool()
    prior.attach_flow(_tiny_flow(SPACE_NATIVE))
    x = torch.zeros(4, len(FEATURES), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="gaussianized flow"):
        prior.log_prob_gaussianized_from_native(x, lambda t: t, param_names=FEATURES)


def test_use_flow_requires_native_flow_attached():
    prior = _prior_with_pool()
    x = torch.zeros(4, len(FEATURES), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="native flow"):
        prior.log_prob(x, param_names=FEATURES, use_flow=True)


def test_snapshot_copies_flows_into_artifacts(tmp_path):
    """prior_source=flow freezes the .pt files into artifacts/empirical/ beside the KDE."""
    from bedcosmo.num_visits.empirical.prior_flow import (
        SED_PRIOR_FLOW_FILENAMES,
        SPACE_GAUSSIANIZED,
        SPACE_NATIVE,
    )
    from bedcosmo.num_visits.empirical.sed_prior import (
        sed_prior_flow_artifact_path,
        sed_prior_kde_artifact_path,
        snapshot_sed_prior,
    )

    src_dir = tmp_path / "empirical_prior"
    src_dir.mkdir()
    (src_dir / "sed_prior_kde_native.joblib").write_bytes(b"kde")  # snapshot only copies bytes
    (src_dir / SED_PRIOR_FLOW_FILENAMES[SPACE_NATIVE]).write_bytes(b"native-flow")
    (src_dir / SED_PRIOR_FLOW_FILENAMES[SPACE_GAUSSIANIZED]).write_bytes(b"gauss-flow")

    artifacts_dir = tmp_path / "artifacts"
    prior_args = {
        "prior_source": "flow",
        "prior_dir": str(src_dir),
    }
    out = snapshot_sed_prior(prior_args, artifacts_dir)

    assert sed_prior_kde_artifact_path(artifacts_dir).is_file()
    assert sed_prior_flow_artifact_path(artifacts_dir, SPACE_NATIVE).is_file()
    assert sed_prior_flow_artifact_path(artifacts_dir, SPACE_GAUSSIANIZED).is_file()
    assert out["prior_dir"] == str(src_dir)


def test_snapshot_flow_missing_native_raises(tmp_path):
    from bedcosmo.num_visits.empirical.sed_prior import snapshot_sed_prior

    src_dir = tmp_path / "empirical_prior"
    src_dir.mkdir()
    (src_dir / "sed_prior_kde_native.joblib").write_bytes(b"kde")  # KDE present, flow absent

    prior_args = {
        "prior_source": "flow",
        "prior_dir": str(src_dir),
    }
    with pytest.raises(FileNotFoundError, match="native flow not found"):
        snapshot_sed_prior(prior_args, tmp_path / "artifacts")
