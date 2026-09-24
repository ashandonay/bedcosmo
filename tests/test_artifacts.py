"""Tests for posterior sample npz save/load helpers."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from bedcosmo.artifacts import (
    load_posterior_samples,
    load_posterior_samples_file,
    make_posterior_samples_path,
    observations_to_numpy,
    read_posterior_samples_meta,
    save_posterior_samples,
)


def _bundle_arrays(n_series=2, n_data=3, n_guide=50, n_params=2, n_obs=4, n_design=3):
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(n_series, n_data, n_guide, n_params))
    y = rng.normal(size=(n_series, n_data, n_obs))
    design = rng.normal(size=(n_series, n_design))
    series_names = [f"s{i}" for i in range(n_series)]
    param_names = [f"p{i}" for i in range(n_params)]
    return theta, y, design, series_names, param_names


def test_observations_to_numpy_flattens():
    arr = np.random.randn(5, 1, 8)
    out = observations_to_numpy(arr, 5)
    assert out.shape == (5, 8)


def test_save_load_roundtrip(tmp_path):
    theta, y, design, series_names, param_names = _bundle_arrays()
    path = tmp_path / "posterior_step10_test.npz"
    meta = {
        "status": "complete",
        "step": 10,
        "run_id": "abc",
        "central": True,
        "seed": 1,
    }
    saved = save_posterior_samples(
        str(path),
        theta=theta,
        y=y,
        design=design,
        series_names=series_names,
        param_names=param_names,
        meta=meta,
    )
    assert saved.endswith(".npz")
    assert path.exists() or (tmp_path / "posterior_step10_test.npz").exists()

    loaded = load_posterior_samples(saved)
    assert loaded["meta"]["status"] == "complete"
    assert loaded["meta"]["step"] == 10
    assert loaded["meta"]["schema_version"] == 1
    np.testing.assert_allclose(loaded["theta"], theta)
    np.testing.assert_allclose(loaded["y"], y)
    np.testing.assert_allclose(loaded["design"], design)
    assert list(loaded["series_names"]) == series_names
    assert list(loaded["param_names"]) == param_names

    # Meta-only read does not require loading arrays eagerly beyond the key
    meta_only = read_posterior_samples_meta(saved)
    assert meta_only["run_id"] == "abc"
    # allow_pickle=False path: meta is JSON text
    with np.load(saved, allow_pickle=False) as z:
        raw = z["meta"]
        assert isinstance(json.loads(str(raw.item() if hasattr(raw, "item") else raw)), dict)


def test_load_posterior_samples_file_picks_newest_matching_step(tmp_path):
    artifacts = tmp_path / "artifacts"
    theta, y, design, series_names, param_names = _bundle_arrays()

    p1 = make_posterior_samples_path(str(artifacts), step=100, timestamp="20200101_000000")
    save_posterior_samples(
        p1,
        theta=theta,
        y=y,
        design=design,
        series_names=series_names,
        param_names=param_names,
        meta={"status": "complete", "step": 100, "tag": "old"},
    )
    time.sleep(0.05)
    p2 = make_posterior_samples_path(str(artifacts), step=100, timestamp="20200101_000100")
    save_posterior_samples(
        p2,
        theta=theta + 1.0,
        y=y,
        design=design,
        series_names=series_names,
        param_names=param_names,
        meta={"status": "complete", "step": 100, "tag": "new"},
    )
    # Different step should be ignored when filtering
    p3 = make_posterior_samples_path(str(artifacts), step=200, timestamp="20200101_000200")
    save_posterior_samples(
        p3,
        theta=theta,
        y=y,
        design=design,
        series_names=series_names,
        param_names=param_names,
        meta={"status": "complete", "step": 200, "tag": "other"},
    )

    bundle = load_posterior_samples_file(str(artifacts), step=100)
    assert bundle["meta"]["tag"] == "new"
    np.testing.assert_allclose(bundle["theta"], theta + 1.0)

    with pytest.raises(FileNotFoundError):
        load_posterior_samples_file(str(artifacts), step=999)


def test_incomplete_status_skipped(tmp_path):
    artifacts = tmp_path / "artifacts"
    theta, y, design, series_names, param_names = _bundle_arrays()
    path = make_posterior_samples_path(str(artifacts), step=5, timestamp="t1")
    save_posterior_samples(
        path,
        theta=theta,
        y=y,
        design=design,
        series_names=series_names,
        param_names=param_names,
        meta={"status": "incomplete", "step": 5},
    )
    with pytest.raises(FileNotFoundError):
        load_posterior_samples_file(str(artifacts), step=5)


def test_load_posterior_samples_file_require_series(tmp_path):
    artifacts = tmp_path / "artifacts"
    theta1, y1, design1, _, param_names = _bundle_arrays(n_series=1)
    # Older file has only "nominal"
    p_old = make_posterior_samples_path(str(artifacts), step=10, timestamp="20200101_000000")
    save_posterior_samples(
        p_old,
        theta=theta1,
        y=y1,
        design=design1,
        series_names=["nominal"],
        param_names=param_names,
        meta={"status": "complete", "step": 10, "tag": "nominal_only"},
    )
    time.sleep(0.05)
    # Newer file has both series — should win when require_series asks for both
    theta2, y2, design2, _, _ = _bundle_arrays(n_series=2)
    p_new = make_posterior_samples_path(str(artifacts), step=10, timestamp="20200101_000100")
    save_posterior_samples(
        p_new,
        theta=theta2,
        y=y2,
        design=design2,
        series_names=["optimal", "nominal"],
        param_names=param_names,
        meta={"status": "complete", "step": 10, "tag": "both"},
    )

    bundle = load_posterior_samples_file(
        str(artifacts), step=10, require_series=("nominal", "optimal")
    )
    assert bundle["meta"]["tag"] == "both"
    assert set(bundle["series_names"]) == {"optimal", "nominal"}

    with pytest.raises(FileNotFoundError):
        load_posterior_samples_file(
            str(artifacts), step=10, require_series=("missing",)
        )


def test_load_posterior_samples_file_skips_newer_without_required_series(tmp_path):
    artifacts = tmp_path / "artifacts"
    param_names = ["p0", "p1"]
    theta2, y2, design2, _, _ = _bundle_arrays(n_series=2)
    p_both = make_posterior_samples_path(str(artifacts), step=3, timestamp="20200101_000000")
    save_posterior_samples(
        p_both,
        theta=theta2,
        y=y2,
        design=design2,
        series_names=["optimal", "nominal"],
        param_names=param_names,
        meta={"status": "complete", "step": 3, "tag": "both_older"},
    )
    time.sleep(0.05)
    theta1, y1, design1, _, _ = _bundle_arrays(n_series=1)
    p_nom = make_posterior_samples_path(str(artifacts), step=3, timestamp="20200101_000100")
    save_posterior_samples(
        p_nom,
        theta=theta1,
        y=y1,
        design=design1,
        series_names=["nominal"],
        param_names=param_names,
        meta={"status": "complete", "step": 3, "tag": "nominal_newer"},
    )

    bundle = load_posterior_samples_file(
        str(artifacts), step=3, require_series=("nominal", "optimal")
    )
    assert bundle["meta"]["tag"] == "both_older"

    # Without require_series, newest wins regardless of series.
    newest = load_posterior_samples_file(str(artifacts), step=3)
    assert newest["meta"]["tag"] == "nominal_newer"


def test_load_posterior_samples_file_generated_by(tmp_path):
    """A newer default-eval central bundle must not shadow a --sample-posterior bundle."""
    artifacts = tmp_path / "artifacts"
    theta, y, design, series_names, param_names = _bundle_arrays(n_series=2)
    for tag, generated_by, ts in (
        ("multi_y", "Evaluator.sample_posterior", "20200101_000000"),
        ("central", "Evaluator.run", "20200101_000100"),
    ):
        save_posterior_samples(
            make_posterior_samples_path(str(artifacts), step=4, timestamp=ts),
            theta=theta,
            y=y,
            design=design,
            series_names=series_names,
            param_names=param_names,
            meta={"status": "complete", "step": 4, "tag": tag, "generated_by": generated_by},
        )
        time.sleep(0.05)

    multi = load_posterior_samples_file(
        str(artifacts), step=4, generated_by="Evaluator.sample_posterior"
    )
    assert multi["meta"]["tag"] == "multi_y"
    central = load_posterior_samples_file(str(artifacts), step=4, generated_by="Evaluator.run")
    assert central["meta"]["tag"] == "central"
    with pytest.raises(FileNotFoundError):
        load_posterior_samples_file(str(artifacts), step=4, generated_by="other")
