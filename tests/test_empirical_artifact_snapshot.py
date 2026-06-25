"""Tests for freezing empirical KDE artifacts into MLflow run directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedcosmo.num_visits.empirical.sed_prior import (
    copy_sed_prior_artifacts,
    resolve_prior_kde_source,
    resolve_runtime_prior_kde_path,
    resolve_runtime_y_prior_kde_path,
    sed_prior_kde_artifact_path,
    sed_prior_y_kde_artifact_path,
    snapshot_sed_prior_kde,
)


def test_resolve_prior_kde_source_null_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))
    default = scratch / "bedcosmo" / "num_visits" / "empirical_prior" / "sed_prior_kde.joblib"
    default.parent.mkdir(parents=True)
    default.write_bytes(b"kde")

    src = resolve_prior_kde_source({"prior_kde_source": None})
    assert src == default


def test_resolve_prior_kde_source_legacy_prior_kde_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))
    legacy = tmp_path / "legacy.joblib"
    legacy.write_bytes(b"kde")

    src = resolve_prior_kde_source({"prior_kde_path": str(legacy)})
    assert src == legacy.resolve()


def test_snapshot_sed_prior_kde_copies_without_pinning_path(tmp_path: Path) -> None:
    src = tmp_path / "build" / "sed_prior_kde.joblib"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"physical-kde-bytes")

    artifacts = tmp_path / "run" / "artifacts"
    prior_args = {"prior_kde_source": str(src), "prior_pool_size": 1024}

    out = snapshot_sed_prior_kde(prior_args, artifacts)

    dest = sed_prior_kde_artifact_path(artifacts)
    assert dest.is_file()
    assert dest.read_bytes() == b"physical-kde-bytes"
    assert "prior_kde_path" not in out
    assert out["prior_kde_source"] == str(src)
    assert out["prior_pool_size"] == 1024


def test_resolve_runtime_prior_kde_path_prefers_artifacts(tmp_path: Path) -> None:
    empirical = tmp_path / "artifacts" / "empirical"
    frozen = empirical / "sed_prior_kde.joblib"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"frozen")

    scratch = tmp_path / "scratch.joblib"
    scratch.write_bytes(b"scratch")

    path = resolve_runtime_prior_kde_path(
        empirical_artifacts_dir=empirical,
        prior_kde_source=str(scratch),
    )
    assert path == frozen.resolve()


def test_resolve_runtime_y_prior_kde_path_from_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "empirical"
    y_path = artifacts / "sed_prior_y_kde.joblib"
    y_path.parent.mkdir(parents=True)
    y_path.write_bytes(b"y-kde")

    resolved = resolve_runtime_y_prior_kde_path(empirical_artifacts_dir=artifacts)
    assert resolved == y_path.resolve()


def test_resolve_runtime_prior_kde_path_from_empirical_subdir(tmp_path: Path) -> None:
    empirical = tmp_path / "artifacts" / "empirical"
    kde_path = empirical / "sed_prior_kde.joblib"
    kde_path.parent.mkdir(parents=True)
    kde_path.write_bytes(b"frozen")

    resolved = resolve_runtime_prior_kde_path(empirical_artifacts_dir=empirical)
    assert resolved == kde_path.resolve()


def test_copy_sed_prior_artifacts(tmp_path: Path) -> None:
    old_run = tmp_path / "old" / "artifacts"
    new_run = tmp_path / "new" / "artifacts"
    src_kde = old_run / "empirical" / "sed_prior_kde.joblib"
    src_kde.parent.mkdir(parents=True)
    src_kde.write_bytes(b"frozen")

    assert copy_sed_prior_artifacts(old_run, new_run)
    assert sed_prior_kde_artifact_path(new_run).read_bytes() == b"frozen"
