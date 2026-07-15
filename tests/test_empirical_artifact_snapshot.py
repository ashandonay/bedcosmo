"""Tests for freezing empirical KDE artifacts into MLflow run directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedcosmo.num_visits.empirical.sed_prior import (
    SED_PRIOR_KDE_GAUSSIANIZED_FILENAME,
    SED_PRIOR_KDE_NATIVE_FILENAME,
    copy_sed_prior_artifacts,
    resolve_prior_dir,
    resolve_runtime_prior_root,
    sed_prior_kde_artifact_path,
    snapshot_sed_prior,
)


def test_resolve_prior_dir_null_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))
    default = scratch / "bedcosmo" / "num_visits" / "empirical_prior"
    default.mkdir(parents=True)

    assert resolve_prior_dir({"prior_dir": None}) == default


def test_resolve_runtime_prior_root_prefers_frozen_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "run" / "artifacts"
    frozen = artifacts / "empirical" / SED_PRIOR_KDE_NATIVE_FILENAME
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"frozen")

    prior_dir = tmp_path / "build"
    prior_dir.mkdir()
    (prior_dir / SED_PRIOR_KDE_NATIVE_FILENAME).write_bytes(b"scratch")

    root = resolve_runtime_prior_root(
        artifacts_dir=artifacts,
        prior_dir=str(prior_dir),
    )
    assert root == frozen.parent.resolve()


def test_resolve_runtime_prior_root_falls_back_to_prior_dir(tmp_path: Path) -> None:
    prior_dir = tmp_path / "build"
    prior_dir.mkdir()
    (prior_dir / SED_PRIOR_KDE_NATIVE_FILENAME).write_bytes(b"scratch")

    root = resolve_runtime_prior_root(prior_dir=str(prior_dir))
    assert root == prior_dir.resolve()


def test_snapshot_sed_prior_copies_from_prior_dir(tmp_path: Path) -> None:
    prior_dir = tmp_path / "build"
    src = prior_dir / SED_PRIOR_KDE_NATIVE_FILENAME
    prior_dir.mkdir(parents=True)
    src.write_bytes(b"physical-kde-bytes")

    artifacts = tmp_path / "run" / "artifacts"
    prior_args = {"prior_dir": str(prior_dir), "prior_pool_size": 1024}

    out = snapshot_sed_prior(prior_args, artifacts)

    dest = sed_prior_kde_artifact_path(artifacts, space="native")
    assert dest.is_file()
    assert dest.read_bytes() == b"physical-kde-bytes"
    assert out["prior_dir"] == str(prior_dir)
    assert out["prior_pool_size"] == 1024


def test_sed_prior_kde_gaussianized_artifact_path(tmp_path: Path) -> None:
    artifacts = tmp_path / "run" / "artifacts"
    path = sed_prior_kde_artifact_path(artifacts, space="gaussianized")
    assert path == artifacts / "empirical" / SED_PRIOR_KDE_GAUSSIANIZED_FILENAME


def test_snapshot_sed_prior_does_not_copy_gaussianized_kde(tmp_path: Path) -> None:
    """Only the native KDE is snapshotted; gaussianized KDE beside it is skipped."""
    prior_dir = tmp_path / "build"
    src = prior_dir / SED_PRIOR_KDE_NATIVE_FILENAME
    gauss_src = prior_dir / SED_PRIOR_KDE_GAUSSIANIZED_FILENAME
    prior_dir.mkdir(parents=True)
    src.write_bytes(b"physical-kde-bytes")
    gauss_src.write_bytes(b"gauss-kde-bytes")

    artifacts = tmp_path / "run" / "artifacts"
    prior_args = {"prior_dir": str(prior_dir), "prior_pool_size": 1024}

    snapshot_sed_prior(prior_args, artifacts)

    kde_dest = sed_prior_kde_artifact_path(artifacts, space="native")
    assert kde_dest.read_bytes() == b"physical-kde-bytes"
    assert not (kde_dest.parent / SED_PRIOR_KDE_GAUSSIANIZED_FILENAME).exists()


def test_copy_sed_prior_artifacts(tmp_path: Path) -> None:
    old_run = tmp_path / "old" / "artifacts"
    new_run = tmp_path / "new" / "artifacts"
    src_kde = old_run / "empirical" / SED_PRIOR_KDE_NATIVE_FILENAME
    src_kde.parent.mkdir(parents=True)
    src_kde.write_bytes(b"frozen")

    assert copy_sed_prior_artifacts(old_run, new_run)
    assert sed_prior_kde_artifact_path(new_run, space="native").read_bytes() == b"frozen"
