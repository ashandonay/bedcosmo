"""Tests for sed_prior SCRATCH-based path defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedcosmo.num_visits.empirical.paths import (
    get_bedcosmo_scratch,
    get_desi_data_dir,
    get_eazy_templates_dir,
    get_healpix_fit_dir,
    get_num_visits_scratch,
    get_prior_build_dir,
    get_prior_kde_path,
    get_prior_weights_csv,
    get_scratch_root,
    resolve_desi_dir,
)


def test_scratch_paths_use_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scratch = tmp_path / "nersc_scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))

    assert get_scratch_root() == scratch
    assert get_bedcosmo_scratch() == scratch / "bedcosmo"
    assert get_num_visits_scratch() == scratch / "bedcosmo" / "num_visits"
    assert get_desi_data_dir() == scratch / "bedcosmo" / "desi" / "tiny_dr1"
    assert get_eazy_templates_dir() == scratch / "bedcosmo" / "eazy"
    assert get_healpix_fit_dir(23040) == (
        scratch / "bedcosmo" / "num_visits" / "empirical_prior" / "healpix" / "hp23040"
    )
    assert get_prior_build_dir() == scratch / "bedcosmo" / "num_visits" / "empirical_prior"
    assert get_prior_weights_csv() == (
        scratch / "bedcosmo" / "num_visits" / "empirical_prior" / "desi_eazy_empirical_weights.csv"
    )
    assert get_prior_kde_path() == (
        scratch / "bedcosmo" / "num_visits" / "empirical_prior" / "sed_prior_kde.joblib"
    )


def test_scratch_paths_fallback_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRATCH", raising=False)
    root = get_scratch_root()
    assert root == Path.home() / "scratch"
    assert get_bedcosmo_scratch() == root / "bedcosmo"
    assert get_num_visits_scratch() == root / "bedcosmo" / "num_visits"


def test_resolve_desi_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scratch = tmp_path / "nersc_scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))
    assert resolve_desi_dir(None) == scratch / "bedcosmo" / "desi" / "tiny_dr1"

    custom = tmp_path / "custom" / "desi"
    custom.mkdir(parents=True)
    assert resolve_desi_dir(custom) == custom.resolve()
