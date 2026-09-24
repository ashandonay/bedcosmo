"""Tests for the unified NumVisits spectral-template layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bedcosmo.num_visits.empirical import templates


@pytest.mark.parametrize(
    ("template_param", "n_components", "source"),
    [
        (templates.DEFAULT_TEMPLATE_PARAM_6D, 6, "eazy6"),
        (templates.DEFAULT_TEMPLATE_PARAM_12D, 12, "eazy12"),
    ],
)
def test_materialize_eazy_bank_uses_source_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    template_param: str,
    n_components: int,
    source: str,
) -> None:
    downloaded: list[tuple[str, Path]] = []

    def fake_download(url: str, path: Path, overwrite: bool = False) -> None:
        del overwrite
        downloaded.append((url, path))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.array([[1000.0, 1.0], [5000.0, 2.0], [9000.0, 1.0]]))

    monkeypatch.setattr(templates, "download", fake_download)
    param_path = templates.materialize_eazy_template_bank(
        template_param,
        template_dir=tmp_path,
    )

    assert param_path == tmp_path / f"{source}.param"
    component_paths = [
        tmp_path / f"component_{index:02d}.dat" for index in range(1, n_components + 1)
    ]
    assert [path for _, path in downloaded] == component_paths
    assert param_path.read_text().splitlines()[1:] == [
        f"{index} component_{index:02d}.dat 1.0" for index in range(1, n_components + 1)
    ]

    waves, fluxes, relative_paths = templates.load_eazy_templates(
        template_param,
        template_dir=tmp_path,
    )
    assert len(waves) == len(fluxes) == n_components
    assert relative_paths == [f"component_{index:02d}.dat" for index in range(1, n_components + 1)]
