from __future__ import annotations

import json

import pandas as pd
import pytest

from bedcosmo.num_visits.empirical.combine_healpix_weights import (
    combine_healpix_weights,
)
from bedcosmo.num_visits.empirical.provenance import (
    read_provenance,
    resolve_template_settings,
    write_provenance,
)


def _template(norm_min: float = 4000.0, norm_max: float = 8000.0) -> dict:
    return {
        "template_param": "templates/example.param",
        "normalization": {
            "method": "integral",
            "wave_min_aa": norm_min,
            "wave_max_aa": norm_max,
        },
    }


def _write_patch(tmp_path, healpix: int, *, norm_min: float = 4000.0) -> None:
    patch_dir = tmp_path / "healpix" / f"hp{healpix}"
    patch_dir.mkdir(parents=True)
    pd.DataFrame({"a1": [1.0], "quality_pass": [True]}).to_csv(
        patch_dir / "desi_eazy_empirical_weights.csv", index=False
    )
    write_provenance(
        patch_dir / "fit_provenance.json",
        {
            "kind": "desi_eazy_patch_fit",
            "template": _template(norm_min=norm_min),
            "parameters": {"healpix": healpix},
        },
    )


def test_write_and_read_provenance_serializes_paths(tmp_path):
    path = write_provenance(tmp_path / "provenance.json", {"input": tmp_path})
    payload = read_provenance(path)
    assert payload["schema_version"] == 1
    assert payload["input"] == str(tmp_path.resolve())
    assert json.loads(path.read_text()) == payload


def test_resolve_template_settings_prefers_and_validates_artifact():
    artifact = {"metadata": {"template_settings": _template()}}
    assert resolve_template_settings(
        artifact,
        configured_template_param="templates/example.param",
    ) == ("templates/example.param", 4000.0, 8000.0)

    with pytest.raises(ValueError, match="template_norm_min"):
        resolve_template_settings(
            artifact,
            configured_template_param="templates/example.param",
            configured_norm_min=3500.0,
        )

    with pytest.raises(ValueError, match="template_param"):
        resolve_template_settings(
            artifact,
            configured_template_param="templates/different.param",
        )


def test_resolve_template_settings_legacy_explicit_values():
    assert resolve_template_settings(
        {},
        configured_template_param="templates/example.param",
        configured_norm_min=3000.0,
        configured_norm_max=9000.0,
    ) == ("templates/example.param", 3000.0, 9000.0)


def test_combine_records_patch_provenance(tmp_path):
    _write_patch(tmp_path, 1)
    _write_patch(tmp_path, 2)

    out = combine_healpix_weights([1, 2], prior_dir=tmp_path)

    assert len(pd.read_csv(out)) == 2
    provenance = read_provenance(tmp_path / "build_provenance.json")
    assert provenance["template"] == _template()
    assert [item["parameters"]["healpix"] for item in provenance["patch_fits"]] == [1, 2]


def test_combine_rejects_incompatible_template_normalization(tmp_path):
    _write_patch(tmp_path, 1, norm_min=4000.0)
    _write_patch(tmp_path, 2, norm_min=4500.0)

    with pytest.raises(ValueError, match="different template bank or normalization"):
        combine_healpix_weights([1, 2], prior_dir=tmp_path)


def test_combine_rejects_stale_fits_for_requested_build(tmp_path):
    _write_patch(tmp_path, 1)
    write_provenance(
        tmp_path / "build_provenance.json",
        {
            "template": _template(),
            "fit": {"fit_method": "wls"},
            "selection": {},
            "quality": {},
        },
    )

    with pytest.raises(ValueError, match="do not match the requested build"):
        combine_healpix_weights([1], prior_dir=tmp_path)
