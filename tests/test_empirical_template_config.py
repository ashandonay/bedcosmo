"""Tests for template_source / reduced_templates prior selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bedcosmo.num_visits.empirical.eazy.build_prior import (
    resolve_eazy_build_selection,
)
from bedcosmo.num_visits.empirical.template_config import (
    empirical_prior_build_name,
    empirical_prior_variant,
    format_reduced_templates,
    materialize_empirical_prior_args,
    n_templates_for,
    parse_reduced_templates,
    resolve_template_param,
)


def _write_prior_build(root: Path, source: str, n_templates: int) -> None:
    build = root / "bedcosmo/num_visits/empirical_prior" / source
    template_param = f"{source}.param"
    (build / "templates").mkdir(parents=True)
    (build / "templates" / template_param).write_text(
        "\n".join(f"{index} component_{index:02d}.dat 1.0" for index in range(1, n_templates + 1))
        + "\n"
    )
    for index in range(1, n_templates + 1):
        (build / "templates" / f"component_{index:02d}.dat").write_text("0 1")
    names = [f"f{i}" for i in range(1, n_templates)] + ["log_c_scale", "z"]
    (build / "prior_args.yaml").write_text(
        yaml.safe_dump(
            {
                "template_param": template_param,
                "template_norm_min": 3600.0,
                "template_norm_max": 4200.0,
                "parameters": {name: {"distribution": {"type": "empirical"}} for name in names},
            }
        )
    )


def test_eazy_builder_source_defaults_and_overrides():
    assert resolve_eazy_build_selection("eazy12") == (
        "empirical_prior/eazy12",
        "eazy12.param",
    )
    assert resolve_eazy_build_selection("eazy6") == (
        "empirical_prior/eazy6",
        "eazy6.param",
    )
    assert resolve_eazy_build_selection(
        "eazy6",
        build_name="empirical_prior/custom",
        template_param="custom/custom.param",
    ) == ("empirical_prior/custom", "custom/custom.param")
    with pytest.raises(ValueError, match="EAZY builder"):
        resolve_eazy_build_selection("desi8")


def test_parse_reduced_templates_variants():
    assert parse_reduced_templates(None) is None
    assert parse_reduced_templates("null") is None
    assert parse_reduced_templates("") is None
    assert parse_reduced_templates("t7,t10") == (7, 10)
    assert parse_reduced_templates("T10+T7") == (7, 10)
    assert parse_reduced_templates("7,10") == (7, 10)


def test_parse_reduced_templates_rejects_singleton():
    with pytest.raises(ValueError, match="at least two"):
        parse_reduced_templates("t7")


def test_variant_and_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    _write_prior_build(tmp_path, "desi8", 8)

    assert empirical_prior_variant("eazy12") == "eazy12"
    assert empirical_prior_variant("eazy12", "t7,t10") == "eazy12-t7-t10"
    assert empirical_prior_variant("eazy6", "T1+T3") == "eazy6-t1-t3"
    assert empirical_prior_variant("desi8") == "desi8"

    assert empirical_prior_build_name("eazy12", "t7,t10") == ("empirical_prior/eazy12-t7-t10")
    assert resolve_template_param("eazy12") == "eazy12.param"
    assert resolve_template_param("eazy6") == "eazy6.param"
    assert resolve_template_param("desi8") == "desi8.param"
    assert resolve_template_param("eazy12", "t7,t10") == ("eazy12_t7-t10.param")
    assert n_templates_for("eazy12") == 12
    assert n_templates_for("eazy6") == 6
    assert n_templates_for("desi8") == 8
    assert n_templates_for("eazy12", "t7,t10") == 2


def test_materialize_empirical_prior_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    _write_prior_build(tmp_path, "desi8", 8)

    base = {
        "template_source": "eazy12",
        "reduced_templates": None,
        "density_type": "flow",
        "flux_unit_scale": 1.0e-17,
        "parameters": {
            "log_c_scale": {
                "distribution": {"type": "empirical"},
                "plot": {"lower": 4.0, "upper": 10.5},
            },
            "z": {
                "distribution": {"type": "empirical"},
                "plot": {"lower": 0.0, "upper": 1.75},
            },
        },
    }

    full = materialize_empirical_prior_args(base)
    assert full["template_source"] == "eazy12"
    assert full["reduced_templates"] is None
    assert full["prior_dir"].endswith("empirical_prior/eazy12")
    assert "template_dir" not in full
    assert full["template_param"] == "eazy12.param"
    assert full["template_norm_min"] == 4000.0
    assert full["template_norm_max"] == 8000.0
    assert list(full["parameters"]) == [f"f{i}" for i in range(1, 12)] + [
        "log_c_scale",
        "z",
    ]
    assert full["parameters"]["log_c_scale"]["plot"]["lower"] == 4.0

    reduced = materialize_empirical_prior_args(base, reduced_templates="t7,t10")
    assert reduced["reduced_templates"] == "t7,t10"
    assert reduced["prior_dir"].endswith("empirical_prior/eazy12-t7-t10")
    assert "template_dir" not in reduced
    assert reduced["template_param"] == "eazy12_t7-t10.param"
    assert list(reduced["parameters"]) == ["f1", "log_c_scale", "z"]
    assert format_reduced_templates((7, 10)) == "t7,t10"

    desi8 = materialize_empirical_prior_args(base, template_source="desi8")
    assert desi8["template_source"] == "desi8"
    assert desi8["prior_dir"].endswith("empirical_prior/desi8")
    assert "template_dir" not in desi8
    assert desi8["template_param"] == "desi8.param"
    assert desi8["template_norm_min"] == 3600.0
    assert desi8["template_norm_max"] == 4200.0
    assert list(desi8["parameters"]) == [f"f{i}" for i in range(1, 8)] + [
        "log_c_scale",
        "z",
    ]


def test_desi8_rejects_eazy_template_reduction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    _write_prior_build(tmp_path, "desi8", 8)
    with pytest.raises(ValueError, match="not supported"):
        empirical_prior_variant("desi8", "t1,t2")


def test_discovers_desi_prior_build_from_scratch_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    _write_prior_build(tmp_path, "desi6", 6)

    assert empirical_prior_variant("desi6") == "desi6"
    assert resolve_template_param("desi6") == "desi6.param"
    assert n_templates_for("desi6") == 6
    materialized = materialize_empirical_prior_args({}, template_source="desi6")
    assert materialized["prior_dir"].endswith("empirical_prior/desi6")
    assert materialized["template_norm_min"] == 3600.0
    assert list(materialized["parameters"]) == ["f1", "f2", "f3", "f4", "f5", "log_c_scale", "z"]


def test_materialize_legacy_without_template_source():
    legacy = {"prior_dir": "/tmp/custom", "template_param": "templates/x.param"}
    out = materialize_empirical_prior_args(legacy)
    assert out["prior_dir"] == "/tmp/custom"
    assert out["template_param"] == "templates/x.param"
    assert "template_source" not in out
