"""Tests for template_source / reduced_templates prior selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedcosmo.num_visits.empirical.template_config import (
    empirical_prior_build_name,
    empirical_prior_variant,
    format_reduced_templates,
    materialize_empirical_prior_args,
    n_templates_for,
    parse_reduced_templates,
    resolve_template_param,
)


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

    assert empirical_prior_variant("eazy12") == "eazy12"
    assert empirical_prior_variant("eazy12", "t7,t10") == "eazy12-t7-t10"
    assert empirical_prior_variant("eazy6", "T1+T3") == "eazy6-t1-t3"

    assert empirical_prior_build_name("eazy12", "t7,t10") == (
        "empirical_prior/eazy12-t7-t10"
    )
    assert resolve_template_param("eazy12") == "templates/fsps_full/fsps_QSF_12_v3.param"
    assert resolve_template_param("eazy6") == "templates/eazy_v1.0.spectra.param"
    assert resolve_template_param("eazy12", "t7,t10") == (
        "templates/reduced/fsps_QSF_12_v3_t7-t10.param"
    )
    assert n_templates_for("eazy12") == 12
    assert n_templates_for("eazy6") == 6
    assert n_templates_for("eazy12", "t7,t10") == 2


def test_materialize_empirical_prior_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))

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
    assert full["template_param"] == "templates/fsps_full/fsps_QSF_12_v3.param"
    assert list(full["parameters"]) == [f"f{i}" for i in range(1, 12)] + [
        "log_c_scale",
        "z",
    ]
    assert full["parameters"]["log_c_scale"]["plot"]["lower"] == 4.0

    reduced = materialize_empirical_prior_args(base, reduced_templates="t7,t10")
    assert reduced["reduced_templates"] == "t7,t10"
    assert reduced["prior_dir"].endswith("empirical_prior/eazy12-t7-t10")
    assert reduced["template_param"] == "templates/reduced/fsps_QSF_12_v3_t7-t10.param"
    assert list(reduced["parameters"]) == ["f1", "log_c_scale", "z"]
    assert format_reduced_templates((7, 10)) == "t7,t10"


def test_materialize_legacy_without_template_source():
    legacy = {"prior_dir": "/tmp/custom", "template_param": "templates/x.param"}
    out = materialize_empirical_prior_args(legacy)
    assert out["prior_dir"] == "/tmp/custom"
    assert out["template_param"] == "templates/x.param"
    assert "template_source" not in out
