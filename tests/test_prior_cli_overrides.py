"""Tests for --prior-<field> CLI overrides."""

from __future__ import annotations

from bedcosmo.util import (
    apply_prior_cli_overrides,
    extract_prior_cli_overrides,
    parse_prior_cli_overrides,
)


def test_parse_prior_cli_overrides_extracts_fields():
    overrides, remaining = parse_prior_cli_overrides(
        [
            "--cosmo-model",
            "empirical",
            "--prior-template-source",
            "eazy6",
            "--prior-reduced-templates",
            "t7,t10",
            "--prior-density-type",
            "kde",
            "--prior-args-path",
            "prior_args_empirical.yaml",
            "--n-transforms",
            "4",
        ]
    )
    assert overrides == {
        "template_source": "eazy6",
        "reduced_templates": "t7,t10",
        "density_type": "kde",
    }
    assert remaining == [
        "--cosmo-model",
        "empirical",
        "--prior-args-path",
        "prior_args_empirical.yaml",
        "--n-transforms",
        "4",
    ]


def test_extract_prior_cli_overrides_from_flat_dict():
    overrides, cleaned = extract_prior_cli_overrides(
        {
            "prior_args_path": "prior_args_empirical.yaml",
            "prior_flow_path": None,
            "prior_template_source": "eazy12",
            "prior_pool_size": 1024,
            "n_transforms": 4,
        }
    )
    assert overrides == {"template_source": "eazy12", "prior_pool_size": 1024}
    assert cleaned["prior_args_path"] == "prior_args_empirical.yaml"
    assert "prior_template_source" not in cleaned
    assert cleaned["n_transforms"] == 4


def test_apply_prior_cli_overrides_merges_and_canonicalizes_density_type():
    out = apply_prior_cli_overrides(
        {"prior_source": "flow", "flux_unit_scale": 1e-17},
        {"density_type": "kde", "prior_pool_size": 128},
    )
    assert out["density_type"] == "kde"
    assert "prior_source" not in out
    assert "source" not in out
    assert out["prior_pool_size"] == 128
    assert out["flux_unit_scale"] == 1e-17
