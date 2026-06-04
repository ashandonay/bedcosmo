"""Tests for central_params CLI/YAML handling."""

from bedcosmo.util import (
    apply_central_param_cli_flags,
    parse_extra_args,
    parse_json_object,
)
from bedcosmo.variable_redshift.experiment import PLANCK18_FIDUCIAL


class TestCentralParamsUtils:
    def test_parse_json_object(self):
        assert parse_json_object({"z": 2.0}) == {"z": 2.0}

    def test_apply_central_param_cli_flags(self):
        kwargs = apply_central_param_cli_flags(
            {"central_param_z": 1.0, "central_param_T": 10000, "device": "cpu"}
        )
        assert kwargs["central_params"] == {"z": 1.0, "T": 10000.0}
        assert "central_param_z" not in kwargs
        assert kwargs["device"] == "cpu"

    def test_apply_merges_with_yaml_central_params(self):
        kwargs = apply_central_param_cli_flags(
            {"central_params": {"z": 0.5}, "central_param_T": 10000}
        )
        assert kwargs["central_params"] == {"z": 0.5, "T": 10000.0}

    def test_parse_extra_args_central_param_flags(self):
        kwargs = parse_extra_args(
            ["--central-param-z", "1.0", "--central-param-T", "10000"]
        )
        assert kwargs["central_params"] == {"z": 1.0, "T": 10000.0}

    def test_apply_central_params_python_repr_string(self):
        """submit.sh used to pass central_params as a Python repr string."""
        kwargs = apply_central_param_cli_flags(
            {"central_params": "{'z': 1.0, 'T': 10000}"}
        )
        assert kwargs["central_params"] == {"z": 1.0, "T": 10000.0}

    def test_parse_extra_args_legacy_central_params_flag(self):
        kwargs = parse_extra_args(
            ["--central-params", '{"z": 1.0, "T": 10000}']
        )
        assert kwargs["central_params"] == {"z": 1.0, "T": 10000.0}


class TestVariableRedshiftFiducials:
    def test_planck18_fiducial_subset(self):
        cosmo_params = ["Om", "hrdrag", "w0"]
        defaults = {p: PLANCK18_FIDUCIAL[p] for p in cosmo_params if p in PLANCK18_FIDUCIAL}
        assert defaults["Om"] == 0.3152
        assert defaults["hrdrag"] == 99.079
        assert defaults["w0"] == -1.0
