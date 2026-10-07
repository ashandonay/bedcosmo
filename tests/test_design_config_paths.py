from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from bedcosmo.artifacts import (
    resolve_design_args_input_path,
    resolve_design_input_path,
    snapshot_design_args_config,
)
from bedcosmo.util import finalize_train_run_args, parse_design_cli_overrides


def test_resolve_design_input_path_expands_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DESIGN_ROOT", str(tmp_path))

    resolved = resolve_design_args_input_path(
        {"input_path": "$DESIGN_ROOT/designs.npy"}
    )

    assert resolved["input_path"] == str((tmp_path / "designs.npy").resolve())


def test_resolve_design_input_path_relative_to_yaml(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "design_args.yaml"

    resolved = resolve_design_args_input_path(
        {"input_path": "../arrays/extreme.npy"},
        config_path,
    )

    assert resolved["input_path"] == str(
        (tmp_path / "arrays" / "extreme.npy").resolve()
    )


def test_snapshot_design_args_freezes_referenced_array(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    designs = np.arange(12, dtype=float).reshape(2, 6)
    np.save(source_dir / "extreme.npy", designs)
    monkeypatch.setenv("DESIGN_FILE", "extreme.npy")
    source_yaml = source_dir / "design_args.yaml"
    source_yaml.write_text(
        "input_type: variable\ninput_path: $DESIGN_FILE\n"
    )
    destination_yaml = tmp_path / "artifacts" / "design_args.yaml"

    snapshot_design_args_config(source_yaml, destination_yaml)

    frozen = yaml.safe_load(destination_yaml.read_text())
    frozen_path = Path(frozen["input_path"])
    assert frozen_path == (destination_yaml.parent / "designs.npy").resolve()
    np.testing.assert_array_equal(np.load(frozen_path), designs)


def test_parse_design_cli_overrides_parses_yaml_values():
    overrides, remaining = parse_design_cli_overrides(
        [
            "--design-args-path", "design_args_ratio.yaml",
            "--design-sum-lower", "1000",
            "--design-lower", "[50,80.5]",
            "--design-input-path", "null",
            "--design-chunk-size", "8",
            "--n-transforms", "4",
        ]
    )

    assert overrides == {"sum_lower": 1000, "lower": [50, 80.5], "input_path": None}
    assert remaining == [
        "--design-args-path", "design_args_ratio.yaml",
        "--design-chunk-size", "8",
        "--n-transforms", "4",
    ]


def test_parse_design_cli_overrides_requires_value():
    with pytest.raises(ValueError, match="--design-sum-lower needs a value"):
        parse_design_cli_overrides(["--design-sum-lower", "--n-transforms", "4"])


def test_finalize_train_run_args_collects_design_overrides():
    run_args = finalize_train_run_args(
        {"cosmo_exp": "num_visits"},
        {"design_args_path": "design_args.yaml"},
        unknown_argv=["--design-args-path", "design_args_ratio.yaml", "--design-sum-upper", "1030"],
    )

    assert run_args["design_args_path"] == "design_args_ratio.yaml"
    assert run_args["design_cli_overrides"] == {"sum_upper": 1030}
    assert "design_sum_upper" not in run_args


def test_snapshot_design_args_applies_overrides(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_yaml = source_dir / "design_args.yaml"
    source_yaml.write_text(
        "input_type: variable\ninput_path: null\nlower: [1.0, 2.0]\nsum_lower: 10\n"
    )
    designs = np.ones((3, 2))
    np.save(tmp_path / "cli.npy", designs)
    # A relative override path is anchored to the cwd, not the YAML directory.
    monkeypatch.chdir(tmp_path)
    destination_yaml = tmp_path / "artifacts" / "design_args.yaml"

    snapshot_design_args_config(
        source_yaml,
        destination_yaml,
        overrides={"sum_lower": 6, "input_path": "cli.npy"},
    )

    frozen = yaml.safe_load(destination_yaml.read_text())
    assert frozen["sum_lower"] == 6
    assert frozen["lower"] == [1.0, 2.0]
    assert Path(frozen["input_path"]) == (destination_yaml.parent / "designs.npy").resolve()
    np.testing.assert_array_equal(np.load(frozen["input_path"]), designs)


def test_snapshot_design_args_rejects_unknown_override(tmp_path):
    source_yaml = tmp_path / "design_args.yaml"
    source_yaml.write_text("input_type: variable\nsum_lower: 10\n")

    with pytest.raises(ValueError, match="sum_lowr"):
        snapshot_design_args_config(
            source_yaml, tmp_path / "artifacts" / "design_args.yaml", overrides={"sum_lowr": 6}
        )


def test_resolve_design_input_path_accepts_old_key(tmp_path):
    resolved = resolve_design_args_input_path(
        {"input_type": "variable", "input_designs_path": "designs.npy"},
        tmp_path / "design_args.yaml",
    )

    assert "input_designs_path" not in resolved
    assert resolved["input_path"] == str((tmp_path / "designs.npy").resolve())


def test_resolve_design_input_path_rejects_both_keys():
    with pytest.raises(ValueError, match="both input_path"):
        resolve_design_args_input_path({"input_path": None, "input_designs_path": None})


def test_snapshot_old_key_yaml_accepts_input_path_override(tmp_path):
    # A pre-rename YAML can still be submitted, including with --design-input-path.
    source_yaml = tmp_path / "design_args.yaml"
    source_yaml.write_text("input_type: variable\ninput_designs_path: null\nsum_lower: 10\n")
    designs = np.ones((3, 2))
    np.save(tmp_path / "cli.npy", designs)
    destination_yaml = tmp_path / "artifacts" / "design_args.yaml"

    snapshot_design_args_config(
        source_yaml, destination_yaml, overrides={"input_path": str(tmp_path / "cli.npy")}
    )

    frozen = yaml.safe_load(destination_yaml.read_text())
    assert "input_designs_path" not in frozen
    np.testing.assert_array_equal(np.load(frozen["input_path"]), designs)


def test_snapshot_design_dir_freezes_array_and_provenance(tmp_path):
    design_dir = tmp_path / "designs" / "pool"
    design_dir.mkdir(parents=True)
    designs = np.arange(6, dtype=float).reshape(3, 2)
    np.save(design_dir / "designs.npy", designs)
    (design_dir / "provenance.json").write_text('{"command": "python -m gen"}\n')
    source_yaml = tmp_path / "design_args.yaml"
    source_yaml.write_text(f"input_type: variable\ninput_path: {design_dir}\n")
    destination_yaml = tmp_path / "artifacts" / "design_args.yaml"

    snapshot_design_args_config(source_yaml, destination_yaml)

    frozen = yaml.safe_load(destination_yaml.read_text())
    assert frozen["input_path"] == str((tmp_path / "artifacts" / "designs.npy").resolve())
    np.testing.assert_array_equal(np.load(frozen["input_path"]), designs)
    assert (tmp_path / "artifacts" / "design_provenance.json").read_text() == '{"command": "python -m gen"}\n'


def test_resolve_design_input_path_expands_dir_with_env_var(monkeypatch, tmp_path):
    # Experiments call this directly, so a raw "$SCRATCH/.../<dir>" from a YAML loads.
    (tmp_path / "pool").mkdir()
    monkeypatch.setenv("DESIGN_ROOT", str(tmp_path))

    assert resolve_design_input_path("$DESIGN_ROOT/pool") == str((tmp_path / "pool" / "designs.npy").resolve())
    assert resolve_design_input_path(None) is None
