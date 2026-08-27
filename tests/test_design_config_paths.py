from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from bedcosmo.util import resolve_design_args_input_path, snapshot_design_args_config


def test_resolve_design_input_path_expands_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DESIGN_ROOT", str(tmp_path))

    resolved = resolve_design_args_input_path(
        {"input_designs_path": "$DESIGN_ROOT/designs.npy"}
    )

    assert resolved["input_designs_path"] == str((tmp_path / "designs.npy").resolve())


def test_resolve_design_input_path_relative_to_yaml(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "design_args.yaml"

    resolved = resolve_design_args_input_path(
        {"input_designs_path": "../arrays/extreme.npy"},
        config_path,
    )

    assert resolved["input_designs_path"] == str(
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
        "input_type: variable\ninput_designs_path: $DESIGN_FILE\n"
    )
    destination_yaml = tmp_path / "artifacts" / "design_args.yaml"

    snapshot_design_args_config(source_yaml, destination_yaml)

    frozen = yaml.safe_load(destination_yaml.read_text())
    frozen_path = Path(frozen["input_designs_path"])
    assert frozen_path == (destination_yaml.parent / "designs.npy").resolve()
    np.testing.assert_array_equal(np.load(frozen_path), designs)
