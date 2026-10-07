from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml

from bedcosmo.artifacts import resolve_design_args_input_path
from bedcosmo.num_tracers import design as tracers_design
from bedcosmo.num_visits import design as visits_design
from bedcosmo.num_visits.experiment import fiducial_nvisits


def test_num_visits_band_subset_sums_to_subset_budget():
    visits = visits_design.generate_designs(
        bands=["g", "r", "i"], n_target=20, ratio_min=0.7, ratio_max=1.3
    )
    nominal = np.array([fiducial_nvisits[b] for b in "gri"])

    assert visits.shape == (20, 3)
    assert np.all(visits.sum(axis=1) == nominal.sum())
    assert np.any(np.all(visits == nominal, axis=1))


def test_num_visits_rejects_unknown_band():
    with pytest.raises(ValueError, match="Unknown bands"):
        visits_design.nominal_visits(["g", "q"])


def test_num_visits_main_writes_design_args(tmp_path):
    out_dir = tmp_path / "experiments"
    designs_dir = tmp_path / "designs"

    visits = visits_design.main([
        "--bands", "ri", "--n-target", "5", "--name", "ri_test",
        "--out-dir", str(out_dir), "--designs-dir", str(designs_dir),
    ])

    design_dir = designs_dir / "ri_test"
    design_args = yaml.safe_load((out_dir / "design_args_ri_test.yaml").read_text())
    assert design_args["labels"] == ["r", "i"]
    assert design_args["input_type"] == "variable"
    assert design_args["input_path"] == str(design_dir)
    # The pipeline resolves the design dir to its designs.npy.
    resolved = resolve_design_args_input_path(design_args)
    assert resolved["input_path"] == str(design_dir / "designs.npy")
    np.testing.assert_array_equal(np.load(resolved["input_path"]), visits)
    assert (design_dir / "designs.png").exists()

    provenance = json.loads((design_dir / "provenance.json").read_text())
    assert provenance["command"].startswith("python -m bedcosmo.num_visits.design --bands ri")
    assert provenance["args"]["n_target"] == 5
    assert provenance["labels"] == ["r", "i"]
    assert provenance["budget"] == int(sum(fiducial_nvisits[b] for b in "ri"))
    assert provenance["shape"] == [5, 2]
    assert provenance["sha256"] == hashlib.sha256((design_dir / "designs.npy").read_bytes()).hexdigest()
    assert len(provenance["git"]["commit"]) == 40


def test_num_tracers_main_writes_design_args(tmp_path):
    out_dir = tmp_path / "experiments"
    designs_dir = tmp_path / "designs"

    pool = tracers_design.main([
        "--out-dir", str(out_dir), "--designs-dir", str(designs_dir), "--no-plot",
        "pool", "--sum-lower", "1.0", "--sum-upper", "1.0", "--n-target", "10",
        "--name", "pool_test",
    ])

    design_args = yaml.safe_load((out_dir / "design_args_pool_test.yaml").read_text())
    assert design_args["labels"] == tracers_design.LABELS
    resolved = resolve_design_args_input_path(design_args)
    np.testing.assert_array_equal(np.load(resolved["input_path"]), pool)
    provenance = json.loads((designs_dir / "pool_test" / "provenance.json").read_text())
    assert provenance["command"].startswith("python -m bedcosmo.num_tracers.design --out-dir")
    assert provenance["args"]["mode"] == "pool"
    assert provenance["args"]["n_target"] == 10


def test_num_tracers_scaled_writes_one_design_dir_per_scale(tmp_path):
    designs_dir = tmp_path / "designs"

    tracers_design.main([
        "--out-dir", str(tmp_path / "experiments"), "--designs-dir", str(designs_dir),
        "scaled", "--scales", "1.0", "1.1",
    ])

    for tag, scale in (("00", 1.0), ("10", 1.1)):
        design_dir = designs_dir / f"nominal_scaled_p{tag}"
        assert np.load(design_dir / "designs.npy").sum() == pytest.approx(scale)
        assert (design_dir / "designs.png").exists()
        assert json.loads((design_dir / "provenance.json").read_text())["scale"] == scale


def test_num_visits_yaml_option_names_design_args(tmp_path):
    out_dir = tmp_path / "experiments"

    visits_design.main([
        "--bands", "ri", "--n-target", "5", "--name", "ri_test",
        "--yaml", "design_args_extreme.yaml",
        "--out-dir", str(out_dir), "--designs-dir", str(tmp_path / "designs"),
    ])

    assert [p.name for p in out_dir.iterdir()] == ["design_args_extreme.yaml"]
    design_args = yaml.safe_load((out_dir / "design_args_extreme.yaml").read_text())
    assert design_args["input_path"] == str(tmp_path / "designs" / "ri_test")


def test_num_visits_yaml_option_rejects_paths(tmp_path):
    with pytest.raises(SystemExit):
        visits_design.main(["--yaml", "sub/design_args_x.yaml", "--out-dir", str(tmp_path)])


def test_num_tracers_pool_yaml_option(tmp_path):
    out_dir = tmp_path / "experiments"

    tracers_design.main([
        "--out-dir", str(out_dir), "--designs-dir", str(tmp_path / "designs"), "--no-plot",
        "pool", "--sum-lower", "1.0", "--sum-upper", "1.0", "--n-target", "10",
        "--name", "pool_test", "--yaml", "design_args_budget.yaml",
    ])

    assert [p.name for p in out_dir.iterdir()] == ["design_args_budget.yaml"]


def test_num_visits_refuses_to_overwrite_yaml(tmp_path):
    out_dir = tmp_path / "experiments"
    out_dir.mkdir()
    existing = out_dir / "design_args_extreme.yaml"
    existing.write_text("hand-written\n")

    with pytest.raises(FileExistsError):
        visits_design.main([
            "--bands", "ri", "--n-target", "5", "--name", "ri_test",
            "--yaml", "design_args_extreme.yaml",
            "--out-dir", str(out_dir), "--designs-dir", str(tmp_path / "designs"),
        ])

    assert existing.read_text() == "hand-written\n"
    assert not (tmp_path / "designs" / "ri_test").exists()


def test_num_visits_refuses_to_overwrite_design_dir(tmp_path):
    args = ["--bands", "ri", "--n-target", "5", "--name", "ri_test",
            "--out-dir", str(tmp_path / "experiments"), "--designs-dir", str(tmp_path / "designs")]
    visits_design.main(args)

    with pytest.raises(FileExistsError):
        visits_design.main([*args, "--yaml", "design_args_other.yaml"])


def test_num_tracers_refuses_to_overwrite_yaml(tmp_path):
    out_dir = tmp_path / "experiments"
    out_dir.mkdir()
    (out_dir / "design_args_pool_test.yaml").write_text("hand-written\n")

    with pytest.raises(FileExistsError):
        tracers_design.main([
            "--out-dir", str(out_dir), "--designs-dir", str(tmp_path / "designs"), "--no-plot",
            "pool", "--sum-lower", "1.0", "--sum-upper", "1.0", "--n-target", "10",
            "--name", "pool_test",
        ])
