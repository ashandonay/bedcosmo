from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

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


def test_num_visits_main_writes_design_dir(tmp_path):
    designs_dir = tmp_path / "designs"

    visits = visits_design.main([
        "--bands", "ri", "--n-target", "5", "--name", "ri_test", "--designs-dir", str(designs_dir),
    ])

    design_dir = designs_dir / "ri_test"
    assert sorted(p.name for p in design_dir.iterdir()) == ["designs.npy", "designs.png", "provenance.json"]
    # The pipeline resolves a design dir passed as input_path to its designs.npy.
    resolved = resolve_design_args_input_path({"input_path": str(design_dir)})
    assert resolved["input_path"] == str(design_dir / "designs.npy")
    np.testing.assert_array_equal(np.load(resolved["input_path"]), visits)

    provenance = json.loads((design_dir / "provenance.json").read_text())
    assert provenance["command"].startswith("python -m bedcosmo.num_visits.design --bands ri")
    assert provenance["args"]["n_target"] == 5
    assert provenance["labels"] == ["r", "i"]
    assert provenance["budget"] == int(sum(fiducial_nvisits[b] for b in "ri"))
    assert provenance["shape"] == [5, 2]
    assert provenance["sha256"] == hashlib.sha256((design_dir / "designs.npy").read_bytes()).hexdigest()
    assert len(provenance["git"]["commit"]) == 40


def test_num_tracers_pool_writes_design_dir(tmp_path):
    designs_dir = tmp_path / "designs"

    pool = tracers_design.main([
        "--designs-dir", str(designs_dir), "--no-plot",
        "pool", "--sum-lower", "1.0", "--sum-upper", "1.0", "--n-target", "10",
        "--name", "pool_test",
    ])

    design_dir = designs_dir / "pool_test"
    assert sorted(p.name for p in design_dir.iterdir()) == ["designs.npy", "provenance.json"]
    np.testing.assert_array_equal(np.load(design_dir / "designs.npy"), pool)
    provenance = json.loads((design_dir / "provenance.json").read_text())
    assert provenance["command"].startswith("python -m bedcosmo.num_tracers.design --designs-dir")
    assert provenance["args"]["mode"] == "pool"
    assert provenance["args"]["n_target"] == 10
    assert provenance["labels"] == tracers_design.LABELS


def test_num_tracers_scaled_writes_one_design_dir_per_scale(tmp_path):
    designs_dir = tmp_path / "designs"

    tracers_design.main(["--designs-dir", str(designs_dir), "scaled", "--scales", "1.0", "1.1"])

    for tag, scale in (("00", 1.0), ("10", 1.1)):
        design_dir = designs_dir / f"nominal_scaled_p{tag}"
        assert np.load(design_dir / "designs.npy").sum() == pytest.approx(scale)
        assert (design_dir / "designs.png").exists()
        assert json.loads((design_dir / "provenance.json").read_text())["scale"] == scale


def test_num_visits_refuses_to_overwrite_design_dir(tmp_path):
    args = ["--bands", "ri", "--n-target", "5", "--name", "ri_test", "--designs-dir", str(tmp_path)]
    visits_design.main(args)
    before = (tmp_path / "ri_test" / "provenance.json").read_text()

    with pytest.raises(FileExistsError):
        visits_design.main([*args, "--seed", "1"])

    assert (tmp_path / "ri_test" / "provenance.json").read_text() == before


def test_num_tracers_refuses_to_overwrite_design_dir(tmp_path):
    args = ["--designs-dir", str(tmp_path), "--no-plot",
            "pool", "--sum-lower", "1.0", "--sum-upper", "1.0", "--n-target", "10", "--name", "pool_test"]
    tracers_design.main(args)

    with pytest.raises(FileExistsError):
        tracers_design.main(args)
