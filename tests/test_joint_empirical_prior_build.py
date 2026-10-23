"""End-to-end self-contained joint prior build and source provenance checks."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_joint_empirical_prior import build

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow


def test_complete_joint_build_from_fitted_population(tmp_path):
    source, out = tmp_path / "source", tmp_path / "joint"
    (source / "templates").mkdir(parents=True)
    rng = np.random.default_rng(11)
    a = rng.uniform(0.1, 0.9, 200)
    rows = np.column_stack(
        [a, 1 - a, np.zeros(200), rng.uniform(3, 5, 200), rng.uniform(0.2, 1, 200)]
    )
    rows = np.vstack([rows, [0, 0.2, 0.8000000000000002, 4, 0.6]])
    table = pd.DataFrame(rows, columns=["a1", "a2", "a3", "log_c_scale", "z"])
    table["quality_pass"] = True
    table.to_csv(source / "desi_eazy_empirical_weights.csv", index=False)
    config = {"template_param": "desi3.param", "flux_unit_scale": 1e-17}
    (source / "prior_args.yaml").write_text(yaml.safe_dump(config))
    (source / "templates" / "desi3.param").write_text(
        "1 component_01.dat 1.0\n2 component_02.dat 1.0\n3 component_03.dat 1.0\n"
    )
    wave = np.linspace(1000, 14000, 300)
    for i in [1, 2, 3]:
        np.savetxt(
            source / "templates" / f"component_{i:02}.dat",
            np.column_stack([wave, (wave / 5000) ** (-i)]),
        )
    provenance = {
        "selection": {
            "prior_z_min": 0.2,
            "prior_z_max": 1,
            "max_chi2_dof": 1.5,
            "n_prior_quality_pass": 201,
        },
        "template": {"rank": 3, "template_param": "desi3.param"},
        "coefficient_scale": {"runtime_flux_unit_scale_cgs": 1e-17},
    }
    (source / "build_provenance.json").write_text(json.dumps(provenance))
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = build(
        source, out, n_samples=1000, epochs=2, validation_samples=1000, plot_samples=80, threads=1
    )
    assert report["status"] == "complete"
    assert report["runtime"]["bed_supported"] is True
    assert report["diagnostics"]["training_aux_boundary_rows_excluded_from_gaussian_plots"] == 4
    for name, contents in before.items():
        assert (source / name).read_bytes() == contents
        assert report["source_sha256"][name] == hashlib.sha256(contents).hexdigest()
    for name, expected in report["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    with np.load(out / "joint_kde_fit.npz") as fitted:
        np.testing.assert_allclose(fitted["training_rows"], rows)
        assert fitted["bandwidth"] == 0.1
    loaded = JointPriorFlow.load(out / "joint_prior_flow.pt")
    assert loaded.metadata["build_provenance"]["status"] == "complete"
    assert np.isfinite(loaded.log_prob(loaded.sample(100, seed=6))).all()
    for name in [
        "build.log",
        "README.md",
        "kde_samples_triangle.png",
        "nf_samples_triangle.png",
        "kde_samples_gaussianized_triangle.png",
        "training_gaussianized_triangle.png",
        "lsst_colors.png",
        "validation/gaussianization.json",
    ]:
        assert (out / name).stat().st_size > 0
    with np.load(out / "lsst_photometry_samples.npz") as photometry:
        assert photometry["kde"].shape == (80, 6)
        assert np.isfinite(photometry["nf"]).all()
    with pytest.raises(FileExistsError):
        build(source, out, n_samples=1000, epochs=2)
    with pytest.raises(ValueError, match="outside"):
        build(source, source / "child")
    with pytest.raises(ValueError, match="positive"):
        build(source, tmp_path / "invalid", bandwidth=float("nan"))
