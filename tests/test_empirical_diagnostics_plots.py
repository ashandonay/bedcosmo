import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bedcosmo.num_visits.empirical.desi.support import lsst_support_limits
from bedcosmo.num_visits.empirical.diagnostics_plots import (
    contributor_density,
    main,
    plot_coverage,
    plot_template_redshifts,
)


def test_contributor_density_includes_endpoints():
    valid = np.array([[True, False], [True, True], [False, True]])
    counts = contributor_density(valid, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(counts, [[1, 0], [1, 2]])
    np.testing.assert_array_equal(counts.sum(axis=0), valid.sum(axis=0))
    with pytest.raises(ValueError, match="entire population"):
        contributor_density(valid, np.array([0.0, 0.5, 2.0]), np.array([0.0, 0.5, 1.0]))


def test_lsst_redshift_limits_match_support_edge_intersections():
    blue, red, lo, hi = lsst_support_limits(1390.0, 9120.0)
    assert 0 < lo < hi
    assert red / (1 + lo) == pytest.approx(9120.0)
    assert blue / (1 + hi) == pytest.approx(1390.0)
    _, _, lo, hi = lsst_support_limits(2000.0, 3000.0)
    assert lo > hi  # No redshift can place every LSST filter inside this support.


def test_coverage_matches_saved_training_split(tmp_path):
    wave = np.array([2000.0, 2010.0, 2020.0])
    weights = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, np.nan]])
    train = np.random.default_rng(42).permutation(3)[:2]
    counts = (np.isfinite(weights) & (weights > 0))[train].sum(axis=0)
    matrix = tmp_path / "matrix.npz"
    np.savez(matrix, wave_rest_aa=wave, relative_ivar=weights, redshift=[0.2, 0.5, 1.0])
    np.savez(tmp_path / "desi_basis.npz", wave_rest_aa=wave[1:], wavelength_contributors=counts)
    (tmp_path / "build_provenance.json").write_text(
        json.dumps(
            {
                "arguments": {"split_seed": 42, "train_fraction": 0.7},
                "selection": {"prior_z_min": 0.2, "prior_z_max": 1.0},
                "factorization": {"required_wavelength_contributors": 1},
            }
        )
    )
    fig = plot_coverage(matrix, tmp_path, 2)
    np.testing.assert_array_equal(fig.axes[1].lines[1].get_ydata(), counts)
    plt.close(fig)
    np.savez(tmp_path / "desi_basis.npz", wave_rest_aa=wave[1:], wavelength_contributors=counts + 1)
    with pytest.raises(ValueError, match="does not match"):
        plot_coverage(matrix, tmp_path)


def test_coverage_reads_data_without_a_prior_build(tmp_path, monkeypatch):
    wave = np.array([1300.0, 1390.0, 9120.0])
    weights = np.tile([0.0, 1.0, 1.0], (150, 1))
    matrix = tmp_path / "matrix.npz"
    np.savez(matrix, wave_rest_aa=wave, relative_ivar=weights, redshift=np.linspace(0.2, 1.0, 150))
    fig = plot_coverage(matrix, redshift_bins=2)
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), [0, 150, 150])
    np.testing.assert_array_equal(fig.axes[1].lines[1].get_ydata(), [0, 105, 105])
    np.testing.assert_array_equal(fig.axes[1].lines[2].get_ydata(), [100, 100])
    np.testing.assert_array_equal(fig.axes[1].lines[3].get_xdata(), [1390, 1390])
    np.testing.assert_array_equal(fig.axes[1].lines[4].get_xdata(), [9120, 9120])
    assert fig.axes[0].get_ylim() == (0.2, 1.0)
    _, _, lo, hi = lsst_support_limits(1390.0, 9120.0)
    np.testing.assert_allclose(fig.axes[0].lines[4].get_ydata(), [lo, lo])
    np.testing.assert_allclose(fig.axes[0].lines[5].get_ydata(), [hi, hi])
    plt.close(fig)
    output = tmp_path / "coverage.png"
    main(["coverage", "--training-matrix", str(matrix), "--output", str(output)])
    assert output.stat().st_size > 0
    data_dir = tmp_path / "bedcosmo/num_visits/desi_training_data"
    data_dir.mkdir(parents=True)
    matrix.rename(data_dir / "desi_rest_frame_training_matrix.npz")
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    main(["coverage", "--output", str(output)])
    assert output.stat().st_size > 0


def test_template_redshifts_selects_real_quality_pass_rows(tmp_path):
    prior = tmp_path / "desi2"
    templates = prior / "templates"
    templates.mkdir(parents=True)
    (templates / "desi2.param").write_text("1 component_01.dat 1.0\n2 component_02.dat 1.0\n")
    for i in (1, 2):
        np.savetxt(templates / f"component_{i:02d}.dat", [[2000, 1], [4000, 2], [9000, 3]])
    pd.DataFrame(
        {
            "z": [0.3, 0.7, 1.0, 1.2],
            "quality_pass": [True, True, True, False],
            "a1": [0.9, 0.05, 0.4, 0.8],
            "a2": [0.1, 0.95, 0.6, 0.2],
        }
    ).to_csv(prior / "desi_eazy_empirical_weights.csv", index=False)
    fig = plot_template_redshifts(prior)
    assert fig.axes[3].get_title() == "N = 2"
    assert fig.axes[7].get_title() == "N = 3"
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [0.5, 1.0, 1.5])
    np.testing.assert_allclose(fig.axes[1].lines[0].get_xdata(), np.array([2000, 4000, 9000]) * 1.3)
    np.testing.assert_allclose(fig.axes[2].lines[0].get_xdata(), np.array([2000, 4000, 9000]) * 2.0)
    plt.close(fig)
    output = tmp_path / "plot.png"
    main(["template-redshifts", "--prior-dir", str(prior), "--output", str(output)])
    assert output.stat().st_size > 0
    with pytest.raises(ValueError, match="threshold"):
        plot_template_redshifts(prior, 0)
