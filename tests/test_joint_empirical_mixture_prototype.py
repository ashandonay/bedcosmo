"""Tests of the joint physical-coordinate mixture and synthetic photometry."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prototype_joint_empirical_mixture import JointPatternMixture, lsst_magnitudes


def test_joint_patterns_and_dependence_are_preserved():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 0.7, 1000)
    rows = np.column_stack([x, 1 - x, np.zeros(1000), 3 + x, 0.1 + x])
    rows[:250, :3] = [0, 0, 1]
    model = JointPatternMixture(rows, bandwidth=0.1)
    draws = model.sample(20000, seed=4)
    np.testing.assert_allclose(draws[:, :3].sum(axis=1), 1)
    patterns, counts = np.unique(draws[:, :3] > 0, axis=0, return_counts=True)
    np.testing.assert_array_equal(patterns, [[False, False, True], [True, True, False]])
    assert abs(counts[0] / len(draws) - 0.25) < 0.015
    active = draws[:, 0] > 0
    assert np.corrcoef(draws[active, 0], draws[active, -1])[0, 1] > 0.97
    np.testing.assert_array_equal(draws, model.sample(20000, seed=4))
    assert (draws[:, -1] >= rows[:, -1].min()).all()
    assert (draws[:, -1] <= rows[:, -1].max()).all()


def test_singleton_patterns_retain_their_full_empirical_rows():
    rows = np.array([[1, 0, 2, 0.1], [0, 1, 4, 0.8]])
    model = JointPatternMixture(rows)
    draws = model.sample(500)
    assert all(any(np.array_equal(row, donor) for donor in rows) for row in draws)
    summary = model.pattern_summary(np.array([[0.5, 0.5, 3, 0.4]]))
    assert summary["mass_in_singleton_patterns"] == 1
    assert summary["heldout_mass_in_unseen_patterns"] == 1


def test_zero_bandwidth_is_joint_empirical_bootstrap():
    rows = np.array([[0.2, 0.8, 2, 0.2], [0.7, 0.3, 4, 0.6]])
    model = JointPatternMixture(rows, bandwidth=0)
    draws = model.sample(30)
    assert all(np.min(np.max(abs(rows - row), axis=1)) < 1e-14 for row in draws)


@pytest.mark.parametrize(
    "rows", [[], [[-1, 2, 3, 0.1]] * 2, [[0.2, 0.2, 3, 0.1]] * 2, [[0.2, 0.8, 3, -0.1]] * 2]
)
def test_invalid_rows_fail(rows):
    with pytest.raises(ValueError):
        JointPatternMixture(rows)


def test_photometry_amplitude_and_color_invariance():
    wave = np.linspace(500, 20000, 1000)
    basis = np.ones((2, len(wave)))
    rows = np.array([[0.3, 0.7, 3, 0.1], [0.3, 0.7, 3 + np.log(2), 0.1]])
    mags, bands = lsst_magnitudes(rows, wave, basis)
    assert bands == [
        "lsst2023-u",
        "lsst2023-g",
        "lsst2023-r",
        "lsst2023-i",
        "lsst2023-z",
        "lsst2023-y",
    ]
    np.testing.assert_allclose(mags[1] - mags[0], -2.5 * np.log10(2))
    np.testing.assert_allclose(np.diff(mags[1]), np.diff(mags[0]))
    with pytest.raises(ValueError, match="cover"):
        lsst_magnitudes(rows, np.linspace(4000, 5000, 100), np.ones((2, 100)))


def test_full_comparison_uses_train_split_and_saves_reloadable_fit(tmp_path, capsys):
    from prototype_joint_empirical_mixture import run

    rng = np.random.default_rng(11)
    weights = rng.dirichlet(np.ones(8), size=150)
    weights[:50, :2] = 0
    weights /= weights.sum(axis=1, keepdims=True)
    rows = np.column_stack([weights, rng.normal(4, 0.2, 150), rng.uniform(0.1, 0.6, 150)])
    source = tmp_path / "prior"
    templates = source / "templates"
    templates.mkdir(parents=True)
    wave = np.linspace(500, 20000, 500)
    for i in range(8):
        np.savetxt(templates / f"t{i}.dat", np.column_stack([wave, (wave / 4000) ** (-0.1 * i)]))
    (templates / "bank.param").write_text("\n".join(f"{i + 1} t{i}.dat 1" for i in range(8)))
    (source / "prior_args.yaml").write_text(
        "flux_unit_scale: 1.0e-17\ntemplate_param: bank.param\n"
    )
    samples = tmp_path / "input.npz"
    np.savez(
        samples, training=rows, feature_names=[f"a{i+1}" for i in range(8)] + ["log_c_scale", "z"]
    )
    report = run(samples, source, tmp_path / "results", n_samples=100)
    assert report["n_train"] == 120
    assert report["n_heldout"] == 30
    assert len(report["lsst_color_ks_vs_heldout"]["joint"]) == 5
    with np.load(tmp_path / "results" / "samples.npz") as saved:
        assert not set(saved["train_indices"]) & set(saved["heldout_indices"])
        with np.load(tmp_path / "results" / "joint_fit.npz") as fitted:
            model = JointPatternMixture(
                fitted["training_rows"], bandwidth=float(fitted["bandwidth"])
            )
        np.testing.assert_array_equal(saved["joint"], model.sample(100, seed=17))
    for name in ["relationships.png", "lsst_colors.png", "pattern_coverage.png"]:
        assert (tmp_path / "results" / name).stat().st_size > 0
