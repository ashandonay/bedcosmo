"""Tests for exact support, simplex coordinates and mixed NF prior scoring."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_empirical_prior_flow import JointPriorFlow, build_flow, decode_rows, encode_rows, train
from prototype_joint_empirical_mixture import JointPatternMixture


def test_stick_roundtrip_and_physical_jacobian():
    lower, upper = np.array([0, 0]), np.array([10, 2])
    rows = np.array([[0.2, 0.3, 0.5, 5, 0.6], [0.1, 0.8, 0.1, 4, 1.5]])
    features, jac = encode_rows(rows, lower, upper)
    np.testing.assert_allclose(decode_rows(features, rows[:, :-2] > 0, lower, upper), rows)
    step = 1e-6
    matrix = np.empty((4, 4))
    for j in range(4):
        plus, minus = rows[:1].copy(), rows[:1].copy()
        index = [0, 1, 3, 4][j]
        plus[0, index] += step
        minus[0, index] -= step
        if j < 2:
            plus[0, 2] -= step
            minus[0, 2] += step
        matrix[:, j] = (
            encode_rows(plus, lower, upper)[0][0] - encode_rows(minus, lower, upper)[0][0]
        ) / (2 * step)
    assert np.linalg.slogdet(matrix)[1] == pytest.approx(jac[0], abs=1e-8)


def test_different_masks_same_size_and_one_active_component_roundtrip():
    lower, upper = np.array([0, 0]), np.array([10, 2])
    rows = np.array([[0.3, 0.7, 0, 5, 0.6], [0, 0.2, 0.8, 4, 1]])
    features, _ = encode_rows(rows, lower, upper)
    decoded = decode_rows(features, rows[:, :-2] > 0, lower, upper)
    np.testing.assert_allclose(decoded, rows)
    np.testing.assert_array_equal(decoded[:, :-2] == 0, rows[:, :-2] == 0)
    singleton = np.array([[0, 1, 0, 3, 0.4]])
    features, _ = encode_rows(singleton, lower, upper)
    assert features.shape == (1, 2)
    np.testing.assert_allclose(
        decode_rows(features, singleton[:, :-2] > 0, lower, upper), singleton
    )


def test_train_reload_sample_and_mixed_scoring(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(11)
    a = rng.uniform(0.1, 0.9, 200)
    rows = np.column_stack(
        [a, 1 - a, np.zeros(200), rng.uniform(3, 5, 200), rng.uniform(0.2, 1, 200)]
    )
    atomic = np.array([[0, 0, 1, 4, 0.6]])
    target = JointPatternMixture(np.vstack([rows, atomic]), bandwidth=0.1)
    model = train(target, n_samples=1000, epochs=2, hidden=8, transforms=1, bins=4, patience=2)
    draws = model.sample(2000, seed=9)
    np.testing.assert_allclose(draws[:, :-2].sum(axis=1), 1)
    assert np.isfinite(model.log_prob(draws)).all()
    assert ((draws[:, 2] == 1).sum()) > 0
    assert model.log_prob(atomic)[0] == pytest.approx(-np.log(201))
    assert np.isneginf(model.log_prob(np.array([[0, 0, 1, 4, 0.7]]))[0])
    assert np.isneginf(model.log_prob(np.array([[1, 0, 0, 4, 0.7]]))[0])
    assert np.isneginf(model.log_prob(np.array([[0.5, 0.5, 0, 20, 0.7]]))[0])
    model.metadata["feature_names"] = ["a1", "a2", "a3", "log_c_scale", "z"]
    model.save(tmp_path / "flow.pt")
    loaded = JointPriorFlow.load(tmp_path / "flow.pt")
    np.testing.assert_array_equal(draws, loaded.sample(2000, seed=9))
    np.testing.assert_array_equal(model.log_prob(draws), loaded.log_prob(draws))
    for physical in [draws, target.sample(2000, seed=43), atomic]:
        ids, groups, jac = loaded.gaussianize(physical)
        restored = loaded.degaussianize(ids, groups)
        np.testing.assert_allclose(restored, physical, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(restored[:, :-2] == 0, physical[:, :-2] == 0)
        base_score = np.log(loaded.probabilities[ids])
        for m, z in groups.items():
            selected = (~loaded.metadata["atomic"][ids]) & (loaded.patterns[ids].sum(axis=1) == m)
            base_score[selected] += -0.5 * (z**2 + np.log(2 * np.pi)).sum(axis=1)
        np.testing.assert_allclose(base_score + jac, loaded.log_prob(physical), atol=3e-6)
    ids, groups, _ = loaded.gaussianize(atomic)
    assert groups == {}
    np.testing.assert_array_equal(loaded.degaussianize(ids, groups), atomic)
    with pytest.raises(ValueError, match="modeled support"):
        loaded.gaussianize(np.array([[1, 0, 0, 4, 0.7]]))
    with pytest.raises(ValueError, match="pattern IDs"):
        loaded.degaussianize(np.array([-1]), {})
    with pytest.raises(ValueError, match="must match"):
        loaded.degaussianize(ids, {2: np.zeros((1, 3))})
    from validate_joint_empirical_prior_flow import validate

    np.savez_compressed(
        tmp_path / "joint_kde_fit.npz", training_rows=target.rows, bandwidth=target.bandwidth
    )
    report = validate(tmp_path / "flow.pt", tmp_path / "validation", n_samples=2000)
    assert report["finite_score_fraction"] == 1
    assert report["number_of_flow_groups"] == 1
    assert (tmp_path / "validation" / "nf_prior_marginals.png").is_file()
    assert (tmp_path / "validation" / "gaussianization_qq.png").is_file()
    assert report["gaussianization"]["kde"]["max_log_density_identity_error"] < 1e-5
    JointPriorFlow(model.metadata, {}).save(tmp_path / "partial.pt")
    with pytest.raises(ValueError, match="Incomplete"):
        JointPriorFlow.load(tmp_path / "partial.pt")


def test_invalid_continuous_boundary_fails_instead_of_epsilon_clipping():
    with pytest.raises(ValueError, match="strictly inside"):
        encode_rows(np.array([[0.3, 0.7, 0, 0.4]]), np.array([0, 0]), np.array([10, 2]))
    with pytest.raises(ValueError, match="same positive"):
        encode_rows(
            np.array([[0.3, 0.7, 3, 0.4], [1, 0, 3, 0.4]]), np.array([0, 0]), np.array([10, 2])
        )


def test_gaussian_coordinates_with_interleaved_counts_masks_and_atoms():
    torch.set_num_threads(1)
    config = {"hidden": [8, 8], "transforms": 2, "bins": 4}
    metadata = {
        "patterns": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
        "counts": np.array([10, 20, 30, 40, 1]),
        "atomic": np.array([False, False, False, False, True]),
        "atomic_rows": np.array([[0, 0, 0, 0, 0]] * 4 + [[0, 0, 1, 5, 1]]),
        "lower": np.array([0, 0]),
        "upper": np.array([10, 2]),
    }
    model = JointPriorFlow(
        metadata,
        {
            m: {
                "flow": build_flow(m + 1, 3, config).eval(),
                "mu": np.zeros(m + 1),
                "std": np.ones(m + 1),
            }
            for m in [1, 2, 3]
        },
    )
    ids = np.random.default_rng(12).integers(0, 5, 500)
    continuous = ~metadata["atomic"][ids]
    counts = model.patterns[ids].sum(axis=1)
    groups = {
        m: np.random.default_rng(m).normal(size=(np.sum(continuous & (counts == m)), m + 1))
        for m in [1, 2, 3]
    }
    rows = model.degaussianize(ids, groups)
    restored_ids, restored_groups, _ = model.gaussianize(rows)
    np.testing.assert_array_equal(restored_ids, ids)
    for m, values in groups.items():
        np.testing.assert_allclose(restored_groups[m], values, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(
        rows[~continuous], np.tile([0, 0, 1, 5, 1], (np.sum(~continuous), 1))
    )
    with pytest.raises(ValueError, match="invalid shape"):
        model.degaussianize(ids, {**groups, 2: np.zeros((1, 3))})
