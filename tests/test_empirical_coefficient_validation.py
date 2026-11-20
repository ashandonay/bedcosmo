"""Checks for nested coefficient-prior evaluation and frozen-run auditing."""

import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_empirical_coefficient_prior import (
    audit_run_snapshots,
    distribution_metrics,
    nested_indices,
    raw_coefficients,
    support_metrics,
)


def test_nested_random_split_is_disjoint_complete_and_reproducible():
    fit, validation, test = nested_indices(1000, 503)
    assert [len(fit), len(validation), len(test)] == [640, 160, 200]
    assert len(set(fit) | set(validation) | set(test)) == 1000
    assert not set(fit) & set(test)
    assert not set(validation) & set(test)
    for a, b in zip((fit, validation, test), nested_indices(1000, 503)):
        np.testing.assert_array_equal(a, b)


def test_region_holdout_is_never_used_in_fit_or_validation():
    heldout = np.arange(30, 65)
    fit, validation, test = nested_indices(1000, 503, heldout=heldout)
    np.testing.assert_array_equal(test, heldout)
    assert not set(heldout) & (set(fit) | set(validation))
    assert len(set(fit) | set(validation) | set(test)) == 1000


def test_physical_joint_metric_detects_dependence_with_identical_marginals():
    rng = np.random.default_rng(17)
    x = rng.uniform(0.1, 0.9, 500)
    reference = np.column_stack([x, 1 - x, x * 2, x * 3])
    shuffled = reference.copy()
    shuffled[:, -2:] = shuffled[rng.permutation(len(shuffled)), -2:]
    exact = distribution_metrics(reference, reference, reference, seed=1)
    changed = distribution_metrics(reference, shuffled, reference, seed=1)
    assert abs(exact["mmd2"]) < 1e-14
    assert changed["max_marginal_ks"] == 0
    assert changed["mmd2"] > 0.005
    assert changed["correlation_error"] > 1


def test_raw_coefficients_and_discrete_pattern_discrepancies():
    reference = np.array([[1, 0, np.log(2), 0.1], [0, 1, np.log(4), 0.2]])
    np.testing.assert_allclose(raw_coefficients(reference), [[2, 0, 0.1], [0, 4, 0.2]])
    assert support_metrics(reference, reference)["pattern_total_variation"] == 0
    smoothed = reference.copy()
    smoothed[:, :2] = [[0.99, 0.01], [0.01, 0.99]]
    assert support_metrics(reference, smoothed)["pattern_total_variation"] == 1
    assert support_metrics(reference, smoothed)["max_absolute_zero_probability_error"] == 0.5


def test_audit_reads_frozen_kde_instead_of_assuming_default(tmp_path):
    run = tmp_path / "exp" / "run"
    empirical = run / "artifacts" / "empirical"
    empirical.mkdir(parents=True)
    (run / "meta.yaml").write_text("start_time: 1000\n")
    (run / "artifacts" / "prior_args.yaml").write_text(
        "template_source: desi8\ndensity_type: flow\n"
    )
    logs = run / "artifacts" / "logs"
    logs.mkdir()
    (logs / "train.log").write_text(
        "density_type=flow; pool n=65536\nloaded flows; rebuilt pool from native flow\n"
    )
    joblib.dump(
        {
            "kde": SimpleNamespace(bandwidth=0.07, kernel="gaussian"),
            "parameterization": "ilr",
            "training_x": np.ones((12, 9)),
        },
        empirical / "sed_prior_kde_native.joblib",
    )
    result = audit_run_snapshots(tmp_path)
    assert len(result) == 1
    assert result[0]["bandwidth"] == 0.07
    assert result[0]["n_training"] == 12
    assert len(result[0]["runtime_evidence"]) == 2
    assert len(result[0]["kde_sha256"]) == 64


def test_small_nested_splits_fail_loudly():
    with pytest.raises(ValueError, match="two rows"):
        nested_indices(5, 1)


def test_comparison_runs_without_tracking_data_and_reports_runtime(tmp_path, monkeypatch):
    import pandas as pd
    import validate_empirical_coefficient_prior as comparison

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        build_feature_matrix,
        fit_sed_prior_kde,
        pack_kde_artifact,
    )

    rng = np.random.default_rng(17)
    coefficients = rng.dirichlet(np.ones(8), size=150)
    table = pd.DataFrame(coefficients, columns=[f"a{i+1}" for i in range(8)])
    table["log_c_scale"] = rng.normal(4, 0.3, len(table))
    table["z"] = rng.uniform(0.1, 1, len(table))
    table["quality_pass"] = True
    table["healpix"] = np.repeat([1, 2, 3], 50)
    for i in range(8):
        table[f"c{i+1}"] = coefficients[:, i] * np.exp(table["log_c_scale"])
    source = tmp_path / "prior"
    source.mkdir()
    table.to_csv(source / "desi_eazy_empirical_weights.csv", index=False)
    features, names, _ = build_feature_matrix(table)
    kde, scaler = fit_sed_prior_kde(features)
    artifact = pack_kde_artifact(
        kde,
        scaler,
        names,
        features,
        metadata={"n_templates": 8, "z_min": None, "max_chi2_dof": 1.5},
    )
    joblib.dump(artifact, source / "sed_prior_kde_native.joblib")
    (source / "build_provenance.json").write_text("{}")
    monkeypatch.setattr(comparison, "BANDWIDTHS", [0.05])
    monkeypatch.setattr(comparison, "METRIC_SEEDS", [304])
    output = tmp_path / "comparison"
    result = comparison.run(source, output, n_samples=700)
    assert len(result["test_results"]) == 24
    assert all(r["fit_and_sample_seconds"] >= 0 for r in result["test_results"])
    assert all(
        r["selected_bandwidth"] == 0.3
        for r in result["test_results"]
        if r["model"] == "ilr_default_0.3"
    )
    assert (output / "comparison.md").is_file()
    assert "Fit + draw seconds" in (output / "comparison.md").read_text()
    assert not (output / "production_audit.json").exists()
