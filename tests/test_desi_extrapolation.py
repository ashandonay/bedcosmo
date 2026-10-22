"""Checks for the retained template-extension and response-holdout workflow."""

import json

import numpy as np
import pytest

from bedcosmo.num_visits.empirical.desi import extrapolation


@pytest.mark.parametrize("method", ["constant", "powerlaw", "powerlaw_anchored"])
def test_actual_extension_preserves_learned_values_and_weights(method):
    wave = np.arange(1390.0, 9130.0, 10.0)
    basis = np.stack([np.ones(len(wave)), 2 * (wave / 4000) ** -0.5])
    ivar = np.full_like(basis, 100.0)
    original_basis, original_ivar = basis.copy(), ivar.copy()
    extended_wave, bank = extrapolation.extend_basis_for_lsst(
        wave, basis, ivar, (0.0117, 1.6971), method=method
    )
    columns = np.searchsorted(extended_wave, wave)
    assert (extended_wave[0], extended_wave[-1]) == (1180, 10870)
    np.testing.assert_array_equal(bank[:, columns], basis)
    np.testing.assert_array_equal(basis, original_basis)
    np.testing.assert_array_equal(ivar, original_ivar)
    assert np.isfinite(bank).all() and (bank >= 0).all()


def test_no_extension_needed_does_not_require_edge_weights():
    wave = np.arange(1390.0, 9130.0, 10.0)
    basis = np.ones((2, len(wave)))
    exported_wave, exported = extrapolation.extend_basis_for_lsst(
        wave, basis, np.zeros_like(basis), (0.3, 1.2)
    )
    np.testing.assert_array_equal(exported_wave, wave)
    np.testing.assert_array_equal(exported, basis)


def test_zero_weight_anchor_rejected_and_zero_flux_anchor_retained():
    wave = np.arange(1000.0, 2010.0, 10.0)
    target = np.array([900.0])
    with pytest.raises(ValueError, match="Local anchor"):
        extrapolation.extrapolate_edge(
            wave, np.ones(len(wave)), np.zeros(len(wave)), 1000, target, side="blue"
        )
    values = extrapolation.extrapolate_edge(
        wave,
        np.zeros(len(wave)),
        np.ones(len(wave)),
        1000,
        target,
        side="blue",
        method="powerlaw_anchored",
    )
    np.testing.assert_array_equal(values, np.zeros(1))


def test_extension_workflow_saves_actual_banks_and_preserves_source(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    wave = np.arange(1390.0, 9130.0, 10.0)
    basis = np.stack([np.ones(len(wave)), np.full(len(wave), 2.0)])
    np.savez(
        prior / "desi_basis.npz",
        wave_rest_aa=wave,
        basis=basis,
        estimated_template_ivar=np.ones_like(basis),
    )
    source = (prior / "desi_basis.npz").read_bytes()
    (prior / "build_provenance.json").write_text(
        json.dumps(
            {
                "selection": {"prior_z_min": 0.0117, "prior_z_max": 1.6971},
            }
        )
    )
    output = tmp_path / "extensions"
    export_wave, banks = extrapolation.compare_template_extensions(prior, output)
    assert set(banks) == {"constant", "powerlaw_500_anchored", "powerlaw_2000_anchored"}
    for bank in banks.values():
        np.testing.assert_allclose(bank[0], 1.0)
        np.testing.assert_allclose(bank[1], 2.0)
        np.testing.assert_array_equal(bank[:, np.searchsorted(export_wave, wave)], basis)
    assert (prior / "desi_basis.npz").read_bytes() == source
    assert (output / "template_extensions.npz").is_file()
    assert (output / "templates_actual.png").is_file()
    assert (output / "templates_actual.pdf").is_file()


@pytest.mark.parametrize("failed_powerlaw", [False, True])
def test_response_holdouts_are_paired_and_count_numerical_failures(
    tmp_path, monkeypatch, failed_powerlaw
):
    from bedcosmo.num_visits.empirical import fit_sed_prior_kde, simplex

    prior = tmp_path / "prior"
    prior.mkdir()
    wave = np.arange(1000.0, 12010.0, 10.0)
    basis = np.array([np.full(len(wave), 2.0), np.full(len(wave), 3.0)])
    np.savez(
        prior / "desi_basis.npz",
        wave_rest_aa=wave,
        basis=basis,
        estimated_template_ivar=np.ones_like(basis),
    )
    (prior / "sed_prior_kde_native.joblib").write_bytes(b"mock KDE input")
    monkeypatch.setattr(
        fit_sed_prior_kde, "load_sed_prior_kde", lambda path: {"parameterization": "ilr"}
    )
    monkeypatch.setattr(
        fit_sed_prior_kde, "sample_sed_prior", lambda kde, n, seed: np.zeros((n, 4))
    )
    monkeypatch.setattr(
        simplex,
        "split_feature_matrix",
        lambda features, rank, parameterization: (
            np.array([[0.4, 0.6], [0.7, 0.3]]),
            np.zeros(2),
            np.array([0.4, 0.8]),
        ),
    )
    real_fit = extrapolation.extrapolate_edge

    def edge_fit(*args, **kwargs):
        if failed_powerlaw and kwargs["method"] == "powerlaw_anchored":
            raise ValueError("Power-law continuum fit failed: synthetic numerical failure")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(extrapolation, "extrapolate_edge", edge_fit)
    output = tmp_path / "evaluation"
    errors = extrapolation.compare_filter_response_holdouts(
        prior,
        output,
        response_fractions=(0.25,),
        powerlaw_windows=(500, 2000),
        n_draws=2,
        max_samples=2,
    )
    assert len(errors) == 12
    for _, group in errors.groupby(["band", "draw_index"]):
        assert len(group) == 3
        assert group.reference_mag.nunique() == 1
    np.testing.assert_allclose(errors.realized_response_fraction, 0.25, atol=1e-5)
    valid = errors[errors.status == "ok"]
    np.testing.assert_allclose(valid.delta_mag, 0, atol=1e-12)
    meta = json.loads((output / "response_comparison.json").read_text())
    assert meta["n_paired_draws_per_panel"] == {"u": 2, "y": 2}
    for row in meta["summary"]:
        expect_failure = failed_powerlaw and row["method"] != "constant"
        assert row["n_failed"] == (2 if expect_failure else 0)
        assert row["fraction_failed_or_abs_above_0.1"] == int(expect_failure)
