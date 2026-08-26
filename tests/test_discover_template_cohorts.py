from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from bedcosmo.num_visits.empirical.discover_template_cohorts import (
    full_fit_chi2_dof_from_statistics,
    solve_subset_nnls,
)
from bedcosmo.num_visits.empirical.fit_eazy_weights_to_desi import _divide_by_continuum
from bedcosmo.num_visits.empirical.plot_template_subset_examples import (
    continuum_sigma_in_observed_frame,
    divide_by_display_continuum,
    ivar_bin_spectrum,
    parse_template_label,
    select_examples,
    valid_display_continuum,
)


def test_batched_subset_solver_matches_pixel_nnls():
    rng = np.random.default_rng(8)
    n_rows = 9
    n_pixels = 80
    n_templates = 5
    design = rng.uniform(0.05, 1.0, size=(n_rows, n_pixels, n_templates))
    truth = rng.uniform(0.0, 2.0, size=(n_rows, n_templates))
    truth[rng.random(truth.shape) < 0.45] = 0.0
    response = np.einsum("npk,nk->np", design, truth) + rng.normal(
        0.0, 0.03, size=(n_rows, n_pixels)
    )
    gram = np.einsum("npi,npj->nij", design, design)
    cross = np.einsum("npi,np->ni", design, response)
    data_norm = np.einsum("np,np->n", response, response)
    subset = (0, 2, 4)

    coefficients, chi2 = solve_subset_nnls(gram, cross, data_norm, subset)

    for row in range(n_rows):
        subset_design = design[row][:, subset]
        expected_coefficients, _ = nnls(subset_design, response[row])
        expected_residual = response[row] - subset_design @ expected_coefficients
        np.testing.assert_allclose(coefficients[row], expected_coefficients, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(chi2[row], expected_residual @ expected_residual, rtol=2e-8)


def test_full_fit_chi2_from_statistics():
    rng = np.random.default_rng(12)
    n_rows, n_pixels, n_templates = 4, 30, 3
    design = rng.normal(size=(n_rows, n_pixels, n_templates))
    coefficients = rng.normal(size=(n_rows, n_templates))
    response = np.einsum("npk,nk->np", design, coefficients) + 0.1
    statistics = {
        "gram": np.einsum("npi,npj->nij", design, design),
        "cross": np.einsum("npi,np->ni", design, response),
        "data_norm": np.einsum("np,np->n", response, response),
        "n_good": np.full(n_rows, n_pixels),
    }
    table = pd.DataFrame({f"c{i + 1}": coefficients[:, i] for i in range(n_templates)})

    actual = full_fit_chi2_dof_from_statistics(table, statistics)
    residual = response - np.einsum("npk,nk->np", design, coefficients)
    expected = np.einsum("np,np->n", residual, residual) / (n_pixels - n_templates)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_subset_example_selection_includes_representative_and_rich_members():
    members = pd.DataFrame(
        {
            "targetid": [1, 2, 3, 4, 5],
            "a_1": [0.34, 0.95, 0.02, 0.03, 0.60],
            "a_2": [0.33, 0.03, 0.95, 0.02, 0.20],
            "a_3": [0.33, 0.02, 0.03, 0.95, 0.20],
            "z": [0.5, 0.2, 0.8, 1.1, 0.6],
            "delta_chi2_dof": [0.001, 0.002, 0.003, 0.004, 0.0015],
        }
    )

    selected = select_examples(members, parse_template_label("T1+T7+T9"))

    assert selected["example_role"].tolist() == [
        "representative",
        "T1-rich",
        "T7-rich",
        "T9-rich",
    ]
    assert selected["targetid"].is_unique


def test_ivar_display_binning():
    wave = np.array([0.2, 0.8, 1.2])
    flux = np.array([1.0, 3.0, 5.0])
    ivar = np.array([1.0, 3.0, 4.0])

    binned_wave, binned_flux, binned_error = ivar_bin_spectrum(
        wave, flux, ivar, np.ones(3, dtype=bool), 1.0
    )

    np.testing.assert_allclose(binned_wave, [0.65, 1.2])
    np.testing.assert_allclose(binned_flux, [2.5, 5.0])
    np.testing.assert_allclose(binned_error, [0.5, 0.5])


def test_display_continuum_division_masks_invalid_pixels_without_warnings():
    values = np.array([2.0, 3.0, 4.0, 5.0])
    continuum = np.array([2.0, 0.0, np.nan, 10.0])
    safe = np.array([True, False, False, True])

    divided = divide_by_display_continuum(values, continuum, safe)

    np.testing.assert_allclose(divided[[0, 3]], [1.0, 0.5])
    assert np.isnan(divided[[1, 2]]).all()


def test_display_continuum_is_edge_safe_and_masks_tiny_values():
    wave = np.arange(1000.0)
    flux = np.ones_like(wave)
    flux[-1] = -1.0
    good = np.ones_like(wave, dtype=bool)

    _, continuum, continuum_ivar = _divide_by_continuum(wave, flux, good, cont_sigma_aa=100.0)
    safe = valid_display_continuum(
        np.array([1.0, 1.0, 0.04, 0.0, np.nan]),
        np.ones(5, dtype=bool),
        min_fraction=0.05,
    )

    assert continuum[-1] > 0.9
    assert continuum_ivar is None
    assert safe.tolist() == [True, True, False, False, False]


def test_display_continuum_downweights_noisy_outlier():
    wave = np.arange(1000.0)
    flux = np.ones_like(wave)
    flux[-1] = -100.0
    ivar = np.ones_like(wave)
    ivar[-1] = 1e-8
    good = np.ones_like(wave, dtype=bool)

    _, continuum, continuum_ivar = _divide_by_continuum(
        wave,
        flux,
        good,
        cont_sigma_aa=100.0,
        ivar=ivar,
    )

    assert continuum[-1] > 0.99
    assert continuum_ivar is not None


def test_display_continuum_masks_low_signal_to_noise_regions():
    continuum = np.ones(3)
    continuum_ivar = np.array([100.0, 4.0, 0.0])
    good = np.ones(3, dtype=bool)

    safe = valid_display_continuum(
        continuum,
        good,
        min_fraction=0.05,
        continuum_ivar=continuum_ivar,
        min_snr=3.0,
    )

    assert safe.tolist() == [True, False, False]


def test_rest_continuum_scale_converts_to_observed_frame():
    assert continuum_sigma_in_observed_frame(250.0, "rest", 1.2) == 550.0
    assert continuum_sigma_in_observed_frame(250.0, "observed", 1.2) == 250.0
