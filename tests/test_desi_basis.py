"""Tests for the direct-DESI nonnegative spectral basis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bedcosmo.num_visits.empirical.desi.build_prior import (
    normalize_basis_for_export,
    physical_prior_coefficients,
    resolve_prior_redshift_limits,
    write_template_bank,
)
from bedcosmo.num_visits.empirical.desi.evaluate_factorization_methods import (
    fit_anls,
    shared_initialization,
)
from bedcosmo.num_visits.empirical.desi.fit_basis import (
    desi_covered_lsst_color_rms,
)
from bedcosmo.num_visits.empirical.desi.support import (
    largest_contiguous_region,
    lsst_demand_weighted_coverage,
    select_wavelength_support,
)
from bedcosmo.num_visits.empirical.desi.training_matrix import (
    bin_rest_frame_spectrum,
    derive_rest_frame_grid,
    discover_desi_manifest,
)
from bedcosmo.num_visits.empirical.desi.weighted_nmf import (
    fit_weighted_nmf,
    infer_coefficients,
    weighted_reconstruction_error,
)


def test_prior_redshift_limits_derived_from_retained_support():
    wave = np.array([1390.0, 4000.0, 9120.0])
    lo, hi = resolve_prior_redshift_limits(wave)
    assert lo == pytest.approx(10990.0 / 9120.0 - 1.0)
    assert hi == pytest.approx(3199.0 / 1390.0 - 1.0)
    assert resolve_prior_redshift_limits(wave, 0.3, 1.2) == (0.3, 1.2)
    assert resolve_prior_redshift_limits(wave, prior_z_max=1.2) == (lo, 1.2)
    with pytest.raises(ValueError, match="within LSST-supported"):
        resolve_prior_redshift_limits(wave, 0.1, 1.2)
    with pytest.raises(ValueError, match="within LSST-supported"):
        resolve_prior_redshift_limits(wave, 0.3, 1.4)
    with pytest.raises(ValueError, match="nonempty"):
        resolve_prior_redshift_limits(np.array([2000.0, 3000.0]))


def test_rest_frame_binning_combines_pixels_and_masks_bad_data():
    grid = np.array([4000.0, 4010.0, 4020.0])
    wave = np.array([8000.0, 8001.0, 8020.0, 8040.0])
    flux = np.array([2.0, 4.0, 6.0, 8.0])
    ivar = np.ones(4)
    mask = np.array([0, 0, 1, 0])
    values, weights, scale = bin_rest_frame_spectrum(wave, flux, ivar, mask, 1.0, grid)
    assert np.isfinite(scale)
    assert weights[0] > 0
    assert values[0] == 3.0 / scale
    assert weights[1] == 0
    assert values[2] == 8.0 / scale


def test_direct_manifest_selects_quality_redrock_galaxies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    coadd = tmp_path / "coadd.fits"
    redrock = tmp_path / "redrock.fits"
    coadd.touch()
    redrock.touch()
    redshifts = np.array(
        [
            (1, 0.4, 0, "GALAXY"),
            (2, 0.5, 1, "GALAXY"),
            (3, 0.6, 0, "QSO"),
            (4, 0.005, 0, "GALAXY"),
            (5, 0.8, 0, "GALAXY"),
            (6, np.nan, 0, "GALAXY"),
        ],
        dtype=[("TARGETID", "i8"), ("Z", "f8"), ("ZWARN", "i8"), ("SPECTYPE", "U10")],
    )
    fibermap = np.array([(1,), (2,), (3,), (4,), (6,)], dtype=[("TARGETID", "i8")])

    monkeypatch.setattr(
        "bedcosmo.num_visits.empirical.desi.training_matrix.get_local_desi_paths",
        lambda *args, **kwargs: (coadd, redrock),
    )

    def fake_getdata(path, extension):
        return redshifts if extension == "REDSHIFTS" else fibermap

    monkeypatch.setattr(
        "bedcosmo.num_visits.empirical.desi.training_matrix.fits.getdata",
        fake_getdata,
    )
    manifest = discover_desi_manifest([23040], desi_dir=tmp_path)
    assert manifest.to_dict("records") == [{"targetid": 1, "healpix": 23040, "z": 0.4}]


def test_weighted_nmf_recovers_nonnegative_low_rank_data_with_missing_pixels():
    rng = np.random.default_rng(12)
    true_basis = np.array(
        [
            np.linspace(0.2, 1.0, 24),
            np.linspace(1.0, 0.2, 24),
        ]
    )
    true_coefficients = rng.uniform(0.2, 2.0, size=(50, 2))
    flux = true_coefficients @ true_basis
    weights = np.ones_like(flux)
    weights[rng.random(weights.shape) < 0.2] = 0
    _, basis, losses = fit_weighted_nmf(
        flux, weights, 2, iterations=5, smooth_sigma_pixels=0, seed=2
    )
    inferred = infer_coefficients(flux, weights, basis)
    error = weighted_reconstruction_error(flux, weights, inferred, basis)
    assert np.all(basis >= 0)
    assert losses[-1] < losses[0]
    assert np.median(error) < 1e-3


def test_signed_data_anls_decreases_weighted_objective_without_clipping():
    rng = np.random.default_rng(19)
    true_coefficients = rng.uniform(0.1, 1.0, size=(35, 2))
    true_basis = rng.uniform(0.1, 1.0, size=(2, 18))
    flux = true_coefficients @ true_basis + rng.normal(0.0, 0.8, size=(35, 18))
    assert np.any(flux < 0)
    weights = rng.uniform(0.5, 2.0, size=flux.shape)
    weights[rng.random(weights.shape) < 0.15] = 0
    coefficient_start, basis_start = shared_initialization(35, 18, 2, seed=4)
    coefficients, basis, losses, _, updates = fit_anls(
        flux,
        weights,
        coefficient_start,
        basis_start,
        max_updates=5,
        relative_tolerance=0.0,
        patience=2,
    )
    assert updates == 5
    assert losses[-1] < losses[0]
    assert np.all(coefficients >= 0)
    assert np.all(basis >= 0)


def test_covered_lsst_color_rms_is_zero_for_exact_reconstruction():
    wave = np.linspace(2500.0, 9000.0, 1000)
    basis = np.vstack([np.ones_like(wave), wave / 5000.0])
    coefficients = np.array([[1.0, 0.4], [0.5, 1.2]])
    flux = coefficients @ basis
    weights = np.ones_like(flux)
    rms, count = desi_covered_lsst_color_rms(
        flux,
        weights,
        coefficients,
        basis,
        wave,
        np.array([0.1, 0.1]),
        minimum_band_coverage=0.5,
    )
    assert np.all(count >= 2)
    assert np.allclose(rms, 0.0, atol=1e-12)


def test_largest_contiguous_region_drops_short_supported_islands():
    mask = np.array([True, True, False, True, True, True, False, True])
    assert np.array_equal(
        largest_contiguous_region(mask),
        np.array([False, False, False, True, True, True, False, False]),
    )


def test_wavelength_support_scales_contributors_with_largest_rank():
    weights = np.zeros((50, 7))
    contributors = [10, 20, 30, 45, 45, 19, 40]
    for column, count in enumerate(contributors):
        weights[:count, column] = 1.0
    selected, measured, required = select_wavelength_support(
        weights, [2, 4], observations_per_component=5, support_rank=4
    )
    assert required == 20
    assert np.array_equal(measured, contributors)
    assert np.array_equal(selected, np.array([False, True, True, True, True, False, False]))


def test_candidate_grid_retains_supported_uv_below_1400(monkeypatch):
    from bedcosmo.num_visits.empirical.desi.fit_basis import parse_args

    monkeypatch.setattr("sys.argv", ["fit_basis"])
    args = parse_args()
    assert args.wave_min is None and args.wave_max is None
    wave = np.arange(1330, 9720 + args.wave_step, args.wave_step)
    weights = np.zeros((120, len(wave)))
    weights[:, (wave >= 1390) & (wave <= 9120)] = 1.0
    weights[:99, wave == 1380] = 1.0
    weights[:99, wave == 9130] = 1.0

    selected, _, required = select_wavelength_support(weights, [8])

    assert required == 100
    assert wave[selected][0] == 1390
    assert wave[selected][-1] == 9120


def test_grid_uses_selected_valid_pixels_and_rounds_outward(tmp_path, monkeypatch):
    import pandas as pd
    from astropy.io import fits

    coadd = tmp_path / "coadd.fits"
    targetid = 39627568982265273
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(
            np.array([(targetid,), (2,)], dtype=[("TARGETID", "i8")]), name="FIBERMAP"
        ),
    ]
    for arm in "BRZ":
        hdus.extend(
            [
                fits.ImageHDU(
                    np.array([2000.0, 2671.0, 6001.0, 8000.0, 9000.0]), name=f"{arm}_WAVELENGTH"
                ),
                fits.ImageHDU(
                    np.array([[1.0, 1.0, 1.0, np.nan, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
                    name=f"{arm}_FLUX",
                ),
                fits.ImageHDU(
                    np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
                    name=f"{arm}_IVAR",
                ),
                fits.ImageHDU(
                    np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32), name=f"{arm}_MASK"
                ),
            ]
        )
    fits.HDUList(hdus).writeto(coadd)
    monkeypatch.setattr(
        "bedcosmo.num_visits.empirical.desi.training_matrix.get_local_desi_paths",
        lambda *args: (coadd, tmp_path / "redrock.fits"),
    )
    manifest = pd.DataFrame({"targetid": [targetid], "healpix": [1], "z": [1.0]})
    grid = derive_rest_frame_grid(manifest, desi_dir=tmp_path, wave_step=10)
    assert grid[0] == 1330
    assert grid[-1] == 3010
    assert np.all(np.diff(grid) == 10)
    override = derive_rest_frame_grid(
        manifest, desi_dir=tmp_path, wave_step=10, wave_min=1400, wave_max=2900
    )
    assert override[0] == 1400 and override[-1] == 2900


def test_lsst_demand_coverage_is_unity_when_every_pixel_is_observed():
    wave = np.array([3500.0, 4500.0, 6000.0])
    redshift = np.array([0.0, 0.2])
    weights = np.ones((2, 3))
    coverage = lsst_demand_weighted_coverage(wave, redshift, weights)
    assert np.allclose(coverage[np.isfinite(coverage)], 1.0)


def test_export_normalization_preserves_factorized_spectra():
    wave = np.linspace(3500.0, 4500.0, 101)
    basis = np.vstack([np.ones_like(wave), wave / 4000.0])
    coefficients = np.array([[0.3, 0.7], [1.2, 0.2]])
    normalized_basis, adjusted_coefficients, integrals = normalize_basis_for_export(
        wave, basis, coefficients, norm_min=3600.0, norm_max=4200.0
    )
    assert np.all(integrals > 0)
    assert np.allclose(coefficients @ basis, adjusted_coefficients @ normalized_basis)
    selected = (wave >= 3600.0) & (wave <= 4200.0)
    assert np.allclose(np.trapz(normalized_basis[:, selected], wave[selected], axis=1), 1.0)


def test_physical_coefficients_round_trip_numvisits_redshifting():
    coefficients = np.array([[1.0, 2.0], [0.5, 0.25]])
    scale = np.array([4.0, 10.0])
    redshift = np.array([0.5, 1.0])
    physical = physical_prior_coefficients(coefficients, scale, redshift)
    runtime_reconstruction = physical / (1.0 + redshift)[:, None]
    assert np.allclose(runtime_reconstruction, coefficients * scale[:, None])


def test_write_template_bank_is_self_contained(tmp_path):
    wave = np.array([1400.0, 1410.0, 1420.0])
    basis = np.array([[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]])
    template_dir = tmp_path / "empirical_prior" / "desi2" / "templates"

    param, component_paths = write_template_bank(
        template_dir,
        Path("desi2.param"),
        wave,
        basis,
    )

    assert param == template_dir / "desi2.param"
    assert component_paths == ["component_01.dat", "component_02.dat"]
    assert param.read_text().splitlines()[1:] == [
        "1 component_01.dat 1.0",
        "2 component_02.dat 1.0",
    ]
    assert all((template_dir / path).is_file() for path in component_paths)
