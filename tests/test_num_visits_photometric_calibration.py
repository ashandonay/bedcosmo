"""Regression tests for num_visits flux and magnitude calibration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bedcosmo.num_visits import NumVisits
from bedcosmo.num_visits.experiment import s0


def _bare_experiment() -> NumVisits:
    exp = object.__new__(NumVisits)
    exp.device = "cpu"
    exp.global_rank = 0
    return exp


def test_zeropoint_magnitude_is_count_rate_of_one():
    exp = _bare_experiment()
    exp.filters_list = ["u", "g"]
    exp.num_filters = len(exp.filters_list)
    exp._wlen_aa_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    exp._transmission_tensor = torch.ones((2, 2), dtype=torch.float64)
    exp._wlen_over_hc_tensor = torch.ones(2, dtype=torch.float64)

    collecting_area_cm2 = (319 / 9.6) * 1e4
    unit_count_rate_flux = torch.full(
        (1, 2), 1.0 / collecting_area_cm2, dtype=torch.float64
    )
    magnitudes = exp._calculate_magnitudes(unit_count_rate_flux)

    expected = torch.tensor([[s0["u"], s0["g"]]], dtype=torch.float64)
    torch.testing.assert_close(magnitudes, expected)


def test_empirical_template_flux_applies_desi_coadd_unit_only():
    exp = _bare_experiment()
    exp._n_eazy_templates = 1
    exp._wlen_aa_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    exp._template_wave_rest = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    exp._template_flux = torch.full((1, 4), 2.0, dtype=torch.float64)
    exp.flux_unit_scale = 1e-17

    flux = exp._observed_spectral_flux(
        torch.tensor([0.0], dtype=torch.float64),
        a=torch.tensor([[1.0]], dtype=torch.float64),
        log_s=torch.tensor([0.0], dtype=torch.float64),
    )

    torch.testing.assert_close(flux, torch.full((1, 2), 2e-17, dtype=torch.float64))


def test_blackbody_flux_does_not_apply_desi_coadd_unit():
    exp = _bare_experiment()
    exp.profile = False
    exp.norm_mode = "bolometric"
    exp._wlen_aa_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    exp._luminosity_distance = lambda z: torch.ones_like(z)
    exp._bolometric_four_pi_R2 = lambda temperature: torch.ones_like(temperature)
    exp._blackbody_flux = lambda wavelength, temperature: torch.ones_like(wavelength)

    flux = exp._observed_spectral_flux(
        torch.tensor([0.0], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )

    expected = torch.full((1, 2), 1.0 / (4.0 * torch.pi), dtype=torch.float64)
    torch.testing.assert_close(flux, expected)


def test_empirical_prior_requires_explicit_flux_unit_scale():
    exp = _bare_experiment()
    with pytest.raises(ValueError, match="explicitly set flux_unit_scale"):
        exp._init_prior_empirical(
            {},
            prior_root=Path("unused"),
            prior_pool_size=1,
            prior_pool_seed=0,
            template_dir=None,
            template_param="unused.param",
            template_norm_min=None,
            template_norm_max=None,
            flux_unit_scale=None,
        )
