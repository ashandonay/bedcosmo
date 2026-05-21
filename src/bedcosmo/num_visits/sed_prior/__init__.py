"""Empirical galaxy SED prior from DESI spectra and EAZY template fits."""

from .build_empirical_sed_prior_kde import (
    coeffs_from_sample_row,
    load_prior_training_table,
    load_sed_prior_kde,
    sample_sed_prior,
    sample_sed_prior_kde,
    samples_to_coeffs,
    save_sed_prior_kde,
)

__all__ = [
    "coeffs_from_sample_row",
    "load_prior_training_table",
    "load_sed_prior_kde",
    "sample_sed_prior",
    "sample_sed_prior_kde",
    "samples_to_coeffs",
    "save_sed_prior_kde",
]
