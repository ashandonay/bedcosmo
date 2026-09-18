"""Learn nonnegative NumVisits SED bases directly from DESI spectra."""

from .training_matrix import build_rest_frame_matrix, load_desi_manifest
from .weighted_nmf import fit_weighted_nmf, infer_coefficients

__all__ = [
    "build_rest_frame_matrix",
    "fit_weighted_nmf",
    "infer_coefficients",
    "load_desi_manifest",
]
