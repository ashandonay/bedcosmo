"""Checks for the summaries displayed in mixture comparison figures."""

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "mixture_plots",
    Path(__file__).resolve().parents[1] / "scripts/plot_empirical_mixture_failures.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_binned_means_include_right_endpoint_and_preserve_empty_bins():
    means, counts = _MODULE.binned_means(
        np.array([0.0, 0.1, 0.9, 1.0]),
        np.array([1.0, 3.0, 5.0, 7.0]),
        np.array([0, 0.25, 0.5, 0.75, 1]),
    )
    np.testing.assert_array_equal(counts, [2, 0, 0, 2])
    np.testing.assert_allclose(means, [2, np.nan, np.nan, 6])
