"""Tests for equal-area probability contours in output-prior figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_empirical_output_priors import contour_thresholds


def test_contour_thresholds_enclose_mass_in_density_order():
    histogram = np.array([[8.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(contour_thresholds(histogram), [1, 8])
    for probability, threshold in zip([0.68, 0.95], [8, 1]):
        assert histogram[histogram >= threshold].sum() / histogram.sum() >= probability
        assert histogram[histogram > threshold].sum() / histogram.sum() < probability
    np.testing.assert_array_equal(contour_thresholds(histogram * 10), [10, 80])


@pytest.mark.parametrize("histogram", [np.zeros((2, 2)), np.array([[2, -1]])])
def test_contour_thresholds_reject_invalid_mass(histogram):
    with pytest.raises(ValueError, match="mass"):
        contour_thresholds(histogram)
