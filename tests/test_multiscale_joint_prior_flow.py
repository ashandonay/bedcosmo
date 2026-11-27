"""Check that the joint loss responds to dependence with fixed marginals."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_multiscale_joint_prior_flow import kernel_mmd2, normalize_scale_weights


def test_full_row_loss_detects_dependence_with_unchanged_marginals():
    x = torch.linspace(-2, 2, 200)
    observed = torch.stack([x, x], dim=1)
    matched = observed.clone()
    wrong = torch.stack([x, -x], dim=1).requires_grad_()
    torch.testing.assert_close(observed.sort(0).values, wrong.sort(0).values)
    bandwidths = torch.tensor([0.35, 0.7, 1.4])
    distance = kernel_mmd2(observed, wrong, bandwidths)
    assert torch.all(distance > kernel_mmd2(observed, matched, bandwidths))
    distance.mean().backward()
    assert torch.isfinite(wrong.grad).all()
    assert wrong.grad.abs().sum() > 0


def test_relative_weights_preserve_loss_scale_and_can_disable_a_scale():
    weights = normalize_scale_weights([0, 3, 5])
    np.testing.assert_allclose(weights, normalize_scale_weights([0, 30, 50]))
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == 0 and weights[2] > weights[1]


@pytest.mark.parametrize("invalid", [[0, 0, 0], [-1, 1, 2], [1, 2], [1, float("nan"), 2]])
def test_invalid_scale_weights_fail(invalid):
    with pytest.raises(ValueError, match="weights"):
        normalize_scale_weights(invalid)
