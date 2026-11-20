"""Check the loss used to fit local probability in pairs of physical coordinates."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tune_joint_prior_local_detail import soft_assignment, soft_histogram


def test_soft_histogram_preserves_mass_and_can_move_a_local_peak():
    edges = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
    far = torch.tensor([-10.0, 10.0], dtype=torch.float64)
    assignment = soft_assignment(far, edges, 0.1)
    torch.testing.assert_close(assignment.sum(1), torch.ones_like(far))
    assert assignment[0, 0] == 1 and assignment[1, -1] == 1

    observed = torch.tensor([[-0.6, 0.6], [-0.65, 0.55]], dtype=torch.float64)
    generated = torch.tensor([[-0.35, 0.35], [-0.4, 0.3]], dtype=torch.float64, requires_grad=True)
    target = soft_histogram(observed, (0, 1), edges, edges, (0.15, 0.15))
    estimate = soft_histogram(generated, (0, 1), edges, edges, (0.15, 0.15))
    torch.testing.assert_close(estimate.sum(), estimate.new_tensor(1))
    loss = (estimate - target).square().sum()
    loss.backward()
    assert torch.isfinite(generated.grad).all()
    assert (generated.grad[:, 0] > 0).all()
    assert (generated.grad[:, 1] < 0).all()
