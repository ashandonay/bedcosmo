"""Check the differentiable path used to fit physical sample distributions."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tune_joint_prior_sample_loss import decode_torch, single_run_schedule

from bedcosmo.num_visits.empirical.joint_prior_flow import decode_rows


def test_torch_decode_matches_physical_decoder_and_backpropagates():
    masks = torch.tensor([[True, False, True, True], [False, True, True, True]])
    x = torch.tensor(
        [[0.5, -0.2, 0.7, 1.1], [-1.2, 3.4, -2, 0.8]], dtype=torch.float64, requires_grad=True
    )
    lower, upper = torch.tensor([1.0, 0.0]), torch.tensor([9.0, 2.0])
    draws = decode_torch(x, masks, lower, upper)
    reference = decode_rows(x.detach().numpy(), masks.numpy(), lower.numpy(), upper.numpy())
    np.testing.assert_allclose(draws.detach().numpy(), reference, atol=1e-12)
    weights = torch.arange(1, draws.shape[1] + 1, dtype=draws.dtype)
    loss = (draws * weights).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 1e-6).all()


def test_reset_weights_recovers_original_architecture_and_discards_fitted_weights():
    from compare_joint_flow_training_schedules import reset_weights
    from prototype_joint_empirical_mixture import JointPatternMixture

    from bedcosmo.num_visits.empirical.joint_prior_flow import initialize_direct_prior, train

    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    a = rng.uniform(0.1, 0.9, 140)
    rows = np.column_stack([a, 1 - a, 2 + rng.random(len(a)), 0.2 + rng.random(len(a))])
    target = JointPatternMixture(rows[:110], bandwidth=0.1)
    model = train(
        target,
        direct=True,
        validation_rows=rows[110:],
        epochs=2,
        batch_size=32,
        hidden=8,
        transforms=1,
        bins=4,
        seed=17,
    )
    cold = reset_weights(model, 17)
    initialized = initialize_direct_prior(target, seed=17, hidden=8, transforms=1, bins=4)
    initial = initialized.groups[2]["flow"]
    np.testing.assert_array_equal(cold.groups[2]["mu"], initialized.groups[2]["mu"])
    np.testing.assert_array_equal(cold.groups[2]["std"], initialized.groups[2]["std"])
    for cold_param, initial_param in zip(cold.groups[2]["flow"].parameters(), initial.parameters()):
        torch.testing.assert_close(cold_param, initial_param, atol=0, rtol=0)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(cold.groups[2]["flow"].parameters(), model.groups[2]["flow"].parameters())
    )
    np.testing.assert_array_equal(cold.probabilities, model.probabilities)
    np.testing.assert_array_equal(cold.groups[2]["mu"], model.groups[2]["mu"])
    np.testing.assert_array_equal(cold.groups[2]["std"], model.groups[2]["std"])


def test_single_run_schedule_continuously_connects_the_training_objectives():
    progress = np.linspace(0, 1, 501)
    nll, sample, lr = np.array([single_run_schedule(t, 0.08, 2e-4) for t in progress]).T
    np.testing.assert_allclose([nll[0], sample[0], lr[0]], [1.0, 0.0, 1e-3])
    np.testing.assert_allclose([nll[-1], sample[-1], lr[-1]], [0.08, 1.0, 2e-4])
    assert (np.diff(nll) <= 0).all()
    assert (np.diff(lr) <= 0).all()
    assert ((sample > 0) & (sample < 1)).sum() > 100
