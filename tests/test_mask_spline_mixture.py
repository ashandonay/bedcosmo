"""Test physical support and marginal likelihood for the offline spline mixture."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_mask_spline_mixture import MaskSplineMixture, MixturePrior, train_mixture
from prototype_joint_empirical_mixture import JointPatternMixture


def test_density_and_gradient_respect_mask_eligibility():
    torch.manual_seed(7)
    eligible = np.zeros(8, dtype=bool)
    eligible[3] = True
    config = {"hidden": [8, 8], "transforms": 1, "bins": 4}
    flow = MaskSplineMixture(3, 3, config, eligible)
    batch = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    values = torch.tensor([[0.2, -0.5, 1.2], [0.5, 0.2, -0.7]], dtype=torch.float32)
    with torch.no_grad():
        flow.locations[3, 1] = 1.5
        flow.log_odds[3] = 0.7
    distribution = flow(batch)
    component = distribution.base.log_prob(values[:, None, :])
    weights = torch.sigmoid(flow.log_odds[3])
    density = (1 - weights) * component[0, 0].exp() + weights * component[0, 1].exp()
    expected = torch.stack([density.log(), component[1, 0]])
    actual = distribution.log_prob(values)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert flow.log_odds.grad[3].abs() > 0
    assert flow.log_odds.grad[5] == 0
    assert torch.isfinite(flow.locations.grad).all()


def test_train_reload_and_sample_preserves_physical_masks(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)

    def rows(n, first_mask):
        fraction = rng.uniform(0.1, 0.9, n)
        zero = np.zeros(n)
        a, b = (1 - fraction, zero) if first_mask else (zero, 1 - fraction)
        amplitude = rng.choice([0.3, 0.7], n) + rng.normal(0, 0.02, n)
        return np.column_stack([fraction, a, b, amplitude, rng.uniform(0.2, 0.9, n)])

    fitted = np.concatenate([rows(60, True), rows(20, False), [[0, 1, 0, 0.5, 0.5]]])
    validation = np.concatenate([rows(20, True), rows(12, False)])
    model = train_mixture(JointPatternMixture(fitted, bandwidth=0.1), validation, 17, 2, 30)
    path = tmp_path / "mixture.pt"
    model.save(path)
    loaded = MixturePrior.load(path)
    draws = loaded.sample(1000, seed=23)
    np.testing.assert_array_equal(draws, model.sample(1000, seed=23))
    np.testing.assert_allclose(draws[:, :-2].sum(axis=1), 1)
    np.testing.assert_array_equal(model.log_prob(draws), loaded.log_prob(draws))
    assert np.isfinite(loaded.log_prob(draws)).all()
    assert {tuple(mask) for mask in (draws[:, :-2] > 0)} == {
        (True, True, False),
        (True, False, True),
        (False, True, False),
    }


def test_multiscale_sample_is_a_hard_draw_with_gate_gradients():
    torch.manual_seed(16)
    eligible = np.zeros(8, dtype=bool)
    eligible[3] = True
    config = {"hidden": [8, 8], "transforms": 1, "bins": 4}
    flow = MaskSplineMixture(3, 3, config, eligible)
    masks = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    z = torch.zeros((3, 3), dtype=torch.float32)
    uniform = torch.tensor([0.2, 0.8, 0.2], dtype=torch.float32)
    with torch.no_grad():
        flow.locations[3, 1] = 5
    dist = flow(masks)
    draws = dist.draw_from_base(z, uniform)
    with torch.no_grad():
        parts = dist.base.transforms[0](
            dist.base.base_dist.transform.inv(z[:, None, :].expand(-1, 2, -1))
        )
        expected = parts[torch.arange(3), torch.tensor([1, 0, 0])]
        torch.testing.assert_close(draws, expected)
    draws.sum().backward()
    assert flow.log_odds.grad[3].abs() > 0
    assert flow.log_odds.grad[5] == 0
    assert torch.isfinite(flow.locations.grad).all()
