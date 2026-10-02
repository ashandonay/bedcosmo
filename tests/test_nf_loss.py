"""nf_loss averages over particles per design and never masks non-finite terms."""

from types import SimpleNamespace

import pytest
import torch

from bedcosmo.pyro_oed_src import nf_loss


class _TableGuide(torch.nn.Module):
    """log q(theta | context) = context[..., 0]: each sample's loss is set by its context."""

    def forward(self, context):
        return SimpleNamespace(log_prob=lambda y: context[:, 0].clone())


def _stub():
    return SimpleNamespace(transform_input=False)


def _neg_log_prob(n_particles=4, n_designs=3):
    return torch.arange(n_particles * n_designs, dtype=torch.float64).view(n_particles, n_designs)


@pytest.mark.parametrize("batch_dim", [False, True])
def test_mean_over_particles_per_design(batch_dim):
    nlp = _neg_log_prob()
    samples = torch.zeros(*nlp.shape, 2, dtype=torch.float64)
    context = (-nlp).unsqueeze(-1)
    if batch_dim:  # the training DataLoader's leading batch dim of 1
        samples, context = samples.unsqueeze(0), context.unsqueeze(0)

    agg_loss, posterior_entropy, neg_log_prob = nf_loss(samples, context, _TableGuide(), _stub())

    expected = nlp.mean(dim=0)  # per design, over the 4 particles
    assert torch.equal(posterior_entropy.reshape(-1), expected)
    assert agg_loss.item() == pytest.approx(expected.sum().item())
    assert torch.equal(neg_log_prob.reshape(nlp.shape), nlp)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
@pytest.mark.parametrize("batch_dim", [False, True])
def test_non_finite_terms_propagate_unchanged(bad, batch_dim):
    nlp = _neg_log_prob()
    nlp[1, 2] = bad
    samples = torch.zeros(*nlp.shape, 2, dtype=torch.float64)
    context = (-nlp).unsqueeze(-1)
    if batch_dim:
        samples, context = samples.unsqueeze(0), context.unsqueeze(0)

    agg_loss, posterior_entropy, neg_log_prob = nf_loss(samples, context, _TableGuide(), _stub())

    posterior_entropy = posterior_entropy.reshape(-1)
    # Design 2 carries the bad term as itself (an inf stays inf, not 0/0 = NaN);
    # the other designs are untouched, and no particle is dropped.
    assert torch.isfinite(posterior_entropy[:2]).all()
    if bad == bad:  # inf
        assert posterior_entropy[2] == bad
    else:
        assert posterior_entropy[2].isnan()
    assert not torch.isfinite(agg_loss)
    assert torch.equal(neg_log_prob.reshape(nlp.shape).isinf(), nlp.isinf())
    assert torch.equal(neg_log_prob.reshape(nlp.shape).isnan(), nlp.isnan())
