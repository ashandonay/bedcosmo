"""Mixed posterior scores, gradients, atoms and independent checkpoint recovery."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, build_flow
from bedcosmo.num_visits.empirical.pattern_posterior import PatternPosterior, pattern_prior_spec
from bedcosmo.num_visits.empirical.sed_prior import (
    EmpiricalSedPrior,
    resolve_runtime_prior_root,
    snapshot_sed_prior,
)
from bedcosmo.util import build_nf_init_config_from_run_args, init_nf, parse_bool_arg


def test_explicit_false_cli_boolean():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transform-input", type=parse_bool_arg, nargs="?", const=True, default=True
    )
    assert parser.parse_args(["--transform-input", "false"]).transform_input is False
    assert parser.parse_args(["--transform-input"]).transform_input is True
    assert parser.parse_args([]).transform_input is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--transform-input", "typo"])


@pytest.fixture
def prior(tmp_path):
    torch.set_num_threads(1)
    config = {"hidden": [8, 8], "transforms": 2, "bins": 4}
    metadata = {
        "patterns": np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        "counts": np.array([10, 20, 30, 1]),
        "atomic": np.array([False, False, False, True]),
        "atomic_rows": np.array([[0, 0, 0, 0, 0]] * 3 + [[0, 0, 1, 5, 1.0]]),
        "lower": np.array([0.0, 0.0]),
        "upper": np.array([10.0, 2.0]),
        "feature_names": ["a1", "a2", "a3", "log_c_scale", "z"],
        "flow_config": config,
        "build_provenance": {"template": {"template_param": "test.param"}},
    }
    groups = {
        m: {
            "flow": build_flow(m + 1, 3, config).eval(),
            "mu": np.zeros(m + 1),
            "std": np.ones(m + 1),
            "history": [1.0],
        }
        for m in [1, 2, 3]
    }
    model = JointPriorFlow(metadata, groups)
    model.save(tmp_path / "joint_prior_flow.pt")
    return model


class StandardGaussian(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, context):
        return torch.distributions.Independent(
            torch.distributions.Normal(
                context.new_zeros((*context.shape[:-1], self.dim)),
                context.new_ones((*context.shape[:-1], self.dim)),
            ),
            1,
        )


def test_same_reference_measure_and_jacobian_as_prior(prior):
    posterior = PatternPosterior(
        pattern_prior_spec(prior), 12, hidden=8, layers=1, transforms=1, bins=4, prior=prior
    )
    for key in posterior.face_flows:
        posterior.face_flows[key] = StandardGaussian(int(key) + 1)
    rows = torch.tensor(prior.sample(1500, seed=19), dtype=torch.float64)
    context = torch.zeros(12, dtype=torch.float64)
    scores = posterior(context).log_prob(rows)
    np.testing.assert_allclose(scores.detach(), prior.log_prob(rows.numpy()), atol=2e-5, rtol=2e-5)
    atomic = torch.tensor(prior.metadata["atomic_rows"][-1:], dtype=torch.float64)
    assert posterior(context).log_prob(atomic).item() == pytest.approx(np.log(1 / 61))
    invalid = atomic.clone()
    invalid[:, -1] += 0.1
    assert torch.isneginf(posterior(context).log_prob(invalid)).all()
    unknown = torch.tensor([[0.5, 0, 0.5, 5, 1]], dtype=torch.float64)
    assert torch.isneginf(posterior(context).log_prob(unknown)).all()
    samples = posterior(context).sample((1000,))
    assert samples.shape == (1000, 5)
    torch.testing.assert_close(samples[:, :3].sum(-1), torch.ones(1000, dtype=torch.float64))
    assert torch.isfinite(posterior(context).log_prob(samples)).all()
    for row in samples:
        assert tuple((row[:3] > 0).tolist()) in prior.lookup


def test_all_branches_receive_gradients_and_checkpoint_rebuild(prior):
    spec = pattern_prior_spec(prior)
    args = {
        "flow_type": "PatternNSF",
        "pattern_prior_spec": spec,
        "transform_input": False,
        "cond_hidden_size": 8,
        "cond_n_layers": 1,
        "n_transforms": 1,
        "spline_bins": 4,
    }
    posterior = init_nf(args, 5, 12, device="cpu", seed=5, pattern_prior=prior)
    # Deliberately omit two face groups and all atoms from the batch.
    rows = torch.tensor([[0.2, 0.3, 0.5, 5, 1]] * 10, dtype=torch.float64)
    context = torch.ones(10, 12, dtype=torch.float64)
    scores = posterior(context).log_prob(rows)
    (-scores.mean()).backward()
    for name, param in posterior.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    restored = init_nf(build_nf_init_config_from_run_args(args), 5, 12, device="cpu")
    restored.load_state_dict(posterior.state_dict())
    torch.testing.assert_close(restored(context).log_prob(rows), scores)
    assert restored(context).sample((3,)).shape == (3, 10, 5)
    with pytest.raises(ValueError, match="internally"):
        init_nf({**args, "transform_input": True}, 5, 12, device="cpu")


def test_joint_prior_pool_snapshot_and_frozen_source_resolution(prior, tmp_path):
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates/test.param").write_text("1 component.dat 1.0\n")
    (tmp_path / "templates/component.dat").write_text("1000 1\n14000 1\n")
    (tmp_path / "build_provenance.json").write_text("{}")
    np.savez(tmp_path / "joint_kde_fit.npz", training_rows=prior.sample(100))
    args = {
        "density_type": "joint_pattern_flow",
        "prior_dir": str(tmp_path),
        "template_param": "test.param",
    }
    dest = tmp_path / "run/artifacts"
    snapshot_sed_prior(args, dest)
    frozen = resolve_runtime_prior_root(artifacts_dir=dest, prior_dir=tmp_path / "missing")
    assert frozen == dest / "empirical"
    loaded = EmpiricalSedPrior.from_joint_path(
        frozen / "joint_prior_flow.pt", pool_size=1000, device="cpu"
    )
    assert loaded.parameterization == "weights"
    assert loaded.pool.pool.shape == (1000, 5)
    assert loaded.uses_flow_prior()
    assert torch.isfinite(loaded.log_prob(loaded.pool.pool, use_flow=True)).all()
    assert loaded.rows_to_param_dict(loaded.pool.pool[:10], loaded.feature_names, (10,))[
        "z"
    ].shape == (10, 1)


def test_num_visits_physical_forward_model_and_joint_eig_terms(prior, tmp_path):
    import pyro

    from bedcosmo.evaluate import Evaluator
    from bedcosmo.num_visits import NumVisits
    from bedcosmo.pyro_oed_src import LikelihoodDataset, nf_loss

    (tmp_path / "templates").mkdir()
    (tmp_path / "templates/test.param").write_text("1 c1.dat 1.0\n2 c2.dat 1.0\n3 c3.dat 1.0\n")
    for i in range(1, 4):
        np.savetxt(tmp_path / f"templates/c{i}.dat", np.array([[900.0, 1.0], [15000.0, 1.0]]))
    args = {
        "density_type": "joint_pattern_flow",
        "prior_dir": str(tmp_path),
        "template_param": "test.param",
        "flux_unit_scale": 1e-17,
        "template_norm_min": 3600.0,
        "template_norm_max": 4200.0,
        "prior_pool_size": 1000,
        "parameters": {
            name: {"distribution": {"type": "empirical"}}
            for name in prior.metadata["feature_names"]
        },
    }
    experiment = NumVisits(
        prior_args=args,
        design_args={"input_type": "nominal"},
        cosmo_model="empirical",
        device="cpu",
        transform_input=False,
    )
    assert experiment.cosmo_params == prior.metadata["feature_names"]
    rows = experiment.sed_prior.sample_batch(100)
    fractions, log_s, z = experiment._empirical_rows_to_physical(rows, (100,))
    torch.testing.assert_close(fractions, rows[:, :3])
    torch.testing.assert_close(log_s, rows[:, 3])
    torch.testing.assert_close(z, rows[:, 4])
    traced = pyro.poutine.trace(experiment.pyro_model).get_trace(experiment.nominal_design)
    assert torch.isfinite(traced.nodes["magnitudes"]["value"]).all()
    spec = pattern_prior_spec(prior)
    guide = PatternPosterior(spec, 12, hidden=8, layers=1, transforms=1, bins=4, prior=prior)
    samples = rows.reshape(100, 1, 5)
    context = experiment.nominal_context.expand(100, 1, 12).contiguous()
    loss, entropy, terms = nf_loss(samples, context, guide, experiment)
    assert torch.isfinite(loss) and torch.isfinite(entropy).all()
    dataset = object.__new__(LikelihoodDataset)
    dataset.experiment = experiment
    scores = dataset._compute_prior_log_probs(samples, None)["joint"]
    torch.testing.assert_close(scores, experiment.sed_prior.log_prob(samples, use_flow=True))
    assert torch.isfinite((-scores - terms).mean())
    evaluator = object.__new__(Evaluator)
    evaluator.experiment = SimpleNamespace(prior_args=args, cosmo_params=experiment.cosmo_params)
    assert evaluator._normalize_marginal_subsets(None) == []
    with pytest.raises(ValueError, match="k-NN"):
        evaluator._normalize_marginal_subsets([["z"]])
