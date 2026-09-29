"""Trainer saves the input-transform check and warns on a noisy CDF table."""

from types import SimpleNamespace
from unittest import mock

import matplotlib
import torch
from pyro import distributions as dist

matplotlib.use("Agg")

from bedcosmo.train import Trainer  # noqa: E402
from bedcosmo.transform import Bijector  # noqa: E402
from tests.test_transform_input import _Stub  # noqa: E402


def _trainer(tmp_path, cdf_samples):
    stub = _Stub({"h": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))}, transform_input=True)
    torch.manual_seed(0)
    stub.param_bijector = Bijector(
        stub, cdf_bins=1000, cdf_samples=cdf_samples, use_prior_flow=False
    )
    (tmp_path / "artifacts" / "plots").mkdir(parents=True)
    return SimpleNamespace(experiment=stub, run_path=str(tmp_path))


def test_saves_plot_and_logs_metrics(tmp_path, capsys):
    trainer = _trainer(tmp_path, cdf_samples=10_000_000)
    with mock.patch("bedcosmo.train.mlflow.log_metrics") as log_metrics:
        Trainer._check_input_transform(trainer)
    assert (tmp_path / "artifacts" / "plots" / "input_transform.png").exists()
    logged = log_metrics.call_args.args[0]
    assert set(logged) == {
        "input_transform/h/slope_scatter",
        "input_transform/h/empty_frac",
        "input_transform/h/y_mean",
        "input_transform/h/y_std",
    }
    assert "WARNING" not in capsys.readouterr().out


def test_warns_on_noisy_table(tmp_path, capsys):
    # 2e4 draws over 1000 segments: ~20 per segment, ~22% slope scatter.
    trainer = _trainer(tmp_path, cdf_samples=20_000)
    with mock.patch("bedcosmo.train.mlflow.log_metrics"):
        Trainer._check_input_transform(trainer)
    assert "WARNING: input transform CDF table for 'h' is noisy" in capsys.readouterr().out
