"""Training-health metrics, the skipped update on a NaN loss, and the NaN dump."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as tdist

from bedcosmo.num_tracers.experiment import NumTracers
from bedcosmo.train import Trainer


@pytest.fixture
def process_group(tmp_path):
    tdist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    yield
    tdist.destroy_process_group()


def _trainer(scale, grad_clip=0.0):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def loss(samples, context):
        per_sample = model(samples).squeeze(-1) * scale
        return per_sample, per_sample.sum()

    return SimpleNamespace(
        profile=False,
        verbose=False,
        global_rank=0,
        posterior_flow=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        run_args={"grad_clip": grad_clip},
        loss=loss,
    )


def _batch():
    return torch.randn(8, 2, dtype=torch.float64), torch.randn(8, 3, dtype=torch.float64)


def test_nan_loss_skips_the_update(process_group):
    trainer = _trainer(scale=float("nan"))
    before = [p.detach().clone() for p in trainer.posterior_flow.parameters()]
    *_, current_step, step_time, grad_norm = Trainer.step(trainer, *_batch(), current_step=5)
    assert current_step == 5
    assert grad_norm is None
    for p, p0 in zip(trainer.posterior_flow.parameters(), before):
        assert torch.equal(p, p0)


@pytest.mark.parametrize("grad_clip", [0.0, 1e-3])
def test_grad_norm_is_measured_before_clipping(process_group, grad_clip):
    samples, context = _batch()
    reference = _trainer(scale=1.0)
    reference.loss(samples, context)[1].backward()
    expected = torch.cat([p.grad.flatten() for p in reference.posterior_flow.parameters()]).norm()

    trainer = _trainer(scale=1.0, grad_clip=grad_clip)
    *_, current_step, _, grad_norm = Trainer.step(trainer, samples, context, current_step=5)
    assert current_step == 6
    assert grad_norm.item() == pytest.approx(expected.item())


def test_log_batch_health(process_group):
    trainer = SimpleNamespace(
        global_rank=0,
        experiment=SimpleNamespace(training_batch_stats=lambda context: {"frac_x": 0.25}),
    )
    loss = torch.tensor([[-2.0, 350.0], [1.0, 0.5]], dtype=torch.float64)
    context = torch.tensor([[0.1, -1.8e8], [3.0, 2.0]], dtype=torch.float64)
    with mock.patch("bedcosmo.train.mlflow.log_metrics") as log_metrics:
        Trainer._log_batch_health(trainer, loss, context, torch.tensor(7.5), current_step=40)
    log_metrics.assert_called_once_with(
        {"loss_max": 350.0, "context_abs_max": 1.8e8, "grad_norm": 7.5, "frac_x": 0.25},
        step=40,
    )


def _rank(tmp_path, rank):
    return SimpleNamespace(
        run_path=str(tmp_path), global_rank=rank, scheduler="sched", save_checkpoint=mock.Mock()
    )


def test_finite_global_loss_continues(tmp_path):
    trainer = _rank(tmp_path, 0)
    loss = torch.zeros(8, dtype=torch.float64)
    assert not Trainer._check_nan_loss(trainer, loss, 0.0, 9, *_batch())
    trainer.save_checkpoint.assert_not_called()


def test_nan_stops_every_rank_and_only_the_failing_one_dumps(tmp_path, capsys):
    samples, context = _batch()
    finite = torch.zeros(8, dtype=torch.float64)
    failing = finite.clone()
    failing[3] = float("nan")
    failing[5] = float("inf")

    # Rank 0's batch is fine: it stops and announces, but saves nothing.
    rank0 = _rank(tmp_path, 0)
    assert Trainer._check_nan_loss(rank0, finite, float("nan"), 9, samples, context)
    rank0.save_checkpoint.assert_not_called()
    assert "ERROR: non-finite global loss (nan) at step 9" in capsys.readouterr().out

    # Rank 2 failed: it saves its own batch and reports itself.
    rank2 = _rank(tmp_path, 2)
    assert Trainer._check_nan_loss(rank2, failing, float("nan"), 9, samples, context)
    (path,), kwargs = rank2.save_checkpoint.call_args
    assert path == f"{tmp_path}/artifacts/nan_dump/checkpoint_rank_2_9.pt"
    assert kwargs["step"] == 9
    state = kwargs["additional_state"]
    assert torch.equal(state["samples"], samples)
    assert torch.equal(state["context"], context)
    assert torch.isnan(state["loss"][3])
    out = capsys.readouterr().out
    assert "ERROR" not in out
    assert (
        f"Rank 2: 1 NaN and 1 Inf of 8 losses; saved the batch and pre-step state to {path}" in out
    )


def test_num_tracers_flags_sigma_ceiling_draws():
    exp = SimpleNamespace(likelihood_mode="emulator", nominal_design=torch.zeros(2))
    exp._SIGMA_CEILING = NumTracers._SIGMA_CEILING
    # Two design columns, then observations: one in-domain row, one at the ceiling.
    context = torch.tensor(
        [
            [0.5, 0.5, 20.0, 1.5e3],
            [0.5, 0.5, 20.0, -1.8e8],
            [0.5, 0.5, 30.0, 40.0],
            [0.5, 0.5, 1.2e8, 40.0],
        ],
        dtype=torch.float64,
    )
    assert NumTracers.training_batch_stats(exp, context) == {"frac_sigma_ceiling": 0.5}

    exp.likelihood_mode = "scaling"
    assert NumTracers.training_batch_stats(exp, context) == {}
