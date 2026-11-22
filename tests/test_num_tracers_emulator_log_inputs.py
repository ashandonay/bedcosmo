"""NumTracers emulator inputs: log-N checkpoints, the N_tracers floor clamp and the cosmology box clamp.

A checkpoint trained with ``train.py --log-inputs N_tracers`` must be fed log N,
standardized with its own stats; feeding it linear N returns wrong σ without
raising. A checkpoint that records its raw training box (``x_train_min`` /
``x_train_max``) evaluates N below the box at the floor, so a class with no
fibers (N = 0) stays finite under the log. Checkpoints without the box keep
the old behavior: linear inputs, no clamp.

Runs ``_load_emulator`` and ``_emulator_predict`` on a stub instance with a small
random network; no DESI data or trained checkpoint needed.
"""
import math

import pytest
import torch

pytest.importorskip("tracer_forecast_emulator")
from tracer_forecast_emulator.util import build_model  # noqa: E402

from bedcosmo.num_tracers.experiment import NumTracers  # noqa: E402

PARAMS = ["N_tracers", "Om", "hrdrag"]
TARGETS = ["sigma_DH_over_rd", "sigma_DM_over_rd", "rho_DH_DM"]
LO, HI = 1.0e3, 1.0e6


def _ckpt(tmp_path, *, log_n, record_box):
    torch.manual_seed(0)
    kwargs = {"width": 8, "n_blocks": 1}
    model = build_model("bao", "resnet", in_dim=3, out_dim=3, **kwargs)
    mu_n = 0.5 * (math.log(LO) + math.log(HI)) if log_n else 0.5 * (LO + HI)
    sg_n = (math.log(HI) - math.log(LO)) / 12 ** 0.5 if log_n else (HI - LO) / 12 ** 0.5
    ckpt = {
        "state_dict": model.state_dict(),
        "model_kwargs": kwargs,
        "architecture": "resnet",
        "analysis": "bao",
        "param_names": PARAMS,
        "target_names": TARGETS,
        "x_mu": torch.tensor([[mu_n, 0.3, 100.0]]),
        "x_sigma": torch.tensor([[sg_n, 0.1, 10.0]]),
        "y_mu": torch.zeros(1, 3),
        "y_sigma": torch.ones(1, 3),
        "log_normalize": False,
        "y_linthresh": None,
        # Production recipe; also the decode path bedcosmo differentiates through.
        "scale_expressions": ["sigma_*:hrdrag"],
    }
    if log_n:
        ckpt["log_inputs"] = ["N_tracers"]
    if record_box:
        ckpt["x_train_min"] = torch.tensor([LO, 0.1, 80.0])
        ckpt["x_train_max"] = torch.tensor([HI, 0.9, 120.0])
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    return path, model, ckpt


@pytest.fixture
def stub():
    exp = NumTracers.__new__(NumTracers)
    exp.device = torch.device("cpu")
    exp._n_extrap_warned = set()
    return exp


def _x(n):
    return torch.tensor([[n, 0.3, 100.0]], dtype=torch.float32)


def _raw_net(model, ckpt, n_feature):
    """What the net returns on hand-standardized inputs (y_mu=0, y_sigma=1)."""
    x = torch.tensor([[n_feature, 0.3, 100.0]])
    with torch.no_grad():
        return model((x - ckpt["x_mu"]) / ckpt["x_sigma"])


def test_log_n_checkpoint_is_fed_log_n(tmp_path, stub):
    path, model, ckpt = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    assert emu["n_floor"] == pytest.approx(LO)
    assert (emu["n_train_lo"], emu["n_train_hi"]) == (pytest.approx(LO), pytest.approx(HI))
    n = 3.0e4
    with torch.no_grad():
        got = stub._emulator_predict(emu, _x(n))
    raw = _raw_net(model, ckpt, math.log(n))
    # rho goes through tanh; check it exactly, the sigmas through the floor/exp decode.
    assert got[0, 2].item() == pytest.approx(math.tanh(raw[0, 2].item()), rel=1e-5)
    wrong = _raw_net(model, ckpt, n)  # linear N into a log-N net
    assert not torch.allclose(raw, wrong)


def test_n_below_the_floor_is_evaluated_at_the_floor(tmp_path, stub):
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    with torch.no_grad():
        at_floor = stub._emulator_predict(emu, _x(LO))
        for n in (0.0, 1.0, 0.5 * LO):
            below = stub._emulator_predict(emu, _x(n))
            assert torch.isfinite(below).all()
            torch.testing.assert_close(below, at_floor)
        above = stub._emulator_predict(emu, _x(2 * LO))
    assert not torch.allclose(above, at_floor)


def test_no_low_n_warning_when_clamped(tmp_path, stub):
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stub._warn_n_extrapolation("QSO", emu, torch.tensor([0.0, 10.0]))


def test_checkpoint_without_box_is_linear_and_unclamped(tmp_path, stub):
    path, model, ckpt = _ckpt(tmp_path, log_n=False, record_box=False)
    emu = stub._load_emulator(path)
    assert emu["n_floor"] is None
    # Box estimated from the uniform-draw stats, as before.
    assert emu["n_train_lo"] == pytest.approx(LO, rel=1e-4)
    with torch.no_grad():
        got = stub._emulator_predict(emu, _x(0.5 * LO))
    raw = _raw_net(model, ckpt, 0.5 * LO)
    assert got[0, 2].item() == pytest.approx(math.tanh(raw[0, 2].item()), rel=1e-5)
    with pytest.warns(RuntimeWarning, match="grossly outside"):
        stub._warn_n_extrapolation("QSO", emu, torch.tensor([0.1 * LO]))


def test_gradient_through_n_is_finite_above_and_zero_below_floor(tmp_path, stub):
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    x = torch.tensor([[3.0e4, 0.3, 100.0], [10.0, 0.3, 100.0]], requires_grad=True)
    stub._emulator_predict(emu, x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, 0] != 0 and x.grad[1, 0] == 0


# ---- cosmology inputs: clipped into the recorded training box -----------------------

def _xc(om, hrdrag, n=3.0e4):
    return torch.tensor([[n, om, hrdrag]], dtype=torch.float32)


def test_cosmology_outside_the_box_is_evaluated_at_its_edge(tmp_path, stub):
    """A prior draw below the trained Om floor gets the floor's covariance, finite."""
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    with torch.no_grad():
        edge = stub._emulator_predict(emu, _xc(0.1, 100.0))
        for om in (0.05, 0.01, -1.0):
            out = stub._emulator_predict(emu, _xc(om, 100.0))
            assert torch.isfinite(out).all()
            torch.testing.assert_close(out, edge)
        inside = stub._emulator_predict(emu, _xc(0.3, 100.0))
        assert not torch.allclose(inside, edge)
        torch.testing.assert_close(stub._emulator_predict(emu, _xc(0.95, 100.0)),
                                   stub._emulator_predict(emu, _xc(0.9, 100.0)))


def test_exact_hrdrag_scaling_uses_the_draws_own_hrdrag(tmp_path, stub):
    """Above the hrdrag box the net sees the edge, and sigma ~ 1/hrdrag uses the true value."""
    path, _, ckpt = _ckpt(tmp_path, log_n=True, record_box=True)
    ckpt["y_mu"] = torch.tensor([[50.0, 50.0, 0.0]])  # scaled sigma labels well above the floor
    torch.save(ckpt, path)
    emu = stub._load_emulator(path)
    with torch.no_grad():
        at_edge = stub._emulator_predict(emu, _xc(0.3, 120.0))
        beyond = stub._emulator_predict(emu, _xc(0.3, 240.0))
    torch.testing.assert_close(beyond[0, :2], at_edge[0, :2] * 120.0 / 240.0)  # sigmas
    torch.testing.assert_close(beyond[0, 2], at_edge[0, 2])                     # rho


def test_checkpoint_without_box_does_not_clamp_cosmology(tmp_path, stub):
    path, _, _ = _ckpt(tmp_path, log_n=False, record_box=False)
    emu = stub._load_emulator(path)
    assert emu["theta_lo"] is None
    with torch.no_grad():
        assert not torch.allclose(stub._emulator_predict(emu, _xc(0.05, 100.0, n=LO)),
                                  stub._emulator_predict(emu, _xc(0.1, 100.0, n=LO)))


def test_gradient_through_om_is_zero_outside_the_box(tmp_path, stub):
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    x = torch.tensor([[3.0e4, 0.3, 100.0], [3.0e4, 0.05, 100.0]], requires_grad=True)
    stub._emulator_predict(emu, x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, 1] != 0 and x.grad[1, 1] == 0


def test_clamped_inputs_are_counted_logged_and_warned_once(tmp_path, stub):
    """frac_emulator_inputs_clamped is the clipped share of evaluations since the last read;
    the first read that sees any warns once for that network, naming the parameter."""
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    emu = stub._load_emulator(path)
    stub._emulators = {"QSO": emu}
    x = torch.tensor([[3.0e4, 0.3, 100.0], [3.0e4, 0.05, 100.0],
                      [3.0e4, 0.3, 240.0], [3.0e4, 0.3, 100.0]])
    with torch.no_grad():
        stub._emulator_predict(emu, x)
    with pytest.warns(RuntimeWarning, match=r"Om below 0\.1 \(1\).*hrdrag above 120 \(1\)"):
        assert stub._input_clamp_stats() == pytest.approx(2 / 4)
    # Counters reset on read; nothing evaluated since -> 0.0, and no second warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert stub._input_clamp_stats() == 0.0
        with torch.no_grad():
            stub._emulator_predict(emu, x)
        assert stub._input_clamp_stats() == pytest.approx(0.5)


def test_clamp_metric_is_always_reported_in_emulator_mode(tmp_path, stub):
    """The trainer averages batch stats over ranks by name, so the key must always be there."""
    path, _, _ = _ckpt(tmp_path, log_n=True, record_box=True)
    stub._emulators = {"QSO": stub._load_emulator(path)}
    stub.likelihood_mode = "emulator"
    stub.nominal_design = torch.zeros(4)
    stats = stub.training_batch_stats(torch.zeros(2, 10))
    assert stats["frac_emulator_inputs_clamped"] == 0.0 and "frac_sigma_ceiling" in stats
