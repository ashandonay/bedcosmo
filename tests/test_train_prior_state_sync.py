"""Trainer gives every DDP rank rank 0's bijector and prior-flow log-acceptance."""

import os
from types import SimpleNamespace

import matplotlib
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp
from pyro import distributions as dist

matplotlib.use("Agg")

from bedcosmo.train import Trainer  # noqa: E402
from bedcosmo.transform import Bijector  # noqa: E402
from tests.test_transform_input import _Stub  # noqa: E402


def _rank(rank, world_size, port, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    tdist.init_process_group("gloo", rank=rank, world_size=world_size)
    # Gamma has no analytic marginal here, so each rank's CDF comes from its own draws.
    stub = _Stub({"h": dist.Gamma(torch.tensor(2.0), torch.tensor(1.0))}, transform_input=True)
    torch.manual_seed(rank)
    stub.param_bijector = Bijector(stub, cdf_bins=200, cdf_samples=20_000, use_prior_flow=False)
    stub.prior_flow_metadata = {"log_acceptance": -0.01 * (rank + 1)}
    before = stub.param_bijector.get_state()
    Trainer._sync_prior_state(SimpleNamespace(experiment=stub, global_rank=rank, device="cpu"))
    torch.save(
        {
            "before": before,
            "after": stub.param_bijector.get_state(),
            "log_acceptance": stub.prior_flow_metadata["log_acceptance"],
        },
        os.path.join(out_dir, f"rank{rank}.pt"),
    )
    tdist.destroy_process_group()


def test_ranks_end_with_rank0_prior_state(tmp_path):
    # fork: spawned children re-import from scratch and pick up the system libstdc++.
    mp.start_processes(_rank, args=(2, 29517, str(tmp_path)), nprocs=2, start_method="fork")
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    cdf = lambda s: s["cdfs"]["h"]["cdf_values"]  # noqa: E731
    assert not torch.equal(cdf(r0["before"]), cdf(r1["before"]))
    assert torch.equal(cdf(r1["after"]), cdf(r0["before"]))
    assert torch.equal(cdf(r0["after"]), cdf(r0["before"]))
    assert r0["log_acceptance"] == r1["log_acceptance"] == -0.01
