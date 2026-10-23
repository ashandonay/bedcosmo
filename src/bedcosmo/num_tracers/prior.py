"""Tempered DESI DR1 BAO-only posterior, sampled as a realistic num_tracers prior.

The target is the box prior times the tempered DR1 likelihood::

    p(theta) ∝ pi_box(theta) * L_DR1(theta) ** (1 / T)

``pi_box`` is the uniform prior and sum constraints of a ``prior_args`` YAML, and
``L_DR1`` the Gaussian likelihood of DESI's compressed DR1 BAO data (``desi_bao_all``:
the ``desi_2024_gaussian_bao_ALL_GCcomb`` mean and covariance). Distances come from
bedcosmo's ``CosmologyMixin`` with the fixed values of the cobaya configs in
``experiments/num_tracers/cobaya/mcmc``: h 0.6736, Neff 3.044, one massive neutrino of
0.06 eV. In near-Gaussian directions every posterior width scales by sqrt(T), the
same as DR1 with 1/T of its data.

Sampling is an affine-invariant ensemble MCMC (Goodman & Weare 2010 stretch move),
vectorized over walkers in torch. Example::

    python -m bedcosmo.num_tracers.prior --cosmo-model base_w_wa --temperature 16 \\
        --out $SCRATCH/bedcosmo/num_tracers/priors/dr1_tempered/base_w_wa_T16.npz
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from astropy import constants

from bedcosmo.cosmology import CosmologyMixin
from bedcosmo.util import get_experiment_config_path

DR1_BAO_DIR = Path(os.environ["HOME"]) / "data/desi/bao_dr1"
DR1_BAO_MEAN = DR1_BAO_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
DR1_BAO_COV = DR1_BAO_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"

# cobaya `ref` centres (experiments/num_tracers/cobaya/mcmc), where walkers start.
_REFERENCE = {"Om": 0.3152, "Ok": 0.0, "w0": -1.0, "wa": 0.0, "hrdrag": 99.079}
_REFERENCE_SCALE = {"Om": 0.01, "Ok": 0.01, "w0": 0.05, "wa": 0.1, "hrdrag": 1.0}


class _Distances(CosmologyMixin):
    def __init__(self, hrdrag_multiplier, device):
        self.hrdrag_multiplier = hrdrag_multiplier
        self.device = device
        self.c = constants.c.to("km/s").value


class TemperedDR1Posterior:
    """Log-density of pi_box * L_DR1^(1/T) over one BAO cosmo_model's parameters.

    Parameters are in the prior_args' raw units (hrdrag before its multiplier), in
    models.yaml order.
    """

    def __init__(self, cosmo_model, temperature, prior_args_path="prior_args.yaml", device="cpu"):
        if temperature < 1:
            raise ValueError(f"temperature must be >= 1, got {temperature}")
        self.cosmo_model = cosmo_model
        self.temperature = float(temperature)
        self.device = device
        self.prior_args_path = Path(get_experiment_config_path("num_tracers", prior_args_path))
        with open(get_experiment_config_path("num_tracers", "models.yaml")) as f:
            model = yaml.safe_load(f)["bao"][cosmo_model]
        with open(self.prior_args_path) as f:
            prior_args = yaml.safe_load(f)
        self.param_names = list(model["parameters"])
        bounds = []
        for name in self.param_names:
            dist = prior_args["parameters"][name]["distribution"]
            if dist["type"] != "uniform":
                raise ValueError(f"{name}: the box prior must be uniform, got {dist['type']!r}")
            bounds.append((float(dist["lower"]), float(dist["upper"])))
        self.bounds = torch.tensor(bounds, dtype=torch.float64, device=device)
        self.constraints = {c: prior_args["constraints"][c] for c in model.get("constraints", [])}
        self.distances = _Distances(float(prior_args["parameters"]["hrdrag"]["multiplier"]), device)

        rows = np.loadtxt(DR1_BAO_MEAN, dtype=str)
        self.z = rows[:, 0].astype(float)
        self.quantity = rows[:, 2]
        self.data = torch.tensor(rows[:, 1].astype(float), dtype=torch.float64, device=device)
        cov = np.loadtxt(DR1_BAO_COV)
        self.precision = torch.tensor(np.linalg.inv(cov), dtype=torch.float64, device=device)

    def in_support(self, theta):
        """Mask of rows of theta (N, D) inside the box and the sum constraints."""
        inside = ((theta > self.bounds[:, 0]) & (theta < self.bounds[:, 1])).all(-1)
        for constraint in self.constraints.values():
            total = sum(
                theta[:, self.param_names.index(p)] for p in constraint["affected_parameters"]
            )
            if "lower" in constraint["bounds"]:
                inside &= total > constraint["bounds"]["lower"]
            if "upper" in constraint["bounds"]:
                inside &= total < constraint["bounds"]["upper"]
        return inside

    def theory(self, theta):
        """DESI DR1 BAO observables (N, 12) at theta (N, D), in the data file's order."""
        params = {name: theta[:, i : i + 1] for i, name in enumerate(self.param_names)}
        out = torch.empty(theta.shape[0], len(self.z), dtype=torch.float64, device=self.device)
        funcs = {
            "DM_over_rs": self.distances.D_M_func,
            "DH_over_rs": self.distances.D_H_func,
            "DV_over_rs": self.distances.D_V_func,
        }
        for quantity, func in funcs.items():
            rows = np.flatnonzero(self.quantity == quantity)
            out[:, rows] = func(self.z[rows], **params)
        return out

    def log_likelihood(self, theta):
        r = self.data - self.theory(theta)
        return -0.5 * torch.einsum("ni,ij,nj->n", r, self.precision, r)

    def log_prob(self, theta):
        """Unnormalized log pi_box + log L / T; -inf outside the support."""
        out = torch.full(theta.shape[:1], -math.inf, dtype=torch.float64, device=self.device)
        inside = self.in_support(theta)
        out[inside] = self.log_likelihood(theta[inside]) / self.temperature
        return out


def sample_ensemble(log_prob, x0, n_steps, generator, a=2.0):
    """Goodman & Weare stretch-move ensemble MCMC; returns the chain (n_steps, W, D) and acceptance.

    Each half of the ensemble moves against the other half, so all walkers update in
    one vectorized ``log_prob`` call per half-step.
    """
    x = x0.clone()
    lp = log_prob(x)
    if not torch.isfinite(lp).all():
        raise ValueError("every starting walker must lie inside the support")
    n_walkers, dim = x.shape
    halves = (torch.arange(n_walkers // 2), torch.arange(n_walkers // 2, n_walkers))
    chain = torch.empty(n_steps, n_walkers, dim, dtype=x.dtype)
    n_accepted = 0
    for step in range(n_steps):
        for moving, fixed in (halves, halves[::-1]):
            u = torch.rand(len(moving), generator=generator, dtype=x.dtype)
            z = ((a - 1.0) * u + 1.0) ** 2 / a
            partner = fixed[torch.randint(len(fixed), (len(moving),), generator=generator)]
            proposal = x[partner] + z[:, None] * (x[moving] - x[partner])
            lp_proposal = log_prob(proposal)
            log_ratio = (dim - 1) * torch.log(z) + lp_proposal - lp[moving]
            accept = (
                torch.log(torch.rand(len(moving), generator=generator, dtype=x.dtype)) < log_ratio
            )
            x[moving[accept]] = proposal[accept]
            lp[moving[accept]] = lp_proposal[accept]
            n_accepted += int(accept.sum())
        chain[step] = x.cpu()
    return chain, n_accepted / (n_steps * n_walkers)


def integrated_autocorr_time(chain, c=5.0):
    """Integrated autocorrelation time per parameter (steps) of chain (n_steps, W, D).

    The walker-averaged autocorrelation function with Sokal's automatic window, as in
    emcee's ``integrated_time``.
    """
    x = chain.numpy()
    n = x.shape[0]
    x = x - x.mean(axis=0)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)
    acf /= acf[0]
    tau = 2.0 * np.cumsum(acf, axis=0) - 1.0
    taus = []
    for d in range(x.shape[2]):
        window = np.flatnonzero(np.arange(n) >= c * tau[:, d])
        taus.append(tau[window[0] if len(window) else n - 1, d])
    return np.array(taus)


def sample_tempered_posterior(
    cosmo_model,
    temperature,
    n_walkers=1024,
    n_steps=4000,
    prior_args_path="prior_args.yaml",
    seed=0,
    device="cpu",
):
    """Draw from pi_box * L_DR1^(1/T); returns (samples (N, D), metadata).

    The first half of the steps is burn-in; the rest is thinned by the largest
    autocorrelation time. Raises if the kept half spans fewer than 50 of them.
    """
    target = TemperedDR1Posterior(cosmo_model, temperature, prior_args_path, device)
    generator = torch.Generator().manual_seed(seed)
    ref = torch.tensor([_REFERENCE[p] for p in target.param_names], dtype=torch.float64)
    scale = torch.tensor([_REFERENCE_SCALE[p] for p in target.param_names], dtype=torch.float64)
    x0 = (
        ref + scale * torch.randn(n_walkers, len(ref), generator=generator, dtype=torch.float64)
    ).to(device)
    chain, acceptance = sample_ensemble(target.log_prob, x0, n_steps, generator)
    kept = chain[n_steps // 2 :]
    tau = integrated_autocorr_time(kept)
    if len(kept) < 50 * tau.max():
        raise RuntimeError(
            f"{len(kept)} post-burn-in steps span only {len(kept) / tau.max():.0f} autocorrelation "
            f"times (tau = {np.round(tau, 1).tolist()}); raise n_steps"
        )
    thin = max(1, int(math.ceil(tau.max())))
    samples = kept[::thin].reshape(-1, len(ref)).numpy()
    metadata = {
        "cosmo_model": cosmo_model,
        "temperature": target.temperature,
        "param_names": target.param_names,
        "prior_args_path": str(target.prior_args_path),
        "prior_args_sha256": _sha256(target.prior_args_path),
        "bao_mean": str(DR1_BAO_MEAN),
        "bao_mean_sha256": _sha256(DR1_BAO_MEAN),
        "bao_cov": str(DR1_BAO_COV),
        "bao_cov_sha256": _sha256(DR1_BAO_COV),
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "burn_in": n_steps // 2,
        "thin": thin,
        "autocorr_time": tau.tolist(),
        "acceptance": acceptance,
        "seed": seed,
        "bedcosmo_commit": _git_commit(),
    }
    return samples, metadata


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    repo = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    )
    return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cosmo-model", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument(
        "--out", required=True, help="Output .npz: samples (N, D) and a JSON metadata string."
    )
    ap.add_argument("--prior-args", default="prior_args.yaml")
    ap.add_argument("--n-walkers", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    samples, metadata = sample_tempered_posterior(
        args.cosmo_model,
        args.temperature,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        prior_args_path=args.prior_args,
        seed=args.seed,
        device=args.device,
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, samples=samples, metadata=json.dumps(metadata))
    print(f"{len(samples)} samples -> {out}")
    for i, name in enumerate(metadata["param_names"]):
        print(f"  {name:7s} mean {samples[:, i].mean():.4f}  std {samples[:, i].std():.4f}")
    print(
        f"  acceptance {metadata['acceptance']:.3f}  tau {np.round(metadata['autocorr_time'], 1).tolist()}"
    )


if __name__ == "__main__":
    main()
