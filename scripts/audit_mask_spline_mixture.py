"""Fit and evaluate a two-component spline prior on observed DESI rows."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from audit_multiscale_joint_prior_flow import local_scores, make_mask_scales, optimize
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.distributions import AffineTransform, TransformedDistribution
from tune_joint_prior_sample_loss import boundary_rates, projection_bank
from zuko.distributions import Mixture

from bedcosmo.num_visits.empirical.joint_prior_flow import (
    JointPriorFlow,
    build_flow,
    encode_rows,
    initialize_direct_prior,
)


class SplineMixtureDistribution(Mixture):
    def draw_from_base(self, z, uniform):
        """Draw a hard component with a straight-through weight gradient."""
        base = self.base.base_dist
        noise = z.unsqueeze(-2).expand(-1, 2, -1)
        components = self.base.transforms[0](base.transform.inv(noise))
        weight = self.logits.softmax(dim=-1)[:, 1]
        choose_second = uniform < weight
        # The correction is zero in the forward pass. It passes a gradient
        # to the gate using the difference between the two component draws.
        gate = (weight - weight.detach()).unsqueeze(-1)
        delta = components[:, 1] - components[:, 0]
        return (
            torch.where(choose_second[:, None], components[:, 1], components[:, 0]) + gate * delta
        )


class MaskSplineMixture(nn.Module):
    """Shared spline network with a component input and mask-specific weights."""

    def __init__(self, dim, n_masks, config, eligible):
        super().__init__()
        self.spline = build_flow(dim, n_masks + 1, config)
        self.locations = nn.Parameter(torch.zeros(len(eligible), 2, dim, dtype=torch.float32))
        self.log_odds = nn.Parameter(torch.zeros(len(eligible), dtype=torch.float32))
        self.register_buffer("eligible", torch.as_tensor(eligible, dtype=torch.bool))
        self.register_buffer("bits", 2 ** torch.arange(n_masks))
        self.register_buffer("component_flags", torch.tensor([False, True]))

    def mask_ids(self, mask):
        return (mask.long() * self.bits).sum(dim=-1)

    def forward(self, mask):
        ids = self.mask_ids(mask)
        flags = (self.eligible[ids, None] & self.component_flags).to(mask)
        context = torch.cat(
            [mask[..., None, :].expand(*mask.shape[:-1], 2, mask.shape[-1]), flags.unsqueeze(-1)],
            dim=-1,
        )
        components = TransformedDistribution(
            self.spline(context), AffineTransform(self.locations[ids], 1.0, event_dim=1)
        )
        second_logit = self.log_odds[ids].masked_fill(~self.eligible[ids], -torch.inf)
        logits = torch.stack([torch.zeros_like(second_logit), second_logit], dim=-1)
        return SplineMixtureDistribution(components, logits)

    def initialize_locations(self, features, mask, seed):
        ids = self.mask_ids(mask).numpy()
        with torch.no_grad():
            for i in np.flatnonzero(self.eligible.numpy()):
                gmm = GaussianMixture(2, covariance_type="diag", random_state=seed)
                gmm.fit(features[ids == i].numpy())
                self.locations[i].copy_(torch.as_tensor(gmm.means_))
                self.log_odds[i] = float(np.log(gmm.weights_[1] / gmm.weights_[0]))


class MixturePrior(JointPriorFlow):
    """Offline density and sampler with no component-free Gaussian inverse."""

    def gaussianize(self, rows):
        raise NotImplementedError("A sampled mixture component is required")

    def degaussianize(self, pattern_ids, gaussian_groups):
        raise NotImplementedError("A sampled mixture component is required")

    def save(self, path):
        torch.save(
            {
                "version": "mask_spline_mixture_v1",
                "metadata": self.metadata,
                "groups": {
                    m: {
                        "mu": g["mu"],
                        "std": g["std"],
                        "eligible": g["flow"].eligible,
                        "flow_state": g["flow"].state_dict(),
                    }
                    for m, g in self.groups.items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["version"] != "mask_spline_mixture_v1":
            raise ValueError("Unsupported spline mixture checkpoint")
        metadata = payload["metadata"]
        groups = {}
        for m, saved in payload["groups"].items():
            flow = MaskSplineMixture(
                m + 1, metadata["patterns"].shape[1], metadata["flow_config"], saved["eligible"]
            )
            flow.load_state_dict(saved["flow_state"])
            groups[m] = {"flow": flow.eval(), "mu": saved["mu"], "std": saved["std"]}
        return cls(metadata, groups)


def train_mixture(target, validation, seed, epochs, min_mask_count):
    initial = initialize_direct_prior(target, seed=seed)
    model = MixturePrior(initial.metadata, initial.groups)
    model.metadata["min_mask_count"] = min_mask_count
    patterns, counts = model.patterns, model.metadata["counts"]
    masks = [rows[:, :-2] > 0 for rows in (target.rows, validation)]
    fit_ok = counts[target.group] > 1
    val_ok = np.all(
        (validation[:, -2:] > model.metadata["lower"])
        & (validation[:, -2:] < model.metadata["upper"]),
        axis=1,
    )
    continuous_masks = {tuple(mask) for mask in patterns[counts > 1]}
    val_ok &= np.array([tuple(mask) in continuous_masks for mask in masks[1]])
    bits = 2 ** np.arange(patterns.shape[1])
    for m, group in model.groups.items():
        torch.manual_seed(seed + m)
        eligible = np.zeros(2 ** len(bits), dtype=bool)
        eligible[patterns[(counts >= min_mask_count) & (patterns.sum(axis=1) == m)] @ bits] = True
        if not eligible.any():
            print(f"active={m} uses fitted single spline for its rare masks", flush=True)
        flow = MaskSplineMixture(m + 1, len(bits), model.metadata["flow_config"], eligible)
        arrays = []
        for rows, mask, included in zip((target.rows, validation), masks, (fit_ok, val_ok)):
            use = included & (mask.sum(axis=1) == m)
            encoded, _ = encode_rows(rows[use], model.metadata["lower"], model.metadata["upper"])
            arrays.extend(
                (
                    torch.as_tensor((encoded - group["mu"]) / group["std"], dtype=torch.float32),
                    torch.as_tensor(mask[use], dtype=torch.float32),
                )
            )
        x, mask, val_x, val_mask = arrays
        flow.initialize_locations(x, mask, seed)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
        best_nll = np.inf
        for epoch in range(1, epochs + 1):
            flow.train()
            for idx in torch.randperm(len(x)).split(256):
                optimizer.zero_grad()
                loss = -flow(mask[idx]).log_prob(x[idx]).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss for {m} active fractions")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 10)
                optimizer.step()
            flow.eval()
            with torch.no_grad():
                nll = -flow(val_mask).log_prob(val_x).mean().item()
            if not np.isfinite(nll):
                raise ValueError(f"Nonfinite validation NLL for {m} active fractions")
            if nll < best_nll:
                best_nll, best_epoch, state = nll, epoch, copy.deepcopy(flow.state_dict())
            if epoch - best_epoch >= 18:
                break
        flow.load_state_dict(state)
        group["flow"] = flow.eval()
        print(
            f"active={m} mixture_masks={int(eligible.sum())} epoch={best_epoch}/{epoch} validation_nll={best_nll:.4f}",
            flush=True,
        )
    return model


def pair_score(fitted, reference, draws):
    def pair(rows):
        selected = rows[:, [1, 6]]
        return selected[(selected > 0).all(axis=1)]

    upper = np.quantile(pair(fitted), 0.995, axis=0)
    grid = [np.r_[np.linspace(0, u, 35)[:-1], 1.000001] for u in upper]
    masses = []
    for rows in (reference, draws):
        selected = pair(rows)
        masses.append(np.histogram2d(*selected.T, bins=grid)[0] / len(selected))
    return {
        name: float(0.5 * np.abs(gaussian_filter(masses[0] - masses[1], width)).sum())
        for name, width in (("fine", 0.8), ("coarse", 2))
    }


def run(args):
    with np.load(args.fit) as f, np.load(args.baseline_dir / "splits.npz") as s:
        fitted, val, test = [f["training_rows"][s[key]] for key in ("fit", "validation", "test")]
        names = f["feature_names"].tolist()
        target = JointPatternMixture(fitted, bandwidth=float(f["bandwidth"]))
    seed = json.loads((args.baseline_dir / "comparison.json").read_text())["seed"]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    single = JointPriorFlow.load(args.baseline_dir / "likelihood_only.pt")
    scale = fitted.std(axis=0)
    scales = make_mask_scales(fitted, target, scale)
    if args.mixture_init:
        initial = MixturePrior.load(args.mixture_init)
        for prior in (initial, single):
            np.testing.assert_array_equal(prior.patterns, target.patterns)
            np.testing.assert_array_equal(prior.metadata["counts"], target.counts)
        for key in ("lower", "upper"):
            np.testing.assert_array_equal(initial.metadata[key], single.metadata[key])
        generator = torch.Generator(device="cpu").manual_seed(seed + 39)

        def draw_latent(dist, z):
            uniform = torch.rand(len(z), generator=generator, dtype=z.dtype)
            return dist.draw_from_base(z, uniform)

        options = {
            "seed": seed + 38,
            "device": torch.device("cpu"),
            "steps": args.steps,
            "batch_size": 2048,
            "weight": 4,
        }
        print("Fine-tuning single spline", flush=True)
        tuned_single = optimize(copy.deepcopy(single), target, val, scale, scales, **options)
        print("Fine-tuning spline mixture", flush=True)
        model = optimize(
            copy.deepcopy(initial), target, val, scale, scales, draw_latent=draw_latent, **options
        )
    else:
        model = train_mixture(target, val, seed, args.epochs, args.min_mask_count)
        tuned_single = JointPriorFlow.load(args.multiscale_prior)
    model.metadata["feature_names"] = names
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "prior_flow.pt"
    model.save(path)
    model = MixturePrior.load(path)
    if args.mixture_init:
        single_path = args.output_dir / "single_prior_flow.pt"
        tuned_single.save(single_path)
        tuned_single = JointPriorFlow.load(single_path)
    priors = {
        "mixture": model,
        "single_likelihood": single,
        "single_multiscale": tuned_single,
        "raw_kde": target,
    }
    if args.mixture_init:
        priors["mixture_likelihood"] = initial
    projection = projection_bank(fitted, np.random.default_rng(seed + 310), 64)
    quantiles = np.linspace(0.001, 0.999, 401)
    results = {}
    if args.mixture_init:
        results["multiscale_training"] = {
            name: priors[name].metadata["multiscale_training"]
            for name in ("mixture", "single_multiscale")
        }
    for split, rows in (("validation", val), ("test", test)):
        results[split] = {}
        for key, prior in priors.items():
            draws = prior.sample(60_000, seed=seed + 101)
            if key != "raw_kde":
                if not np.isfinite(prior.log_prob(draws)).all():
                    raise ValueError("Prior does not score its own samples")
            risks, covered = local_scores(target, scale, scales, rows, draws[:24000], seed + 395)
            q_obs, q_sim = (
                np.quantile(x @ projection, quantiles, axis=0) for x in (rows, draws[:24000])
            )
            scores = {
                "mmd2": risks.tolist(),
                "mmd2_mean": float(risks.mean()),
                "evaluated_local_rows": covered,
                "joint_distance": float(np.abs(q_sim - q_obs).mean()),
                "near_zero_mae_pp": float(
                    100 * np.abs(boundary_rates(draws[:24000]) - boundary_rates(rows)).mean()
                ),
                "a2_a7": pair_score(fitted, rows, draws),
            }
            if key != "raw_kde":
                lp = prior.log_prob(rows)
                supported = np.isfinite(lp)
                scores.update(
                    mean_log_density_nats=float(lp[supported].mean()),
                    density_rows=int(supported.sum()),
                )
            results[split][key] = scores
    (args.output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(results["test"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--multiscale-prior", type=Path)
    modes.add_argument("--mixture-init", type=Path, help="Fit both priors with the multiscale loss")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-mask-count", type=int, default=75)
    args = parser.parse_args()
    if min(args.epochs, args.steps) < 1 or args.min_mask_count < 2:
        parser.error("Use a positive epoch limit and at least two rows per mixture mask")
    torch.set_num_threads(2)
    run(args)
