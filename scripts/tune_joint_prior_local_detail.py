"""Audit a sample loss for local structure in the all-active DESI population."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from prototype_joint_empirical_mixture import JointPatternMixture
from tune_joint_prior_sample_loss import decode_torch, evaluate, projection_bank

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, encode_rows

PAIR_NAMES = ("a3_vs_a8", "log_amplitude_vs_redshift")
SCALES = ((12, 0.28), (28, 0.10))


def soft_assignment(values, edges, bandwidth):
    """Assign each observation to bins using a Gaussian CDF, including both tails."""
    cdf = torch.special.ndtr((edges[None, :] - values[:, None]) / bandwidth)
    zeros = torch.zeros_like(cdf[:, :1])
    return torch.diff(torch.cat([zeros, cdf, 1 - zeros], dim=1), dim=1)


def soft_histogram(rows, pair, x_edges, y_edges, bandwidth):
    x = soft_assignment(rows[:, pair[0]], x_edges, bandwidth[0])
    y = soft_assignment(rows[:, pair[1]], y_edges, bandwidth[1])
    return x.T @ y / len(rows)


def build_grids(rows):
    k = rows.shape[1] - 2
    if k < 8:
        raise ValueError("This audit targets the eight-template DESI prior")
    edges = np.quantile(rows, [0.01, 0.99], axis=0)
    scale = rows.std(axis=0)
    grids = {}
    for name, pair in zip(PAIR_NAMES, [(2, 7), (k, k + 1)]):
        grids[name] = []
        for bins, width in SCALES:
            inside = [
                torch.as_tensor(np.linspace(edges[0, j], edges[1, j], bins + 1)[1:-1]) for j in pair
            ]
            grids[name].append((pair, *inside, tuple(width * scale[list(pair)])))
    return grids


def histograms(rows, grids):
    return {
        name: [soft_histogram(rows, *grid) for grid in pair_grids]
        for name, pair_grids in grids.items()
    }


def histogram_discrepancies(generated, reference, *, train=False):
    discrepancies = {}
    for name in PAIR_NAMES:
        distances = []
        for a, b in zip(generated[name], reference[name]):
            diff = a - b
            # Smooth the absolute error only for optimization.
            error = torch.sqrt(diff.square() + 1e-10) - 1e-5 if train else diff.abs()
            distances.append(error.sum() / 2)
        discrepancies[name] = torch.stack(distances)
    return discrepancies


def conditional_draws(model, n, seed):
    ids = np.full(n, model.lookup[(True,) * (model.patterns.shape[1])], dtype=int)
    rng = np.random.default_rng(seed)
    return model.degaussianize(
        ids, {model.patterns.shape[1]: rng.normal(size=(n, model.patterns.shape[1] + 1))}
    )


def fit_detail(base, fit, val, grids, *, seed, steps, local_weight):
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(base)
    m = model.patterns.shape[1]
    group = model.groups[m]
    flow = group["flow"]
    lo, hi = (model.metadata[k] for k in ("lower", "upper"))
    e, _ = encode_rows(fit, lo, hi)
    v, _ = encode_rows(val, lo, hi)
    scale = torch.as_tensor(group["std"])
    center = torch.as_tensor(group["mu"])
    observed = torch.as_tensor((e - group["mu"]) / group["std"], dtype=torch.float32)
    v_obs = torch.as_tensor((v - group["mu"]) / group["std"], dtype=torch.float32)
    lo, hi = torch.as_tensor(lo), torch.as_tensor(hi)
    target = histograms(torch.as_tensor(fit), grids)
    validation_target = histograms(torch.as_tensor(val), grids)
    direction = projection_bank(fit, rng, 128)
    probabilities = (np.arange(1536) + 0.5) / 1536
    fit_quantiles = torch.as_tensor(np.quantile(fit @ direction, probabilities, axis=0))
    direction = torch.as_tensor(direction)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

    def check(step):
        flow.eval()
        with torch.no_grad():
            pred = torch.as_tensor(conditional_draws(model, 8192, seed + 100))
            local = histogram_discrepancies(histograms(pred, grids), validation_target)
            mask = torch.ones((len(v_obs), m), dtype=torch.float32)
            nll = float(-flow(mask).log_prob(v_obs).mean())
            detail = float(torch.cat(list(local.values())).mean())
        return {"step": step, "local_distance": detail, "conditional_encoded_nll": nll}

    history = [check(0)]
    best = history[0]
    best_state = copy.deepcopy(flow.state_dict())
    context = torch.ones((len(probabilities), m), dtype=torch.bool)
    for step in range(1, steps + 1):
        flow.train()
        dist = flow(context.float())
        z = torch.as_tensor(rng.normal(size=(len(context), m + 1)), dtype=torch.float32)
        pred = decode_torch(dist.transform.inv(z) * scale + center, context, lo, hi)
        batch = rng.integers(len(observed), size=len(pred))
        nll = -dist.log_prob(observed[batch]).mean()
        proj_ids = rng.choice(direction.shape[1], size=40, replace=False)
        d = (pred @ direction[:, proj_ids]).sort(0).values - fit_quantiles[:, proj_ids]
        loss = torch.sqrt(d.square() + 0.005**2).mean() + 0.08 * nll
        if local_weight:
            local = histogram_discrepancies(histograms(pred, grids), target, train=True)
            loss = loss + local_weight * torch.cat(list(local.values())).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite local detail training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == steps:
            state = check(step)
            history.append(state)
            if (
                state["local_distance"] < best["local_distance"]
                and state["conditional_encoded_nll"] <= history[0]["conditional_encoded_nll"] + 0.15
            ):
                best = state
                best_state = copy.deepcopy(flow.state_dict())
            print(local_weight, state, flush=True)
    flow.load_state_dict(best_state)
    flow.eval()
    model.metadata["local_detail_audit"] = {
        "pairs": list(PAIR_NAMES),
        "local_weight": local_weight,
        "selected_step": best["step"],
        "history": history,
    }
    return model


def run(fit_path, baseline_dir, out, steps, local_weight):
    baseline = json.loads((baseline_dir / "comparison.json").read_text())
    seed = baseline["seed"]
    with np.load(fit_path) as saved, np.load(baseline_dir / "splits.npz") as split:
        rows = saved["training_rows"]
        training, validation, test = [rows[split[k]] for k in ("fit", "validation", "test")]
        h = float(saved["bandwidth"])
    base = JointPriorFlow.load(baseline_dir / "likelihood_then_combined.pt")
    fit, val, heldout = [x[(x[:, :-2] > 0).all(1)] for x in (training, validation, test)]
    lo, hi = (base.metadata[key] for key in ("lower", "upper"))
    val = val[((val[:, -2:] > lo) & (val[:, -2:] < hi)).all(1)]
    grids = build_grids(fit)
    out.mkdir(parents=True, exist_ok=False)
    models = {"starting_flow": base}
    for label, weight in [("continued_global_loss", 0.0), ("added_local_loss", local_weight)]:
        model = fit_detail(base, fit, val, grids, seed=seed + 77, steps=steps, local_weight=weight)
        model.save(out / f"{label}.pt")
        models[label] = JointPriorFlow.load(out / f"{label}.pt")
    report = {
        "seed": seed,
        "all_active_counts": [len(fit), len(val), len(heldout)],
        "training": {k: v.metadata.get("local_detail_audit") for k, v in models.items()},
    }
    draws = {label: conditional_draws(f, 24000, seed + 400) for label, f in models.items()}
    kde = JointPatternMixture(training, bandwidth=h).sample(100000, seed=seed + 400)
    draws["raw_kde"] = kde[(kde[:, :-2] > 0).all(1)]
    reference = histograms(torch.as_tensor(heldout), grids)
    report["conditional_test"] = {}
    for label, x in draws.items():
        np.testing.assert_allclose(x[:, :-2].sum(1), 1)
        assert (x[:, :-2] > 0).all()
        local = histogram_discrepancies(histograms(torch.as_tensor(x), grids), reference)
        values = {k: (100 * a).tolist() for k, a in local.items()}
        values["mean_pct"] = float(100 * torch.cat(list(local.values())).mean())
        report["conditional_test"][label] = values
    report["full_test"] = evaluate({**models, "raw_kde": kde}, training, test, seed)
    np.savez_compressed(out / "all_active_samples.npz", held_out_desi=heldout, **draws)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["conditional_test"], indent=2), flush=True)
    print(json.dumps(report["full_test"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", required=True, type=Path)
    p.add_argument("--baseline-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--local-weight", type=float, default=0.55)
    a = p.parse_args()
    if a.steps < 1 or not np.isfinite(a.local_weight) or a.local_weight < 0:
        p.error("Use positive steps and a finite, nonnegative local weight")
    torch.set_num_threads(2)
    run(a.fit, a.baseline_dir, a.output_dir, a.steps, a.local_weight)
