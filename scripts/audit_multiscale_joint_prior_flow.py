"""Compare spline capacity and a multiscale loss on full DESI rows."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tune_joint_prior_sample_loss import boundary_rates, decode_torch, projection_bank

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, encode_rows, train

VARIANTS = ("current_old", "current_multiscale", "larger_multiscale")
SCALES = np.array([0.5, 1.0, 2.0])


def kernel_mmd2(observed, generated, bandwidths):
    """Unbiased squared MMD for Gaussian kernels, one value per bandwidth."""
    if min(len(observed), len(generated)) < 2:
        raise ValueError("MMD requires at least two points per population")
    xx = torch.cdist(observed, observed).square()
    yy = torch.cdist(generated, generated).square()
    xy = torch.cdist(observed, generated).square()
    lhs = 2 * bandwidths[:, None, None].square()
    return (
        (torch.exp(-xx / lhs).sum((1, 2)) - len(observed)) / (len(observed) * (len(observed) - 1))
        + (torch.exp(-yy / lhs).sum((1, 2)) - len(generated))
        / (len(generated) * (len(generated) - 1))
        - 2 * torch.exp(-xy / lhs).mean((1, 2))
    )


def normalize_scale_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != SCALES.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Provide three finite, nonnegative scale weights")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Scale weights must have a finite positive sum")
    return weights / total


def make_mask_scales(training, target, scale):
    masks = {}
    for i, (mask, count) in enumerate(zip(target.patterns, target.counts)):
        if count < 75:
            continue
        cols = np.flatnonzero(np.r_[mask, True, True])
        rows = training[np.ix_(target.members[i], cols)] / scale[cols]
        typical = np.median(cKDTree(rows).query(rows, k=11)[0][:, -1])
        if typical <= 0:
            raise ValueError("A typical neighborhood radius must be positive")
        masks[i] = (cols, typical * SCALES)
    if not masks:
        raise ValueError("No mask has at least 75 fitted rows")
    return masks


def local_scores(target, scale, mask_scales, real, simulated, seed):
    """Compare whole physical rows at three scales, holding the mask fixed."""
    real_masks, sim_masks = real[:, :-2] > 0, simulated[:, :-2] > 0
    aggregate = np.zeros(len(SCALES))
    n_total = 0
    rng = np.random.default_rng(seed)
    for i, (cols, widths) in mask_scales.items():
        mask = target.patterns[i]
        x = real[np.all(real_masks == mask, axis=1)][:, cols] / scale[cols]
        y = simulated[np.all(sim_masks == mask, axis=1)][:, cols] / scale[cols]
        if len(x) < 20:
            continue
        if len(y) < 2:
            raise ValueError("A common validation mask has too few simulated samples")
        y = y[rng.choice(len(y), size=min(len(y), 512), replace=False)]
        xx, yy, xy = (cdist(a, b, "sqeuclidean") for a, b in ((x, x), (y, y), (x, y)))
        for j, width in enumerate(widths):
            v = 2 * width**2
            a = (np.exp(-xx / v).sum() - len(x)) / (len(x) * (len(x) - 1))
            b = (np.exp(-yy / v).sum() - len(y)) / (len(y) * (len(y) - 1))
            aggregate[j] += len(x) * (a + b - 2 * np.exp(-xy / v).mean())
        n_total += len(x)
    if not n_total:
        raise ValueError("No held-out mask has enough rows for the multiscale score")
    return aggregate / n_total, n_total


def optimize(
    model,
    target,
    val,
    scale,
    mask_scales,
    *,
    seed,
    device,
    steps,
    batch_size,
    weight,
    mask_batch_cap=128,
    scale_weights=(1, 1, 1),
    draw_latent=None,
):
    if draw_latent is None:

        def draw_latent(dist, z):
            return dist.transform.inv(z)

    rng = np.random.default_rng(seed)
    training = target.rows
    lower, upper = [model.metadata[k] for k in ("lower", "upper")]
    lo, hi = [torch.as_tensor(x, device=device) for x in (lower, upper)]
    physical = torch.as_tensor(training, device=device)
    masks = torch.as_tensor(training[:, :-2] > 0, device=device)
    features = {}
    gpu_flows = {}
    for m, group in model.groups.items():
        chosen = np.flatnonzero(
            (target.patterns[target.group].sum(1) == m) & (target.counts[target.group] > 1)
        )
        encoded, _ = encode_rows(training[chosen], lower, upper)
        t = torch.as_tensor(
            (encoded - group["mu"]) / group["std"], device=device, dtype=torch.float32
        )
        mu, std = [torch.as_tensor(group[k], device=device) for k in ("mu", "std")]
        features[m] = (chosen, t, mu, std)
        gpu_flows[m] = copy.deepcopy(group["flow"]).to(device)
    params = [p for flow in gpu_flows.values() for p in flow.parameters()]
    opt = torch.optim.Adam(params, lr=2e-4)
    positions = (np.arange(batch_size) + 0.5) / batch_size
    bank = projection_bank(training, rng, 192)
    raw_quantiles = torch.as_tensor(np.quantile(training @ bank, positions, axis=0), device=device)
    bank = torch.as_tensor(bank, device=device)
    std_device = torch.as_tensor(scale, device=device)
    scale_weights = normalize_scale_weights(scale_weights)
    kernel_weights = torch.as_tensor(scale_weights, device=device, dtype=torch.float32)
    metric_projection = projection_bank(training, np.random.default_rng(seed + 1), 64)
    q = np.linspace(0.001, 0.999, 401)
    val_q = np.quantile(val @ metric_projection, q, axis=0)
    training_scales = {
        i: (
            torch.as_tensor(cols, device=device),
            torch.as_tensor(widths, device=device, dtype=torch.float32),
        )
        for i, (cols, widths) in mask_scales.items()
    }

    def check(step):
        for m, flow in gpu_flows.items():
            model.groups[m]["flow"].load_state_dict(flow.state_dict())
        sample = model.sample(16000, seed=seed + 3)
        distance = float(np.abs(np.quantile(sample @ metric_projection, q, axis=0) - val_q).mean())
        local, coverage = local_scores(target, scale, mask_scales, val, sample, seed + 4)
        lp = model.log_prob(val)
        mean = float(lp[np.isfinite(lp)].mean())
        return {
            "step": step,
            "joint_distance": distance,
            "multiscale_mmd2": local.tolist(),
            "joint_selection_score": distance + 4 * float(local.mean()),
            "mean_log_density": mean,
            "local_validation_count": coverage,
        }

    history = [check(0)]
    floor = history[0]["mean_log_density"] - 0.15
    best = history[0]
    best_states = {m: copy.deepcopy(g["flow"].state_dict()) for m, g in model.groups.items()}
    for step in range(1, steps + 1):
        ids = rng.integers(len(training), size=batch_size)
        ids_device = torch.as_tensor(ids, device=device)
        patterns = target.group[ids]
        empirical = physical[ids_device]
        simulated = empirical.clone()
        nll = torch.zeros((), device=device)
        total = 0
        for m, flow in gpu_flows.items():
            chosen, observed, mu, std = features[m]
            mask_idx = np.flatnonzero(
                (target.patterns[patterns].sum(1) == m) & (target.counts[patterns] > 1)
            )
            if not len(mask_idx):
                continue
            select = torch.as_tensor(mask_idx, device=device)
            data_idx = ids_device[select]
            flow.train()
            dist = flow(masks[data_idx].float())
            z = torch.as_tensor(
                rng.normal(size=(len(select), m + 1)), device=device, dtype=torch.float32
            )
            simulated[select] = decode_torch(
                draw_latent(dist, z) * std + mu, masks[data_idx], lo, hi
            )
            index = torch.as_tensor(np.searchsorted(chosen, ids[mask_idx]), device=device)
            nll = nll - dist.log_prob(observed[index]).sum()
            total += len(select)
        proj_ids = torch.as_tensor(rng.choice(bank.shape[1], size=40, replace=False), device=device)
        delta = (simulated @ bank[:, proj_ids]).sort(0).values - raw_quantiles[:, proj_ids]
        loss = torch.sqrt(delta.square() + 0.005**2).mean() + 0.08 * nll / total
        if weight:
            local = torch.zeros((), device=device)
            n_local = 0
            for i, (cols, widths) in training_scales.items():
                mask_idx = np.flatnonzero(patterns == i)
                if len(mask_idx) < 2:
                    continue
                selected = torch.as_tensor(mask_idx[:mask_batch_cap], device=device)
                x = (empirical[selected][:, cols] / std_device[cols]).float()
                y = (simulated[selected][:, cols] / std_device[cols]).float()
                local = local + len(selected) * torch.dot(kernel_mmd2(x, y, widths), kernel_weights)
                n_local += len(selected)
            if n_local:
                loss = loss + weight * local / n_local
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite multiscale training loss")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True)
        opt.step()
        if step % 30 == 0 or step == steps:
            state = check(step)
            history.append(state)
            if (
                state["joint_selection_score"] < best["joint_selection_score"]
                and state["mean_log_density"] >= floor
            ):
                best = state
                best_states = {
                    m: copy.deepcopy(g["flow"].state_dict()) for m, g in model.groups.items()
                }
            print(state, flush=True)
    for m, state in best_states.items():
        model.groups[m]["flow"].load_state_dict(state)
    model.metadata["multiscale_training"] = {
        "steps": steps,
        "batch_size": batch_size,
        "mask_batch_cap": mask_batch_cap,
        "mmd_weight": weight,
        "scale_weights_fine_to_broad": scale_weights.tolist(),
        "selected_step": best["step"],
        "history": history,
        "validation_density_floor": floor,
        "device": str(device),
        "n_parameters": sum(p.numel() for p in params),
    }
    return model


def run(args):
    start_time = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this host")
    base = json.loads((args.baseline_dir / "comparison.json").read_text())
    seed = base["seed"]
    with np.load(args.fit) as f, np.load(args.baseline_dir / "splits.npz") as s:
        rows = f["training_rows"]
        fitted, val, test = [rows[s[k]] for k in ("fit", "validation", "test")]
        h = float(f["bandwidth"])
        names = f["feature_names"].tolist()
    target = JointPatternMixture(fitted, bandwidth=h)
    if args.variant == "larger_multiscale":
        if args.large_init:
            initial = JointPriorFlow.load(args.large_init)
        else:
            initial = train(
                target,
                direct=True,
                validation_rows=val,
                seed=seed,
                n_samples=len(fitted),
                hidden=96,
                transforms=5,
                bins=12,
                epochs=100,
                batch_size=256,
                patience=18,
            )
    else:
        initial = JointPriorFlow.load(args.baseline_dir / "likelihood_only.pt")
    np.testing.assert_array_equal(initial.patterns, target.patterns)
    np.testing.assert_array_equal(initial.metadata["counts"], target.counts)
    scale = fitted.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Physical coordinates must have nonzero training spread")
    mask_scales = make_mask_scales(fitted, target, scale)
    weight = 0 if args.variant == "current_old" else args.mmd_weight
    model = optimize(
        initial,
        target,
        val,
        scale,
        mask_scales,
        seed=seed + 38,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        mask_batch_cap=args.mask_batch_cap,
        weight=weight,
        scale_weights=args.scale_weights,
    )
    model.metadata["feature_names"] = names
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "prior_flow.pt"
    model.save(path)
    model = JointPriorFlow.load(path)
    model_draws = model.sample(24000, seed=seed + 101)
    assert np.isfinite(model.log_prob(model_draws)).all()
    np.testing.assert_allclose(model_draws[:, :-2].sum(1), 1)
    kde = target.sample(24000, seed=seed + 101)
    p = projection_bank(fitted, np.random.default_rng(seed + 310), 64)
    q = np.linspace(0.001, 0.999, 401)
    test_q = np.quantile(test @ p, q, axis=0)
    ref_cdf = boundary_rates(test)
    report = {
        "seed": seed,
        "variant": args.variant,
        "flow_config": model.metadata["flow_config"],
        "training": model.metadata["multiscale_training"],
        "eligible_training_masks": len(mask_scales),
        "seconds": time.perf_counter() - start_time,
        "test": {},
    }
    for label, draws in [("flow", model_draws), ("raw_kde", kde)]:
        values, covered = local_scores(target, scale, mask_scales, test, draws, seed + 395)
        report["test"][label] = {
            "mmd2": values.tolist(),
            "mmd2_mean": float(values.mean()),
            "evaluated_local_rows": covered,
            "joint_distance": float(np.abs(np.quantile(draws @ p, q, axis=0) - test_q).mean()),
            "near_zero_mae_pp": float(100 * np.abs(boundary_rates(draws) - ref_cdf).mean()),
        }
    lp = model.log_prob(test)
    report["test"]["flow"]["mean_log_density_nats"] = float(lp[np.isfinite(lp)].mean())
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--large-init", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--mask-batch-cap", type=int, default=128)
    p.add_argument("--mmd-weight", type=float, default=4.0)
    p.add_argument(
        "--scale-weights",
        type=float,
        nargs=3,
        default=(1, 1, 1),
        metavar=("FINE", "MEDIUM", "BROAD"),
    )
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 128 or a.mask_batch_cap < 2:
        p.error("Use positive steps, at least 128 rows per batch, and at least two rows per mask")
    if a.large_init and a.variant != "larger_multiscale":
        p.error("--large-init is only for the larger model")
    if not np.isfinite(a.mmd_weight) or a.mmd_weight < 0:
        p.error("--mmd-weight must be finite and nonnegative")
    try:
        normalize_scale_weights(a.scale_weights)
    except ValueError as error:
        p.error(str(error))
    torch.set_num_threads(2)
    run(a)
