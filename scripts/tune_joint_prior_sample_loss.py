"""Fine-tune a joint-pattern spline flow against physical DESI sample distributions."""

import argparse
import copy
import hashlib
import json
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.special import ndtri
from scipy.stats import qmc
from torch.nn.functional import logsigmoid

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, encode_rows, train


def decode_torch(features, masks, lower, upper):
    """Differentiable counterpart of the physical stick/auxiliary decode."""
    m = features.shape[1] - 1
    stick = features[:, : m - 1].double()
    remaining = logsigmoid(-stick).cumsum(dim=1)
    prefix = torch.cat([stick.new_zeros((len(stick), 1)), remaining], dim=1)
    active = torch.cat([prefix[:, :-1] + logsigmoid(stick), prefix[:, -1:]], dim=1).softmax(1)
    fractions = active.new_zeros(masks.shape).masked_scatter(masks, active.reshape(-1))
    aux = lower + (upper - lower) * features[:, -2:].double().sigmoid()
    return torch.cat([fractions, aux], dim=1)


def projection_bank(rows, rng, count):
    directions = rng.normal(size=(rows.shape[1], count))
    directions /= np.linalg.norm(directions, axis=0)
    scale = rows.std(axis=0)
    return directions / np.where(scale > 0, scale, 1)[:, None]


def boundary_rates(rows):
    a = rows[:, :-2]
    return np.stack(
        [((a > 0) & (a <= t)).sum(0) / (a > 0).sum(0) for t in [0.001, 0.005, 0.01, 0.02, 0.05]]
    )


def single_run_schedule(progress, nll_weight, learning_rate):
    """Transition from likelihood to the combined loss in one optimizer run."""
    blend = np.clip((progress - 0.22) / 0.23, 0, 1)
    blend = blend**2 * (3 - 2 * blend)
    return 1 - (1 - nll_weight) * blend, blend, (1 - blend) * 0.001 + blend * learning_rate


def finetune(
    model,
    fitted,
    validation,
    *,
    seed,
    steps,
    nll_weight,
    batch_size=1024,
    full_reference=False,
    low_noise=False,
    lr=2e-4,
    moment_weight=0.0,
    cpu_budget=None,
    minimum_log_density=None,
    ramp_sample_loss=False,
    group_warmup=False,
):
    started = time.process_time()
    rng = np.random.default_rng(seed)
    mask_array = fitted[:, :-2] > 0
    ids = np.array([model.lookup[tuple(mask)] for mask in mask_array])
    lower, upper = (model.metadata[key] for key in ("lower", "upper"))
    lo, hi = map(torch.as_tensor, (lower, upper))
    observed = torch.as_tensor(fitted)
    masks = torch.as_tensor(mask_array)
    atomic = model.metadata["atomic"][ids]
    data = {}
    for m, group in model.groups.items():
        indices = np.flatnonzero((mask_array.sum(1) == m) & ~atomic)
        encoded, _ = encode_rows(fitted[indices], lower, upper)
        z = torch.as_tensor((encoded - group["mu"]) / group["std"], dtype=torch.float32)
        data[m] = (indices, z, torch.as_tensor(group["mu"]), torch.as_tensor(group["std"]))

    val_projection = projection_bank(fitted, np.random.default_rng(seed + 700), 96)
    quantiles = np.linspace(0.001, 0.999, 401)
    target_q = np.quantile(validation @ val_projection, quantiles, axis=0)
    val_masks = validation[:, :-2] > 0
    valid_density = np.array(
        [
            tuple(mask) in model.lookup and not model.metadata["atomic"][model.lookup[tuple(mask)]]
            for mask in val_masks
        ]
    )
    valid_density &= ((validation[:, -2:] > lower) & (validation[:, -2:] < upper)).all(1)
    assert valid_density.any()

    def validation_metrics():
        draws = model.sample(10000, seed=seed + 800)
        distance = np.abs(np.quantile(draws @ val_projection, quantiles, axis=0) - target_q).mean()
        log_density = model.log_prob(validation[valid_density]).mean()
        if not (np.isfinite(distance) and np.isfinite(log_density)):
            raise ValueError("Nonfinite validation metric")
        return {"distance": float(distance), "log_density": float(log_density)}

    params = [
        parameter for group in model.groups.values() for parameter in group["flow"].parameters()
    ]
    optimizer = torch.optim.Adam(params, lr=lr)
    baseline = validation_metrics()
    history = [{"step": 0, **baseline}]
    calibrate_floor = ramp_sample_loss and minimum_log_density is None
    if minimum_log_density is None:
        minimum_log_density = baseline["log_density"] - 0.15
    best = history[0]
    best_state = {m: copy.deepcopy(group["flow"].state_dict()) for m, group in model.groups.items()}
    floor_met = best["log_density"] >= minimum_log_density
    row_count = len(fitted)
    if moment_weight:
        standardized = (fitted - fitted.mean(axis=0)) / np.maximum(fitted.std(axis=0), 1e-10)
        center = torch.as_tensor(fitted.mean(axis=0))
        scale = torch.as_tensor(np.maximum(fitted.std(axis=0), 1e-10))
        target_cov = torch.as_tensor(standardized.T @ standardized / row_count)
        pair_i, pair_j = np.triu_indices(fitted.shape[1], k=1)
    if full_reference:
        # Remove reference minibatch noise. Vary the projections while reusing
        # the corresponding quantiles of the entire fitted population.
        bank = projection_bank(fitted, rng, 256)
        probabilities = (np.arange(batch_size) + 0.5) / batch_size
        reference_bank = torch.as_tensor(np.quantile(fitted @ bank, probabilities, axis=0))
        bank = torch.as_tensor(bank)
    if low_noise:
        mask_weights = np.cumsum(model.probabilities)
        mask_weights[-1] = 1
        mask_donors = [np.flatnonzero(ids == i) for i in range(len(model.patterns))]
    if group_warmup:
        groups = list(data)
        weights = np.array([np.ceil(len(data[m][0]) / 256) for m in groups])
        group_prob = weights / weights.sum()
    last_validation = time.process_time()
    for step in count(1) if cpu_budget is not None else range(1, steps + 1):
        progress = (
            (time.process_time() - started) / cpu_budget if cpu_budget else (step - 1) / steps
        )
        weight_nll, weight_sample = nll_weight, 1.0
        active_batch_size = batch_size
        if ramp_sample_loss:
            weight_nll, weight_sample, learning_rate = single_run_schedule(progress, nll_weight, lr)
            optimizer.param_groups[0]["lr"] = learning_rate
            if weight_sample == 0:
                active_batch_size = min(512, batch_size)
        selected_group = None
        if group_warmup and weight_sample == 0:
            selected_group = rng.choice(groups, p=group_prob)
            donor_rows = data[selected_group][0]
            batch = rng.choice(donor_rows, size=min(256, len(donor_rows)), replace=False)
        else:
            if low_noise:
                grid = np.arange(active_batch_size) / active_batch_size
                mask_ids = np.searchsorted(mask_weights, grid + rng.random() / active_batch_size)
                batch = np.array([rng.choice(mask_donors[i]) for i in mask_ids])
                if not ramp_sample_loss:
                    if cpu_budget is None:
                        progress = step / steps
                    optimizer.param_groups[0]["lr"] = lr * (
                        0.1 + 0.9 * (1 + np.cos(np.pi * min(progress, 1))) / 2
                    )
            else:
                batch = rng.integers(row_count, size=active_batch_size)
        reference = observed[batch]
        fake = reference.clone()
        nll_sum = torch.zeros((), dtype=torch.float32)
        continuous_count = 0
        for m, group in model.groups.items():
            if selected_group is not None and m != selected_group:
                continue
            all_indices, x_obs, mu, std = data[m]
            subset = np.flatnonzero((mask_array[batch].sum(1) == m) & ~atomic[batch])
            if not len(subset):
                continue
            indices = batch[subset]
            context = masks[indices].float()
            group["flow"].train()
            dist = group["flow"](context)
            if weight_sample > 0:
                if low_noise:
                    # Randomized low-discrepancy latent draws, randomly assigned to masks.
                    sobol = qmc.Sobol(d=m + 1, scramble=True, seed=int(rng.integers(2**32)))
                    points = sobol.random_base2((len(subset) - 1).bit_length())[: len(subset)]
                    shuffled = points[rng.permutation(len(points))]
                    eps = np.finfo(float).eps
                    z_base = ndtri(np.clip(shuffled, eps, 1 - eps))
                else:
                    z_base = rng.normal(size=(len(subset), m + 1))
                base = torch.as_tensor(z_base, dtype=torch.float32)
                latent = dist.transform.inv(base)
                fake[subset] = decode_torch(latent * std + mu, masks[indices], lo, hi)
            nll_sum = nll_sum - dist.log_prob(x_obs[np.searchsorted(all_indices, indices)]).sum()
            continuous_count += len(subset)
        loss = weight_nll * nll_sum / continuous_count
        if weight_sample > 0:
            if full_reference:
                selected = rng.choice(bank.shape[1], size=48, replace=False)
                projection = bank[:, selected]
                reference_q = reference_bank[:, selected]
            else:
                projection = torch.as_tensor(projection_bank(fitted, rng, 48))
                reference_q = (reference @ projection).sort(0).values
            delta = (fake @ projection).sort(0).values - reference_q
            # Smooth the absolute penalty near zero to limit minibatch-quantile noise.
            sample_loss = torch.sqrt(delta.square() + 0.005**2).mean()
            loss = loss + weight_sample * sample_loss
            if moment_weight:
                standardized = (fake - center) / scale
                centered = standardized - standardized.mean(0)
                cov = centered.T @ centered / len(centered)
                covariance_error = cov[pair_i, pair_j] - target_cov[pair_i, pair_j]
                loss = loss + weight_sample * moment_weight * torch.sqrt(
                    covariance_error.square().mean() + 1e-8
                )
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite fine-tuning loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True)
        optimizer.step()
        finished = (
            time.process_time() - started >= cpu_budget if cpu_budget is not None else step == steps
        )
        should_validate = (
            time.process_time() - last_validation >= 15
            if ramp_sample_loss and cpu_budget is not None
            else step % 20 == 0
        )
        if should_validate or finished:
            state = {"step": step, "sample_weight": float(weight_sample), **validation_metrics()}
            last_validation = time.process_time()
            history.append(state)
            # Prefer sample fidelity among candidates that meet the common density floor.
            # If none meet it, report the highest-density available checkpoint as a fallback.
            meets_floor = state["log_density"] >= minimum_log_density
            improves = (
                meets_floor and (not floor_met or state["distance"] < best["distance"])
            ) or (not meets_floor and not floor_met and state["log_density"] > best["log_density"])
            if calibrate_floor and weight_sample == 0:
                improves = state["log_density"] > best["log_density"]
                if improves:
                    minimum_log_density = state["log_density"] - 0.15
            if improves:
                best = state
                floor_met = meets_floor
                best_state = {
                    m: copy.deepcopy(group["flow"].state_dict())
                    for m, group in model.groups.items()
                }
            print(state, flush=True)
        if finished:
            break
    for m, group in model.groups.items():
        group["flow"].load_state_dict(best_state[m])
        group["flow"].eval()
    model.metadata["sample_finetuning"] = {
        "nll_weight": nll_weight,
        "batch_size": batch_size,
        "full_reference": full_reference,
        "low_noise": low_noise,
        "learning_rate": lr,
        "moment_weight": moment_weight,
        "selected_step": best["step"],
        "ramp_sample_loss": ramp_sample_loss,
        "group_warmup": group_warmup,
        "steps_completed": step,
        "cpu_seconds": time.process_time() - started,
        "minimum_validation_log_density": minimum_log_density,
        "density_floor_met": floor_met,
        "history": history,
    }
    return model


def evaluate(models, fitted, reference, seed):
    projection = projection_bank(fitted, np.random.default_rng(seed + 310), 64)
    q = np.linspace(0.001, 0.999, 401)
    target_q = np.quantile(reference @ projection, q, axis=0)
    target_near_zero = boundary_rates(reference)
    report = {}
    for name, model in models.items():
        if isinstance(model, np.ndarray):
            rows = model
        else:
            rows = model.sample(24000, seed=seed + 101)
            assert np.isfinite(model.log_prob(rows)).all()
        np.testing.assert_allclose(rows[:, :-2].sum(axis=1), 1)
        report[name] = {
            "sliced_wasserstein": float(
                np.abs(np.quantile(rows @ projection, q, axis=0) - target_q).mean()
            ),
            "near_zero_mae_pp": float(100 * np.abs(boundary_rates(rows) - target_near_zero).mean()),
        }
        if not isinstance(model, np.ndarray):
            log_density = model.log_prob(reference)
            finite = np.isfinite(log_density)
            report[name]["mean_log_density_nats"] = float(log_density[finite].mean())
    return report


def run(args):
    torch.set_num_threads(2)
    if (
        args.steps < 1
        or args.batch_size < 2
        or not all(
            np.isfinite(value) and value >= 0 for value in [args.nll_weight, args.moment_weight]
        )
        or not np.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError(
            "Use positive steps and learning rate, at least two rows per batch, and nonnegative loss weights"
        )
    with np.load(args.fit) as data:
        rows = data["training_rows"]
        h = float(data["bandwidth"])
        names = data["feature_names"].tolist()
    rng = np.random.default_rng(args.seed)
    ids = np.split(rng.permutation(len(rows)), [int(0.6 * len(rows)), int(0.8 * len(rows))])
    if args.baseline_dir:
        with np.load(args.baseline_dir / "splits.npz") as saved:
            ids = [saved[k] for k in ["fit", "validation", "test"]]
    fitted, validation, test = [rows[i] for i in ids]
    target = JointPatternMixture(fitted, bandwidth=h)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(args.output_dir / "splits.npz", fit=ids[0], validation=ids[1], test=ids[2])
    if args.baseline_dir:
        base = JointPriorFlow.load(args.baseline_dir / "direct_prior_flow.pt")
    else:
        base = train(
            target,
            n_samples=len(fitted),
            epochs=100,
            batch_size=256,
            patience=18,
            seed=args.seed,
            direct=True,
            validation_rows=validation,
        )
        base.metadata["feature_names"] = names
        base.save(args.output_dir / "direct_prior_flow.pt")
    warm = JointPriorFlow.load(args.warm_start) if args.warm_start else base
    for model in (base, warm):
        if not np.array_equal(model.patterns, target.patterns) or not np.array_equal(
            model.metadata["counts"], target.counts
        ):
            raise ValueError("Starting checkpoints must be fitted on the same galaxy split")
    tuned = finetune(
        copy.deepcopy(warm),
        fitted,
        validation,
        seed=args.seed,
        steps=args.steps,
        nll_weight=args.nll_weight,
        batch_size=args.batch_size,
        full_reference=args.full_reference,
        low_noise=args.low_noise,
        lr=args.lr,
        moment_weight=args.moment_weight,
    )
    path = args.output_dir / "sample_loss_prior_flow.pt"
    tuned.save(path)
    tuned = JointPriorFlow.load(path)
    models = {
        "bootstrap": fitted[rng.integers(len(fitted), size=24000)],
        "raw_kde": target.sample(24000, seed=args.seed + 101),
        "direct_flow": base,
        "fine_tuned_flow": tuned,
    }
    report = {
        "seed": args.seed,
        "fit_sha256": hashlib.sha256(args.fit.read_bytes()).hexdigest(),
        "starting_checkpoint": str(args.warm_start) if args.warm_start else None,
        "training": tuned.metadata["sample_finetuning"],
        "validation": evaluate(models, fitted, validation, args.seed),
        "test": evaluate(models, fitted, test, args.seed),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--nll-weight", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--full-reference", action="store_true")
    p.add_argument("--low-noise", action="store_true")
    p.add_argument("--warm-start", type=Path)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--moment-weight", type=float, default=0.0)
    run(p.parse_args())
