"""Pattern-preserving normalizing flow for the empirical SED prior."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

VERSION = "joint_pattern_flow_v1"


def encode_rows(rows, lower, upper):
    """Active simplex -> sequential stick logits, followed by bounded aux logits.

    All rows must have the same number of active templates, in bank order.
    The returned Jacobian uses the first m-1 active fractions as face coordinates.
    """
    rows = np.asarray(rows, dtype=float)
    masks = rows[:, :-2] > 0
    counts = masks.sum(axis=1)
    if len(rows) == 0 or not np.all(counts == counts[0]) or counts[0] < 1:
        raise ValueError("Rows need the same positive number of active templates")
    if (
        not np.isfinite(rows).all()
        or np.any(rows[:, :-2] < 0)
        or not np.allclose(rows[:, :-2].sum(axis=1), 1)
    ):
        raise ValueError("Expected finite simplex rows")
    active = rows[:, :-2][masks].reshape(len(rows), int(counts[0]))
    tails = np.cumsum(active[:, ::-1], axis=1)[:, ::-1]
    stick = np.log(active[:, :-1]) - np.log(tails[:, 1:])
    aux = rows[:, -2:]
    left, right = aux - lower, upper - aux
    if np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("Continuous auxiliary values must be strictly inside training bounds")
    aux_logits = np.log(left) - np.log(right)
    log_jac = -np.log(active).sum(axis=1)
    log_jac += (np.log(upper - lower) - np.log(left) - np.log(right)).sum(axis=1)
    return np.column_stack([stick, aux_logits]), log_jac


def decode_rows(features, masks, lower, upper):
    """Invert the stick/aux logits without epsilon smoothing or inactive weights."""
    features, masks = np.asarray(features, dtype=float), np.asarray(masks, dtype=bool)
    m = int(masks[0].sum())
    if not np.all(masks.sum(axis=1) == m) or features.shape != (len(masks), m + 1):
        raise ValueError("Features and same-size masks do not match")
    stick = features[:, : m - 1]
    log_v, log_one_minus_v = -np.logaddexp(0, -stick), -np.logaddexp(0, stick)
    prefix = np.column_stack([np.zeros(len(stick)), np.cumsum(log_one_minus_v, axis=1)])
    log_weights = np.column_stack([prefix[:, :-1] + log_v, prefix[:, -1]])
    active = np.exp(log_weights)
    if np.any(active == 0) or not np.isfinite(active).all():
        raise ValueError("Flow tail underflowed an active fraction")
    active /= active.sum(axis=1, keepdims=True)
    rows = np.zeros((len(features), masks.shape[1] + 2))
    rows[:, :-2][masks] = active.ravel()
    rows[:, -2:] = lower + (upper - lower) * expit(features[:, -2:])
    if np.any(rows[:, -2:] <= lower) or np.any(rows[:, -2:] >= upper):
        raise ValueError("Flow tail rounded onto an auxiliary boundary")
    return rows


def build_flow(dim, k, config):
    import torch
    import zuko

    return zuko.flows.NSF(
        features=dim,
        context=k,
        transforms=config["transforms"],
        bins=config["bins"],
        hidden_features=tuple(config["hidden"]),
        randperm=True,
    ).to(dtype=torch.float32)


class JointPriorFlow:
    """Discrete pattern probabilities + conditional face flows + singleton atoms."""

    def __init__(self, metadata, groups):
        self.metadata = metadata
        self.groups = groups
        self.patterns = metadata["patterns"]
        self.probabilities = metadata["counts"] / np.sum(metadata["counts"])
        self.lookup = {tuple(p): i for i, p in enumerate(self.patterns)}

    def sample(self, n, *, seed=0):
        import torch

        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        chosen = rng.choice(len(self.patterns), n, p=self.probabilities)
        masks = self.patterns[chosen]
        output = np.empty((n, masks.shape[1] + 2))
        atomic = self.metadata["atomic"][chosen]
        output[atomic] = self.metadata["atomic_rows"][chosen[atomic]]
        for m in np.unique(masks[~atomic].sum(axis=1)):
            selected = (~atomic) & (masks.sum(axis=1) == m)
            group = self.groups[int(m)]
            group["flow"].eval()
            indices = np.flatnonzero(selected)
            with torch.no_grad():
                for start in range(0, len(indices), 2048):
                    idx = indices[start : start + 2048]
                    context = torch.as_tensor(masks[idx], dtype=torch.float32)
                    latent = group["flow"](context).sample().numpy()
                    latent = latent * group["std"] + group["mu"]
                    output[idx] = decode_rows(
                        latent, masks[idx], self.metadata["lower"], self.metadata["upper"]
                    )
        return output

    def log_prob(self, rows):
        """Pattern-weighted densities, or singleton masses, on their own measures."""
        import torch

        rows = np.asarray(rows, dtype=float)
        if (
            rows.ndim != 2
            or rows.shape[1] != self.patterns.shape[1] + 2
            or not np.isfinite(rows).all()
            or np.any(rows[:, :-2] < 0)
            or not np.allclose(rows[:, :-2].sum(axis=1), 1)
        ):
            raise ValueError("Expected finite physical simplex rows")
        masks = rows[:, :-2] > 0
        ids = np.array([self.lookup.get(tuple(mask), -1) for mask in masks])
        result = np.full(len(rows), -np.inf)
        for i in np.unique(ids[ids >= 0]):
            selected = ids == i
            if self.metadata["atomic"][i]:
                equal = np.all(rows[selected] == self.metadata["atomic_rows"][i], axis=1)
                result[np.flatnonzero(selected)[equal]] = np.log(self.probabilities[i])
        eligible = ids >= 0
        eligible[ids >= 0] &= ~self.metadata["atomic"][ids[ids >= 0]]
        eligible &= np.all(
            (rows[:, -2:] > self.metadata["lower"]) & (rows[:, -2:] < self.metadata["upper"]),
            axis=1,
        )
        for m in np.unique(masks[eligible].sum(axis=1)):
            indices = np.flatnonzero(eligible & (masks.sum(axis=1) == m))
            group = self.groups[int(m)]
            group["flow"].eval()
            with torch.no_grad():
                for start in range(0, len(indices), 2048):
                    idx = indices[start : start + 2048]
                    features, jac = encode_rows(
                        rows[idx], self.metadata["lower"], self.metadata["upper"]
                    )
                    latent = torch.as_tensor(
                        (features - group["mu"]) / group["std"], dtype=torch.float32
                    )
                    context = torch.as_tensor(masks[idx], dtype=torch.float32)
                    score = group["flow"](context).log_prob(latent).numpy()
                    result[idx] = (
                        score
                        - np.log(group["std"]).sum()
                        + jac
                        + np.log(self.probabilities[ids[idx]])
                    )
        return result

    def gaussianize(self, rows):
        """Return pattern IDs, Gaussian arrays by active count, and log Jacobians.

        Each array keeps the input order within its non-atomic active-count
        group. Atoms have no continuous coordinates and a zero log Jacobian.
        """
        import torch

        rows = np.asarray(rows, dtype=float)
        if not np.isfinite(self.log_prob(rows)).all():
            raise ValueError("Gaussianization requires rows on the modeled support")
        ids = np.array([self.lookup[tuple(row[:-2] > 0)] for row in rows], dtype=int)
        masks = self.patterns[ids]
        continuous = ~self.metadata["atomic"][ids]
        groups, jacobian = {}, np.zeros(len(rows))
        for m, group in self.groups.items():
            indices = np.flatnonzero(continuous & (masks.sum(axis=1) == m))
            if len(indices) == 0:
                continue
            values = np.empty((len(indices), m + 1))
            with torch.no_grad():
                for start in range(0, len(indices), 2048):
                    idx = indices[start : start + 2048]
                    features, jac = encode_rows(
                        rows[idx], self.metadata["lower"], self.metadata["upper"]
                    )
                    x = torch.as_tensor(
                        (features - group["mu"]) / group["std"], dtype=torch.float32
                    )
                    context = torch.as_tensor(masks[idx], dtype=torch.float32)
                    z, ladj = group["flow"](context).transform.call_and_ladj(x)
                    values[start : start + len(idx)] = z.numpy()
                    jacobian[idx] = jac - np.log(group["std"]).sum() + ladj.numpy()
            groups[m] = values
        return ids, groups, jacobian

    def degaussianize(self, pattern_ids, gaussian_groups):
        """Invert Gaussian arrays, restoring exact masks and singleton atoms."""
        import torch

        ids = np.asarray(pattern_ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= len(self.patterns))
        ):
            raise ValueError("Expected valid integer pattern IDs")
        masks = self.patterns[ids]
        atomic = self.metadata["atomic"][ids]
        counts = masks.sum(axis=1)
        required = set(counts[~atomic].astype(int))
        if set(gaussian_groups) != required:
            raise ValueError("Gaussian groups must match non-atomic active counts")
        rows = np.empty((len(ids), masks.shape[1] + 2))
        rows[atomic] = self.metadata["atomic_rows"][ids[atomic]]
        for m in required:
            indices = np.flatnonzero((~atomic) & (counts == m))
            values = np.asarray(gaussian_groups[m], dtype=float)
            if values.shape != (len(indices), m + 1) or not np.isfinite(values).all():
                raise ValueError("Gaussian group has invalid shape or non-finite coordinates")
            group = self.groups[m]
            with torch.no_grad():
                for start in range(0, len(indices), 2048):
                    idx = indices[start : start + 2048]
                    z = torch.as_tensor(values[start : start + len(idx)], dtype=torch.float32)
                    context = torch.as_tensor(masks[idx], dtype=torch.float32)
                    x = group["flow"](context).transform.inv(z).numpy()
                    rows[idx] = decode_rows(
                        x * group["std"] + group["mu"],
                        masks[idx],
                        self.metadata["lower"],
                        self.metadata["upper"],
                    )
        return rows

    def save(self, path):
        import torch

        payload = {
            "version": VERSION,
            "metadata": self.metadata,
            "groups": {
                m: {
                    "mu": g["mu"],
                    "std": g["std"],
                    "history": g["history"],
                    "flow_state": g["flow"].state_dict(),
                }
                for m, g in self.groups.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["version"] != VERSION:
            raise ValueError("Unsupported joint prior-flow artifact version")
        metadata = payload["metadata"]
        required = set(metadata["patterns"][~metadata["atomic"]].sum(axis=1).astype(int))
        if set(payload["groups"]) != required:
            raise ValueError("Incomplete joint prior-flow checkpoint; load the completed artifact")
        groups = {}
        for m, saved in payload["groups"].items():
            flow = build_flow(m + 1, metadata["patterns"].shape[1], metadata["flow_config"])
            flow.load_state_dict(saved["flow_state"])
            flow.eval()
            groups[m] = {**saved, "flow": flow}
        return cls(metadata, groups)


def train(
    target,
    *,
    n_samples=120000,
    epochs=150,
    batch_size=1024,
    hidden=64,
    transforms=3,
    bins=8,
    patience=25,
    seed=23,
    output_dir=None,
):
    import torch

    if n_samples < 100 or min(epochs, batch_size, hidden, transforms, bins, patience) < 1:
        raise ValueError("Training counts/configuration must be positive; use at least 100 samples")
    atomic = target.counts == 1
    if np.all(atomic):
        raise ValueError("No continuous patterns to train a flow")
    atomic_rows = np.zeros((len(target.patterns), target.rows.shape[1]))
    for i, members in enumerate(target.members):
        if atomic[i]:
            atomic_rows[i] = target.rows[members[0]]
            atomic_rows[i, :-2] /= atomic_rows[i, :-2].sum()
        elif np.any(target.widths[i, -2:] == 0) or (
            target.patterns[i].sum() > 1 and np.any(target.widths[i, :-2][target.patterns[i]] == 0)
        ):
            raise ValueError(
                "Non-singleton degenerate pattern requires explicit lower-dimensional modeling"
            )
    config = {"hidden": [hidden, hidden], "transforms": transforms, "bins": bins}
    metadata = {
        "patterns": target.patterns,
        "counts": target.counts,
        "atomic": atomic,
        "atomic_rows": atomic_rows,
        "lower": target.lower[-2:],
        "upper": target.upper[-2:],
        "bandwidth": target.bandwidth,
        "flow_config": config,
        "seed": seed,
        "n_samples": n_samples,
        "epochs_requested": epochs,
        "measure": "Counting over patterns; first m-1 active fractions plus log scale and redshift for continuous faces; singleton point masses otherwise.",
    }
    model = JointPriorFlow(metadata, {})
    training = target.sample(n_samples, seed=seed)
    validation = target.sample(max(20000, n_samples // 5), seed=seed + 1)
    train_ids = np.array([model.lookup[tuple(row[:-2] > 0)] for row in training])
    val_ids = np.array([model.lookup[tuple(row[:-2] > 0)] for row in validation])
    for m in np.unique(target.patterns[~atomic].sum(axis=1)):
        tr_idx = (~atomic[train_ids]) & ((training[:, :-2] > 0).sum(axis=1) == m)
        ev_idx = (~atomic[val_ids]) & ((validation[:, :-2] > 0).sum(axis=1) == m)
        if min(tr_idx.sum(), ev_idx.sum()) < 2:
            raise ValueError(f"Insufficient KDE draws for {m}-active group")
        tr, _ = encode_rows(training[tr_idx], metadata["lower"], metadata["upper"])
        ev, _ = encode_rows(validation[ev_idx], metadata["lower"], metadata["upper"])
        mu, std = tr.mean(axis=0), tr.std(axis=0)
        if np.any(std <= 0):
            raise ValueError("Continuous flow coordinate has zero training variance")
        x = torch.as_tensor((tr - mu) / std, dtype=torch.float32)
        v = torch.as_tensor((ev - mu) / std, dtype=torch.float32)
        ctx = torch.as_tensor(training[tr_idx, :-2] > 0, dtype=torch.float32)
        vctx = torch.as_tensor(validation[ev_idx, :-2] > 0, dtype=torch.float32)
        torch.manual_seed(seed + int(m))
        flow = build_flow(int(m) + 1, target.patterns.shape[1], config)
        opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
        best, best_state, history, stale = np.inf, None, [], 0
        started = time.perf_counter()
        for epoch in range(epochs):
            flow.train()
            order = torch.randperm(len(x))
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                opt.zero_grad()
                loss = -flow(ctx[idx]).log_prob(x[idx]).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite training NLL in group {m}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 10)
                opt.step()
            flow.eval()
            with torch.no_grad():
                nll = float(-flow(vctx).log_prob(v).mean())
            if not np.isfinite(nll):
                raise ValueError("Nonfinite validation NLL")
            history.append(nll)
            if nll < best:
                best, best_state, stale = nll, copy.deepcopy(flow.state_dict()), 0
            else:
                stale += 1
            if epoch % 10 == 0:
                print(
                    f"active={m} epoch={epoch} validation_nll={nll:.4f} best={best:.4f} seconds={time.perf_counter()-started:.1f}",
                    flush=True,
                )
            if stale >= patience:
                break
        flow.load_state_dict(best_state)
        flow.eval()
        model.groups[int(m)] = {"flow": flow, "mu": mu, "std": std, "history": history}
        metadata.setdefault("training_groups", {})[int(m)] = {
            "n_train": len(tr),
            "n_validation": len(ev),
            "best_nll": best,
            "epochs": len(history),
            "seconds": time.perf_counter() - started,
        }
        if output_dir is not None:
            # Partial checkpoint documents which groups are complete. The final
            # artifact is written only after every required group has trained.
            model.save(Path(output_dir) / f"completed_through_active{m}.pt")
        print(f"completed active={m}, best={best:.4f}, epochs={len(history)}", flush=True)
    return model
