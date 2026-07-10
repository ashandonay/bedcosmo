"""Train and persist a normalizing flow over the empirical SED prior.

This module does one thing: fit an unconditional normalizing flow (zuko NSF) to
samples drawn from the frozen KDE artifact and save it to disk. It trains in
either of the two spaces the pipeline uses:

* ``native`` : the native/ILR prior rows straight from ``sample_sed_prior``.
* ``gaussianized`` : those rows pushed through the production gaussianizer
  (``matrix_to_gaussian``, ``input_transform_type="joint"``) -- the ``z`` space
  that ``transform_input=True`` EIG lives in.

The flow is a *product*: a learned density you can reload and score anywhere. It
is deliberately agnostic to what you do with it. In particular this module does
**not** compute entropy; a consumer computes ``H = -E[log p]`` by drawing fresh
prior samples and calling :meth:`PriorFlow.log_prob` on the loaded file::

    pf = PriorFlow.load(path)
    x = sample_sed_prior(artifact, 100_000, seed=123)   # fresh, not the fit set
    H_nats = -pf.log_prob(x).mean()

Read-only w.r.t. ``src/bedcosmo`` state: draws prior samples and applies the
transform, modifies nothing but the output files. NSF training is CPU-heavy --
run on a compute node or cap threads. Example::

    OMP_NUM_THREADS=8 python -m bedcosmo.num_visits.empirical.prior_flow \\
        --space both --n 400000 --epochs 250 --transforms 6 --bins 16
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

PRIOR_FLOW_VERSION = "prior_flow_v1"
SPACE_NATIVE = "native"
SPACE_GAUSSIANIZED = "gaussianized"
SED_PRIOR_FLOW_FILENAMES = {
    SPACE_NATIVE: "sed_prior_flow_native.pt",
    SPACE_GAUSSIANIZED: "sed_prior_flow_gaussianized.pt",
}
PRIOR_FLOW_TRAINING_PLOT = "prior_flow_training.png"


def _build_nsf(dim: int, config: dict[str, Any]):
    """Construct the zuko NSF from a stored config (shared by train and load)."""
    import zuko

    return zuko.flows.NSF(
        features=dim,
        context=0,
        transforms=int(config["transforms"]),
        bins=int(config["bins"]),
        hidden_features=tuple(config["hidden_features"]),
        randperm=bool(config.get("randperm", True)),
    )


class PriorFlow:
    """A trained flow over the prior plus the standardization it was fit under.

    ``log_prob`` returns the density in the *raw* variable's units (nats): inputs
    are standardized with the stored train-split mean/std, scored by the flow, and
    corrected by the constant ``-sum(log std)`` change-of-variables term.
    """

    def __init__(self, flow, meta: dict[str, Any]):
        self.flow = flow
        self.meta = meta

    @property
    def space(self) -> str:
        return self.meta["space"]

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta.get("feature_names", []))

    @property
    def dim(self) -> int:
        return int(self.meta["dim"])

    def log_prob(self, x: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        """Log density (nats, raw-variable units) for rows ``x`` of shape ``(N, D)``."""
        import torch

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}) rows, got {x.shape}")
        mu = self.meta["standardize"]["mu"]
        std = self.meta["standardize"]["std"]
        log_std_sum = float(np.sum(np.log(std)))
        z = (x - mu) / std
        self.flow.eval()
        out = np.empty(x.shape[0], dtype=np.float64)
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                zt = torch.as_tensor(z[start : start + batch_size], dtype=torch.float32)
                out[start : start + batch_size] = self.flow().log_prob(zt).cpu().numpy()
        return out - log_std_sum

    def sample(self, n: int, *, seed: int | None = None, batch_size: int = 8192) -> np.ndarray:
        """Draw ``n`` rows in raw-variable units; shape ``(n, D)``.

        Exact inverse of :meth:`log_prob`'s standardization: sample the flow in
        standardized coordinates, then de-standardize ``x = z*std + mu``. Only a
        native-space flow may sample -- a gaussianized flow generates ``y``, and
        recovering physical rows from it needs the (tail-sensitive) bijector
        inverse, which this class deliberately does not perform.
        """
        import torch

        if self.space != SPACE_NATIVE:
            raise ValueError(f"sample() requires a {SPACE_NATIVE!r}-space flow, got {self.space!r}")
        if int(n) <= 0:
            raise ValueError("n must be positive")
        mu = self.meta["standardize"]["mu"]
        std = self.meta["standardize"]["std"]
        if seed is not None:
            torch.manual_seed(int(seed))
        self.flow.eval()
        out = np.empty((int(n), self.dim), dtype=np.float64)
        with torch.no_grad():
            filled = 0
            while filled < n:
                b = min(batch_size, n - filled)
                z = self.flow().sample((b,)).cpu().numpy().astype(np.float64)
                out[filled : filled + b] = z * std + mu
                filled += b
        return out

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.meta)
        payload["flow_state"] = {k: v.cpu() for k, v in self.flow.state_dict().items()}
        payload["version"] = PRIOR_FLOW_VERSION
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> PriorFlow:
        import torch

        payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
        if payload.get("version") != PRIOR_FLOW_VERSION:
            raise ValueError(
                f"prior-flow version mismatch: {payload.get('version')} != {PRIOR_FLOW_VERSION}"
            )
        flow = _build_nsf(int(payload["dim"]), payload["flow_config"]).to(dtype=torch.float32)
        flow.load_state_dict(payload["flow_state"])
        flow.eval()
        meta = {k: v for k, v in payload.items() if k != "flow_state"}
        return cls(flow, meta)


def train_prior_flow(
    samples: np.ndarray,
    *,
    space: str,
    feature_names: list[str] | None = None,
    whitening: str | None = None,
    n_train: int | None = None,
    transforms: int = 3,
    bins: int = 8,
    hidden_features: tuple[int, ...] = (128, 128),
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 4096,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
    log_path: str | Path | None = None,
) -> PriorFlow:
    """Fit a zuko NSF on ``samples`` and return a :class:`PriorFlow`.

    Inputs are standardized on a train split; the flow is scored each epoch on a
    disjoint eval split and the best-eval checkpoint is kept (early stopping so it
    does not overfit its own fit points). The returned flow scores any external
    rows via :meth:`PriorFlow.log_prob`; entropy is left to the caller.

    If ``log_path`` is given, the full per-epoch eval-NLL history and hyperparameters
    are written there (one file per call, so concurrent per-space training does not
    interleave), independent of the coarser ``verbose`` stdout cadence.
    """
    import torch
    import zuko  # noqa: F401  (ensures the dependency is present before training)

    log_fh = None
    if log_path is not None:
        log_path = Path(log_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w")

    def _emit(line: str, *, stdout: bool = True) -> None:
        if log_fh is not None:
            log_fh.write(line + "\n")
            log_fh.flush()
        if stdout and verbose:
            print(line, flush=True)

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {x.shape}")
    n, d = x.shape
    if n_train is None:
        n_train = n // 2
    if not (0 < n_train < n):
        raise ValueError("n_train must split the sample set")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    x_tr = x[perm[:n_train]]
    x_ev = x[perm[n_train:]]

    mu = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)

    dev = torch.device(device)
    torch.manual_seed(seed)
    xt = torch.as_tensor((x_tr - mu) / std, dtype=torch.float32, device=dev)
    xe = torch.as_tensor((x_ev - mu) / std, dtype=torch.float32, device=dev)

    config = {
        "transforms": transforms,
        "bins": bins,
        "hidden_features": tuple(hidden_features),
        "randperm": True,
    }
    # Pin float32: importing bedcosmo can set torch's default dtype to float64.
    flow = _build_nsf(d, config).to(dev, dtype=torch.float32)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    best_eval = math.inf
    best_state = None
    history: list[float] = []
    n_tr = xt.shape[0]
    try:
        _emit(
            f"[prior-flow] space={space} dim={d} n_train={n_train} n_eval={n - n_train} "
            f"transforms={transforms} bins={bins} hidden={tuple(hidden_features)} "
            f"epochs={epochs} lr={lr} batch_size={batch_size} seed={seed}"
        )
        for epoch in range(epochs):
            flow.train()
            idx = torch.randperm(n_tr, device=dev)
            for start in range(0, n_tr, batch_size):
                batch = xt[idx[start : start + batch_size]]
                opt.zero_grad()
                loss = -flow().log_prob(batch).mean()
                loss.backward()
                opt.step()
            flow.eval()
            with torch.no_grad():
                eval_nll = float(-flow().log_prob(xe).mean().item())
            history.append(eval_nll)
            if eval_nll < best_eval:
                best_eval = eval_nll
                best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            # Every epoch to the log file; every 50th (and last) to stdout.
            _emit(
                f"  [{space}] epoch {epoch:4d}  eval_nll={eval_nll:.4f}  best={best_eval:.4f}",
                stdout=(epoch % 50 == 0 or epoch == epochs - 1),
            )
        _emit(f"[prior-flow] done space={space}  best_eval_nll={best_eval:.4f}")
    finally:
        if log_fh is not None:
            log_fh.close()

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval().to("cpu")

    meta = {
        "space": space,
        "feature_names": list(feature_names or []),
        "whitening": whitening,
        "dim": d,
        "flow_config": config,
        "standardize": {"mu": mu, "std": std},
        "train": {
            "epochs": epochs,
            "n_train": int(n_train),
            "n_eval": int(n - n_train),
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "best_eval_nll": best_eval,
            "eval_nll_history": history,
        },
    }
    return PriorFlow(flow, meta)


def _draw_space_samples(artifact, space: str, n: int, seed: int, whitening: str | None):
    """Draw ``n`` prior rows in the requested space; returns (rows, whitening_used)."""
    from bedcosmo.transform import _whitening_to_apply_joint

    from .fit_sed_prior_kde import (
        get_empirical_gaussianizer,
        get_gaussianizer_whitening,
        sample_sed_prior,
    )

    x = sample_sed_prior(artifact, int(n), seed=int(seed))
    if space == SPACE_NATIVE:
        return x, None
    if space == SPACE_GAUSSIANIZED:
        gaussianizer = get_empirical_gaussianizer(artifact)
        if whitening is None:
            whitening = get_gaussianizer_whitening(artifact)
        y = gaussianizer.matrix_to_gaussian(x, apply_joint=_whitening_to_apply_joint(whitening))
        return y.detach().cpu().numpy(), whitening
    raise ValueError(f"unknown space {space!r}")


def plot_training_convergence(flow_paths: dict[str, Path], out_path: str | Path) -> Path:
    """Plot held-out eval NLL vs epoch for each trained flow, one subplot per space.

    A convergence sanity check saved beside the ``.pt`` files. Loads only the flow
    metadata (``meta["train"]["eval_nll_history"]``); no retraining, no mlflow run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    fig, axes = plt.subplots(
        1, len(flow_paths), figsize=(6.0 * len(flow_paths), 4.4), squeeze=False
    )
    for ax, (space, path) in zip(axes.ravel(), flow_paths.items()):
        tr = PriorFlow.load(path).meta["train"]
        hist = np.asarray(tr["eval_nll_history"], dtype=float)
        epochs = np.arange(len(hist))
        best_epoch = int(hist.argmin())
        best = float(hist[best_epoch])

        ax.plot(epochs, hist, color="C0", lw=1.3, label="held-out eval NLL")
        ax.axhline(best, color="C3", ls="--", lw=1.0, alpha=0.7)
        ax.plot([best_epoch], [best], "o", color="C3", ms=7, label=f"best (ep {best_epoch})")
        ax.annotate(
            f"best NLL={best:.3f}",
            xy=(best_epoch, best),
            xytext=(0.5, 0.85),
            textcoords="axes fraction",
            fontsize=9,
            ha="center",
        )
        # settle-in ylim: ignore the first few noisy epochs so the plateau is visible
        tail = hist[min(5, len(hist) - 1) :]
        ax.set_ylim(best - 0.05 * (tail.max() - best + 1e-9), tail.max() + 0.1)
        ax.set_title(
            f"{space} flow\nn_train={tr['n_train']}  lr={tr['lr']}  epochs={tr['epochs']}",
            fontsize=10,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("eval NLL (standardized coords, nats)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("Empirical SED prior-flow training convergence", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def train_and_save_prior_flows_for_kde(
    kde_path: str | Path,
    *,
    spaces: tuple[str, ...] = (SPACE_NATIVE, SPACE_GAUSSIANIZED),
    out_dir: str | Path | None = None,
    n_samples: int = 100_000,
    seed: int = 0,
    whitening: str | None = None,
    train_kwargs: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Train and save one flow per requested space beside the KDE artifact."""
    from .fit_sed_prior_kde import load_sed_prior_kde

    kde_path = Path(kde_path).expanduser()
    artifact = load_sed_prior_kde(kde_path)
    names = list(artifact.get("feature_names", []))
    out_dir = Path(out_dir).expanduser() if out_dir else kde_path.parent
    train_kwargs = dict(train_kwargs or {})

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for space in spaces:
        rows, whitening_used = _draw_space_samples(artifact, space, n_samples, seed, whitening)
        log_path = out_dir / f"{SED_PRIOR_FLOW_FILENAMES[space].removesuffix('.pt')}_train.log"
        print(f"[train] space={space}  rows={rows.shape}  log={log_path}", flush=True)
        pf = train_prior_flow(
            rows,
            space=space,
            feature_names=names,
            whitening=whitening_used,
            seed=seed,
            log_path=log_path,
            **train_kwargs,
        )
        pf.meta["source_kde_path"] = str(kde_path)
        pf.meta["n_samples"] = int(n_samples)
        dest = out_dir / SED_PRIOR_FLOW_FILENAMES[space]
        pf.save(dest)
        written[space] = dest
        print(
            f"[save]  {space}: {dest}  (best_eval_nll={pf.meta['train']['best_eval_nll']:.4f})",
            flush=True,
        )
    plot_path = plot_training_convergence(written, out_dir / PRIOR_FLOW_TRAINING_PLOT)
    print(f"[plot]  training convergence: {plot_path}", flush=True)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--kde-path", default=None, help="Override KDE joblib path (default: production artifact)."
    )
    ap.add_argument(
        "--space",
        choices=[SPACE_NATIVE, SPACE_GAUSSIANIZED, "both"],
        default="both",
        help="Which space(s) to train a flow in.",
    )
    ap.add_argument(
        "--out-dir", default=None, help="Output directory (default: beside the KDE artifact)."
    )
    ap.add_argument("--n", type=int, default=400_000, help="Prior rows to draw for training.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--transforms", type=int, default=6)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--whitening",
        choices=["default", "cholesky", "none"],
        default="default",
        help="Gaussianized-space whitening: default = artifact setting (Cholesky); none = marginal only.",
    )
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(int(args.threads))

    from .paths import get_prior_kde_path

    kde_path = Path(args.kde_path).expanduser() if args.kde_path else get_prior_kde_path()
    spaces = (SPACE_NATIVE, SPACE_GAUSSIANIZED) if args.space == "both" else (args.space,)
    whitening = None if args.whitening == "default" else args.whitening
    train_kwargs = {
        "epochs": args.epochs,
        "hidden_features": tuple(args.hidden),
        "transforms": args.transforms,
        "bins": args.bins,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    print(f"[load] KDE artifact: {kde_path}", flush=True)
    written = train_and_save_prior_flows_for_kde(
        kde_path,
        spaces=spaces,
        out_dir=args.out_dir,
        n_samples=int(args.n),
        seed=int(args.seed),
        whitening=whitening,
        train_kwargs=train_kwargs,
    )
    print("\n===== trained prior flows =====", flush=True)
    for space, dest in written.items():
        print(f"  {space:14s} -> {dest}")
    print(
        "\n  Entropy is computed elsewhere: load a file and score fresh prior draws,\n"
        "  e.g. H_nats = -PriorFlow.load(path).log_prob(sample_sed_prior(artifact, N)).mean()",
        flush=True,
    )


if __name__ == "__main__":
    main()
