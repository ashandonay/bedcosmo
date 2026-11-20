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

    OMP_NUM_THREADS=8 python -m bedcosmo.num_visits.empirical.train_sed_prior_flow \\
        --space both --n 400000 --epochs 250 --transforms 6 --bins 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bedcosmo.prior_flow import (
    SPACE_GAUSSIANIZED,
    SPACE_NATIVE,
    plot_training_convergence,
    train_prior_flow,
)

from .sed_prior import SED_PRIOR_FLOW_FILENAMES

SED_PRIOR_FLOW_TRAINING_PLOT_FILENAMES = {
    SPACE_NATIVE: "prior_flow_training_native.png",
    SPACE_GAUSSIANIZED: "prior_flow_training_gaussianized.png",
}


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
        plot_path = plot_training_convergence(
            dest, out_dir / SED_PRIOR_FLOW_TRAINING_PLOT_FILENAMES[space]
        )
        print(f"[plot]  {space} training convergence: {plot_path}", flush=True)
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
        "--device",
        default="cpu",
        help="Torch device used to fit the flow (for example cpu or cuda:0).",
    )
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
        "device": args.device,
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
