#!/usr/bin/env python3
"""
Fit and save a KDE prior on DESI EAZY template-fit coordinates.

Training targets (per galaxy, quality_pass only):
    a1..aK  (normalized coefficients a_k; c_k = exp(log s) * a_k)
    log_c_scale  (log s = log sum_k |c_k|)
    z

Sample with load_sed_prior_kde() / sample_sed_prior(), then reconstruct:
    c_k = exp(log_s) * a_k

NNLS priors are sparse (many a_k == 0). Sampling: global KDE, clip to training
bounds, zero inactive a_k using a random training galaxy's support mask, then
renormalize onto the simplex.

See README.md. Example:

  python build_empirical_sed_prior_kde.py \\
    --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

try:
    from .fit_eazy_weights_to_desi import (
        DEFAULT_MAX_CHI2_DOF,
        apply_quality_cuts,
        build_prior_parameter_samples,
        n_template_coeff_columns,
        prior_a_column_names,
        prior_quality_mask,
        save_triangle_plot,
    )
except ImportError:
    from fit_eazy_weights_to_desi import (
        DEFAULT_MAX_CHI2_DOF,
        apply_quality_cuts,
        build_prior_parameter_samples,
        n_template_coeff_columns,
        prior_a_column_names,
        prior_quality_mask,
        save_triangle_plot,
    )

PRIOR_KDE_VERSION = 1
DEFAULT_KDE_BANDWIDTH = 0.3
A_ZERO_EPS = 1e-12


def prior_feature_names(n_templates: int) -> list[str]:
    return [f"a{k + 1}" for k in range(n_templates)] + ["log_c_scale", "z"]


def load_prior_training_table(
    weights_csv: Path,
    *,
    max_chi2_dof: float | None = DEFAULT_MAX_CHI2_DOF,
    require_quality_pass: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(weights_csv)
    if "quality_pass" not in df.columns:
        df = apply_quality_cuts(df, max_chi2_dof=max_chi2_dof)
    if require_quality_pass:
        df = df[df["quality_pass"].astype(bool)].copy()
    elif max_chi2_dof is not None:
        df = df[prior_quality_mask(df)].copy()
    if df.empty:
        raise ValueError(f"No training rows in {weights_csv}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n_templates = n_template_coeff_columns(df)
    a_cols = prior_a_column_names(df)
    names = prior_feature_names(n_templates)
    if a_cols != names[:n_templates]:
        raise ValueError(f"Expected columns {names[:n_templates]}, got {a_cols}")
    extra = ["log_c_scale", "z"]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in weights table: {missing}")
    x = df[a_cols + extra].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values in prior feature matrix")
    return x, names


def renormalize_a_rows(x: np.ndarray, n_templates: int) -> np.ndarray:
    """Project a rows onto sum_k |a_k| = 1 (WLS / signed L1 shell)."""
    out = np.array(x, dtype=float, copy=True)
    a = out[:, :n_templates]
    l1 = np.sum(np.abs(a), axis=1, keepdims=True)
    scale = np.where((l1 > 0) & np.isfinite(l1), l1, 1.0)
    out[:, :n_templates] = a / scale
    return out


def project_a_simplex(x: np.ndarray, n_templates: int) -> np.ndarray:
    """NNLS: a_k >= 0 and sum_k a_k = 1."""
    out = np.array(x, dtype=float, copy=True)
    a = np.clip(out[:, :n_templates], 0.0, None)
    s = np.sum(a, axis=1)
    bad = (~np.isfinite(s)) | (s <= 0)
    a = a / np.where(bad[:, None], 1.0, s[:, None])
    if np.any(bad):
        a[bad] = 1.0 / n_templates
    out[:, :n_templates] = a
    return out


def _artifact_enforces_nonnegative_a(artifact: dict[str, Any]) -> bool:
    if artifact.get("enforce_nonnegative_a", False):
        return True
    return artifact.get("metadata", {}).get("fit_method") == "nnls"


def apply_training_support_mask(
    a: np.ndarray,
    training_a: np.ndarray,
    rng: np.random.Generator,
    *,
    eps: float = A_ZERO_EPS,
) -> np.ndarray:
    """Zero a_k wherever a random training galaxy had a_k == 0."""
    ref_idx = rng.integers(0, training_a.shape[0], size=a.shape[0])
    mask = training_a[ref_idx] > eps
    return np.array(a, dtype=float) * mask


def get_training_matrix(artifact: dict[str, Any]) -> np.ndarray:
    if "training_x" in artifact:
        return np.asarray(artifact["training_x"], dtype=float)
    weights_csv = artifact.get("metadata", {}).get("weights_csv")
    if not weights_csv:
        raise KeyError("Artifact has no training_x and no metadata['weights_csv'] to reload.")
    df = load_prior_training_table(Path(weights_csv).expanduser())
    x, _ = build_feature_matrix(df)
    return x


def fit_sed_prior_kde(
    x: np.ndarray,
    *,
    bandwidth: float | str = DEFAULT_KDE_BANDWIDTH,
    kernel: str = "gaussian",
) -> tuple[KernelDensity, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(x_scaled)
    return kde, scaler


def pack_kde_artifact(
    kde: KernelDensity,
    scaler: StandardScaler,
    feature_names: list[str],
    training_x: np.ndarray,
    *,
    metadata: dict[str, Any],
    enforce_nonnegative_a: bool = False,
) -> dict[str, Any]:
    n_templates = len(feature_names) - 2
    return {
        "version": PRIOR_KDE_VERSION,
        "kde": kde,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_templates": n_templates,
        "training_x": np.asarray(training_x, dtype=float),
        "feature_bounds_min": np.asarray(training_x.min(axis=0), dtype=float),
        "feature_bounds_max": np.asarray(training_x.max(axis=0), dtype=float),
        "enforce_nonnegative_a": bool(enforce_nonnegative_a),
        "metadata": metadata,
    }


def postprocess_kde_samples(
    x: np.ndarray,
    artifact: dict[str, Any],
    *,
    renormalize_a: bool = True,
    apply_support_mask: bool | None = None,
    seed: int | None = None,
) -> np.ndarray:
    n_templates = int(artifact["n_templates"])
    bounds_min = artifact.get("feature_bounds_min")
    bounds_max = artifact.get("feature_bounds_max")
    if bounds_min is not None and bounds_max is not None:
        x = np.clip(x, bounds_min, bounds_max)

    if not renormalize_a:
        return np.asarray(x, dtype=float)

    out = np.array(x, dtype=float, copy=True)
    if apply_support_mask is None:
        apply_support_mask = _artifact_enforces_nonnegative_a(artifact)
    if apply_support_mask:
        rng = np.random.default_rng(seed)
        training_a = get_training_matrix(artifact)[:, :n_templates]
        out[:, :n_templates] = apply_training_support_mask(
            out[:, :n_templates], training_a, rng
        )

    if _artifact_enforces_nonnegative_a(artifact):
        return project_a_simplex(out, n_templates)
    return renormalize_a_rows(out, n_templates)


def save_sed_prior_kde(path: Path, artifact: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_sed_prior_kde(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if artifact.get("version") != PRIOR_KDE_VERSION:
        raise ValueError(
            f"Unsupported sed prior KDE version {artifact.get('version')!r}, "
            f"expected {PRIOR_KDE_VERSION}"
        )
    return artifact


def sample_sed_prior_kde(
    artifact: dict[str, Any],
    n_samples: int,
    *,
    seed: int | None = None,
    renormalize_a: bool = True,
    apply_support_mask: bool | None = None,
) -> np.ndarray:
    kde = artifact["kde"]
    scaler = artifact["scaler"]
    x_scaled = kde.sample(n_samples, random_state=seed)
    x = scaler.inverse_transform(x_scaled)
    return postprocess_kde_samples(
        x,
        artifact,
        renormalize_a=renormalize_a,
        apply_support_mask=apply_support_mask,
        seed=seed,
    )


def sample_sed_prior(
    artifact: dict[str, Any],
    n_samples: int,
    seed: int | None = None,
    *,
    apply_support_mask: bool | None = None,
    renormalize_a: bool = True,
) -> np.ndarray:
    """Alias for sample_sed_prior_kde (masked KDE prior)."""
    return sample_sed_prior_kde(
        artifact,
        n_samples,
        seed=seed,
        renormalize_a=renormalize_a,
        apply_support_mask=apply_support_mask,
    )


def samples_to_coeffs(
    x: np.ndarray,
    n_templates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = x[:, :n_templates]
    log_s = x[:, n_templates]
    z = x[:, n_templates + 1]
    return a, log_s, z


def coeffs_from_sample_row(a: np.ndarray, log_s: float) -> np.ndarray:
    return np.exp(log_s) * np.asarray(a, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit KDE prior on DESI EAZY fit table.")
    parser.add_argument(
        "--weights-csv",
        type=Path,
        default=Path.home() / "scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .joblib path (default: <weights-dir>/sed_prior_kde.joblib).",
    )
    parser.add_argument("--max-chi2-dof", type=float, default=DEFAULT_MAX_CHI2_DOF)
    parser.add_argument("--no-quality-cuts", action="store_true")
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_KDE_BANDWIDTH)
    parser.add_argument(
        "--bandwidth-rule",
        choices=("scott", "silverman"),
        default=None,
    )
    parser.add_argument(
        "--kernel",
        default="gaussian",
        choices=("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"),
    )
    parser.add_argument(
        "--fit-method",
        choices=("wls", "nnls"),
        default="nnls",
        help="nnls: nonnegative simplex + support mask after sampling.",
    )
    parser.add_argument("--no-renormalize-a", action="store_true")
    parser.add_argument(
        "--no-support-mask",
        action="store_true",
        help="Skip zeroing inactive a_k after KDE (not recommended for nnls).",
    )
    parser.add_argument("--sample", type=int, default=0, help="Test draws after save.")
    parser.add_argument(
        "--plot-kde-triangle",
        action="store_true",
        help="Save kde_samples_triangle.png when --sample > 0.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    weights_csv = Path(args.weights_csv).expanduser()
    if not weights_csv.exists():
        raise FileNotFoundError(weights_csv)

    out_path = (
        Path(args.out).expanduser()
        if args.out is not None
        else weights_csv.parent / "sed_prior_kde.joblib"
    )

    max_chi2 = None if args.no_quality_cuts else args.max_chi2_dof
    df = load_prior_training_table(weights_csv, max_chi2_dof=max_chi2)
    x, feature_names = build_feature_matrix(df)
    n_templates = len(feature_names) - 2

    bandwidth: float | str = (
        args.bandwidth_rule if args.bandwidth_rule is not None else float(args.bandwidth)
    )
    if isinstance(bandwidth, float) and bandwidth <= 0:
        raise ValueError(f"--bandwidth must be > 0, got {bandwidth}")

    print(f"Training KDE on {x.shape[0]} galaxies, {x.shape[1]} features")
    kde, scaler = fit_sed_prior_kde(x, bandwidth=bandwidth, kernel=args.kernel)

    enforce_nonnegative_a = args.fit_method == "nnls"
    metadata = {
        "weights_csv": str(weights_csv.resolve()),
        "n_train": int(x.shape[0]),
        "max_chi2_dof": max_chi2,
        "fit_method": args.fit_method,
        "bandwidth": kde.bandwidth,
        "bandwidth_rule": bandwidth if isinstance(bandwidth, str) else None,
        "kernel": args.kernel,
        "enforce_nonnegative_a": enforce_nonnegative_a,
        "apply_support_mask_default": enforce_nonnegative_a and not args.no_support_mask,
    }

    artifact = pack_kde_artifact(
        kde,
        scaler,
        feature_names,
        x,
        metadata=metadata,
        enforce_nonnegative_a=enforce_nonnegative_a,
    )
    save_sed_prior_kde(out_path, artifact)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved KDE prior: {out_path}")

    if args.sample > 0:
        mask = False if args.no_support_mask else None
        draws = sample_sed_prior(
            artifact,
            args.sample,
            seed=args.seed,
            renormalize_a=not args.no_renormalize_a,
            apply_support_mask=mask,
        )
        a, log_s, z = samples_to_coeffs(draws, n_templates)
        train_a = x[:, :n_templates]
        fz = lambda arr, k: float((arr[:, k] <= A_ZERO_EPS).mean())
        print(f"\nTest sample n={args.sample}:")
        print(f"  z in [{z.min():.3f}, {z.max():.3f}]  log s in [{log_s.min():.2f}, {log_s.max():.2f}]")
        if enforce_nonnegative_a:
            print(f"  frac a4==0: train {fz(train_a, 3):.3f}  sample {fz(a, 3):.3f}")
            print(f"  frac a10==0: train {fz(train_a, 9):.3f}  sample {fz(a, 9):.3f}")

        if args.plot_kde_triangle:
            joint, labels = build_prior_parameter_samples(a, log_s, z)
            save_triangle_plot(
                out_path.parent,
                joint,
                labels,
                filename="kde_samples_triangle.png",
                title=rf"KDE prior samples ($N={args.sample}$, masked NNLS)",
                panel_size=1.35,
            )
            print(f"Saved {out_path.parent / 'kde_samples_triangle.png'}")


if __name__ == "__main__":
    main()
