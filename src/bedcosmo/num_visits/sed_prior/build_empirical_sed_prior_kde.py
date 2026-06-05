#!/usr/bin/env python3
"""
Fit and save an empirical KDE prior for DESI-fitted EAZY template SED weights.

Recommended mode: a smooth CLR prior.

The input table is produced by fit_eazy_weights_to_desi.py and contains fitted
normalized EAZY template weights a1..aK, log_c_scale, z, and quality columns.
For NNLS fits many a_k are exactly zero. A direct KDE on those sparse simplex
weights is awkward: the distribution lives on simplex faces, has boundary atoms,
and post-hoc support masking creates discontinuities.

This script therefore defaults to:

    1. take fitted a1..aK from DESI spectra fits,
    2. add a small positive floor eps to every template weight,
    3. renormalize onto the simplex,
    4. map weights to centered log-ratio coordinates
       f_i = log(a_i) - mean_j log(a_j),
    5. fit a KDE in [f1..f_K, log_c_scale, z],
    6. optionally fit a gaussianizer either on training features or on a large
       KDE reference sample (--gaussianizer-fit-source kde),
    7. sample in CLR space and decode by softmax.

This produces strictly positive simplex weights and avoids exact sparse masks.
Legacy sparse/masked behavior is still available with --support-mode masked or
--parameterization weights, but it is not recommended for normalizing-flow input.

Example:

  python build_empirical_sed_prior_kde.py \
    --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from bedcosmo.transform import Bijector, _whitening_to_apply_joint

from .fit_eazy_weights_to_desi import (
    DEFAULT_MAX_CHI2_DOF,
    apply_quality_cuts,
    build_prior_parameter_samples,
    n_template_coeff_columns,
    prior_a_column_names,
    prior_quality_mask,
    save_triangle_plot,
)
from .paths import DEFAULT_EMPIRICAL_PRIOR_DIR, get_prior_kde_path, get_prior_weights_csv
from .simplex import (
    PARAMETERIZATION_CLR,
    PARAMETERIZATION_LOGITS,
    PARAMETERIZATION_WEIGHTS,
    clr_to_weights,
    prior_clr_feature_names,
    prior_feature_names,
    prior_weights_feature_names,
    split_feature_matrix,
    weights_to_clr,
    weights_to_logits,
)

PRIOR_KDE_VERSION = 3
PRIOR_KDE_VERSION_SMOOTH_LOGIT = 3
PRIOR_KDE_VERSION_PREVIOUS = 2
PRIOR_KDE_VERSION_LEGACY = 1

DEFAULT_KDE_BANDWIDTH = 0.3
DEFAULT_KDE_DIAGNOSTIC_SAMPLES = 20_000
DEFAULT_Z_MIN = 0.01
DEFAULT_SIMPLEX_SMOOTHING_EPS = 1e-5
DEFAULT_GAUSSIANIZER_WHITENING = "cholesky"
DEFAULT_GAUSSIANIZER_FIT_SOURCE = "kde"
A_ZERO_EPS = 1e-12

SupportMode = Literal["smooth", "masked", "none"]


def fit_empirical_gaussianizer(
    x: np.ndarray,
    feature_names: list[str],
    *,
    shrinkage: float = 1e-3,
    eps: float = 1e-6,
    max_rows: int | None = 50_000,
    seed: int = 0,
) -> Bijector:
    """Fit a torch ``Bijector`` on the reference feature matrix (joint whitening by default)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match x.shape[1]")

    max_fit = max_rows if max_rows is not None else x.shape[0]
    return Bijector.fit_from_matrix(
        x,
        feature_names,
        input_transform_type="joint",
        cdf_eps=float(eps),
        shrinkage=float(shrinkage),
        max_rows=int(max_fit),
        seed=int(seed),
    )


def get_empirical_gaussianizer(artifact: dict[str, Any]) -> Bijector:
    """Reconstruct the empirical gaussianizer from artifact state."""
    state = artifact.get("gaussianizer_state")
    if state is None:
        raise KeyError(
            "Artifact does not contain gaussianizer_state; rebuild without --no-gaussianizer"
        )
    bj = Bijector.from_state(state)
    if bj.matrix_columns is None and artifact.get("feature_names"):
        bj.matrix_columns = list(artifact["feature_names"])
    return bj


def gaussianize_with(
    gaussianizer: Bijector,
    x: np.ndarray,
    whitening: str = "cholesky",
) -> np.ndarray:
    """Apply either marginal-only or joint-whitened gaussianization."""
    y = gaussianizer.matrix_to_gaussian(
        x, apply_joint=_whitening_to_apply_joint(whitening)
    )
    return y.detach().cpu().numpy()


def degaussianize_with(
    gaussianizer: Bijector,
    y: np.ndarray,
    whitening: str = "cholesky",
) -> np.ndarray:
    """Invert either marginal-only or joint-whitened gaussianization."""
    x = gaussianizer.matrix_from_gaussian(
        y, apply_joint=_whitening_to_apply_joint(whitening)
    )
    return x.detach().cpu().numpy()


def get_gaussianizer_whitening(artifact: dict[str, Any]) -> str:
    """Return the artifact default gaussianizer whitening mode.

    ``none`` means marginal normal scores only. ``cholesky`` means apply the
    additional Gaussian-copula/linear whitening rotation (production default).
    """
    mode = artifact.get("metadata", {}).get(
        "gaussianizer_whitening", DEFAULT_GAUSSIANIZER_WHITENING
    )
    if mode not in ("none", "cholesky"):
        return "none"
    return mode


def gaussianize_sed_prior_features(
    artifact: dict[str, Any],
    x: np.ndarray,
    *,
    whitening: str | None = None,
) -> np.ndarray:
    gaussianizer = get_empirical_gaussianizer(artifact)
    if whitening is None:
        whitening = get_gaussianizer_whitening(artifact)
    return gaussianize_with(gaussianizer, x, whitening=whitening)


def degaussianize_sed_prior_features(
    artifact: dict[str, Any],
    y: np.ndarray,
    *,
    whitening: str | None = None,
) -> np.ndarray:
    gaussianizer = get_empirical_gaussianizer(artifact)
    if whitening is None:
        whitening = get_gaussianizer_whitening(artifact)
    x = degaussianize_with(gaussianizer, y, whitening=whitening)
    return postprocess_kde_samples(x, artifact)


def _prior_feature_names_any(n_templates: int, *, parameterization: str) -> list[str]:
    return prior_feature_names(n_templates, parameterization=parameterization)


def _split_feature_matrix_any(
    x: np.ndarray,
    n_templates: int,
    *,
    parameterization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_feature_matrix(x, n_templates, parameterization=parameterization)


def load_prior_training_table(
    weights_csv: Path,
    *,
    max_chi2_dof: float | None = DEFAULT_MAX_CHI2_DOF,
    require_quality_pass: bool = True,
    z_min: float | None = DEFAULT_Z_MIN,
) -> pd.DataFrame:
    """Load fitted-weight CSV and apply quality/physical cuts used by the prior.

    z_min defaults to 0.01 to drop near-zero redshifts (likely stellar
    contaminants in DESI GALAXY samples). The comparison is inclusive: z >= z_min.
    """
    df = pd.read_csv(weights_csv)
    n_raw = len(df)

    if z_min is not None:
        if "z" not in df.columns:
            raise KeyError("Missing z column needed for --z-min filtering")
        z = df["z"].to_numpy(dtype=float)
        keep_z = np.isfinite(z) & (z >= float(z_min))
        n_removed = int((~keep_z).sum())
        if n_removed:
            print(
                f"Filtered {n_removed} / {n_raw} rows with non-finite z or z < {float(z_min):g}"
            )
        df = df.loc[keep_z].copy()

    if "quality_pass" not in df.columns:
        df = apply_quality_cuts(df, max_chi2_dof=max_chi2_dof)
    if require_quality_pass:
        before = len(df)
        df = df[df["quality_pass"].astype(bool)].copy()
        if before != len(df):
            print(f"Filtered {before - len(df)} / {before} rows failing quality_pass")
    elif max_chi2_dof is not None:
        before = len(df)
        df = df[prior_quality_mask(df)].copy()
        if before != len(df):
            print(f"Filtered {before - len(df)} / {before} rows failing prior quality mask")
    if df.empty:
        raise ValueError(f"No training rows in {weights_csv}")
    return df


def smooth_simplex_weights(a: np.ndarray, eps: float = DEFAULT_SIMPLEX_SMOOTHING_EPS) -> np.ndarray:
    """
    Make fitted template weights strictly positive and renormalized.

    This is the key smooth-logit step. Exact NNLS zeros become tiny positive
    weights, so log-ratios remain finite and the KDE sees a continuous target.
    """
    if eps < 0:
        raise ValueError(f"simplex smoothing eps must be >= 0, got {eps}")
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {a.shape}")
    a = np.clip(a, 0.0, None)
    if eps > 0:
        a = a + eps
    s = np.sum(a, axis=1, keepdims=True)
    bad = (~np.isfinite(s)) | (s <= 0)
    a = a / np.where(bad, 1.0, s)
    if np.any(bad):
        a[bad[:, 0]] = 1.0 / a.shape[1]
    return a


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    parameterization: str = PARAMETERIZATION_LOGITS,
    simplex_smoothing_eps: float = DEFAULT_SIMPLEX_SMOOTHING_EPS,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build KDE training features.

    Returns
    -------
    x
        Feature matrix used for KDE.
    names
        Feature names matching x.
    a_model
        The simplex weights represented by x. For logit mode these are the
        smoothed weights; for legacy weights mode these are projected weights.
    """
    n_templates = n_template_coeff_columns(df)
    a_cols = prior_a_column_names(df)
    expected = prior_weights_feature_names(n_templates)[:n_templates]
    if a_cols != expected:
        raise ValueError(f"Expected columns {expected}, got {a_cols}")

    missing = [c for c in ("log_c_scale", "z") if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in weights table: {missing}")

    a_raw = df[a_cols].to_numpy(dtype=float)
    log_s = df["log_c_scale"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    if parameterization == PARAMETERIZATION_LOGITS:
        a_model = smooth_simplex_weights(a_raw, eps=simplex_smoothing_eps)
        eta = weights_to_logits(a_model, eps=max(simplex_smoothing_eps, A_ZERO_EPS))
        x = np.column_stack([eta, log_s, z])
    elif parameterization == PARAMETERIZATION_CLR:
        a_model = smooth_simplex_weights(a_raw, eps=simplex_smoothing_eps)
        clr = weights_to_clr(a_model, eps=max(simplex_smoothing_eps, A_ZERO_EPS))
        x = np.column_stack([clr, log_s, z])
    elif parameterization == PARAMETERIZATION_WEIGHTS:
        a_model = smooth_simplex_weights(a_raw, eps=0.0)
        x = np.column_stack([a_model, log_s, z])
    else:
        raise ValueError(f"Unknown parameterization {parameterization!r}")

    names = _prior_feature_names_any(n_templates, parameterization=parameterization)
    if not np.all(np.isfinite(x)):
        bad = np.argwhere(~np.isfinite(x))[:10]
        raise ValueError(f"Non-finite values in prior feature matrix; first bad entries: {bad}")
    return x, names, a_model


def project_a_simplex(a: np.ndarray) -> np.ndarray:
    """Project rows to a_k >= 0 and sum_k a_k = 1."""
    return smooth_simplex_weights(a, eps=0.0)


def renormalize_signed_l1(a: np.ndarray) -> np.ndarray:
    """Legacy WLS normalization: project rows to sum_k |a_k| = 1."""
    a = np.asarray(a, dtype=float)
    l1 = np.sum(np.abs(a), axis=1, keepdims=True)
    return a / np.where((l1 > 0) & np.isfinite(l1), l1, 1.0)


def apply_training_support_mask(
    a: np.ndarray,
    training_a: np.ndarray,
    rng: np.random.Generator,
    *,
    eps: float = A_ZERO_EPS,
) -> np.ndarray:
    """Legacy NNLS behavior: zero a_k wherever a random training row has a_k == 0."""
    ref_idx = rng.integers(0, training_a.shape[0], size=a.shape[0])
    mask = training_a[ref_idx] > eps
    return np.asarray(a, dtype=float) * mask


def get_parameterization(artifact: dict[str, Any]) -> str:
    return artifact.get("parameterization", PARAMETERIZATION_WEIGHTS)


def get_support_mode(artifact: dict[str, Any]) -> SupportMode:
    mode = artifact.get("support_mode", artifact.get("metadata", {}).get("support_mode"))
    if mode in ("smooth", "masked", "none"):
        return mode
    # Backward compatibility with old artifacts.
    if artifact.get("metadata", {}).get("apply_support_mask_default", False):
        return "masked"
    return "smooth" if get_parameterization(artifact) == PARAMETERIZATION_LOGITS else "none"


def get_training_matrix(artifact: dict[str, Any]) -> np.ndarray:
    if "training_x" in artifact:
        return np.asarray(artifact["training_x"], dtype=float)
    weights_csv = artifact.get("metadata", {}).get("weights_csv")
    if not weights_csv:
        raise KeyError("Artifact has no training_x and no metadata['weights_csv'] to reload.")
    df = load_prior_training_table(Path(weights_csv).expanduser())
    x, _, _ = build_feature_matrix(
        df,
        parameterization=get_parameterization(artifact),
        simplex_smoothing_eps=float(
            artifact.get("metadata", {}).get(
                "simplex_smoothing_eps", DEFAULT_SIMPLEX_SMOOTHING_EPS
            )
        ),
    )
    return x


def get_training_weights(artifact: dict[str, Any]) -> np.ndarray:
    """Template weights a_k represented by the artifact training matrix."""
    x = get_training_matrix(artifact)
    n_templates = int(artifact["n_templates"])
    a, _, _ = _split_feature_matrix_any(
        x,
        n_templates,
        parameterization=get_parameterization(artifact),
    )
    return a


def _marginal_central_value_1d(values: np.ndarray) -> float:
    """Central value for one feature column from KDE prior draws.

    Uses the 1D Gaussian-KDE mode when it agrees with the sample median (same sign).
    For bimodal marginals where the global KDE mode sits on a secondary peak, falls
    back to the median so central params match triangle-plot bulk (e.g. positive f1).
    """
    col = np.asarray(values, dtype=np.float64)
    col = col[np.isfinite(col)]
    if col.size == 0:
        raise ValueError("cannot compute marginal central value from empty column")
    median = float(np.median(col))
    if col.size < 2:
        return median
    kde1d = stats.gaussian_kde(col)
    grid = np.linspace(float(col.min()), float(col.max()), 4000)
    mode = float(grid[int(np.argmax(kde1d(grid)))])
    if np.sign(mode) == np.sign(median) or median == 0.0:
        return mode
    return median


def mode_central_params_from_artifact(
    artifact: dict[str, Any],
    *,
    n_samples: int = DEFAULT_KDE_DIAGNOSTIC_SAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """
    Per-feature central values from marginals of the fitted KDE prior.

    Draws ``n_samples`` from the artifact KDE (same distribution as diagnostic
    triangle plots), then estimates each marginal mode with a median fallback for
    bimodal columns where the global 1D mode lies on a low-mass secondary peak.
    """
    draws = sample_sed_prior(artifact, n_samples, seed=seed)
    names = list(artifact["feature_names"])
    if draws.shape[1] != len(names):
        raise ValueError(
            f"KDE sample columns {draws.shape[1]} != len(feature_names) {len(names)}"
        )
    return {
        name: _marginal_central_value_1d(draws[:, i])
        for i, name in enumerate(names)
    }


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
    parameterization: str = PARAMETERIZATION_LOGITS,
    support_mode: SupportMode = "smooth",
    gaussianizer: Bijector | None = None,
) -> dict[str, Any]:
    if "n_templates" in metadata:
        n_templates = int(metadata["n_templates"])
    elif parameterization == PARAMETERIZATION_LOGITS:
        n_templates = len(feature_names) - 1
    elif parameterization == PARAMETERIZATION_CLR:
        n_templates = len(feature_names) - 2
    else:
        n_templates = len(feature_names) - 2

    return {
        "version": PRIOR_KDE_VERSION,
        "parameterization": parameterization,
        "support_mode": support_mode,
        "kde": kde,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_templates": n_templates,
        "training_x": np.asarray(training_x, dtype=float),
        "feature_bounds_min": np.asarray(training_x.min(axis=0), dtype=float),
        "feature_bounds_max": np.asarray(training_x.max(axis=0), dtype=float),
        # Retained for old downstream code that checks this key.
        "enforce_nonnegative_a": parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR) or support_mode in ("smooth", "masked"),
        "metadata": metadata,
        "gaussianizer_state": None if gaussianizer is None else gaussianizer.get_state(),
    }


def postprocess_kde_samples(
    x: np.ndarray,
    artifact: dict[str, Any],
    *,
    renormalize_a: bool = True,
    apply_support_mask: bool | None = None,
    seed: int | None = None,
    clip_to_training_bounds: bool = True,
) -> np.ndarray:
    """
    Convert raw KDE draws into valid prior feature rows.

    In recommended smooth-logit mode, this does not mask or project the KDE
    samples. It only clips features to training bounds, because logits already
    decode to valid positive simplex weights via softmax.
    """
    n_templates = int(artifact["n_templates"])
    parameterization = get_parameterization(artifact)
    support_mode = get_support_mode(artifact)
    out = np.asarray(x, dtype=float)

    if clip_to_training_bounds:
        bounds_min = artifact.get("feature_bounds_min")
        bounds_max = artifact.get("feature_bounds_max")
        if bounds_min is not None and bounds_max is not None:
            out = np.clip(out, bounds_min, bounds_max)

    if not renormalize_a:
        return np.asarray(out, dtype=float)

    # Recommended path: KDE is already in smooth simplex coordinates.
    if parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR) and support_mode == "smooth":
        if parameterization == PARAMETERIZATION_CLR:
            # Numerical guard: keep CLR rows centered after KDE/clipping.
            out = np.asarray(out, dtype=float).copy()
            out[:, :n_templates] -= out[:, :n_templates].mean(axis=1, keepdims=True)
        return np.asarray(out, dtype=float)

    a, log_s, z = _split_feature_matrix_any(out, n_templates, parameterization=parameterization)

    # Backward compatibility: callers can force masking on/off.
    if apply_support_mask is None:
        apply_support_mask = support_mode == "masked"
    if apply_support_mask:
        rng = np.random.default_rng(seed)
        training_a = get_training_weights(artifact)
        a = apply_training_support_mask(a, training_a, rng)

    if support_mode in ("smooth", "masked") or parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR):
        a = project_a_simplex(a)
    else:
        a = renormalize_signed_l1(a)

    if parameterization == PARAMETERIZATION_LOGITS:
        eps = float(artifact.get("metadata", {}).get("simplex_smoothing_eps", DEFAULT_SIMPLEX_SMOOTHING_EPS))
        a = smooth_simplex_weights(a, eps=eps if support_mode == "smooth" else 0.0)
        eta = weights_to_logits(a, eps=max(eps, A_ZERO_EPS))
        return np.column_stack([eta, log_s, z])

    if parameterization == PARAMETERIZATION_CLR:
        eps = float(artifact.get("metadata", {}).get("simplex_smoothing_eps", DEFAULT_SIMPLEX_SMOOTHING_EPS))
        a = smooth_simplex_weights(a, eps=eps if support_mode == "smooth" else 0.0)
        clr = weights_to_clr(a, eps=max(eps, A_ZERO_EPS))
        return np.column_stack([clr, log_s, z])

    return np.column_stack([a, log_s, z])


def save_sed_prior_kde(path: Path, artifact: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_sed_prior_kde(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    version = artifact.get("version")
    if version not in (
        PRIOR_KDE_VERSION,
        PRIOR_KDE_VERSION_PREVIOUS,
        PRIOR_KDE_VERSION_LEGACY,
    ):
        raise ValueError(
            f"Unsupported sed prior KDE version {version!r}; "
            f"expected one of {PRIOR_KDE_VERSION}, {PRIOR_KDE_VERSION_PREVIOUS}, "
            f"or {PRIOR_KDE_VERSION_LEGACY}"
        )
    if version == PRIOR_KDE_VERSION_LEGACY and "parameterization" not in artifact:
        artifact = dict(artifact)
        artifact["parameterization"] = PARAMETERIZATION_WEIGHTS
    if "support_mode" not in artifact:
        artifact = dict(artifact)
        artifact["support_mode"] = get_support_mode(artifact)
    return artifact


def sample_sed_prior_kde(
    artifact: dict[str, Any],
    n_samples: int,
    *,
    seed: int | None = None,
    renormalize_a: bool = True,
    apply_support_mask: bool | None = None,
    clip_to_training_bounds: bool = True,
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
        clip_to_training_bounds=clip_to_training_bounds,
    )


def sample_sed_prior(
    artifact: dict[str, Any],
    n_samples: int,
    seed: int | None = None,
    *,
    apply_support_mask: bool | None = None,
    renormalize_a: bool = True,
    clip_to_training_bounds: bool = True,
) -> np.ndarray:
    """Alias used by prior_sampler.py and downstream code."""
    return sample_sed_prior_kde(
        artifact,
        n_samples,
        seed=seed,
        renormalize_a=renormalize_a,
        apply_support_mask=apply_support_mask,
        clip_to_training_bounds=clip_to_training_bounds,
    )




def sample_sed_prior_gaussianized(
    artifact: dict[str, Any],
    n_samples: int,
    seed: int | None = None,
    *,
    whitening: str | None = None,
    apply_support_mask: bool | None = None,
    renormalize_a: bool = True,
    clip_to_training_bounds: bool = True,
) -> np.ndarray:
    """Draw from the KDE prior and return gaussianized feature rows.

    By default this uses the artifact's configured gaussianizer whitening mode.
    The recommended default is whitening="none", i.e. marginal normal scores.
    Pass whitening="cholesky" to additionally use Gaussian-copula linear
    whitening. Use degaussianize_sed_prior_features() to map them back to
    CLR/log_s/z feature space, then samples_to_coeffs() for weights.
    """
    x = sample_sed_prior(
        artifact,
        n_samples,
        seed=seed,
        apply_support_mask=apply_support_mask,
        renormalize_a=renormalize_a,
        clip_to_training_bounds=clip_to_training_bounds,
    )
    return gaussianize_sed_prior_features(artifact, x, whitening=whitening)

def samples_to_coeffs(
    x: np.ndarray,
    n_templates: int,
    *,
    parameterization: str = PARAMETERIZATION_LOGITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _split_feature_matrix_any(x, n_templates, parameterization=parameterization)


def coeffs_from_sample_row(a: np.ndarray, log_s: float) -> np.ndarray:
    return np.exp(log_s) * np.asarray(a, dtype=float)


def _format_weight_summary(a: np.ndarray, label: str) -> str:
    n = a.shape[0]
    exact_zero = (a <= A_ZERO_EPS).mean(axis=0)
    tiny = (a <= DEFAULT_SIMPLEX_SMOOTHING_EPS).mean(axis=0)
    entropy = -np.sum(np.clip(a, A_ZERO_EPS, None) * np.log(np.clip(a, A_ZERO_EPS, None)), axis=1)
    return (
        f"  {label}: n={n}, "
        f"sum range=[{a.sum(axis=1).min():.6f}, {a.sum(axis=1).max():.6f}], "
        f"mean entropy={entropy.mean():.3f}, "
        f"mean exact-zero frac={exact_zero.mean():.3f}, "
        f"mean <=1e-4 frac={tiny.mean():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit KDE prior on DESI EAZY fit table.")
    parser.add_argument(
        "--build-name",
        default=DEFAULT_EMPIRICAL_PRIOR_DIR,
        help="Prior build directory name under num_visits (used when --weights-csv/--out omitted).",
    )
    parser.add_argument(
        "--weights-csv",
        type=Path,
        default=None,
        help="Combined fit weights CSV (default: .../num_visits/<build-name>/desi_eazy_empirical_weights.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .joblib path (default: <weights-dir>/sed_prior_kde.joblib).",
    )
    parser.add_argument("--max-chi2-dof", type=float, default=DEFAULT_MAX_CHI2_DOF)
    parser.add_argument("--no-quality-cuts", action="store_true")
    parser.add_argument(
        "--z-min",
        type=float,
        default=DEFAULT_Z_MIN,
        help=(
            "Minimum redshift to use when training the empirical prior. "
            "Rows with non-finite z or z < z_min are filtered before KDE fitting. "
            f"Default {DEFAULT_Z_MIN:g} drops near-zero redshifts."
        ),
    )
    parser.add_argument(
        "--no-z-filter",
        action="store_true",
        help="Disable redshift filtering in the KDE training table.",
    )
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_KDE_BANDWIDTH)
    parser.add_argument(
        "--bandwidth-rule",
        choices=("scott", "silverman"),
        default=None,
        help="Use sklearn's named bandwidth rule instead of --bandwidth.",
    )
    parser.add_argument(
        "--kernel",
        default="gaussian",
        choices=("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"),
    )
    parser.add_argument(
        "--parameterization",
        choices=(PARAMETERIZATION_CLR, PARAMETERIZATION_LOGITS, PARAMETERIZATION_WEIGHTS),
        default=PARAMETERIZATION_CLR,
        help=(
            "clr: recommended centered log-ratio features f1..fK; "
            "logits: K-1 log-ratios against the last template; "
            "weights: legacy a1..aK."
        ),
    )
    parser.add_argument(
        "--simplex-smoothing-eps",
        type=float,
        default=DEFAULT_SIMPLEX_SMOOTHING_EPS,
        help=(
            "Positive floor added to every a_k before logit training. "
            "Use 1e-5 (default) or 1e-4 for smooth NF-friendly priors."
        ),
    )
    parser.add_argument(
        "--support-mode",
        choices=("smooth", "masked", "none"),
        default="smooth",
        help=(
            "smooth: recommended, no exact-zero support mask; "
            "masked: legacy NNLS random support mask after sampling; "
            "none: do not enforce sparse support."
        ),
    )
    parser.add_argument(
        "--fit-method",
        choices=("wls", "nnls"),
        default="nnls",
        help="Stored as metadata only; nnls inputs are still smoothed in recommended mode.",
    )
    parser.add_argument("--no-renormalize-a", action="store_true")
    parser.add_argument(
        "--no-support-mask",
        action="store_true",
        help="Backward-compatible alias for --support-mode smooth in logit mode / none in weights mode.",
    )
    parser.add_argument(
        "--no-clip-to-training-bounds",
        action="store_true",
        help="Do not clip raw KDE samples to per-feature training min/max before decoding.",
    )
    parser.add_argument(
        "--no-gaussianizer",
        action="store_true",
        help="Do not fit/store the empirical CDF + Gaussian-copula gaussianizer.",
    )
    parser.add_argument(
        "--gaussianizer-shrinkage",
        type=float,
        default=1e-3,
        help="Diagonal shrinkage used to make the Gaussian-copula correlation matrix positive definite.",
    )
    parser.add_argument(
        "--gaussianizer-eps",
        type=float,
        default=1e-3,
        help=(
            "CDF clipping epsilon for empirical gaussianization. Larger values "
            "compress extreme rank tails; default 1e-3 is usually safer for NF training."
        ),
    )
    parser.add_argument(
        "--gaussianizer-max-rows",
        type=int,
        default=50000,
        help=(
            "Maximum number of rows used to fit gaussianizer quantile grids/correlations; "
            "<=0 uses all rows from the selected gaussianizer fit source."
        ),
    )
    parser.add_argument(
        "--gaussianizer-fit-source",
        choices=("training", "kde"),
        default=DEFAULT_GAUSSIANIZER_FIT_SOURCE,
        help=(
            "Data used to fit the empirical gaussianizer. "
            "training uses the filtered input table features directly. "
            "kde (default) first fits the KDE, draws --gaussianizer-fit-samples "
            "reference samples from it, and fits the gaussianizer on those samples."
        ),
    )
    parser.add_argument(
        "--gaussianizer-fit-samples",
        type=int,
        default=100000,
        help=(
            "Number of KDE reference samples used when --gaussianizer-fit-source=kde. "
            "This is distinct from --sample, which only controls diagnostic draws."
        ),
    )
    parser.add_argument(
        "--gaussianizer-fit-seed",
        type=int,
        default=None,
        help="Random seed for KDE reference samples used to fit the gaussianizer; default uses --seed + 100003.",
    )
    parser.add_argument(
        "--gaussianizer-whitening",
        choices=("none", "cholesky"),
        default=DEFAULT_GAUSSIANIZER_WHITENING,
        help=(
            "Which gaussianized coordinate to use by default. "
            "cholesky (default) = marginal normal scores plus Gaussian-copula "
            "linear whitening; none = marginal normal scores only."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_KDE_DIAGNOSTIC_SAMPLES,
        help=(
            "Diagnostic KDE draws after save (console stats and triangle plots). "
            f"Default {DEFAULT_KDE_DIAGNOSTIC_SAMPLES}. Use 0 to skip."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    weights_csv = (
        Path(args.weights_csv).expanduser()
        if args.weights_csv is not None
        else get_prior_weights_csv(args.build_name)
    )
    if not weights_csv.exists():
        raise FileNotFoundError(weights_csv)

    if args.simplex_smoothing_eps < 0:
        raise ValueError("--simplex-smoothing-eps must be >= 0")
    if args.parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR) and args.simplex_smoothing_eps <= 0:
        raise ValueError(
            "--simplex-smoothing-eps must be > 0 for smooth simplex-coordinate priors. "
            "Use a small value like 1e-5 or 1e-4."
        )

    support_mode: SupportMode = args.support_mode
    if args.no_support_mask:
        support_mode = "smooth" if args.parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR) else "none"
    if args.parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR) and support_mode == "none":
        # Smooth simplex coordinates always imply valid simplex weights through softmax.
        support_mode = "smooth"

    out_path = (
        Path(args.out).expanduser()
        if args.out is not None
        else get_prior_kde_path(args.build_name)
    )

    max_chi2 = None if args.no_quality_cuts else args.max_chi2_dof
    z_min = None if args.no_z_filter else args.z_min
    df = load_prior_training_table(
        weights_csv,
        max_chi2_dof=max_chi2,
        z_min=z_min,
    )
    x, feature_names, train_a_model = build_feature_matrix(
        df,
        parameterization=args.parameterization,
        simplex_smoothing_eps=args.simplex_smoothing_eps,
    )
    n_templates = n_template_coeff_columns(df)

    bandwidth: float | str = (
        args.bandwidth_rule if args.bandwidth_rule is not None else float(args.bandwidth)
    )
    if isinstance(bandwidth, float) and bandwidth <= 0:
        raise ValueError(f"--bandwidth must be > 0, got {bandwidth}")

    print(f"Training KDE on {x.shape[0]} galaxies, {x.shape[1]} features")
    print(f"  parameterization: {args.parameterization}")
    print(f"  support_mode:     {support_mode}")
    print(f"  smoothing eps:    {args.simplex_smoothing_eps:g}")
    print(_format_weight_summary(train_a_model, "training a_model"))

    kde, scaler = fit_sed_prior_kde(x, bandwidth=bandwidth, kernel=args.kernel)

    gaussianizer = None
    gaussianizer_fit_x = None
    gaussianizer_fit_seed = (
        int(args.gaussianizer_fit_seed)
        if args.gaussianizer_fit_seed is not None
        else int(args.seed) + 100003
    )
    if not args.no_gaussianizer:
        max_rows = None if args.gaussianizer_max_rows <= 0 else int(args.gaussianizer_max_rows)

        if args.gaussianizer_fit_source == "training":
            gaussianizer_fit_x = x
        elif args.gaussianizer_fit_source == "kde":
            if args.gaussianizer_fit_samples <= 0:
                raise ValueError("--gaussianizer-fit-samples must be > 0 when --gaussianizer-fit-source=kde")
            # Build a temporary artifact without the gaussianizer so that the
            # reference distribution is exactly the same postprocessed KDE
            # distribution used by sample_sed_prior(...). This makes the
            # gaussianizer's empirical CDF support match the samples it will
            # later be asked to transform, reducing hard CDF-clipping edges.
            temp_metadata = {
                "n_templates": int(n_templates),
                "parameterization": args.parameterization,
                "support_mode": support_mode,
                "simplex_smoothing_eps": float(args.simplex_smoothing_eps),
                "clip_to_training_bounds_default": not args.no_clip_to_training_bounds,
                "apply_support_mask_default": support_mode == "masked",
            }
            temp_artifact = pack_kde_artifact(
                kde,
                scaler,
                feature_names,
                x,
                metadata=temp_metadata,
                parameterization=args.parameterization,
                support_mode=support_mode,
                gaussianizer=None,
            )
            gaussianizer_fit_x = sample_sed_prior(
                temp_artifact,
                int(args.gaussianizer_fit_samples),
                seed=gaussianizer_fit_seed,
                renormalize_a=not args.no_renormalize_a,
                apply_support_mask=True if support_mode == "masked" else False,
                clip_to_training_bounds=not args.no_clip_to_training_bounds,
            )
        else:
            raise ValueError(f"Unknown gaussianizer fit source {args.gaussianizer_fit_source!r}")

        gaussianizer = fit_empirical_gaussianizer(
            gaussianizer_fit_x,
            feature_names,
            shrinkage=float(args.gaussianizer_shrinkage),
            eps=float(args.gaussianizer_eps),
            max_rows=max_rows,
            seed=args.seed,
        )
        y_train = gaussianize_with(gaussianizer, x, args.gaussianizer_whitening)
        y_train_marginal = gaussianize_with(gaussianizer, x, "none")
        y_train_whitened = gaussianize_with(gaussianizer, x, "cholesky")
        print("  gaussianizer:    enabled")
        print(f"  gaussianizer fit source: {args.gaussianizer_fit_source}")
        if args.gaussianizer_fit_source == "kde":
            print(
                f"  gaussianizer KDE reference: n={gaussianizer_fit_x.shape[0]}, "
                f"seed={gaussianizer_fit_seed}"
            )
        print(f"  gaussianizer whitening default: {args.gaussianizer_whitening}")
        print(
            "  marginal gaussianized training: "
            f"mean |mu|={np.abs(y_train_marginal.mean(axis=0)).mean():.3f}, "
            f"mean sigma={y_train_marginal.std(axis=0).mean():.3f}, "
            f"max |corr offdiag|={np.max(np.abs(np.corrcoef(y_train_marginal, rowvar=False) - np.eye(y_train_marginal.shape[1]))):.3f}"
        )
        print(
            "  whitened gaussianized training: "
            f"mean |mu|={np.abs(y_train_whitened.mean(axis=0)).mean():.3f}, "
            f"mean sigma={y_train_whitened.std(axis=0).mean():.3f}, "
            f"max |corr offdiag|={np.max(np.abs(np.corrcoef(y_train_whitened, rowvar=False) - np.eye(y_train_whitened.shape[1]))):.3f}"
        )
    else:
        print("  gaussianizer:    disabled")

    metadata = {
        "weights_csv": str(weights_csv.resolve()),
        "n_train": int(x.shape[0]),
        "n_templates": int(n_templates),
        "parameterization": args.parameterization,
        "support_mode": support_mode,
        "simplex_smoothing_eps": float(args.simplex_smoothing_eps),
        "max_chi2_dof": max_chi2,
        "z_min": z_min,
        "z_filter_enabled": z_min is not None,
        "fit_method": args.fit_method,
        "bandwidth": kde.bandwidth,
        "bandwidth_rule": bandwidth if isinstance(bandwidth, str) else None,
        "kernel": args.kernel,
        "clip_to_training_bounds_default": not args.no_clip_to_training_bounds,
        "gaussianizer_enabled": gaussianizer is not None,
        "gaussianizer_shrinkage": None if gaussianizer is None else float(args.gaussianizer_shrinkage),
        "gaussianizer_eps": None if gaussianizer is None else float(args.gaussianizer_eps),
        "gaussianizer_whitening": None if gaussianizer is None else args.gaussianizer_whitening,
        "gaussianizer_fit_source": None if gaussianizer is None else args.gaussianizer_fit_source,
        "gaussianizer_fit_samples": (
            None
            if gaussianizer is None or args.gaussianizer_fit_source != "kde"
            else int(args.gaussianizer_fit_samples)
        ),
        "gaussianizer_fit_seed": None if gaussianizer is None else int(gaussianizer_fit_seed),
        "gaussianizer_fit_n_rows": None if gaussianizer_fit_x is None else int(gaussianizer_fit_x.shape[0]),
        # Backward-compatible metadata keys.
        "enforce_nonnegative_a": args.parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR) or support_mode in ("smooth", "masked"),
        "apply_support_mask_default": support_mode == "masked",
        "notes": (
            "Recommended artifact is smooth CLR KDE: epsilon-smoothed simplex "
            "weights are mapped to centered log-ratio coordinates and sampled "
            "without random support masks."
        ),
    }

    artifact = pack_kde_artifact(
        kde,
        scaler,
        feature_names,
        x,
        metadata=metadata,
        parameterization=args.parameterization,
        support_mode=support_mode,
        gaussianizer=gaussianizer,
    )
    save_sed_prior_kde(out_path, artifact)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved KDE prior: {out_path}")

    if args.sample > 0:
        mask_arg = True if support_mode == "masked" else False
        draws = sample_sed_prior(
            artifact,
            args.sample,
            seed=args.seed,
            renormalize_a=not args.no_renormalize_a,
            apply_support_mask=mask_arg,
            clip_to_training_bounds=not args.no_clip_to_training_bounds,
        )
        a, log_s, z = samples_to_coeffs(
            draws,
            n_templates,
            parameterization=args.parameterization,
        )
        print(f"\nTest sample n={args.sample}:")
        print(f"  z in [{z.min():.3f}, {z.max():.3f}]")
        print(f"  log s in [{log_s.min():.2f}, {log_s.max():.2f}]")
        print(_format_weight_summary(a, "sample a"))

        gaussian_draws = None
        training_gaussian_draws = None
        gaussian_draws_marginal = None
        training_gaussian_draws_marginal = None
        gaussian_draws_whitened = None
        training_gaussian_draws_whitened = None
        if gaussianizer is not None:
            gaussian_draws_marginal = gaussianize_with(gaussianizer, draws, "none")
            training_gaussian_draws_marginal = gaussianize_with(gaussianizer, x, "none")
            gaussian_draws_whitened = gaussianize_with(gaussianizer, draws, "cholesky")
            training_gaussian_draws_whitened = gaussianize_with(gaussianizer, x, "cholesky")

            gaussian_draws = (
                gaussian_draws_marginal
                if args.gaussianizer_whitening == "none"
                else gaussian_draws_whitened
            )
            training_gaussian_draws = (
                training_gaussian_draws_marginal
                if args.gaussianizer_whitening == "none"
                else training_gaussian_draws_whitened
            )

            for label, arr in (
                ("marginal gaussianized sample", gaussian_draws_marginal),
                ("whitened gaussianized sample", gaussian_draws_whitened),
            ):
                corr_g = np.corrcoef(arr, rowvar=False)
                offdiag = corr_g - np.eye(corr_g.shape[0])
                print(
                    f"  {label}: "
                    f"mean |mu|={np.abs(arr.mean(axis=0)).mean():.3f}, "
                    f"mean sigma={arr.std(axis=0).mean():.3f}, "
                    f"max |corr offdiag|={np.max(np.abs(offdiag)):.3f}"
                )

        # Plot 1: decoded physical simplex weights a_k plus log scale and z.
        # This is the plot you should use to inspect the actual SED mixture
        # coefficients that will be passed downstream. Even when the KDE is
        # trained in logit space, this plot is in decoded weight space.
        joint, labels = build_prior_parameter_samples(a, log_s, z)
        decoded_name = "kde_samples_triangle.png"
        save_triangle_plot(
            out_path.parent,
            joint,
            labels,
            filename=decoded_name,
            title=(
                rf"Decoded KDE prior weights "
                rf"($N={args.sample}$, {args.parameterization}, {support_mode})"
            ),
            panel_size=1.35,
        )
        print(f"Saved decoded-weight triangle: {out_path.parent / decoded_name}")

        # Plot 2: actual KDE feature coordinates. For the recommended CLR
        # parameterization this shows f_i = log(a_i) - mean_j log(a_j),
        # log_c_scale, z. This is the space where smoothness/Gaussianization
        # should be judged.
        if args.parameterization in (PARAMETERIZATION_LOGITS, PARAMETERIZATION_CLR):
            suffix = "clr" if args.parameterization == PARAMETERIZATION_CLR else "logit"
            feature_name = f"kde_samples_{suffix}_triangle.png"
            save_triangle_plot(
                out_path.parent,
                draws,
                feature_names,
                filename=feature_name,
                title=(
                    rf"KDE prior feature/{suffix} samples "
                    rf"($N={args.sample}$, {support_mode})"
                ),
                panel_size=1.35,
            )
            print(f"Saved {suffix}-feature triangle: {out_path.parent / feature_name}")

        if gaussianizer is not None:
            gaussian_names = [f"g_{name}" for name in feature_names]

            def _subsample_training(arr: np.ndarray) -> np.ndarray:
                n_plot = min(args.sample, arr.shape[0])
                rng = np.random.default_rng(args.seed)
                if arr.shape[0] > n_plot:
                    idx = rng.choice(arr.shape[0], size=n_plot, replace=False)
                    return arr[idx]
                return arr

            # Plot 3: gaussianized coordinates using --gaussianizer-whitening
            # (none = marginal normal scores; cholesky = copula whitening).
            if gaussian_draws is not None:
                gaussian_name = "kde_samples_gaussianized_triangle.png"
                save_triangle_plot(
                    out_path.parent,
                    gaussian_draws,
                    gaussian_names,
                    filename=gaussian_name,
                    title=(
                        rf"Gaussianized KDE prior samples "
                        rf"($N={args.sample}$, {args.parameterization}, {support_mode}, "
                        rf"whitening={args.gaussianizer_whitening})"
                    ),
                    panel_size=1.35,
                )
                print(f"Saved default gaussianized triangle: {out_path.parent / gaussian_name}")

                train_plot = _subsample_training(training_gaussian_draws)
                training_gaussian_name = "training_gaussianized_triangle.png"
                save_triangle_plot(
                    out_path.parent,
                    train_plot,
                    gaussian_names,
                    filename=training_gaussian_name,
                    title=(
                        rf"Gaussianized training prior data "
                        rf"($N={train_plot.shape[0]}$, {args.parameterization}, {support_mode}, "
                        rf"whitening={args.gaussianizer_whitening})"
                    ),
                    panel_size=1.35,
                )
                print(
                    f"Saved default gaussianized training triangle: "
                    f"{out_path.parent / training_gaussian_name}"
                )


if __name__ == "__main__":
    main()
