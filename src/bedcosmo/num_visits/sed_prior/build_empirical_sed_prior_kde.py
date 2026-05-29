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
    --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv \
    --parameterization clr \
    --simplex-smoothing-eps 1e-4 \
    --sample 10000 \
    --plot-kde-triangle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.special import ndtr, ndtri
from scipy.linalg import cholesky

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

try:
    from .simplex import (
        PARAMETERIZATION_LOGITS,
        PARAMETERIZATION_WEIGHTS,
        prior_feature_names,
        prior_weights_feature_names,
        split_feature_matrix,
        weights_to_logits,
    )
except ImportError:
    from simplex import (
        PARAMETERIZATION_LOGITS,
        PARAMETERIZATION_WEIGHTS,
        prior_feature_names,
        prior_weights_feature_names,
        split_feature_matrix,
        weights_to_logits,
    )

PRIOR_KDE_VERSION = 3
PRIOR_KDE_VERSION_SMOOTH_LOGIT = 3
PRIOR_KDE_VERSION_PREVIOUS = 2
PRIOR_KDE_VERSION_LEGACY = 1

DEFAULT_KDE_BANDWIDTH = 0.3
DEFAULT_SIMPLEX_SMOOTHING_EPS = 1e-5
A_ZERO_EPS = 1e-12

PARAMETERIZATION_CLR = "clr"

SupportMode = Literal["smooth", "masked", "none"]


class EmpiricalGaussianizer:
    """Empirical-CDF Gaussianization with optional Gaussian-copula whitening.

    The map is fit on the KDE feature space x, e.g.
        [clr_1..clr_K, log_c_scale, z]

    Forward:
        x_j -> u_j = Fhat_j(x_j) -> g_j = Phi^{-1}(u_j)
        optionally, y = L^{-1} g

    Inverse:
        y -> optionally g = L y -> u = Phi(g) -> x_j = Fhat_j^{-1}(u_j)

    Marginal-only gaussianization makes each feature close to N(0, 1).
    Cholesky whitening additionally removes linear correlation in the
    normal-score space, but it will not erase genuine multimodality and may make
    nonlinear structure look visually more complicated.

    eps controls tail clipping in CDF space. Values like 1e-3 deliberately
    compress tiny empirical tail populations instead of mapping them to very
    large Gaussian coordinates, which is often better conditioned for NF input.
    """

    def __init__(
        self,
        feature_names: list[str],
        quantile_x: list[np.ndarray],
        quantile_u: list[np.ndarray],
        corr: np.ndarray,
        L: np.ndarray,
        L_inv: np.ndarray,
        shrinkage: float,
        eps: float,
    ):
        self.feature_names = list(feature_names)
        self.quantile_x = [np.asarray(q, dtype=float) for q in quantile_x]
        self.quantile_u = [np.asarray(q, dtype=float) for q in quantile_u]
        self.corr = np.asarray(corr, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self.L_inv = np.asarray(L_inv, dtype=float)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)

    @staticmethod
    def _monotone_cdf_grid(x: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            raise ValueError("Need at least two finite values to fit empirical CDF")
        xs = np.sort(x)
        n = xs.size
        u = (np.arange(n, dtype=float) + 0.5) / n
        u = np.clip(u, eps, 1.0 - eps)

        # np.interp needs an increasing x grid. For tied x values, collapse to
        # one point and use the midpoint of the tied CDF range. This avoids flat
        # CDF artifacts and duplicate-grid interpolation surprises.
        uniq, start, counts = np.unique(xs, return_index=True, return_counts=True)
        if uniq.size == xs.size:
            return xs, u
        u_mid = np.empty_like(uniq, dtype=float)
        for i, (s, c) in enumerate(zip(start, counts)):
            u_mid[i] = 0.5 * (u[s] + u[s + c - 1])
        u_mid = np.maximum.accumulate(u_mid)
        u_mid = np.clip(u_mid, eps, 1.0 - eps)
        return uniq, u_mid

    @staticmethod
    def _make_pd_corr(corr: np.ndarray, shrinkage: float) -> np.ndarray:
        corr = np.asarray(corr, dtype=float)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        d = corr.shape[0]
        if d == 1:
            return np.ones((1, 1), dtype=float)
        lam = float(shrinkage)
        # Diagonal loading until Cholesky succeeds.
        for _ in range(8):
            try:
                cholesky(corr, lower=True)
                return corr
            except Exception:
                corr = (1.0 - lam) * corr + lam * np.eye(d)
                np.fill_diagonal(corr, 1.0)
                lam = min(0.5, lam * 2.0)
        eigval, eigvec = np.linalg.eigh(corr)
        eigval = np.clip(eigval, 1e-6, None)
        corr = (eigvec * eigval) @ eigvec.T
        sd = np.sqrt(np.diag(corr))
        corr = corr / sd[:, None] / sd[None, :]
        np.fill_diagonal(corr, 1.0)
        return corr

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        feature_names: list[str],
        *,
        shrinkage: float = 1e-3,
        eps: float = 1e-6,
        max_rows: int | None = 50_000,
        seed: int = 0,
    ) -> "EmpiricalGaussianizer":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")
        if x.shape[1] != len(feature_names):
            raise ValueError("feature_names length does not match x.shape[1]")

        fit_x = x
        if max_rows is not None and x.shape[0] > max_rows:
            rng = np.random.default_rng(seed)
            idx = rng.choice(x.shape[0], size=int(max_rows), replace=False)
            fit_x = x[idx]

        qx: list[np.ndarray] = []
        qu: list[np.ndarray] = []
        gcols = []
        for j in range(fit_x.shape[1]):
            xj, uj = cls._monotone_cdf_grid(fit_x[:, j], eps=eps)
            qx.append(xj)
            qu.append(uj)
            u_all = np.interp(fit_x[:, j], xj, uj, left=eps, right=1.0 - eps)
            gcols.append(ndtri(np.clip(u_all, eps, 1.0 - eps)))
        g = np.column_stack(gcols)
        corr = np.corrcoef(g, rowvar=False)
        if corr.ndim == 0:
            corr = np.ones((1, 1), dtype=float)
        corr = cls._make_pd_corr(corr, shrinkage=shrinkage)
        L = cholesky(corr, lower=True)
        L_inv = np.linalg.inv(L)
        return cls(feature_names, qx, qu, corr, L, L_inv, shrinkage, eps)

    def to_normal_scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x, dtype=float)
        for j, (qx, qu) in enumerate(zip(self.quantile_x, self.quantile_u)):
            u = np.interp(x[:, j], qx, qu, left=self.eps, right=1.0 - self.eps)
            out[:, j] = ndtri(np.clip(u, self.eps, 1.0 - self.eps))
        return out

    def to_gaussian(self, x: np.ndarray, *, whitening: str = "none") -> np.ndarray:
        """Map feature rows to gaussianized coordinates.

        whitening="none" returns marginal normal scores only:
            g_j = Phi^{-1}(Fhat_j(x_j))

        whitening="cholesky" additionally applies the Gaussian-copula linear
        decorrelation y = L^{-1} g. This can be useful for nearly elliptical
        copulas, but for the SED CLR prior it may expose/rotate nonlinear
        manifold structure into visually odd branches.
        """
        g = self.to_normal_scores(x)
        if whitening == "none":
            return g
        if whitening == "cholesky":
            return g @ self.L_inv.T
        raise ValueError(f"Unknown whitening mode {whitening!r}; expected 'none' or 'cholesky'")

    def from_gaussian(self, y: np.ndarray, *, whitening: str = "none") -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if whitening == "none":
            g = y
        elif whitening == "cholesky":
            g = y @ self.L.T
        else:
            raise ValueError(f"Unknown whitening mode {whitening!r}; expected 'none' or 'cholesky'")
        x = np.empty_like(g, dtype=float)
        for j, (qx, qu) in enumerate(zip(self.quantile_x, self.quantile_u)):
            u = np.clip(ndtr(g[:, j]), self.eps, 1.0 - self.eps)
            x[:, j] = np.interp(u, qu, qx, left=qx[0], right=qx[-1])
        return x


def fit_empirical_gaussianizer(
    x: np.ndarray,
    feature_names: list[str],
    *,
    shrinkage: float = 1e-3,
    eps: float = 1e-6,
    max_rows: int | None = 50_000,
    seed: int = 0,
) -> EmpiricalGaussianizer:
    return EmpiricalGaussianizer.fit(
        x,
        feature_names,
        shrinkage=shrinkage,
        eps=eps,
        max_rows=max_rows,
        seed=seed,
    )


def get_gaussianizer_whitening(artifact: dict[str, Any]) -> str:
    """Return the artifact default gaussianizer whitening mode.

    ``none`` means marginal normal scores only. ``cholesky`` means apply the
    additional Gaussian-copula/linear whitening rotation. Marginal-only is the
    recommended default for the SED prior because the CLR features have strong
    nonlinear/multimodal dependencies.
    """
    mode = artifact.get("metadata", {}).get("gaussianizer_whitening", "none")
    if mode not in ("none", "cholesky"):
        return "none"
    return mode


def gaussianize_sed_prior_features(
    artifact: dict[str, Any],
    x: np.ndarray,
    *,
    whitening: str | None = None,
) -> np.ndarray:
    gaussianizer = artifact.get("gaussianizer")
    if gaussianizer is None:
        raise KeyError("Artifact does not contain a gaussianizer; rebuild without --no-gaussianizer")
    if whitening is None:
        whitening = get_gaussianizer_whitening(artifact)
    return gaussianizer.to_gaussian(x, whitening=whitening)


def degaussianize_sed_prior_features(
    artifact: dict[str, Any],
    y: np.ndarray,
    *,
    whitening: str | None = None,
) -> np.ndarray:
    gaussianizer = artifact.get("gaussianizer")
    if gaussianizer is None:
        raise KeyError("Artifact does not contain a gaussianizer; rebuild without --no-gaussianizer")
    if whitening is None:
        whitening = get_gaussianizer_whitening(artifact)
    x = gaussianizer.from_gaussian(y, whitening=whitening)
    return postprocess_kde_samples(x, artifact)


def prior_clr_feature_names(n_templates: int) -> list[str]:
    """Feature names for CLR rows: f1..fK, log_c_scale, z."""
    if n_templates < 2:
        raise ValueError("n_templates must be >= 2 for CLR simplex parameterization")
    return [f"f{k + 1}" for k in range(n_templates)] + ["log_c_scale", "z"]


def weights_to_clr(w: np.ndarray, eps: float = DEFAULT_SIMPLEX_SMOOTHING_EPS) -> np.ndarray:
    """Map simplex weights (..., K) to centered log-ratio coordinates (..., K).

    CLR uses all templates symmetrically:
        clr_i = log(a_i) - mean_j log(a_j)

    The output has K coordinates but lies in a K-1 dimensional subspace because
    each row sums to zero. This avoids choosing a fragile reference template.
    """
    w = np.asarray(w, dtype=float)
    w = np.clip(w, eps, None)
    s = w.sum(axis=-1, keepdims=True)
    w = w / np.where(s > 0, s, 1.0)
    logw = np.log(w)
    return logw - logw.mean(axis=-1, keepdims=True)


def clr_to_weights(clr: np.ndarray) -> np.ndarray:
    """Map centered log-ratio coordinates (..., K) back to simplex weights."""
    clr = np.asarray(clr, dtype=float)
    x = clr - clr.max(axis=-1, keepdims=True)
    expx = np.exp(x)
    return expx / expx.sum(axis=-1, keepdims=True)


def _prior_feature_names_any(n_templates: int, *, parameterization: str) -> list[str]:
    if parameterization == PARAMETERIZATION_CLR:
        return prior_clr_feature_names(n_templates)
    return prior_feature_names(n_templates, parameterization=parameterization)


def _split_feature_matrix_any(
    x: np.ndarray,
    n_templates: int,
    *,
    parameterization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if parameterization == PARAMETERIZATION_CLR:
        clr = x[:, :n_templates]
        a = clr_to_weights(clr)
        log_s = x[:, n_templates]
        z = x[:, n_templates + 1]
        return a, log_s, z
    return split_feature_matrix(x, n_templates, parameterization=parameterization)


def load_prior_training_table(
    weights_csv: Path,
    *,
    max_chi2_dof: float | None = DEFAULT_MAX_CHI2_DOF,
    require_quality_pass: bool = True,
    z_min: float | None = 0.0,
) -> pd.DataFrame:
    """Load fitted-weight CSV and apply quality/physical cuts used by the prior.

    z_min defaults to 0.0 so unphysical negative-redshift rows are not used
    to train the empirical SED prior. The comparison is inclusive: z >= z_min.
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
    gaussianizer: EmpiricalGaussianizer | None = None,
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
        "gaussianizer": gaussianizer,
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
        "--weights-csv",
        type=Path,
        default=Path.home()
        / "scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv",
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
        default=0.0,
        help=(
            "Minimum redshift to use when training the empirical prior. "
            "Rows with non-finite z or z < z_min are filtered before KDE fitting. "
            "Default 0.0 removes unphysical negative-redshift rows."
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
        default="clr",
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
            "Use 1e-4 or 1e-3 for smooth NF-friendly priors."
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
        default="kde",
        help=(
            "Data used to fit the empirical gaussianizer. "
            "training uses the filtered input table features directly. "
            "kde first fits the KDE, draws --gaussianizer-fit-samples reference "
            "samples from it, and fits the gaussianizer on those samples. "
            "Use kde if gaussianized KDE samples show hard CDF-clipping edges."
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
        default="cholesky",
        help=(
            "Which gaussianized coordinate to use by default. "
            "none = marginal normal scores only, recommended for the SED prior; "
            "cholesky = additionally apply Gaussian-copula linear whitening."
        ),
    )
    parser.add_argument("--sample", type=int, default=0, help="Test draws after save.")
    parser.add_argument(
        "--plot-kde-triangle",
        action="store_true",
        help=("Save diagnostic triangle plots when --sample > 0: "
              "decoded weights in kde_samples_triangle.png and, for clr/logits, "
              "feature space in kde_samples_clr_triangle.png or kde_samples_logit_triangle.png."),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    weights_csv = Path(args.weights_csv).expanduser()
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
        else weights_csv.parent / "sed_prior_kde.joblib"
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
        y_train = gaussianizer.to_gaussian(x, whitening=args.gaussianizer_whitening)
        y_train_marginal = gaussianizer.to_gaussian(x, whitening="none")
        y_train_whitened = gaussianizer.to_gaussian(x, whitening="cholesky")
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
            gaussian_draws_marginal = gaussianizer.to_gaussian(draws, whitening="none")
            training_gaussian_draws_marginal = gaussianizer.to_gaussian(x, whitening="none")
            gaussian_draws_whitened = gaussianizer.to_gaussian(draws, whitening="cholesky")
            training_gaussian_draws_whitened = gaussianizer.to_gaussian(x, whitening="cholesky")

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

        if args.plot_kde_triangle:
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

                # Plot 3a: marginal-only gaussianized KDE samples. This is the
                # recommended NF/BED coordinate for this SED prior: each axis is
                # a standard-normal marginal, but we do not force a linear
                # whitening rotation on nonlinear/multimodal dependencies.
                if gaussian_draws_marginal is not None:
                    marginal_name = "kde_samples_marginal_gaussianized_triangle.png"
                    save_triangle_plot(
                        out_path.parent,
                        gaussian_draws_marginal,
                        gaussian_names,
                        filename=marginal_name,
                        title=(
                            rf"Marginal gaussianized KDE prior samples "
                            rf"($N={args.sample}$, {args.parameterization}, {support_mode})"
                        ),
                        panel_size=1.35,
                    )
                    print(f"Saved marginal gaussianized triangle: {out_path.parent / marginal_name}")

                    train_plot = _subsample_training(training_gaussian_draws_marginal)
                    training_marginal_name = "training_marginal_gaussianized_triangle.png"
                    save_triangle_plot(
                        out_path.parent,
                        train_plot,
                        gaussian_names,
                        filename=training_marginal_name,
                        title=(
                            rf"Marginal gaussianized training prior data "
                            rf"($N={train_plot.shape[0]}$, {args.parameterization}, {support_mode})"
                        ),
                        panel_size=1.35,
                    )
                    print(
                        f"Saved marginal gaussianized training triangle: "
                        f"{out_path.parent / training_marginal_name}"
                    )

                # Plot 3b: Cholesky-whitened gaussianized coordinates. This is
                # useful as a diagnostic, but it is not the default for the SED
                # prior because a single linear whitening transform can make
                # nonlinear CLR structure look like branches/starfish patterns.
                if gaussian_draws_whitened is not None:
                    whitened_name = "kde_samples_whitened_gaussianized_triangle.png"
                    save_triangle_plot(
                        out_path.parent,
                        gaussian_draws_whitened,
                        gaussian_names,
                        filename=whitened_name,
                        title=(
                            rf"Whitened gaussianized KDE prior samples "
                            rf"($N={args.sample}$, {args.parameterization}, {support_mode})"
                        ),
                        panel_size=1.35,
                    )
                    print(f"Saved whitened gaussianized triangle: {out_path.parent / whitened_name}")

                    train_plot = _subsample_training(training_gaussian_draws_whitened)
                    training_whitened_name = "training_whitened_gaussianized_triangle.png"
                    save_triangle_plot(
                        out_path.parent,
                        train_plot,
                        gaussian_names,
                        filename=training_whitened_name,
                        title=(
                            rf"Whitened gaussianized training prior data "
                            rf"($N={train_plot.shape[0]}$, {args.parameterization}, {support_mode})"
                        ),
                        panel_size=1.35,
                    )
                    print(
                        f"Saved whitened gaussianized training triangle: "
                        f"{out_path.parent / training_whitened_name}"
                    )

                # Backward-compatible aliases using the configured default mode.
                # With the new default --gaussianizer-whitening none, these are
                # identical to the marginal plots above.
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
