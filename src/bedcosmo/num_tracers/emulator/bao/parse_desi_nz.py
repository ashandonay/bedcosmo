#!/usr/bin/env python3
"""
Parse DESI LSS *_nz.txt files into BAO-bin redshift slices.

The DESI nz files have columns:

    zmid zlow zhigh n(z) Nbin Vol_bin

where n(z) is in h^3 Mpc^-3, Nbin is the weighted number in the bin,
and Vol_bin is in (Mpc/h)^3. The DESI data model says Nbin includes
weights, so this script defaults to using Nbin only as the redshift-shape
weight, then rescaling to a chosen design N if provided.

Typical files:
    BGS_ANY_SGC_nz.txt
    LRG_SGC_nz.txt
    ELG_LOPnotqso_SGC_nz.txt
    ELGnotqso_SGC_nz.txt
    QSO_SGC_nz.txt

Example:
    python parse_desi_nz.py /home/ashandonay/data/desi/nz_data \\
        --caps SGC \\
        --elg-target ELG_LOPnotqso \\
        --design-counts-csv /home/ashandonay/data/desi/bao_dr1/desi_data.csv \\
        --output-dir /home/ashandonay/data/desi/nz_slices
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Match these to your BAO tracer-bin definitions in util/tracers.yaml.
BAO_BINS: dict[str, tuple[float, float]] = {
    "BGS": (0.1, 0.4),
    "LRG1": (0.4, 0.6),
    "LRG2": (0.6, 0.8),
    "LRG3_ELG1": (0.8, 1.1),
    "ELG2": (1.1, 1.6),
    "QSO": (0.8, 2.1),
}


# Maps your BAO bin names to the DESI nz target files.
# ELG target can be swapped by --elg-target.
DEFAULT_TARGETS_BY_BIN: dict[str, list[str]] = {
    "BGS": ["BGS_ANY"],
    "LRG1": ["LRG"],
    "LRG2": ["LRG"],
    "LRG3_ELG1": ["LRG", "ELG_LOPnotqso"],
    "ELG2": ["ELG_LOPnotqso"],
    "QSO": ["QSO"],
}


# Maps your internal bin names to labels used in desi_data.csv.
DESI_DATA_TRACER_LABELS: dict[str, str] = {
    "BGS": "BGS",
    "LRG1": "LRG1",
    "LRG2": "LRG2",
    "LRG3_ELG1": "LRG3+ELG1",
    "ELG2": "ELG2",
    "QSO": "QSO",
}


def parse_area_from_header(path: Path) -> dict[str, float | None]:
    """
    Parse area/effective area from header lines when present.

    Example header:
        area is 8656.4828square degrees
        #effective area is 8602.811062753179square degrees
    """
    area = None
    effective_area = None

    area_re = re.compile(r"area\s+is\s+([0-9.eE+-]+)\s*square", re.IGNORECASE)
    eff_re = re.compile(r"effective\s+area\s+is\s+([0-9.eE+-]+)\s*square", re.IGNORECASE)

    with path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Stop after header-like lines.
            first = stripped.lstrip("#").strip().split(maxsplit=1)[0]
            try:
                float(first)
                break
            except ValueError:
                pass

            m_eff = eff_re.search(stripped)
            if m_eff:
                effective_area = float(m_eff.group(1))

            m_area = area_re.search(stripped)
            if m_area and "effective" not in stripped.lower():
                area = float(m_area.group(1))

    return {"area": area, "effective_area": effective_area}


def read_nz_file(path: Path) -> pd.DataFrame:
    """Read one DESI *_nz.txt file."""
    arr = np.loadtxt(path, comments="#")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < 6:
        raise ValueError(f"{path} has {arr.shape[1]} columns; expected at least 6")

    header = parse_area_from_header(path)

    df = pd.DataFrame(
        arr[:, :6],
        columns=["zmid", "zlow", "zhigh", "nbar_file", "Nbin_file", "Vol_bin_file"],
    )
    df["source_file"] = path.name
    df["file_area_deg2"] = header["area"]
    df["file_effective_area_deg2"] = header["effective_area"]

    # Internal consistency diagnostic: usually Nbin ~= nbar * Vol_bin.
    df["Nbin_from_nbar_volume"] = df["nbar_file"] * df["Vol_bin_file"]

    return df


def find_nz_files(
    nz_dir: Path,
    target: str,
    caps: Iterable[str],
) -> list[Path]:
    """Find TARGET_CAP_nz.txt files."""
    files: list[Path] = []

    for cap in caps:
        path = nz_dir / f"{target}_{cap}_nz.txt"
        if path.exists():
            files.append(path)

    return files


def combine_nz_files(files: list[Path]) -> pd.DataFrame:
    """
    Combine multiple nz files, e.g. NGC+SGC or LRG+ELG.

    Assumes compatible redshift binning. Sums weighted Nbin and Vol_bin,
    then recomputes nbar_file_combined = Nbin / Vol.
    """
    if not files:
        raise ValueError("No nz files provided")

    frames = [read_nz_file(path) for path in files]

    for df in frames:
        df["_zmid_key"] = df["zmid"].round(8)
        df["_zlow_key"] = df["zlow"].round(8)
        df["_zhigh_key"] = df["zhigh"].round(8)

    all_df = pd.concat(frames, ignore_index=True)

    grouped = (
        all_df.groupby(["_zmid_key", "_zlow_key", "_zhigh_key"], as_index=False)
        .agg(
            zmid=("zmid", "mean"),
            zlow=("zlow", "mean"),
            zhigh=("zhigh", "mean"),
            Nbin_file=("Nbin_file", "sum"),
            Vol_bin_file=("Vol_bin_file", "sum"),
            source_file=("source_file", lambda xs: ",".join(sorted(set(xs)))),
            file_area_deg2=("file_area_deg2", "sum"),
            file_effective_area_deg2=("file_effective_area_deg2", "sum"),
        )
        .sort_values("zmid")
        .reset_index(drop=True)
    )

    grouped["nbar_file"] = np.where(
        grouped["Vol_bin_file"] > 0.0,
        grouped["Nbin_file"] / grouped["Vol_bin_file"],
        0.0,
    )

    return grouped[
        [
            "zmid",
            "zlow",
            "zhigh",
            "nbar_file",
            "Nbin_file",
            "Vol_bin_file",
            "source_file",
            "file_area_deg2",
            "file_effective_area_deg2",
        ]
    ]


def load_reference_nz_for_bin(
    nz_dir: Path,
    tracer_bin: str,
    targets_by_bin: dict[str, list[str]],
    caps: Iterable[str],
    allow_missing_targets: bool = False,
) -> pd.DataFrame:
    """Load combined reference n(z) for one BAO tracer bin."""
    targets = targets_by_bin[tracer_bin]
    files: list[Path] = []

    for target in targets:
        target_files = find_nz_files(nz_dir, target=target, caps=caps)
        if not target_files:
            msg = f"No nz files found for target={target!r}, caps={list(caps)}, dir={nz_dir}"
            if allow_missing_targets:
                print(f"[warn] {msg}; skipping this target")
                continue
            raise FileNotFoundError(msg)

        files.extend(target_files)

    if not files:
        raise FileNotFoundError(f"No nz files found for tracer_bin={tracer_bin!r}")

    return combine_nz_files(files)


def slice_nz_shape_rescaled(
    nz: pd.DataFrame,
    zrange: tuple[float, float],
    N_design: float | None = None,
    use_absolute_file_nbar: bool = False,
    trim_partial_bins: bool = True,
) -> pd.DataFrame:
    """
    Restrict an n(z) table to a BAO z range.

    Default behavior:
        Use DESI Nbin_file only as the shape weight.
        If N_design is supplied, rescale slice numbers so sum(number_design)=N_design.

    If use_absolute_file_nbar=True:
        Use the file's weighted Nbin and nbar as the absolute normalization.
        In that mode, N_design is ignored.
    """
    zlo, zhi = zrange
    df = nz[(nz["zhigh"] > zlo) & (nz["zlow"] < zhi)].copy()

    if df.empty:
        raise ValueError(f"No rows overlap zrange={zrange}")

    if trim_partial_bins:
        old_width = df["zhigh"].to_numpy() - df["zlow"].to_numpy()
        new_zlow = np.maximum(df["zlow"].to_numpy(), zlo)
        new_zhigh = np.minimum(df["zhigh"].to_numpy(), zhi)
        new_width = np.maximum(new_zhigh - new_zlow, 0.0)
        overlap_fraction = np.where(old_width > 0.0, new_width / old_width, 0.0)

        df["zlow"] = new_zlow
        df["zhigh"] = new_zhigh
        df["zmid"] = 0.5 * (df["zlow"] + df["zhigh"])
    else:
        overlap_fraction = np.ones(len(df), dtype=float)

    # The file volume corresponds to the fiducial cosmology and file area.
    # For your Fisher pipeline, it is safest to use this only for diagnostics
    # and let CutskyFootprint recompute volume for the chosen area/cosmology.
    df["volume_file_trimmed"] = df["Vol_bin_file"] * overlap_fraction
    df["shape_weight"] = df["Nbin_file"] * overlap_fraction

    total_shape = float(df["shape_weight"].sum())
    total_volume_file = float(df["volume_file_trimmed"].sum())

    if total_shape <= 0.0:
        raise ValueError(f"Non-positive total n(z) shape weight in zrange={zrange}")

    df["slice_fraction"] = df["shape_weight"] / total_shape

    if use_absolute_file_nbar:
        df["number_design"] = df["shape_weight"]
        df["nbar_design_file_volume"] = np.where(
            df["volume_file_trimmed"] > 0.0,
            df["number_design"] / df["volume_file_trimmed"],
            0.0,
        )
        N_used = total_shape
        scale_to_design = 1.0
    else:
        if N_design is None:
            # No external design count supplied: keep file weighted normalization.
            # This is useful for diagnostics, but may not equal raw object counts.
            N_design = total_shape

        N_used = float(N_design)
        scale_to_design = N_used / total_shape

        df["number_design"] = N_used * df["slice_fraction"]
        df["nbar_design_file_volume"] = np.where(
            df["volume_file_trimmed"] > 0.0,
            df["number_design"] / df["volume_file_trimmed"],
            0.0,
        )

    df["nbar_shape_file_volume"] = np.where(
        df["volume_file_trimmed"] > 0.0,
        df["shape_weight"] / df["volume_file_trimmed"],
        0.0,
    )

    df.attrs["N_shape_file_weighted"] = total_shape
    df.attrs["V_file_trimmed"] = total_volume_file
    df.attrs["N_design"] = N_used
    df.attrs["scale_to_design"] = scale_to_design
    df.attrs["zrange"] = zrange

    return df[
        [
            "zmid",
            "zlow",
            "zhigh",
            "slice_fraction",
            "number_design",
            "shape_weight",
            "volume_file_trimmed",
            "nbar_design_file_volume",
            "nbar_shape_file_volume",
            "nbar_file",
            "Nbin_file",
            "Vol_bin_file",
            "source_file",
            "file_area_deg2",
            "file_effective_area_deg2",
        ]
    ].reset_index(drop=True)

def coarsen_slices(df: pd.DataFrame, dz: float | None) -> pd.DataFrame:
    """
    Combine fine n(z) rows into coarser redshift chunks.

    This preserves:
      - total slice_fraction
      - total number_design
      - total shape_weight
      - total volume_file_trimmed

    It recomputes:
      - zlow, zhigh
      - zmid as fraction-weighted mean
      - nbar columns from summed number / summed volume
    """
    if dz is None or dz <= 0:
        return df.reset_index(drop=True)

    df = df.copy()
    z_start = float(df["zlow"].min())

    # Group by coarse redshift bins starting at this BAO bin's low edge.
    df["_coarse_group"] = np.floor((df["zmid"].to_numpy() - z_start) / dz).astype(int)

    rows = []
    for _, g in df.groupby("_coarse_group", sort=True):
        frac_sum = float(g["slice_fraction"].sum())
        number_sum = float(g["number_design"].sum())
        shape_sum = float(g["shape_weight"].sum())
        volume_sum = float(g["volume_file_trimmed"].sum())

        if frac_sum <= 0 or number_sum <= 0:
            continue

        # Fraction-weighted zmid is best for the Fisher slice effective redshift.
        zmid = float(np.sum(g["zmid"].to_numpy() * g["slice_fraction"].to_numpy()) / frac_sum)

        rows.append({
            "zmid": zmid,
            "zlow": float(g["zlow"].min()),
            "zhigh": float(g["zhigh"].max()),
            "slice_fraction": frac_sum,
            "number_design": number_sum,
            "shape_weight": shape_sum,
            "volume_file_trimmed": volume_sum,
            "nbar_design_file_volume": number_sum / volume_sum if volume_sum > 0 else 0.0,
            "nbar_shape_file_volume": shape_sum / volume_sum if volume_sum > 0 else 0.0,
            "nbar_file": shape_sum / volume_sum if volume_sum > 0 else 0.0,
            "Nbin_file": float(g["Nbin_file"].sum()),
            "Vol_bin_file": float(g["Vol_bin_file"].sum()),
            "source_file": ",".join(sorted(set(",".join(g["source_file"]).split(",")))),
            "file_area_deg2": float(g["file_area_deg2"].sum()) if "file_area_deg2" in g else np.nan,
            "file_effective_area_deg2": (
                float(g["file_effective_area_deg2"].sum())
                if "file_effective_area_deg2" in g
                else np.nan
            ),
        })

    out = pd.DataFrame(rows).sort_values("zmid").reset_index(drop=True)

    # Renormalize against tiny floating point drift.
    total_frac = float(out["slice_fraction"].sum())
    total_N = float(out["number_design"].sum())
    if total_frac > 0:
        out["slice_fraction"] /= total_frac
    if total_N > 0:
        # Keep number_design exactly consistent with the renormalized fractions.
        out["number_design"] = total_N * out["slice_fraction"]

    # Recompute density after the final renormalization.
    out["nbar_design_file_volume"] = np.where(
        out["volume_file_trimmed"] > 0,
        out["number_design"] / out["volume_file_trimmed"],
        0.0,
    )

    out.attrs.update(df.attrs)
    out.attrs["coarsen_dz"] = dz
    return out

def load_design_counts_from_desi_data_csv(path: Path) -> dict[str, float]:
    """
    Load design N values from your desi_data.csv.

    Expected columns:
        tracer, passed

    It deduplicates by tracer, like your plotting script.
    """
    df = pd.read_csv(path)
    required = {"tracer", "passed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    dedup = df[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    by_label = {str(row["tracer"]): float(row["passed"]) for _, row in dedup.iterrows()}

    out: dict[str, float] = {}
    for bin_name, label in DESI_DATA_TRACER_LABELS.items():
        if label in by_label:
            out[bin_name] = by_label[label]

    return out


def parse_manual_design_counts(items: list[str] | None) -> dict[str, float]:
    """
    Parse manual design counts from CLI entries like:
        --design-count BGS=300017 --design-count ELG2=1500000
    """
    out: dict[str, float] = {}
    if not items:
        return out

    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --design-count {item!r}; expected TRACER=N")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in BAO_BINS:
            raise ValueError(f"Unknown tracer bin {key!r}; expected one of {list(BAO_BINS)}")
        out[key] = float(val)

    return out


def build_targets_by_bin(elg_target: str, bgs_target: str) -> dict[str, list[str]]:
    """Build target mapping after applying CLI choices."""
    targets = dict(DEFAULT_TARGETS_BY_BIN)
    targets["BGS"] = [bgs_target]
    targets["LRG3_ELG1"] = ["LRG", elg_target]
    targets["ELG2"] = [elg_target]
    return targets


def build_all_tracer_slices(
    nz_dir: Path,
    targets_by_bin: dict[str, list[str]],
    caps: Iterable[str],
    design_counts: dict[str, float] | None = None,
    use_absolute_file_nbar: bool = False,
    allow_missing_targets: bool = False,
    coarsen_dz: float | None = 0.02,
) -> dict[str, pd.DataFrame]:
    """Build sliced n(z) tables for all configured BAO bins."""
    design_counts = design_counts or {}
    out: dict[str, pd.DataFrame] = {}

    for tracer_bin, zrange in BAO_BINS.items():
        nz = load_reference_nz_for_bin(
            nz_dir=nz_dir,
            tracer_bin=tracer_bin,
            targets_by_bin=targets_by_bin,
            caps=caps,
            allow_missing_targets=allow_missing_targets,
        )

        N_design = design_counts.get(tracer_bin)

        sliced = slice_nz_shape_rescaled(
            nz=nz,
            zrange=zrange,
            N_design=N_design,
            use_absolute_file_nbar=use_absolute_file_nbar,
            trim_partial_bins=True,
        )

        out[tracer_bin] = coarsen_slices(sliced, dz=coarsen_dz)

    return out


def print_summary(slices: dict[str, pd.DataFrame]) -> None:
    print("\n=== DESI n(z) tracer slices ===")
    print(
        "tracer      zrange        rows  N_shape_file     N_design        scale       "
        "V_file          nbar_design_avg   sources"
    )
    print("-" * 130)

    for tracer_bin, df in slices.items():
        zlo, zhi = BAO_BINS[tracer_bin]
        N_shape = df.attrs["N_shape_file_weighted"]
        N_design = df.attrs["N_design"]
        scale = df.attrs["scale_to_design"]
        V_file = df.attrs["V_file_trimmed"]
        nbar_avg = N_design / V_file if V_file > 0 else np.nan

        sources = ",".join(sorted(set(",".join(df["source_file"]).split(","))))
        if len(sources) > 55:
            sources = sources[:52] + "..."

        print(
            f"{tracer_bin:<10} "
            f"[{zlo:.2f},{zhi:.2f}] "
            f"{len(df):>5d} "
            f"{N_shape:>15.6g} "
            f"{N_design:>15.6g} "
            f"{scale:>10.4g} "
            f"{V_file:>14.6g} "
            f"{nbar_avg:>16.6e} "
            f"{sources}"
        )

def plot_normalized_nz_curves(
    slices: dict[str, pd.DataFrame],
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Plot normalized n(z)-shape curves for each BAO bin.

    This plots slice_fraction / dz, so the area under each curve is ~1.
    That makes bins with different dz directly comparable as normalized
    redshift distributions.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)

    for tracer_bin, df in slices.items():
        dz = df["zhigh"].to_numpy() - df["zlow"].to_numpy()
        zmid = df["zmid"].to_numpy()
        y = df["slice_fraction"].to_numpy() / np.maximum(dz, 1.0e-30)

        ax.step(
            zmid,
            y,
            where="mid",
            linewidth=2.0,
            label=tracer_bin,
        )

        # Lightly mark the BAO bin edges.
        zlo, zhi = BAO_BINS[tracer_bin]
        ax.axvline(zlo, linewidth=0.6, alpha=0.15)
        ax.axvline(zhi, linewidth=0.6, alpha=0.15)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"Normalized $n(z)$ shape, $f_i / \Delta z$")
    ax.set_title(title or "Normalized DESI n(z) shapes by BAO bin")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="best", fontsize=9, ncol=2)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_nz_panels(
    slices: dict[str, pd.DataFrame],
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Same information as plot_normalized_nz_curves, but one panel per tracer.
    Useful when BGS dominates the y-scale.
    """
    tracer_bins = list(slices.keys())
    n = len(tracer_bins)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.0, 2.8 * nrows),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, tracer_bin in zip(axes, tracer_bins):
        df = slices[tracer_bin]
        dz = df["zhigh"].to_numpy() - df["zlow"].to_numpy()
        zmid = df["zmid"].to_numpy()
        y = df["slice_fraction"].to_numpy() / np.maximum(dz, 1.0e-30)

        ax.step(zmid, y, where="mid", linewidth=2.0)
        ax.set_title(tracer_bin)
        ax.set_xlabel("z")
        ax.set_ylabel(r"$f_i / \Delta z$")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

        zlo, zhi = BAO_BINS[tracer_bin]
        ax.axvline(zlo, linewidth=0.8, alpha=0.25)
        ax.axvline(zhi, linewidth=0.8, alpha=0.25)

        integral = np.sum(y * dz)
        ax.text(
            0.03,
            0.95,
            rf"$\sum f_i={df['slice_fraction'].sum():.3f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title or "Normalized DESI n(z) shapes by BAO bin", fontsize=13)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_unnormalized_nz_curves(
    slices: dict[str, pd.DataFrame],
    output_path: Path,
    title: str | None = None,
    use_design_rescaled: bool = True,
) -> None:
    """
    Plot non-bin-normalized n(z) curves.

    If use_design_rescaled=True, plots the density after rescaling each BAO bin
    to N_design:
        nbar_design_file_volume = number_design / volume_file_trimmed

    If False, plots the raw DESI file weighted density:
        nbar_shape_file_volume = shape_weight / volume_file_trimmed

    Both are in (h/Mpc)^3. The plot multiplies by 1e4 to match the style of
    the DESI figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)

    ycol = "nbar_design_file_volume" if use_design_rescaled else "nbar_shape_file_volume"
    suffix = "design-rescaled" if use_design_rescaled else "file-weighted"

    for tracer_bin, df in slices.items():
        zmid = df["zmid"].to_numpy()
        y = 1.0e4 * df[ycol].to_numpy()

        ax.step(
            zmid,
            y,
            where="mid",
            linewidth=2.0,
            label=tracer_bin,
        )

    # Draw BAO redshift-bin boundaries once.
    boundaries = sorted({edge for zr in BAO_BINS.values() for edge in zr})
    for z in boundaries:
        ax.axvline(z, color="black", linestyle=":", linewidth=0.8, alpha=0.35)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"number density $10^{-4}\,(h/\mathrm{Mpc})^3$")
    ax.set_title(title or f"DESI n(z) curves, {suffix}")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="best", fontsize=9, ncol=2)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse DESI *_nz.txt files into BAO-bin redshift slices."
    )
    parser.add_argument(
        "nz_dir",
        type=Path,
        help="Directory containing DESI *_nz.txt files.",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        default=["NGC", "SGC"],
        choices=["NGC", "SGC"],
        help="Galactic caps to include. Use '--caps SGC' for SGC-only.",
    )
    parser.add_argument(
        "--elg-target",
        default="ELG_LOPnotqso",
        choices=["ELG_LOPnotqso", "ELGnotqso", "ELG", "ELG_HIPnotqso", "ELG_HIP"],
        help="Which ELG nz target to use for ELG bins.",
    )
    parser.add_argument(
        "--bgs-target",
        default="BGS_ANY",
        choices=["BGS_ANY", "BGS_BRIGHT", "BGS_BRIGHT-20.2", "BGS_BRIGHT-21.35", "BGS_BRIGHT-21.5"],
        help="Which BGS nz target to use for the BGS bin.",
    )
    parser.add_argument(
        "--design-counts-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to desi_data.csv with columns tracer,passed. "
            "If provided, each BAO bin is normalized to its passed count."
        ),
    )
    parser.add_argument(
        "--design-count",
        action="append",
        default=None,
        help=(
            "Manual design count, e.g. --design-count ELG2=1200000. "
            "Can be supplied multiple times. Overrides CSV for that bin."
        ),
    )
    parser.add_argument(
        "--use-absolute-file-nbar",
        action="store_true",
        help=(
            "Use the file's weighted Nbin/n(z) as the absolute normalization. "
            "By default, Nbin is used only as a shape weight and rescaled "
            "to design counts if provided."
        ),
    )
    parser.add_argument(
        "--allow-missing-targets",
        action="store_true",
        help="Skip missing targets when a combined bin requests multiple targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write one CSV per BAO bin.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional path for a single combined CSV containing all BAO bins.",
    )
    parser.add_argument(
        "--coarsen-dz",
        type=float,
        default=0.02,
        help=(
            "Coarsen output n(z) slices to this redshift width before writing. "
            "Use 0 to keep native nz-file bins. Default: 0.05."
        ),
    )
    args = parser.parse_args()

    nz_dir = args.nz_dir.expanduser().resolve()
    if not nz_dir.exists():
        raise FileNotFoundError(f"Directory not found: {nz_dir}")

    targets_by_bin = build_targets_by_bin(
        elg_target=args.elg_target,
        bgs_target=args.bgs_target,
    )

    design_counts: dict[str, float] = {}

    if args.design_counts_csv is not None:
        design_counts.update(
            load_design_counts_from_desi_data_csv(args.design_counts_csv.expanduser().resolve())
        )

    # Manual entries override CSV entries.
    design_counts.update(parse_manual_design_counts(args.design_count))

    slices = build_all_tracer_slices(
        nz_dir=nz_dir,
        targets_by_bin=targets_by_bin,
        caps=args.caps,
        design_counts=design_counts,
        use_absolute_file_nbar=args.use_absolute_file_nbar,
        allow_missing_targets=args.allow_missing_targets,
        coarsen_dz=None if args.coarsen_dz <= 0 else args.coarsen_dz,
    )

    print_summary(slices)

    # Save normalized n(z) plots in the current working directory, i.e. where
    # the command is run from, not necessarily next to the script.

    cwd = Path.cwd()

    caps_tag = "".join(args.caps)
    plot_base = f"normalized_nz_{args.elg_target}_{caps_tag}"
    overlay_path = cwd / f"{plot_base}_overlay.png"
    panels_path = cwd / f"{plot_base}_panels.png"

    plot_title = (
        f"Normalized DESI n(z) shapes | ELG={args.elg_target} | "
        f"caps={'+'.join(args.caps)}"
    )

    plot_normalized_nz_curves(
        slices=slices,
        output_path=overlay_path,
        title=plot_title,
    )

    plot_normalized_nz_panels(
        slices=slices,
        output_path=panels_path,
        title=plot_title,
    )

    print(f"wrote {overlay_path}")
    print(f"wrote {panels_path}")

    unnorm_design_path = Path.cwd() / f"{plot_base}_unnormalized_design_nbar.png"
    unnorm_file_path = Path.cwd() / f"{plot_base}_unnormalized_file_nbar.png"

    plot_unnormalized_nz_curves(
        slices=slices,
        output_path=unnorm_design_path,
        title=(
            f"DESI n(z), design-rescaled | "
            f"caps={'+'.join(args.caps)}"
        ),
        use_design_rescaled=True,
    )

    plot_unnormalized_nz_curves(
        slices=slices,
        output_path=unnorm_file_path,
        title=(
            f"DESI n(z), file-weighted | ELG={args.elg_target} | "
            f"caps={'+'.join(args.caps)}"
        ),
        use_design_rescaled=False,
    )

    print(f"wrote {unnorm_design_path}")
    print(f"wrote {unnorm_file_path}")

    if args.output_dir is not None:
        outdir = args.output_dir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        for tracer_bin, df in slices.items():
            path = outdir / f"{tracer_bin}_nz_slices.csv"
            df.to_csv(path, index=False)
            print(f"wrote {path}")

    if args.combined_output is not None:
        combined_path = args.combined_output.expanduser().resolve()
        combined_path.parent.mkdir(parents=True, exist_ok=True)

        frames = []
        for tracer_bin, df in slices.items():
            tmp = df.copy()
            tmp.insert(0, "tracer_bin", tracer_bin)
            tmp.insert(1, "bao_zlow", BAO_BINS[tracer_bin][0])
            tmp.insert(2, "bao_zhigh", BAO_BINS[tracer_bin][1])
            frames.append(tmp)

        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()