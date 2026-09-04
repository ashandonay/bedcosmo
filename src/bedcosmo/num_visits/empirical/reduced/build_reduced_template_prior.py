#!/usr/bin/env python3
"""Build an empirical prior from one reduced-template cohort."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import (
    BUILD_PROVENANCE_FILENAME,
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    SED_PRIOR_KDE_NATIVE_FILENAME,
    get_prior_build_dir,
    get_template_dir,
)
from ..provenance import write_provenance

KDE_MODULE = "bedcosmo.num_visits.empirical.fit_sed_prior_kde"


def parse_template_subset(label: str) -> tuple[int, ...]:
    """Parse labels such as ``T1+T7`` into one-based template indices."""
    pieces = label.upper().replace(",", "+").split("+")
    try:
        subset = tuple(int(piece.strip().removeprefix("T")) for piece in pieces)
    except ValueError as error:
        raise ValueError(f"Invalid template subset {label!r}") from error
    if not subset or len(subset) != len(set(subset)) or any(index < 1 for index in subset):
        raise ValueError(f"Invalid template subset {label!r}")
    return tuple(sorted(subset))


def template_label(subset: tuple[int, ...]) -> str:
    return "+".join(f"T{index}" for index in subset)


def reduced_weights_table(
    memberships: pd.DataFrame,
    subset: tuple[int, ...],
    *,
    n_full_templates: int,
    max_chi2_dof: float,
) -> pd.DataFrame:
    """Convert cohort memberships into the fit-table schema consumed by the KDE."""
    label = template_label(subset)
    selected = memberships.loc[memberships["templates"] == label].copy()
    if selected.empty:
        raise ValueError(f"No cohort memberships found for {label}")
    if selected["targetid"].duplicated().any():
        raise ValueError(f"Cohort {label} contains duplicate TARGETIDs")

    n_reduced = len(subset)
    required = [
        "targetid",
        "healpix",
        "z",
        "dof",
        "chi2_dof",
        "reduced_chi2_dof",
        "reduced_log_c_scale",
        "delta_chi2_dof",
        "lsst_color_rms",
        *(f"template_{position + 1}" for position in range(n_reduced)),
        *(f"c_{position + 1}" for position in range(n_reduced)),
        *(f"a_{position + 1}" for position in range(n_reduced)),
    ]
    missing = [column for column in required if column not in selected]
    if missing:
        raise KeyError(f"Membership table is missing columns: {missing}")

    recorded = tuple(int(selected[f"template_{i + 1}"].iloc[0]) for i in range(n_reduced))
    if recorded != subset:
        raise ValueError(f"Membership columns record {recorded}, expected {subset}")
    for position, expected in enumerate(subset, start=1):
        if not np.all(selected[f"template_{position}"].to_numpy(int) == expected):
            raise ValueError(f"template_{position} is inconsistent within cohort {label}")

    reduced_dof = selected["dof"].to_numpy(float) + n_full_templates - n_reduced
    reduced_chi2_dof = selected["reduced_chi2_dof"].to_numpy(float)
    output = selected[["targetid", "healpix", "z"]].copy()
    output["success"] = True
    output["dof"] = reduced_dof.astype(int)
    output["chi2_dof"] = reduced_chi2_dof
    output["chi2"] = reduced_chi2_dof * reduced_dof
    output["log_c_scale"] = selected["reduced_log_c_scale"].to_numpy(float)
    for position in range(1, n_reduced + 1):
        output[f"c{position}"] = selected[f"c_{position}"].to_numpy(float)
        output[f"a{position}"] = selected[f"a_{position}"].to_numpy(float)
    finite = np.all(
        np.isfinite(
            output[
                [
                    "z",
                    "chi2_dof",
                    "log_c_scale",
                    *(f"c{i}" for i in range(1, n_reduced + 1)),
                    *(f"a{i}" for i in range(1, n_reduced + 1)),
                ]
            ].to_numpy(float)
        ),
        axis=1,
    )
    output["quality_pass"] = finite & (output["chi2_dof"] <= max_chi2_dof)
    output["source_full_chi2_dof"] = selected["chi2_dof"].to_numpy(float)
    output["delta_chi2_dof"] = selected["delta_chi2_dof"].to_numpy(float)
    output["lsst_color_rms"] = selected["lsst_color_rms"].to_numpy(float)
    output["source_subset"] = label
    return output


def write_reduced_template_param(
    path: Path,
    template_paths: list[str],
    subset: tuple[int, ...],
) -> Path:
    """Write an EAZY parameter file containing only the selected templates."""
    if any(index > len(template_paths) for index in subset):
        raise ValueError(f"Subset {subset} exceeds the {len(template_paths)}-template source bank")
    lines = [
        "# Reduced template bank generated from cohort discovery.",
        *(
            f"{position} {template_paths[index - 1]} 1.0"
            for position, index in enumerate(subset, 1)
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=None,
        help="Fixed-N cohort directory; inferred from --source-build-name and subset size",
    )
    parser.add_argument(
        "--source-build-name",
        default=DEFAULT_EMPIRICAL_PRIOR_DIR,
        help="Full-template empirical build used when --cohort-dir is omitted",
    )
    parser.add_argument("--templates", required=True, help="Subset label, e.g. T1+T7")
    parser.add_argument("--build-name", default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--z-min", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kde-sample", type=int, default=2000)
    parser.add_argument("--skip-kde", action="store_true")
    args = parser.parse_args()

    subset = parse_template_subset(args.templates)
    label = template_label(subset)
    slug = label.lower().replace("+", "-")
    build_name = args.build_name or f"empirical_prior/eazy12-{slug}"
    prior_dir = get_prior_build_dir(build_name)
    template_dir = (args.template_dir or get_template_dir()).expanduser().resolve()
    cohort_dir = (
        args.cohort_dir
        if args.cohort_dir is not None
        else get_prior_build_dir(args.source_build_name)
        / "reduced_template_cohorts"
        / f"n{len(subset)}"
    ).expanduser().resolve()

    discovery_path = cohort_dir / "discovery_parameters.json"
    memberships_path = cohort_dir / "subset_memberships.csv"
    discovery = json.loads(discovery_path.read_text())
    template_paths = [str(path) for path in discovery["template_paths"]]
    n_full_templates = len(template_paths)
    max_chi2_dof = float(discovery["max_chi2_dof"])
    memberships = pd.read_csv(memberships_path)
    weights = reduced_weights_table(
        memberships,
        subset,
        n_full_templates=n_full_templates,
        max_chi2_dof=max_chi2_dof,
    )

    source_stem = Path(str(discovery["template_param"])).stem
    reduced_param_rel = Path("templates/reduced") / f"{source_stem}_{slug}.param"
    reduced_param_path = write_reduced_template_param(
        template_dir / reduced_param_rel, template_paths, subset
    )
    prior_dir.mkdir(parents=True, exist_ok=True)
    weights_path = prior_dir / "desi_eazy_empirical_weights.csv"
    weights.to_csv(weights_path, index=False)

    provenance_path = write_provenance(
        prior_dir / BUILD_PROVENANCE_FILENAME,
        {
            "kind": "reduced_template_empirical_sed_prior_build",
            "build_name": build_name,
            "template": {
                "template_param": str(reduced_param_rel),
                "source_template_param": discovery["template_param"],
                "selected_template_indices": list(subset),
                "selected_template_labels": [f"T{index}" for index in subset],
                "selected_template_paths": [template_paths[index - 1] for index in subset],
                "normalization": {
                    "method": "integral",
                    "wave_min_aa": float(discovery["norm_min"]),
                    "wave_max_aa": float(discovery["norm_max"]),
                },
            },
            "selection": {
                "source_cohort_dir": cohort_dir,
                "source_build_name": args.source_build_name,
                "source_memberships": memberships_path,
                "subset": label,
                "n_rows": int(len(weights)),
                "n_quality_pass": int(weights["quality_pass"].sum()),
                "z_min_for_kde": float(args.z_min),
                "cohort_thresholds": {
                    key: discovery[key]
                    for key in (
                        "max_chi2_dof",
                        "max_delta_chi2_dof",
                        "max_color_rms",
                        "min_component_weight",
                        "min_good_pixels",
                    )
                },
            },
            "inputs": {
                "discovery_parameters": discovery_path,
                "subset_memberships": memberships_path,
                "template_dir": template_dir,
            },
            "kde_request": {"sample": int(args.kde_sample), "seed": int(args.seed)},
        },
    )

    print(f"Wrote {len(weights):,} {label} prior rows to {weights_path}")
    print(f"Wrote reduced template bank to {reduced_param_path}")
    print(f"Wrote build provenance to {provenance_path}")
    if args.skip_kde:
        return

    kde_path = prior_dir / SED_PRIOR_KDE_NATIVE_FILENAME
    command = [
        sys.executable,
        "-m",
        KDE_MODULE,
        "--weights-csv",
        str(weights_path),
        "--out",
        str(kde_path),
        "--build-provenance",
        str(provenance_path),
        "--max-chi2-dof",
        str(max_chi2_dof),
        "--z-min",
        str(args.z_min),
        "--sample",
        str(args.kde_sample),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
