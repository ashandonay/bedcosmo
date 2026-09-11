#!/usr/bin/env python
"""Plot completeness and purity for every tested subset of one spectral family."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from ..paths import DEFAULT_EMPIRICAL_PRIOR_DIR, get_prior_build_dir  # noqa: E402


def select_family_candidates(
    candidates: pd.DataFrame,
    family: str,
    *,
    min_completeness: float = 0.0,
    min_purity: float = 0.0,
    max_templates: int | None = None,
) -> pd.DataFrame:
    """Return a reproducibly ordered table of subsets tested for one family."""
    family = family.upper()
    selected = candidates.loc[candidates["family"].str.upper() == family].copy()
    if selected.empty:
        available = ", ".join(sorted(candidates["family"].unique()))
        raise ValueError(f"Unknown family {family!r}; available families: {available}")
    selected = selected.loc[
        (selected["coverage_fraction"] >= min_completeness)
        & (selected["purity_fraction"].fillna(0.0) >= min_purity)
    ]
    if max_templates is not None:
        selected = selected.loc[selected["n_templates"] <= max_templates]
    return selected.sort_values(
        ["n_templates", "coverage_fraction", "purity_fraction", "templates"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def make_family_candidate_report(
    candidates: pd.DataFrame,
    family: str,
    *,
    output: Path,
    page_size: int = 40,
) -> None:
    """Write a paginated PDF containing paired bars for every supplied subset."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if candidates.empty:
        raise ValueError("No family subsets remain after filtering")
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = math.ceil(len(candidates) / page_size)
    with PdfPages(output) as pdf:
        for page in range(pages):
            shown = candidates.iloc[page * page_size : (page + 1) * page_size]
            height = max(4.5, 0.31 * len(shown) + 1.5)
            fig, ax = plt.subplots(figsize=(10.5, height), constrained_layout=True)
            y = range(len(shown))
            bar_height = 0.34
            complete = ax.barh(
                [value - bar_height / 1.8 for value in y],
                shown["coverage_fraction"],
                height=bar_height,
                color="#3366CC",
                label="Completeness",
            )
            pure = ax.barh(
                [value + bar_height / 1.8 for value in y],
                shown["purity_fraction"].fillna(0.0),
                height=bar_height,
                color="#E07A3F",
                label="Purity",
            )
            ax.bar_label(
                complete,
                labels=[f"{value:.1%}" for value in shown["coverage_fraction"]],
                padding=2,
                fontsize=7,
            )
            ax.bar_label(
                pure,
                labels=[f"{value:.1%}" for value in shown["purity_fraction"].fillna(0.0)],
                padding=2,
                fontsize=7,
            )
            ax.set_yticks(list(y), shown["templates"])
            ax.invert_yaxis()
            ax.set_xlim(0, 1.12)
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xlabel("Spectra passing quality threshold")
            ax.set_title(
                f"{family}: tested template subsets ({page + 1}/{pages})",
                loc="left",
            )
            ax.legend(loc="lower right", frameon=False)
            ax.grid(axis="x", color="#D9DDE3", alpha=0.7, linewidth=0.7)
            ax.set_axisbelow(True)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--family", required=True, help="Family label, for example F01")
    parser.add_argument("--build-name", default=DEFAULT_EMPIRICAL_PRIOR_DIR)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--page-size", type=int, default=40)
    parser.add_argument("--min-completeness", type=float, default=0.0)
    parser.add_argument("--min-purity", type=float, default=0.0)
    parser.add_argument("--max-templates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_completeness <= 1 or not 0 <= args.min_purity <= 1:
        raise ValueError("minimum completeness and purity must lie in [0, 1]")
    family = args.family.upper()
    family_dir = get_prior_build_dir(args.build_name) / "reduced_template_cohorts/spectral_families"
    candidate_csv = args.candidate_csv or family_dir / "family_basis_candidates.csv"
    output = args.output or family_dir / f"{family.lower()}_subset_tradeoffs.pdf"
    selected = select_family_candidates(
        pd.read_csv(candidate_csv),
        family,
        min_completeness=args.min_completeness,
        min_purity=args.min_purity,
        max_templates=args.max_templates,
    )
    table_output = output.with_suffix(".csv")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(table_output, index=False)
    make_family_candidate_report(
        selected,
        family,
        output=output,
        page_size=args.page_size,
    )
    print(f"Wrote {len(selected):,} {family} subset rows to {table_output}")
    print(f"Wrote paired-bar report to {output}")


if __name__ == "__main__":
    main()
