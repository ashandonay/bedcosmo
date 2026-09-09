"""Resolve empirical prior builds from ``template_source`` + ``reduced_templates``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import EMPIRICAL_PRIOR_ROOT_DIR, get_prior_build_dir
from .simplex import prior_ilr_feature_names
from .templates import DEFAULT_TEMPLATE_PARAM_6D, DEFAULT_TEMPLATE_PARAM_12D

TEMPLATE_SOURCES = ("eazy12", "eazy6")

_SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "eazy12": {
        "n_templates": 12,
        "template_param": DEFAULT_TEMPLATE_PARAM_12D,
    },
    "eazy6": {
        "n_templates": 6,
        "template_param": DEFAULT_TEMPLATE_PARAM_6D,
    },
}

_DEFAULT_F_PLOT = {"lower": -8.0, "upper": 8.0}
_DEFAULT_LOG_S_PLOT = {"lower": 4.0, "upper": 10.5}
_DEFAULT_Z_PLOT = {"lower": 0.0, "upper": 1.75}


def normalize_template_source(value: str | None) -> str:
    """Return a validated ``eazy12`` / ``eazy6`` source name."""
    if value is None:
        raise ValueError("template_source is required (eazy12 or eazy6)")
    source = str(value).strip().lower()
    if source not in _SOURCE_CONFIG:
        raise ValueError(
            f"template_source must be one of {list(_SOURCE_CONFIG)}, got {value!r}"
        )
    return source


def parse_reduced_templates(value: Any) -> tuple[int, ...] | None:
    """Parse ``reduced_templates`` from YAML/CLI.

    Accepts ``null`` / empty (no reduction), or a string like ``\"t7,t10\"``,
    ``\"T7+T10\"``, or ``\"7,10\"``. Returns sorted one-based indices, or
    ``None`` when the full template bank is used.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        pieces = [str(item) for item in value]
        label = "+".join(pieces)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "~"}:
            return None
        label = text
    subset = _parse_template_subset_label(label)
    if len(subset) < 2:
        raise ValueError(
            f"reduced_templates must list at least two templates for ILR, got {value!r}"
        )
    return subset


def format_reduced_templates(subset: tuple[int, ...] | None) -> str | None:
    """Canonical storage form: ``\"t7,t10\"`` (sorted), or ``None``."""
    if subset is None:
        return None
    return ",".join(f"t{index}" for index in subset)


def reduced_template_slug(subset: tuple[int, ...]) -> str:
    """Directory / filename slug, e.g. ``t7-t10``."""
    return "-".join(f"t{index}" for index in subset)


def empirical_prior_variant(
    template_source: str,
    reduced_templates: Any = None,
) -> str:
    """Scratch subdirectory under ``empirical_prior/``, e.g. ``eazy12-t7-t10``."""
    source = normalize_template_source(template_source)
    subset = parse_reduced_templates(reduced_templates)
    if subset is None:
        return source
    return f"{source}-{reduced_template_slug(subset)}"


def empirical_prior_build_name(
    template_source: str,
    reduced_templates: Any = None,
) -> str:
    """Build name relative to the num_visits scratch root."""
    return f"{EMPIRICAL_PRIOR_ROOT_DIR}/{empirical_prior_variant(template_source, reduced_templates)}"


def resolve_template_param(
    template_source: str,
    reduced_templates: Any = None,
) -> str:
    """EAZY ``.param`` path relative to the template cache directory."""
    source = normalize_template_source(template_source)
    full_param = str(_SOURCE_CONFIG[source]["template_param"])
    subset = parse_reduced_templates(reduced_templates)
    if subset is None:
        return full_param
    stem = Path(full_param).stem
    slug = reduced_template_slug(subset)
    return f"templates/reduced/{stem}_{slug}.param"


def n_templates_for(
    template_source: str,
    reduced_templates: Any = None,
) -> int:
    source = normalize_template_source(template_source)
    subset = parse_reduced_templates(reduced_templates)
    if subset is None:
        return int(_SOURCE_CONFIG[source]["n_templates"])
    return len(subset)


def default_empirical_parameters(
    n_templates: int,
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``parameters:`` block for an ILR prior with ``n_templates`` SEDs."""
    existing = existing or {}
    names = prior_ilr_feature_names(int(n_templates))
    out: dict[str, Any] = {}
    for name in names:
        if name in existing:
            out[name] = existing[name]
            continue
        if name == "log_c_scale":
            plot = dict(_DEFAULT_LOG_S_PLOT)
        elif name == "z":
            plot = dict(_DEFAULT_Z_PLOT)
        else:
            plot = dict(_DEFAULT_F_PLOT)
        out[name] = {
            "distribution": {"type": "empirical"},
            "plot": plot,
        }
    return out


def materialize_empirical_prior_args(
    prior_args: dict[str, Any] | None,
    *,
    template_source: Any = None,
    reduced_templates: Any = None,
) -> dict[str, Any]:
    """Fill ``prior_dir`` / ``template_param`` / ``parameters`` from the two selectors.

    ``template_source`` / ``reduced_templates`` keyword args override values already
    present in ``prior_args`` when not ``None``. Pass the string ``\"null\"`` (or an
    empty string) for ``reduced_templates`` to clear a YAML reduction from the CLI.

    Legacy configs without ``template_source`` are returned unchanged (aside from a
    shallow copy), so explicit ``prior_dir`` / ``template_param`` files still work.
    """
    out = dict(prior_args or {})

    source_raw = template_source if template_source is not None else out.get("template_source")
    if source_raw is None:
        return out

    source = normalize_template_source(source_raw)
    if reduced_templates is not None:
        subset = parse_reduced_templates(reduced_templates)
    elif "reduced_templates" in out:
        subset = parse_reduced_templates(out.get("reduced_templates"))
    else:
        subset = None

    n_templates = n_templates_for(source, subset)
    build_name = empirical_prior_build_name(source, subset)
    prior_dir = get_prior_build_dir(build_name)
    template_param = resolve_template_param(source, subset)

    out["template_source"] = source
    out["reduced_templates"] = format_reduced_templates(subset)
    out["prior_dir"] = str(prior_dir)
    out["template_param"] = template_param
    out["parameters"] = default_empirical_parameters(
        n_templates,
        existing=out.get("parameters") if isinstance(out.get("parameters"), dict) else None,
    )
    return out


def _parse_template_subset_label(label: str) -> tuple[int, ...]:
    """Parse labels such as ``T1+T7`` / ``t7,t10`` into sorted one-based indices."""
    pieces = label.upper().replace(",", "+").split("+")
    try:
        subset = tuple(int(piece.strip().removeprefix("T")) for piece in pieces if piece.strip())
    except ValueError as error:
        raise ValueError(f"Invalid reduced_templates {label!r}") from error
    if not subset or len(subset) != len(set(subset)) or any(index < 1 for index in subset):
        raise ValueError(f"Invalid reduced_templates {label!r}")
    return tuple(sorted(subset))
