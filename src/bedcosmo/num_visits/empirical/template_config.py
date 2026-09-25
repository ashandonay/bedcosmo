"""Resolve empirical prior builds from ``template_source`` + ``reduced_templates``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .paths import (
    EMPIRICAL_PRIOR_ROOT_DIR,
    get_prior_build_dir,
)
from .simplex import prior_ilr_feature_names
from .templates import (
    DEFAULT_TEMPLATE_NORM_MAX_AA,
    DEFAULT_TEMPLATE_NORM_MIN_AA,
    DEFAULT_TEMPLATE_PARAM_6D,
    DEFAULT_TEMPLATE_PARAM_12D,
    read_template_param,
)

_BUILTIN_SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "eazy12": {
        "n_templates": 12,
        "template_param": DEFAULT_TEMPLATE_PARAM_12D,
        "template_norm_min": DEFAULT_TEMPLATE_NORM_MIN_AA,
        "template_norm_max": DEFAULT_TEMPLATE_NORM_MAX_AA,
    },
    "eazy6": {
        "n_templates": 6,
        "template_param": DEFAULT_TEMPLATE_PARAM_6D,
        "template_norm_min": DEFAULT_TEMPLATE_NORM_MIN_AA,
        "template_norm_max": DEFAULT_TEMPLATE_NORM_MAX_AA,
    },
}

_DEFAULT_F_PLOT = {"lower": -8.0, "upper": 8.0}
_DEFAULT_LOG_S_PLOT = {"lower": 4.0, "upper": 10.5}
_DEFAULT_Z_PLOT = {"lower": 0.0, "upper": 1.75}


def normalize_template_source(value: str | None) -> str:
    """Return a validated empirical spectral-template source name."""
    if value is None:
        raise ValueError("template_source is required")
    source = str(value).strip().lower()
    if source not in _BUILTIN_SOURCE_CONFIG and not _prior_build_config(source):
        available = sorted(set(_BUILTIN_SOURCE_CONFIG) | _discover_prior_sources())
        raise ValueError(f"No empirical prior build found for {value!r}; available: {available}")
    return source


def _discover_prior_sources() -> set[str]:
    root = get_prior_build_dir(EMPIRICAL_PRIOR_ROOT_DIR)
    if not root.is_dir():
        return set()
    return {
        path.name for path in root.iterdir() if path.is_dir() and _prior_build_config(path.name)
    }


def _prior_build_config(source: str) -> dict[str, Any] | None:
    path = get_prior_build_dir(f"{EMPIRICAL_PRIOR_ROOT_DIR}/{source}") / "prior_args.yaml"
    if not path.is_file():
        return None
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid empirical prior config: {path}")
    template_param = config.get("template_param")
    if not isinstance(template_param, str) or not template_param:
        raise ValueError(f"{path} must define template_param")
    template_dir = get_prior_build_dir(f"{EMPIRICAL_PRIOR_ROOT_DIR}/{source}") / "templates"
    template_path = template_dir / template_param
    if not template_path.is_file() and (template_dir / Path(template_param).name).is_file():
        template_path = template_dir / Path(template_param).name
        config["template_param"] = template_path.name
    if not template_path.is_file():
        raise FileNotFoundError(f"Template parameter file is missing: {template_path}")
    missing_components = [
        rel for rel in read_template_param(template_path) if not (template_dir / rel).is_file()
    ]
    if missing_components:
        raise FileNotFoundError(
            f"Template components referenced by {template_path} are missing: {missing_components}"
        )
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} must define parameters")
    n_templates = sum(bool(re.fullmatch(r"f\d+", name)) for name in parameters)
    if n_templates < 2 or "log_c_scale" not in parameters or "z" not in parameters:
        raise ValueError(f"{path} has an invalid empirical parameter set")
    config["n_templates"] = n_templates + 1
    return config


def _source_config(source: str) -> dict[str, Any]:
    if source in _BUILTIN_SOURCE_CONFIG:
        return _BUILTIN_SOURCE_CONFIG[source]
    config = _prior_build_config(source)
    if config is None:
        raise ValueError(f"No prior_args.yaml found for empirical prior source {source!r}")
    return {
        "n_templates": config["n_templates"],
        "template_param": config["template_param"],
        "template_norm_min": config["template_norm_min"],
        "template_norm_max": config["template_norm_max"],
        "parameters": config["parameters"],
    }


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
    if source not in _BUILTIN_SOURCE_CONFIG and subset is not None:
        raise ValueError(f"reduced_templates is not supported for template_source={source!r}")
    if subset is None:
        return source
    return f"{source}-{reduced_template_slug(subset)}"


def empirical_prior_build_name(
    template_source: str,
    reduced_templates: Any = None,
) -> str:
    """Build name relative to the num_visits scratch root."""
    return (
        f"{EMPIRICAL_PRIOR_ROOT_DIR}/{empirical_prior_variant(template_source, reduced_templates)}"
    )


def resolve_template_param(
    template_source: str,
    reduced_templates: Any = None,
) -> str:
    """Template-bank filename relative to ``<prior_dir>/templates``."""
    source = normalize_template_source(template_source)
    full_param = str(_source_config(source)["template_param"])
    subset = parse_reduced_templates(reduced_templates)
    if source not in _BUILTIN_SOURCE_CONFIG and subset is not None:
        raise ValueError(f"reduced_templates is not supported for template_source={source!r}")
    if subset is None:
        return full_param
    slug = reduced_template_slug(subset)
    return f"{source}_{slug}.param"


def n_templates_for(
    template_source: str,
    reduced_templates: Any = None,
) -> int:
    source = normalize_template_source(template_source)
    subset = parse_reduced_templates(reduced_templates)
    if source not in _BUILTIN_SOURCE_CONFIG and subset is not None:
        raise ValueError(f"reduced_templates is not supported for template_source={source!r}")
    if subset is None:
        return int(_source_config(source)["n_templates"])
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

    source_config = _source_config(source)
    n_templates = n_templates_for(source, subset)
    build_name = empirical_prior_build_name(source, subset)
    prior_dir = get_prior_build_dir(build_name)
    template_param = resolve_template_param(source, subset)

    out["template_source"] = source
    out["reduced_templates"] = format_reduced_templates(subset)
    out["prior_dir"] = str(prior_dir)
    out["template_param"] = template_param
    out.pop("template_dir", None)
    out["template_norm_min"] = float(source_config["template_norm_min"])
    out["template_norm_max"] = float(source_config["template_norm_max"])
    existing_parameters = dict(source_config.get("parameters", {}))
    if isinstance(out.get("parameters"), dict):
        existing_parameters.update(out["parameters"])
    out["parameters"] = default_empirical_parameters(n_templates, existing=existing_parameters)
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
