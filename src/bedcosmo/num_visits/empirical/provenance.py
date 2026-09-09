"""Serializable provenance for DESI/EAZY empirical-prior builds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import BUILD_PROVENANCE_FILENAME, FIT_PROVENANCE_FILENAME
from .templates import (
    DEFAULT_TEMPLATE_NORM_MAX_AA,
    DEFAULT_TEMPLATE_NORM_MIN_AA,
)

PROVENANCE_SCHEMA_VERSION = 1


def json_safe(value: Any) -> Any:
    """Recursively convert argparse/build values to JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_provenance(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a stable, human-readable provenance document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        **json_safe(payload),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def read_provenance(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Provenance must be a JSON object: {path}")
    return payload


def build_provenance_path(prior_dir: str | Path) -> Path:
    return Path(prior_dir) / BUILD_PROVENANCE_FILENAME


def fit_provenance_path(fit_dir: str | Path) -> Path:
    return Path(fit_dir) / FIT_PROVENANCE_FILENAME


def template_settings_from_artifact(artifact: dict[str, Any]) -> dict[str, Any] | None:
    """Return recorded template settings from a KDE artifact, if available."""
    metadata = artifact.get("metadata", {})
    direct = metadata.get("template_settings")
    if isinstance(direct, dict):
        return direct
    provenance = metadata.get("build_provenance")
    if not isinstance(provenance, dict):
        return None
    template = provenance.get("template")
    if isinstance(template, dict):
        return template
    fit = provenance.get("fit")
    if isinstance(fit, dict) and isinstance(fit.get("template"), dict):
        return fit["template"]
    return None


def resolve_template_settings(
    artifact: dict[str, Any],
    *,
    configured_template_param: str,
    configured_norm_min: float | None = None,
    configured_norm_max: float | None = None,
) -> tuple[str, float, float]:
    """Resolve and validate the template bank used to interpret prior coefficients.

    New artifacts are authoritative. Explicit runtime values are retained for
    legacy artifacts and are checked against provenance when it is present.
    """
    recorded = template_settings_from_artifact(artifact)
    recorded_param = None if recorded is None else recorded.get("template_param")
    normalization = None if recorded is None else recorded.get("normalization")
    if normalization is not None and not isinstance(normalization, dict):
        raise ValueError("Artifact template normalization provenance must be an object")

    if recorded_param is not None and str(recorded_param) != str(configured_template_param):
        raise ValueError(
            "Configured template_param does not match the KDE artifact: "
            f"{configured_template_param!r} != {recorded_param!r}"
        )

    recorded_min = None if normalization is None else normalization.get("wave_min_aa")
    recorded_max = None if normalization is None else normalization.get("wave_max_aa")
    for label, configured, stored in (
        ("template_norm_min", configured_norm_min, recorded_min),
        ("template_norm_max", configured_norm_max, recorded_max),
    ):
        if configured is not None and stored is not None and float(configured) != float(stored):
            raise ValueError(
                f"Configured {label}={configured:g} does not match the KDE artifact "
                f"value {float(stored):g}"
            )

    norm_min = (
        float(recorded_min)
        if recorded_min is not None
        else (
            float(configured_norm_min)
            if configured_norm_min is not None
            else DEFAULT_TEMPLATE_NORM_MIN_AA
        )
    )
    norm_max = (
        float(recorded_max)
        if recorded_max is not None
        else (
            float(configured_norm_max)
            if configured_norm_max is not None
            else DEFAULT_TEMPLATE_NORM_MAX_AA
        )
    )
    if norm_min >= norm_max:
        raise ValueError(
            f"Template normalization minimum must be below maximum, got {norm_min:g} >= {norm_max:g}"
        )
    return str(recorded_param or configured_template_param), norm_min, norm_max
