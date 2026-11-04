"""Build a self-contained joint-pattern KDE/NF prior from fitted DESI coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prototype_joint_empirical_mixture import JointPatternMixture, lsst_magnitudes
from validate_joint_empirical_prior_flow import validate

from bedcosmo.num_visits.empirical.eazy.fit_eazy_weights_to_desi import save_triangle_plot
from bedcosmo.num_visits.empirical.fit_sed_prior_kde import load_prior_training_table
from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, train
from bedcosmo.num_visits.empirical.templates import load_two_column_template, read_template_param


def build(
    prior_dir,
    output_dir,
    *,
    bandwidth=0.1,
    n_samples=120000,
    epochs=150,
    validation_samples=20000,
    plot_samples=2000,
    threads=2,
    seed=23,
):
    """Reuse the measured spectral fits; rebuild every statistical prior stage."""
    prior_dir, out = Path(prior_dir).resolve(), Path(output_dir).resolve()
    if out == prior_dir or prior_dir in out.parents:
        raise ValueError("Output must be a new directory outside the source build")
    if (
        not np.isfinite(bandwidth)
        or bandwidth <= 0
        or min(n_samples, validation_samples) < 100
        or min(epochs, threads) < 1
        or plot_samples < 3
    ):
        raise ValueError("Use positive settings, at least 100 training/validation and 3 plot draws")
    source_provenance = json.loads((prior_dir / "build_provenance.json").read_text())
    config = yaml.safe_load((prior_dir / "prior_args.yaml").read_text())
    template_param = Path(config["template_param"])
    if template_param.is_absolute() or ".." in template_param.parts:
        raise ValueError("Template parameter file must be inside the source template bank")
    bank_dir = prior_dir / "templates"
    param = bank_dir / template_param
    component_paths = [param.parent / p for p in read_template_param(param)]
    if any(not p.resolve().is_relative_to(bank_dir.resolve()) for p in component_paths):
        raise ValueError("All template components must belong to the source template bank")
    bank = [load_two_column_template(p) for p in component_paths]
    wave = bank[0][0]
    if any(not np.array_equal(wave, item[0]) for item in bank):
        raise ValueError("Expected template components on a shared wavelength grid")
    if config["flux_unit_scale"] != 1e-17:
        raise ValueError("DESI photometry validation expects flux_unit_scale=1e-17")
    selection = source_provenance["selection"]
    request = {
        "prior_dir": str(prior_dir),
        "output_dir": str(out),
        "bandwidth": bandwidth,
        "n_samples": n_samples,
        "epochs": epochs,
        "validation_samples": validation_samples,
        "plot_samples": plot_samples,
        "threads": threads,
        "seed": seed,
    }
    source_files = [
        prior_dir / name
        for name in ["build_provenance.json", "prior_args.yaml", "desi_eazy_empirical_weights.csv"]
    ] + sorted(p for p in bank_dir.rglob("*") if p.is_file())
    source_hashes = {
        str(p.relative_to(prior_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    provenance = {
        "schema_version": 1,
        "kind": "joint_pattern_empirical_sed_prior_build",
        "build_name": out.name,
        "status": "building",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": request,
        "source_directory": str(prior_dir),
        "source_sha256": source_hashes,
        "template": source_provenance["template"],
        "selection": selection,
        "coefficient_scale": source_provenance["coefficient_scale"],
        "upstream_stages": "Reuse fitted coefficients and extrapolated templates; no NMF or spectral refit.",
        "kde": {
            "method": "joint_pattern_reflected_gaussian_then_simplex_closure",
            "bandwidth": bandwidth,
        },
        "gaussianization": "Trained conditional NF transform to standard Gaussian bases; patterns and singleton atoms remain discrete.",
        "runtime": {
            "bed_supported": True,
            "density_type": "joint_pattern_flow",
            "posterior_flow_type": "PatternNSF",
            "loader": "bedcosmo.num_visits.empirical.joint_prior_flow:JointPriorFlow.load",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    provenance_path = out / "build_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    (out / "request.json").write_text(json.dumps(request, indent=2))
    shutil.copytree(bank_dir, out / "templates")
    for original, copied in [
        ("build_provenance.json", "source_build_provenance.json"),
        ("prior_args.yaml", "source_prior_args.yaml"),
        ("desi_eazy_empirical_weights.csv", "desi_eazy_empirical_weights.csv"),
    ]:
        shutil.copy2(prior_dir / original, out / copied)
    for name in ["template_extrapolation.png", "template_extrapolation.pdf", "desi_basis.npz"]:
        if (prior_dir / name).is_file():
            shutil.copy2(prior_dir / name, out / name)
            source_hashes[name] = hashlib.sha256((prior_dir / name).read_bytes()).hexdigest()
    torch.set_num_threads(threads)
    with (out / "build.log").open("w", buffering=1) as log, redirect_stdout(log):
        table = load_prior_training_table(
            out / "desi_eazy_empirical_weights.csv",
            z_min=selection["prior_z_min"],
            max_chi2_dof=selection["max_chi2_dof"],
        )
        table = table[table["z"] <= selection["prior_z_max"]].copy()
        if len(table) != selection["n_prior_quality_pass"]:
            raise ValueError("Selected population differs from source build provenance")
        names = [f"a{i+1}" for i in range(len(bank))] + ["log_c_scale", "z"]
        rows = table[names].to_numpy()
        table.to_csv(out / "prior_training_table.csv", index=False)
        target = JointPatternMixture(rows, bandwidth=bandwidth)
        np.savez_compressed(
            out / "joint_kde_fit.npz", training_rows=rows, bandwidth=bandwidth, feature_names=names
        )
        (out / "joint_kde_metadata.json").write_text(
            json.dumps(
                {
                    "feature_names": names,
                    "bandwidth": bandwidth,
                    "pattern_summary": target.pattern_summary(rows),
                    "lower": target.lower.tolist(),
                    "upper": target.upper.tolist(),
                    "source_sha256": source_hashes,
                },
                indent=2,
            )
        )
        model = train(target, n_samples=n_samples, epochs=epochs, seed=seed, output_dir=out)
        model.metadata.update(
            {"feature_names": names, "source_sha256": source_hashes, "build_provenance": provenance}
        )
        model.save(out / "joint_prior_flow.pt")
        (out / "training.json").write_text(json.dumps(model.metadata["training_groups"], indent=2))
        report = validate(
            out / "joint_prior_flow.pt",
            out / "validation",
            n_samples=validation_samples,
            seed=seed + 1000,
        )
        with np.load(out / "validation" / "fresh_samples.npz") as samples:
            kde, nf = samples["joint_kde"], samples["nf"]
        plot_indices = np.random.default_rng(seed).choice(
            len(rows), min(plot_samples, len(rows)), replace=False
        )
        for data, filename, title in [
            (
                rows[plot_indices],
                "training_samples_triangle.png",
                "Fitted DESI coefficient population",
            ),
            (
                kde[:plot_samples],
                "kde_samples_triangle.png",
                "Joint-pattern KDE samples (physical coordinates)",
            ),
            (
                nf[:plot_samples],
                "nf_samples_triangle.png",
                "Joint-pattern NF samples (physical coordinates)",
            ),
        ]:
            save_triangle_plot(out, data, names, filename=filename, title=title, panel_size=1.35)
        ids = np.array([model.lookup[tuple(row[:-2] > 0)] for row in rows])
        interior = model.metadata["atomic"][ids] | np.all(
            (rows[:, -2:] > model.metadata["lower"]) & (rows[:, -2:] < model.metadata["upper"]),
            axis=1,
        )
        gaussian_reference = rows[interior].copy()
        reference_ids = ids[interior]
        reference_atomic = model.metadata["atomic"][reference_ids]
        # The KDE closes the simplex even for singleton donors. Match those
        # exact saved points rather than raw CSV roundoff when plotting atoms.
        gaussian_reference[reference_atomic] = model.metadata["atomic_rows"][
            reference_ids[reference_atomic]
        ]
        gaussian_sets = {
            "kde": model.gaussianize(kde)[1],
            "training": model.gaussianize(gaussian_reference)[1],
        }
        primary = int(
            max(
                model.metadata["training_groups"],
                key=lambda m: model.metadata["training_groups"][m]["n_train"],
            )
        )
        for label, groups in gaussian_sets.items():
            for m, values in groups.items():
                filename = f"{label}_samples_gaussianized_active{m}_triangle.png"
                title = f"Gaussianized {label} rows: {m} active templates (conditional group; N={len(values)})"
                save_triangle_plot(
                    out / "gaussianized_by_active_count",
                    values[:plot_samples],
                    [f"g{i+1}" for i in range(m + 1)],
                    filename=filename,
                    title=title,
                    panel_size=1.35,
                )
                if m == primary:
                    root_name = (
                        "kde_samples_gaussianized_triangle.png"
                        if label == "kde"
                        else "training_gaussianized_triangle.png"
                    )
                    shutil.copy2(out / "gaussianized_by_active_count" / filename, out / root_name)
        provenance["diagnostics"] = {
            "primary_gaussianized_active_count": primary,
            "training_aux_boundary_rows_excluded_from_gaussian_plots": int((~interior).sum()),
            "note": "Gaussian plots are conditional on active count; atoms have no continuous coordinates. No ILR coordinates are used.",
        }
        photometry, bands = {}, None
        for label, data in [("kde", kde), ("nf", nf)]:
            photometry[label], bands = lsst_magnitudes(
                data[:plot_samples], wave, np.stack([b[1] for b in bank])
            )
        np.savez_compressed(out / "lsst_photometry_samples.npz", **photometry, bands=bands)
        fig, axes = plt.subplots(1, len(bands) - 1, figsize=(16, 3), layout="constrained")
        for j, ax in enumerate(np.atleast_1d(axes)):
            for label, color in [("kde", "#168556"), ("nf", "#b14988")]:
                ax.hist(
                    photometry[label][:, j] - photometry[label][:, j + 1],
                    bins=40,
                    histtype="step",
                    density=True,
                    label=label,
                    color=color,
                )
            ax.set(xlabel=f"{bands[j]} − {bands[j+1]}", ylabel="Density")
            ax.legend()
        fig.savefig(out / "lsst_colors.png", dpi=170)
        plt.close(fig)
        # Exercise the completed artifact from its final location.
        reloaded = JointPriorFlow.load(out / "joint_prior_flow.pt")
        check = reloaded.sample(1000, seed=seed + 2000)
        if not np.isfinite(reloaded.log_prob(check)).all():
            raise ValueError("Saved prior failed its sampling/scoring check")
        provenance["status"] = "complete"
        provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["nf"] = {
            "groups": model.metadata["training_groups"],
            "flow_config": model.metadata["flow_config"],
        }
        provenance["validation_summary"] = report["metrics"]
        (out / "README.md").write_text(
            "# Joint-pattern empirical prior\n\n"
            "This build reuses the source DESI coefficient fits and extrapolated template bank, "
            "then rebuilds the joint KDE, conditional NF, Gaussianization and diagnostics.\n\n"
            "Load `joint_prior_flow.pt` with `JointPriorFlow.load` from "
            "`bedcosmo.num_visits.empirical.joint_prior_flow`. Reconstruct the joint KDE from "
            "`joint_kde_fit.npz` using `JointPatternMixture(training_rows, bandwidth=bandwidth)`.\n\n"
            "`build_provenance.json` records this build; `source_build_provenance.json` "
            "and `source_prior_args.yaml` preserve upstream history. The source YAML "
            "describes the original ILR backend and is not a configuration for this method.\n\n"
            "`kde_samples_triangle.png` shows the full physical population. Gaussian "
            "triangles are conditional on active count, with all groups under "
            "`gaussianized_by_active_count/`; the root copies show the most populous "
            "continuous group. Atoms have no continuous Gaussian coordinates. "
            "No ILR triangle is applicable to this method.\n\n"
            "`validation/` includes NF/KDE/DESI comparisons, Gaussianization diagnostics "
            "and fresh samples. `lsst_colors.png` checks photometry through the copied bank. "
            "BED full-parameter training and joint EIG use density_type=joint_pattern_flow "
            "with flow_type=PatternNSF and transform_input=false.\n"
        )
        provenance["outputs"] = {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and p.name not in ["build.log", "build_provenance.json", "joint_prior_flow.pt"]
        }
        provenance_path.write_text(json.dumps(provenance, indent=2))
        # Embed final provenance as well as keeping the human-readable sidecar.
        model.metadata["build_provenance"] = provenance
        model.save(out / "joint_prior_flow.pt")
        # The flow embeds provenance; its own hash belongs only in the sidecar.
        provenance["outputs"]["joint_prior_flow.pt"] = hashlib.sha256(
            (out / "joint_prior_flow.pt").read_bytes()
        ).hexdigest()
        provenance_path.write_text(json.dumps(provenance, indent=2))
        print(f"Completed self-contained joint prior build: {out}")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--validation-samples", type=int, default=20000)
    parser.add_argument("--plot-samples", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    result = build(**vars(parser.parse_args()))
    print(f"Completed: {result['arguments']['output_dir']}")
