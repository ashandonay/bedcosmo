# Reduced EAZY template bases

This package contains the search, diagnostics, and build tools used to replace
the full EAZY template bank with a smaller subset of the original templates for
the `num_visits` empirical SED prior.

The important distinction is:

- PCA and HDBSCAN are analysis tools used to discover populated spectral-shape
  families and identify promising original-template subsets.
- The BED forward model does **not** use principal components as spectra. A
  reduced prior still uses original EAZY templates such as `T1+T7` or
  `T7+T10`, together with one overall flux scale and redshift.
- `build_template_prior` uses every DESI spectrum that passes the requested
  subset's quality cuts. It does not restrict the prior to one HDBSCAN family.

For a bank of `K` templates, the prior contains `K-1` independent ILR shape
coordinates plus `log_c_scale` and `z`. Thus a two-template bank has three KDE
features: one shape coordinate, scale, and redshift.

## Tools

| Module | Purpose |
|---|---|
| `discover_template_cohorts` | Exhaustively refit the DESI sample with every exact-`N` subset and save fit/color quality matrices. |
| `build_template_prior` | Build the reduced coefficient table, EAZY `.param` file, provenance, KDE, and diagnostic samples for one subset. |
| `build_reduced_template_prior` | Historical alias for `build_template_prior`. |
| `plot_template_subset_examples` | Plot observed DESI spectra, full fits, reduced fits, and individual template contributions. |
| `summarize_template_composition` | Summarize integrated flux shares and template-dominance fractions within fixed-`N` cohorts. |
| `discover_eazy_spectral_families` | Transform full-fit weights to ILR space, apply PCA, cluster with HDBSCAN, and decode families back to sparse original-template subsets. |
| `compare_eazy_family_embeddings` | Test PCA variance cutoffs and standardized versus raw PC scores. |
| `plot_family_spectral_features` | Compare reconstructed family spectra, continua, absorption indices, emission EWs, and the Balmer decrement. |
| `plot_eazy_basis_representativeness` | Diagnose full-bank template usage, omission losses, and PCA reconstruction fidelity. |
| `plot_eazy_dominant_cohort_traits` | Older exploratory T1/T7 majority-cohort diagnostic; not part of the recommended discovery workflow. |

All commands below are Python modules, so they work from an installed checkout
without depending on the repository's `experiments/` directory.

## Paths and inputs

The path helpers default to:

```text
$SCRATCH/bedcosmo/eazy/
$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/
```

The full EAZY12 build must contain `desi_eazy_empirical_weights.csv`. The EAZY
root must contain `templates/fsps_full/fsps_QSF_12_v3.param` and its referenced
spectra.

## 1. Search all fixed-size subsets

Run once for every subset size of interest. The sufficient-statistics cache is
shared, so later searches do not reread and reproject every DESI spectrum.

```bash
python -m bedcosmo.num_visits.empirical.reduced.discover_template_cohorts \
  --n-templates 2 \
  --build-name empirical_prior/eazy12

python -m bedcosmo.num_visits.empirical.reduced.discover_template_cohorts \
  --n-templates 3 \
  --build-name empirical_prior/eazy12
```

The default quality definition is:

```text
reduced chi2/dof <= 1.2
extra chi2/original dof relative to the full fit <= 0.05
LSST ugrizy color RMS relative to the full fit <= 0.02 mag
```

Important outputs under
`eazy12/reduced_template_cohorts/nN/` include:

- `subset_summary.csv`: coverage and fit-quality summaries for every subset.
- `subset_memberships.csv`: every passing spectrum/subset pair and its reduced
  coefficients.
- `subset_quality_matrices.npz`: per-spectrum quality values for every subset.
- `spectrum_assignments.csv`: optional disjoint best-subset assignment.
- `discovery_parameters.json`: complete thresholds, normalization, and input
  provenance.

Subset memberships overlap by design. A spectrum may be accurately represented
by several candidate subsets.

## 2. Optional population-family discovery

After searches through the largest subset size that should be considered (the
current analysis used `N=1,...,5`), run:

```bash
python -m bedcosmo.num_visits.empirical.reduced.discover_eazy_spectral_families \
  --build-name empirical_prior/eazy12 \
  --variance-threshold 0.90 \
  --pca-scaling standardized
```

The analysis maps the twelve compositional weights to eleven-dimensional ILR
shape space, retains enough PCs to reach the requested cumulative variance,
standardizes the retained PC scores, and applies HDBSCAN. The saved fixed-`N`
quality matrices then identify the smallest original-template subset that
accurately represents most members of each family.

In the current EAZY12/DESI run, the 90% setting retained eight PCs and found
thirteen dense families. This result motivated compact candidates including
`T1+T7`, `T7+T10`, and `T1+T8`; the PC vectors themselves are not passed to
EAZY or BED.

## 3. Build reduced empirical priors

If the default full build and cohort layout are present, only the template
label is required:

```bash
python -m bedcosmo.num_visits.empirical.reduced.build_template_prior \
  --templates T1+T7 \
  --kde-sample 2000

python -m bedcosmo.num_visits.empirical.reduced.build_template_prior \
  --templates T7+T10 \
  --kde-sample 2000
```

These infer the source cohort directories from subset size and write builds to:

```text
$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12-t1-t7/
$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12-t7-t10/
```

Use `--source-build-name`, `--cohort-dir`, `--template-dir`, or `--build-name`
only for a nonstandard layout. A custom `--build-name` controls the output
directory and must match the intended subset to avoid misleading paths.

Each build writes:

- `desi_eazy_empirical_weights.csv`
- `build_provenance.json`
- `sed_prior_kde_native.joblib`
- `sed_prior_kde_gaussianized.joblib`
- KDE diagnostic triangle plots
- `templates/reduced/<source>_<subset>.param` under the EAZY root

The coefficient scale and KDE `log_c_scale` inherit the full-fit template
normalization recorded in `discovery_parameters.json`. DESI coadd fluxes are
converted with the configured `flux_unit_scale=1e-17` when used by NumVisits.

## 4. Train BED with a reduced prior

Select a built prior without maintaining a separate YAML file:

```bash
./submit.sh train num_visits empirical \
  --prior-template-source eazy12 \
  --prior-reduced-templates t1,t7

./submit.sh train num_visits empirical \
  --prior-template-source eazy12 \
  --prior-reduced-templates t7,t10
```

`density_type: flow` requires the corresponding trained prior-flow artifacts in
the selected build. Use `--prior-density-type kde` for a KDE-backed test or
train the prior flow following the parent empirical-prior README.

## Interpretation and recommended comparisons

- `T1+T7` is the largest two-template cohort and provides the common reduced
  baseline.
- `T7+T10` is a strongly UV/emission-line population and is spectrally distinct
  from `T1+T7` while retaining the same parameter dimension.
- A prior built from `--templates T1+T7` includes all spectra passing that
  subset, not only the PCA/HDBSCAN family labeled F12. Family-specific priors
  would require an additional intersection with
  `spectrum_family_assignments.csv`.
- To isolate spectral-type effects in BED, match or condition the family priors
  in redshift and scale. Native priors intentionally mix spectral differences
  with their empirical redshift/scale distributions.

## Reproducibility

Keep `discovery_parameters.json`, the reduced build's
`build_provenance.json`, and the exact full-template coefficient table together.
The provenance records the source template paths, normalization interval,
selection thresholds, subset labels, sample counts, and KDE request.
