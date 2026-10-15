# Direct DESI spectral basis

This package learns nonnegative rest-frame galaxy components directly from
DESI B/R/Z coadds. NMF and coefficient fitting use **measured data only**.
Finished templates are extrapolated afterward to cover LSST photometry over
the requested prior redshift interval.

## Build the measured training matrix

```bash
python -m bedcosmo.num_visits.empirical.desi.build_matrix \
  --max-spectra 0 \
  --z-min 0.3 --z-max 1.4
```

`--z-min` and `--z-max` are optional hard cuts on **training galaxies**. Omit
them to use the full eligible DESI population. They do not impose LSST coverage
or add redshift buffers. These population cuts are independent of the target
prior bounds passed to `build_prior`.

The default output is the shared DESI training-data directory:

```text
$SCRATCH/bedcosmo/num_visits/desi_training_data/
├── desi_candidate_manifest.csv
├── desi_sample_manifest.csv
├── desi_rest_frame_training_matrix.npz
├── desi_training_matrix_provenance.json
└── rest_wavelength_coverage.csv
```

`--output-dir` overrides that directory. The shared full catalog candidate
manifest is written alongside the matrix in `desi_training_data/`; with an
explicit output directory, its candidate manifest is written there. The candidate list precedes optional
training redshift and pilot cuts and remains suitable for EAZY selection.
The default `--max-spectra 1500` provides a bounded pilot; zero uses all eligible
candidates. `--manifest` deliberately overrides catalog discovery with a table
containing `targetid`, `healpix`, and `z`.

The matrix builder:

1. selects Redrock `GALAXY` rows with finite `z >= 0.01`, `ZWARN == 0`, and a
   matching coadd target;
2. applies optional training redshift bounds and pilot sampling;
3. bins valid, unmasked coadd pixels by inverse variance onto a common
   rest-frame grid (`--wave-step`, default 10 Å), covering measured DESI pixels;
4. removes one robust multiplicative flux scale per galaxy;
5. applies the permissive exterior quality cut described below and retains
   galaxies with at least `--min-good-pixels` (100) measured bins;
6. saves signed fluxes and their inverse variances in normalized units.

No exterior tails or interior gaps are filled. Unobserved bins keep zero
weight and are ignored by factorization. `--wave-min`/`--wave-max` override
native grid bounds without requiring LSST coverage. The NPZ and provenance
identify the matrix with `coverage_kind: measured`. Full prior builds reject
old matrices without this marker, including extrapolated matrices: regenerate
from coadds rather than reusing assumed fluxes as measured data.

### Exterior quality cut

The 5% IVAR exterior quality cut is always applied by the default matrix builder.

At each native endpoint, use 5% of the median positive inverse variance
50–300 **observed-frame** Å inward, and retain the first run of three
consecutive bins passing the threshold. Only exterior bins are trimmed;
retained fluxes, weights, and internal gaps are unchanged. With fewer than
three reference bins, retain the endpoint. If a reference exists but no
three-bin run passes, reject the spectrum. Minimum-good-pixel selection counts
retained measured bins only. This fixed working rule is recorded as
`edge_quality_cut` and is not an optimized continuum criterion.

### Measured wavelength support

`build_prior` chooses the longest contiguous interval meeting its contributor
threshold on the planned training split. The default requires 100 measured
spectra per wavelength bin (ten per component for support rank 10).
`--minimum-wavelength-contributors`, `--observations-per-component`, and
`--wavelength-support-rank` control this selection. The threshold selects the
**learned wavelength interval**; it does not shrink the requested prior
redshift range or require synthetic training contributors.

`rest_wavelength_coverage.csv` reports measured contributor counts, measured
fractions, and LSST-demand-weighted measured coverage at each galaxy's redshift.

### Optional: compare ranks

```bash
python -m bedcosmo.num_visits.empirical.desi.compare_ranks \
  --ranks 2 3 4 5 6 7 8 9 10 11 12
```

This exploratory diagnostic reads the measured matrix, learns quick bases,
and reports held-out spectral and measured-band LSST color errors. Its bases
are not used by `build_prior`, which learns the production basis separately.

## Build a rank-K NumVisits prior

`build_prior` performs the production build in one command. It selects the best
of several Nearly-NMF initializations on validation spectra, reports the result
on an untouched test split, and then refits the selected basis on all spectra.
The training population is the matrix's selected DESI galaxies. NMF support,
validation/test metrics, the final refit, coefficient solves, and fit-quality
cuts all use measured pixels within the selected learned interval.

After factorization and unit-integral normalization, each finished component
is extended to cover full tabulated LSST `ugrizy` bandpasses over
`--prior-z-min` to `--prior-z-max`. These bounds default to the matrix's actual
training redshift extrema and may be set independently. They are never reduced
by the measured contributor threshold. Only fitted galaxies inside the requested
prior interval contribute to the coefficient prior: extending templates cannot
create missing galaxy populations outside the available data.

`--edge-extrapolation constant` is the default. It fills each template's
exterior tails with its mean weighted by estimated template IVAR of the unsmoothed learned values
over the nearest 100 rest-frame Å. Zero-IVAR samples contribute no weight. `--edge-extrapolation powerlaw` instead robustly fits a positive
power law over the nearest 500 Å, using the same estimated template IVAR weights, and continues it beyond the learned edge. These are assumptions about
unmeasured template shapes; synthetic values never enter the NMF objective.
The exported grid includes the learned interval and LSST demand
`[lambda_LSST_blue / (1 + prior_z_max), lambda_LSST_red / (1 + prior_z_min)]`,
rounded outward to the native grid spacing. Learned template values and their
normalization are preserved exactly.

`desi_basis.npz` stores learned `wave_rest_aa`/`basis` separately from
`export_wave_rest_aa`/`export_basis`, plus `estimated_template_ivar` on the learned grid. Template `.dat` files contain the extended
bank used for photometry. Provenance records learned/exported bounds, the edge
method, and extension distances.

### Estimated template IVAR used for edge fits

“Estimated template IVAR” means conditional template inverse variance, holding
the fitted galaxy coefficients fixed.

After normalization and the final coefficient solve, estimate each wavelength's
component covariance as `inv(C.T @ diag(measurement_ivar[:, j]) @ C)`, holding
fitted galaxy coefficients `C` fixed. The inverse diagonal variances supply the
constant and power-law edge-fit weights. The full component matrix is inverted
so component correlations are included; this is not a sum of DESI IVAR values.
The estimate transforms consistently with template normalization. Numerically
unidentifiable components receive zero IVAR and a window with no positive-IVAR
samples raises an error. Actual learned endpoints stay fixed even when their
IVAR is zero, and all learned flux values remain unchanged.

This is a conditional linearized measurement-noise estimate, not a full NMF
posterior: coefficient uncertainty, positivity constraints, alternative NMF
solutions, model mismatch, and unmeasured tails are not included. The formula
and assumptions are recorded in template provenance. It follows the weighted
least-squares covariance construction described in the
[NIST ODRPACK guide](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4834.pdf).
The constant estimator uses unsmoothed flux because median filtering would
change the errors and introduce correlations not represented by these weights.

Existing datasets and templates are not modified automatically. `--prior-only`
reuses a completed measured-data build's saved coefficients and exported bank.
It can change prior bounds within that bank's LSST coverage but cannot enlarge
the bank; a new full build exports templates for a broader request.
A full factorization build checks its request against saved checkpoints.

Install the shared empirical-prior and pinned Nearly-NMF dependencies before
building:

```bash
pip install -e '.[sed-prior,desi-basis]'
```

### Rebuild only the prior from an existing basis

```bash
python -m bedcosmo.num_visits.empirical.desi.build_prior \
  --build-name desi8 \
  --prior-only

python -m bedcosmo.num_visits.empirical.prior_flow \
  --kde-path "$SCRATCH/bedcosmo/num_visits/empirical_prior/desi8/sed_prior_kde_native.joblib" \
  --out-dir "$SCRATCH/bedcosmo/num_visits/empirical_prior/desi8" \
  --space both
```

`--prior-only` overwrites the fit table, selection provenance, `prior_args.yaml`,
KDEs, and diagnostic triangles in the existing build. It loads the learned and exported arrays from `desi_basis.npz`
and the training matrix recorded in provenance (or `--training-matrix`), checks
the matrix's SHA-256 fingerprint, target-ID order, wavelengths and the exported
template bank, and reuses the
saved coefficients. Rank, normalization and template paths come from the saved
build; no support reselection, coefficient solve, or factorization is performed.
Templates, basis arrays and factorization checkpoints remain untouched.
Full builds save the training-matrix fingerprint in provenance and checkpoint
requests. Changing any matrix contents prevents reuse of cached coefficients
or checkpoints. Older builds without a fingerprint require a new full build;
no automatic metadata migration is performed.
Redshift overrides and `--max-chi2-dof` control the new prior selection;
`--skip-kde` updates the table, provenance, runtime YAML and template-redshift plot.
Like the full build, this does not train prior flows. Rebuild them with the
second command before BED use; old flows no longer describe the updated KDE.

### Full basis and prior build

```bash
python -m bedcosmo.num_visits.empirical.desi.build_prior \
  --rank 8
```

With no `--training-matrix` override, a full build reads
`$SCRATCH/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz`.
Build that measured matrix first. To train on the full DESI population while
requesting a z=0.3–1.4 prior:

```bash
python -m bedcosmo.num_visits.empirical.desi.build_matrix --max-spectra 0
python -m bedcosmo.num_visits.empirical.desi.build_prior \
  --rank 8 --build-name desi8_measured \
  --prior-z-min 0.3 --prior-z-max 1.4 \
  --edge-extrapolation constant
```

Use a new build name to preserve previous templates for comparison.
`--prior-only` uses the matrix recorded in the existing basis provenance.

`--rank` controls both the number of learned spectral components and the
dimension of the generated prior. Because the default build name is `desi8`,
give other ranks their own build name. For example, the production K=4 build is:

```bash
python -m bedcosmo.num_visits.empirical.desi.build_prior \
  --rank 4 \
  --build-name desi4
```

That command writes the prior artifacts and four-component template bank under
`$SCRATCH/bedcosmo/num_visits/empirical_prior/desi4/`, with the components in
its `templates/` subdirectory.

`--build-name` is a single directory name, not a relative path. The builder
adds `empirical_prior/` internally: `--build-name desi6` resolves to
`$SCRATCH/bedcosmo/num_visits/empirical_prior/desi6/`. For a custom filesystem
location, use `--output-dir /path/to/build` instead. This applies to both full
builds and `--prior-only`.

With no path overrides, the prior artifacts are written to
`$SCRATCH/bedcosmo/num_visits/empirical_prior/desi8/` and the learned template
bank to its `templates/` subdirectory. EAZY builds use the same self-contained
component-plus-parameter-file layout.

The command writes an EAZY-compatible component bank, the standard
`desi_eazy_empirical_weights.csv`, build provenance, native and gaussianized KDE
artifacts, and diagnostic triangle plots.

Whenever the saved template bank extends beyond its learned wavelength support,
full builds and `--prior-only` rebuilds automatically save
`template_extrapolation.png` and `template_extrapolation.pdf` in the build
output directory, including with `--skip-kde`. The plot shows each full template,
close-ups of each extended edge, learned-support boundaries, connected saved
samples at those boundaries, and conditional ±1σ bands derived from estimated
template IVAR. Plot amplitudes and uncertainties share the same display
normalization. Only edges with actual extension get close-ups. Banks without
extension produce no extrapolation plot. Generated plot paths are recorded in
build provenance. Rendering uses the saved arrays and does not refit NMF or
change the templates.

Every full build and `--prior-only` rebuild also saves `template_redshifts.png`
in the prior directory, including when `--skip-kde` is used. Each component
shows its mean-normalized rest-frame shape in black, observed-frame curves at
coefficient-weighted redshift percentiles (5th, median, 95th), and the
coefficient-weighted redshift histogram alongside the full fitted population.
The template panels use linear axes capped at 12; histogram axes are uncapped.
This diagnostic uses real quality-passing fitted coefficients, not KDE draws.

The component spectra are normalized
to unit integrated flux over 3600--4200 Angstrom and the coefficients are
rescaled exactly, so this convention does not change any reconstructed spectrum.
Because the rest-frame matrix retains the observed DESI `f_lambda` values while
NumVisits applies `1 / (1 + z)` during redshifting, stored coefficient scales
also include `(1 + z)`. NumVisits should use `flux_unit_scale: 1.0e-17` to apply
the DESI coadd FLUX unit conversion.

The build also writes an explicit KDE-backed `prior_args.yaml` beside the KDE.
It contains the resolved prior/template directories, seven ILR shape parameters
for K=8, `log_c_scale`, `z`, and `flux_unit_scale: 1.0e-17`. This is useful for
custom output paths and immediate KDE tests. The standard DESI8 runtime instead
uses `template_source: desi8` in
`experiments/num_visits/prior_args_empirical.yaml`, which resolves the standard
scratch paths and currently selects `density_type: flow`. Train both prior-flow
spaces after the KDE build before using that default.

## Signed-data factorization evaluation

DESI coadds contain statistically valid negative flux measurements even though
the latent spectra are nonnegative. The method comparison keeps those values
and tests two optimizers for the same inverse-variance-weighted objective:

- ANLS, meaning alternating exact inverse-variance-weighted NNLS block solves;
  and
- Green & Bailey's Nearly-NMF multiplicative updates.

If it was not installed for the production build above, install the pinned
Nearly-NMF implementation before running this comparison:

```bash
pip install -e '.[desi-basis]'
```

The generated basis, template bank, and KDE artifacts do not require
`nearly_nmf` merely to be loaded by NumVisits; it is a build-time dependency.

Then run, for example:

```bash
python -m bedcosmo.num_visits.empirical.desi.evaluate_factorization_methods \
  --training-matrix /path/to/desi_rest_frame_training_matrix.npz \
  --output-dir /path/to/factorization_evaluation \
  --max-spectra 3000 \
  --ranks 4 6 8 \
  --starts 5 \
  --polish-selected
```

The evaluator uses a fixed 70/15/15 train/validation/test split and shared
strictly positive initial factors. No basis smoothing is applied. Validation
selects one initialization for each method and rank; only that selected solution
is evaluated on the test spectra. Held-out coefficients are always inferred by
the same SciPy NNLS solver so the comparison isolates the learned bases.

### Plot measured coverage and template support

```bash
python -m bedcosmo.num_visits.empirical.diagnostics_plots coverage \
  --training-matrix /path/to/measured/desi_rest_frame_training_matrix.npz \
  --prior-dir /path/to/completed/build \
  --output /path/to/desi-coverage.png
```

The heatmap and contributor curves count measured pixels only. A completed
build overlays the actual learned support and the extended template limits;
extrapolation does not add contributors to these curves. Without `--prior-dir`,
the plot uses the default 70% seed-42 split and a threshold of 100, showing the
training population's redshift bounds. `--population-matrix` can show the full
measured DESI population in the heatmap while contributor counts use only the
selected training population; that matrix must also be measured-only.
