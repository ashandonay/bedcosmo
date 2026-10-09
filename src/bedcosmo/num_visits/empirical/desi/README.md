# Direct DESI spectral basis

This package is a new empirical-prior source parallel to the EAZY template
pipeline. It learns nonnegative rest-frame component spectra from DESI B/R/Z
coadd fluxes rather than selecting or combining EAZY templates.

The DESI sample is selected directly from each patch's Redrock `REDSHIFTS`
table and coadd `FIBERMAP`; it does not use an EAZY fit table or EAZY selection.
The defaults retain Redrock `GALAXY` rows with finite `z >= 0.01`, `ZWARN == 0`,
a matching coadd target, and at least 100 usable pixels on the rest-frame grid.

The training path:

1. reads DESI flux, inverse variance, and masks from the coadds;
2. shifts valid pixels to the rest frame and bins them by inverse variance;
3. removes one robust multiplicative scale per object;
4. trims unsupported exterior bins with a permissive 5% relative-ivar cut,
   then extends each spectrum beyond its retained blue and red endpoints using a
   selected edge method, only to that galaxy’s LSST limits at its redshift;
   inferred bins receive 10% of the endpoint window's
   median weight;
5. preserves inverse variance in those normalized-flux units so high-S/N
   spectra determine component shapes more strongly than noisy spectra;
6. fits nonnegative components with alternating weighted NNLS while treating
   unobserved wavelengths as missing, not zero;
7. evaluates reconstruction on a fixed held-out galaxy sample; and
8. saves simplex-normalized component coefficients for eventual prior fitting.

First create the direct-DESI sample manifests and rest-frame matrix. This is
the only step `build_prior` depends on; it has no rank argument:

```bash
python -m bedcosmo.num_visits.empirical.desi.build_matrix \
  --max-spectra 0 \
  --z-min 0.3 --z-max 1.4 \
  --edge-extrapolation constant \
  --output-dir "$SCRATCH/bedcosmo/num_visits/desi_training_data_extrapolated"
```

Before fitting either edge, matrix building applies a fixed permissive quality
cut: use 5% of the median positive inverse variance 50–300 **observed-frame**
Angstroms inward from the original endpoint, and retain the first run of three
consecutive grid bins passing that threshold. Trim only bins outside the
resulting endpoints. Retained fluxes, weights, and internal gaps are unchanged.
If fewer than three reference bins exist, retain that endpoint; if a supported
reference exists but no three-bin run passes, reject the spectrum. Apply the
minimum-good-pixel requirement after trimming and before extrapolation.
The rule is recorded as `edge_quality_cut` in matrix provenance. It is a
conservative working choice, not an optimized or validated extrapolation cut.

`--edge-extrapolation constant` (the default) extends each retained endpoint with the
mean of the median-smoothed spectrum over the nearest 100 Angstroms.
`--edge-extrapolation linear` estimates the broadband continuum over the
nearest 500 Angstroms: it median-smooths the spectrum, takes 50-Angstrom median
bins, and fits a weighted line while iteratively clipping outlying bins. It
then continues that local continuum beyond the DESI edge. Long extrapolations
remain uncertain, so inspect the output before using it for factorization. The
selected method is recorded in `desi_training_matrix_provenance.json`.

`--edge-extrapolation powerlaw` fits a positive continuum
`f_lambda = A * (lambda / lambda_edge)**alpha` separately at each edge, over
the nearest 500 rest-frame Angstroms. It fits measured flux directly, including
negative measurements, with inverse-variance weighting and a soft-L1 robust
loss to reduce the influence of spectral lines and outliers. The amplitude is
parameterized as `exp(log_A)`; flux data are never log-transformed or clipped.
The fitted continuum is continued into the missing wavelengths. Fit failures
raise an error. After the quality cut, this option does not alter retained measured pixels or internal gaps.
The default grid covers full LSST bandpasses at every selected galaxy redshift.
`--wave-min` and `--wave-max` can request broader bounds; bounds excluding that
LSST coverage are rejected. Constant remains the default;
power-law continuation is an extrapolation assumption to assess on held-out
measured wavelength regions before a production NMF build.

The default output is:

```text
$SCRATCH/bedcosmo/num_visits/desi_training_data_extrapolated/
├── desi_candidate_manifest.csv
├── desi_sample_manifest.csv
├── desi_rest_frame_training_matrix.npz
├── desi_training_matrix_provenance.json
└── rest_wavelength_coverage.csv
```

The candidate manifest records the shared catalog-level population passing the
Redrock/FIBERMAP cuts. Both EAZY prior builds and direct-DESI basis builds use
this population. EAZY fits its native observed-frame coadd pixels; the direct
DESI path additionally creates the rest-frame matrix. The sample manifest is
the final direct-basis subset with enough usable matrix pixels. Pass
`--manifest /path/to/table.csv` only when deliberately overriding the direct
selection with a table containing `targetid`, `healpix`, and `z`.

The default `--max-spectra 1500` is useful for a bounded pilot; set it to zero
for a production matrix.

### Optional: compare ranks

`compare_ranks` is an exploratory diagnostic for choosing a rank. It reads the
saved matrix, fits a quick weighted NMF per rank on a training split, and
reports held-out spectral and LSST-color RMS. Its bases are not used by
`build_prior`, which learns its own production basis:

```bash
python -m bedcosmo.num_visits.empirical.desi.compare_ranks \
  --ranks 2 3 4 5 6 7 8 9 10 11 12
```

It writes `rank_comparison.csv`, `desi_basis_rank{K}.csv`,
`desi_basis_coefficients.csv`, `rank_comparison_support.csv`,
`rank_comparison_provenance.json`, and `desi_basis_rank_comparison.png` beside
the matrix (override with `--output-dir`).

DESI rest-frame wavelength support is strongly redshift-dependent. By default,
the usual contributor threshold is sized for rank 10 with ten spectra per
component (100 per wavelength bin). `--minimum-wavelength-contributors`
overrides that count. `build_matrix --z-min 0.3 --z-max 1.4` requests the
**final prior** interval, not a hard training-catalog cut. The builder starts
with galaxies in that interval and grows the low- and high-redshift buffers
independently. At each step it adds a batch of nearby galaxies on the side
covering the most currently deficient wavelength columns, rechecking the exact
planned training split. The final batch is refined galaxy by galaxy. It stops
when all required LSST wavelength bins have at least 100 training contributors;
the two buffers need not have the same redshift width. Those additional galaxies train the NMF basis but are excluded
from the prior population. Each galaxy still receives only its own LSST edge
extensions; the buffer does not extend every spectrum to a common rectangle.

`--train-fraction` (0.7), `--split-seed` (42), and
`--minimum-wavelength-contributors` (100) control this support planning.
The matrix stores `prior_redshift_bounds`; provenance records the actual
training bounds, buffer size, and minimum contributor count. If either prior
bound is omitted, it is inferred from threshold-supported coverage of the
available candidate sample, bounded by its measured redshift range. A manifest
or pilot sample must include enough galaxies outside an explicit requested
interval to supply the buffer. Insufficient available coverage raises an error;
the requested prior interval is never silently narrowed. Contributor counts
include both measured and downweighted extrapolated bins. Fluxes in the
validation/test splits are not used to learn templates.

`build_prior` reads the saved requested bounds by default and rechecks coverage
on its actual training split and contributor threshold. Changing those settings
can require rebuilding the matrix with a different buffer.

The candidate rest-frame grid covers both the selected coadds' valid pixels
(finite wavelength, flux and positive inverse variance, with zero mask) and
all full tabulated LSST `ugrizy` bandpasses over the selected redshift range.
For selected extrema `z_min` and `z_max`, the LSST requirements are
`lambda_min <= lambda_LSST_blue / (1 + z_max)` and
`lambda_max >= lambda_LSST_red / (1 + z_min)`. Taking the union with measured
DESI coverage preserves the original data used to estimate endpoint levels.
Bounds are rounded outward to the 10-Angstrom spacing (`--wave-step`).
Explicit `--wave-min` and `--wave-max` overrides must still cover those LSST
limits; nonaligned upper bounds are rounded outward to the next grid bin.
Within that common grid, missing exterior bins are filled only over each
individual galaxy's LSST interval, from `lambda_LSST_blue / (1 + z)` to
`lambda_LSST_red / (1 + z)`. One grid center bracketing each exact endpoint is
included so interpolation covers the whole filter. Exterior bins beyond that
interval retain zero weight. Measured DESI pixels outside LSST demand remain
measured, and internal masked gaps remain missing.
The default constant endpoint level is the mean of a median-filtered spectrum
over the nearest 100 Angstroms. The optional linear mode fits a robust,
inverse-variance-weighted continuum to 50-Angstrom median bins over the nearest
500 Angstroms, with iterative outlier clipping. Inferred bins are downweighted
to 10% of the measured edge window's median inverse variance. Minimum-good-pixel
selection counts retained measured pixels only. The grid now includes LSST demand as well as valid DESI coverage. Existing
saved matrices retain their original grid and values, so regenerate the matrix
before building a basis with edge extrapolation. This includes matrices built
with the earlier DESI-only default grid.

The matrix's `rest_wavelength_coverage.csv` reports contributor counts, the
catalog-wide observed fraction, and an LSST-demand-weighted conditional
coverage evaluated at each object's redshift. `compare_ranks` writes its
training-split contributor counts and selected support separately.
Internal missing pixels retain zero weight during factorization. The prior
redshift selection still checks that the learned rest-frame grid spans the
full tabulated LSST `ugrizy` bandpasses.

## Build a rank-K NumVisits prior

`build_prior` performs the production build in one command. It selects the best
of several Nearly-NMF initializations on validation spectra, reports the result
on an untouched test split, and then refits the selected basis on all spectra.
The basis is learned from the full buffered, quality-selected DESI population.
Only coefficient/redshift rows inside the requested matrix
`prior_redshift_bounds` are used to fit the prior. The learned wavelength support
must span all full tabulated LSST bandpasses throughout that exact interval.
`--prior-z-min` and `--prior-z-max` override the saved request; unsupported bounds
are rejected. Resolved bounds are printed and recorded in build provenance.
Existing prior files are not changed automatically. `--prior-only` reuses the
saved prior bounds unless explicitly overridden and does not retrain templates.
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
KDEs, and diagnostic triangles in the existing build. It loads `desi_basis.npz`
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
`$SCRATCH/bedcosmo/num_visits/desi_training_data_extrapolated/desi_rest_frame_training_matrix.npz`.
Build that matrix with the command above. The original `desi_training_data`
matrix is preserved. Full builds require a newly generated matrix containing
`prior_redshift_bounds`; regenerate older matrices before using them.
`--prior-only` continues to use the matrix recorded in the existing basis provenance.

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

### Plot matrix coverage and the redshift buffer

```bash
python -m bedcosmo.num_visits.empirical.diagnostics_plots coverage \
  --training-matrix "$SCRATCH/bedcosmo/num_visits/desi_training_data_extrapolated/desi_rest_frame_training_matrix.npz" \
  --output /path/to/desi-coverage.png
```

The heatmap marks requested prior bounds and actual buffered training bounds,
with labels showing each redshift extension and its galaxy count. The bottom
panel compares wavelength contributor counts for the requested range (purple
dash-dot), the extended range in the training split (green solid), and all
selected splits (gray). Both range labels give their redshift bounds. The
requested-only comparison uses its own split with the same seed and training
fraction. The middle panels show how the low-redshift buffer supplies the required red wavelength
endpoint and how the high-redshift buffer supplies the blue endpoint. They hold
the final training split fixed and accumulate contributors as each buffer is
included, with the threshold and selected redshift bounds marked. These curves
explain the final sample, rather than replaying the search (which recomputes the
split). The build checks every required wavelength, not only the endpoints.
Counts include measured and downweighted extrapolated bins.

Use `--population-matrix /path/to/full_population_matrix.npz` to show the full
quality-selected DESI population in the heatmap, including galaxies outside the
buffer. The requested and buffered boundaries remain overlaid; lower-panel
counts continue to use only the selected training matrix. Without this argument,
the heatmap shows the training matrix population.
