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
4. preserves inverse variance in those normalized-flux units so high-S/N
   spectra determine component shapes more strongly than noisy spectra;
5. fits nonnegative components with alternating weighted NNLS while treating
   unobserved wavelengths as missing, not zero;
6. evaluates reconstruction on a fixed held-out galaxy sample; and
7. saves simplex-normalized component coefficients for eventual prior fitting.

First create the direct-DESI sample manifests and rest-frame matrix. This
example processes the complete selected population and evaluates rank 8 in the
exploratory basis diagnostic:

```bash
python -m bedcosmo.num_visits.empirical.desi.fit_basis \
  --max-spectra 0 \
  --ranks 8
```

The default output is:

```text
$SCRATCH/bedcosmo/num_visits/desi_samples/
├── desi_candidate_manifest.csv
├── desi_sample_manifest.csv
└── desi_rest_frame_training_matrix.npz
```

The candidate manifest records objects passing the Redrock/FIBERMAP cuts. The
sample manifest is the final subset with enough usable spectral pixels. Pass
`--manifest /path/to/table.csv` only when deliberately overriding the direct
selection with a table containing `targetid`, `healpix`, and `z`.

The default `--max-spectra 1500` is useful for a bounded pilot; set it to zero
for a production matrix. A development rank scan can use
`--ranks 2 3 4 5 6 7 8 9 10 11 12`.

DESI rest-frame wavelength support is strongly redshift-dependent. A global
catalog-fraction cutoff is inappropriate: low-redshift spectra supply the red
wavelengths needed by LSST at low redshift, while high-redshift spectra supply
the UV wavelengths needed at high redshift. Instead, the fitter retains the
largest contiguous interval having enough actual contributors to constrain
every wavelength column. By default it sizes this requirement for rank 10 and
requires ten observed spectra per component, or 100 contributors per bin.
`--minimum-wavelength-contributors` can override that explicit count.
Support is selected from the training split only; validation and test masks do
not influence the fitted wavelength interval.

The saved `rest_wavelength_coverage.csv` reports contributor counts, the
catalog-wide observed fraction, and an LSST-demand-weighted conditional
coverage evaluated at each object's redshift. Missing pixels always retain zero
weight during factorization. These components must not be extrapolated by
silently clamping their endpoint values; the portions of LSST `u` and `y`
outside DESI's observed-frame range still require an explicit external anchor.

## Build a rank-K NumVisits prior

`build_prior` performs the production build in one command. It selects the best
of several Nearly-NMF initializations on validation spectra, reports the result
on an untouched test split, and then refits the selected basis on all spectra.
The basis is learned from the full quality-selected DESI population. Only the
coefficient/redshift rows used to train the prior are restricted to the default
`0.21 <= z <= 1.28`, where the learned rest-frame support covers the full
tabulated LSST `ugrizy` bandpasses without endpoint extrapolation.

```bash
python -m bedcosmo.num_visits.empirical.desi.build_prior \
  --rank 8
```

With no `--training-matrix` override, this reads
`$SCRATCH/bedcosmo/num_visits/desi_samples/desi_rest_frame_training_matrix.npz`.

`--rank` controls both the number of learned spectral components and the
dimension of the generated prior. Because the default build name is `desi8`,
give other ranks their own build name. For example, the production K=4 build is:

```bash
python -m bedcosmo.num_visits.empirical.desi.build_prior \
  --rank 4 \
  --build-name empirical_prior/desi4
```

That command writes the prior artifacts under
`$SCRATCH/bedcosmo/num_visits/empirical_prior/desi4/` and the four-component
template bank under `$SCRATCH/bedcosmo/num_visits/spectral_templates/desi4/`.

With no path overrides, the prior artifacts are written to
`$SCRATCH/bedcosmo/num_visits/empirical_prior/desi8/` and the learned template
bank to `$SCRATCH/bedcosmo/num_visits/spectral_templates/desi8/`. EAZY banks
use the parallel `spectral_templates/eazy6/` and `spectral_templates/eazy12/`
directories with the same flat component-plus-parameter-file layout.

The command writes an EAZY-compatible component bank, the standard
`desi_eazy_empirical_weights.csv`, build provenance, native and gaussianized KDE
artifacts, and diagnostic triangle plots. The component spectra are normalized
to unit integrated flux over 3600--4200 Angstrom and the coefficients are
rescaled exactly, so this convention does not change any reconstructed spectrum.
Because the rest-frame matrix retains the observed DESI `f_lambda` values while
NumVisits applies `1 / (1 + z)` during redshifting, stored coefficient scales
also include `(1 + z)`. NumVisits should use `flux_unit_scale: 1.0e-17` to apply
the DESI coadd FLUX unit conversion.

The build also writes an explicit `prior_args.yaml` beside the KDE. It contains
the resolved prior/template directories, seven ILR shape parameters for K=8,
`log_c_scale`, `z`, and `flux_unit_scale: 1.0e-17`. Because these paths are
machine-specific, run the builder on the machine that will execute NumVisits
and pass that generated file as the empirical prior configuration.

## Signed-data factorization evaluation

DESI coadds contain statistically valid negative flux measurements even though
the latent spectra are nonnegative. The method comparison keeps those values
and tests two optimizers for the same inverse-variance-weighted objective:

- alternating exact nonnegative least squares (ANLS); and
- Green & Bailey's Nearly-NMF multiplicative updates.

Install the pinned implementation in the bedcosmo environment before fitting a
basis:

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
