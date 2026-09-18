# Direct DESI spectral basis

This package is a new empirical-prior source parallel to the EAZY template
pipeline. It learns nonnegative rest-frame component spectra from DESI B/R/Z
coadd fluxes rather than selecting or combining EAZY templates.

The current implementation uses an existing fit table only as a manifest of `TARGETID`,
HEALPix, redshift, and the quality-selected population. EAZY coefficient and
template columns are never read by the basis fitter. A later input adapter can
replace this manifest with a selection made directly from Redrock.

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

Run a bounded pilot with:

```bash
python -m bedcosmo.num_visits.empirical.desi_basis.fit_basis \
  --manifest "$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/desi_eazy_empirical_weights.csv" \
  --max-spectra 1500 \
  --ranks 2 3 4 5 6
```

Use `--max-spectra 0` to load the complete manifest. For example, the full
12,531-spectrum rank scan used during development was run with
`--max-spectra 0 --ranks 2 3 4 5 6 7 8 9 10 11 12`.

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

## Build the K=8 NumVisits prior

`build_prior` performs the production build in one command. It selects the best
of several Nearly-NMF initializations on validation spectra, reports the result
on an untouched test split, and then refits the selected basis on all spectra.
The basis is learned from the full quality-selected DESI population. Only the
coefficient/redshift rows used to train the prior are restricted to the default
`0.21 <= z <= 1.28`, where the learned rest-frame support covers the full
tabulated LSST `ugrizy` bandpasses without endpoint extrapolation.

```bash
python -m bedcosmo.num_visits.empirical.desi_basis.build_prior \
  --training-matrix /path/to/full12531/desi_rest_frame_training_matrix.npz \
  --rank 8
```

With no path overrides, the prior artifacts are written to
`$SCRATCH/bedcosmo/num_visits/empirical_prior/desi8/` and the learned template
bank to `$SCRATCH/bedcosmo/num_visits/spectral_templates/desi8/`. This keeps the direct
DESI components separate from the cached EAZY templates under
`$SCRATCH/bedcosmo/eazy/templates/`.

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
python -m bedcosmo.num_visits.empirical.desi_basis.evaluate_factorization_methods \
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
