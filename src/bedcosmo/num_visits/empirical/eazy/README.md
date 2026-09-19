# EAZY spectral-template prior

This package builds the NumVisits empirical SED prior by fitting nonnegative
mixtures of EAZY templates to DESI galaxy spectra. It is the EAZY-template
counterpart to the directly learned [DESI basis](../desi/README.md).

The production workflow:

1. downloads the requested EAZY template bank when it is first needed;
2. loads DESI B/R/Z coadd spectra and their inverse variances and masks;
3. fixes each galaxy redshift to its DESI Redrock value;
4. fits nonnegative template coefficients to each spectrum;
5. separates the normalized template mixture from its overall flux scale;
6. applies the configured spectrum-quality cuts across the selected HEALPix
   patches; and
7. trains native and gaussianized empirical priors over the ILR mixture
   coordinates, `log_c_scale`, and redshift.

## Spectral-template layout

Downloaded EAZY banks and learned DESI banks share one root:

```text
$SCRATCH/bedcosmo/num_visits/spectral_templates/
├── eazy6/
│   ├── eazy6.param
│   └── component_01.dat ... component_06.dat
├── eazy12/
│   ├── eazy12.param
│   ├── component_01.dat ... component_12.dat
│   └── reduced/
└── desiK/
    ├── desiK.param
    └── component_*.dat
```

The upstream EAZY repository uses several nested template directories. The
loader downloads those spectra and materializes them into the flat
component-bank layout above. Parameter-file paths are relative to the shared
`spectral_templates/` root:

| Source | Parameter file | Components |
|---|---|---:|
| EAZY12 | `eazy12/eazy12.param` | 12 |
| EAZY6 | `eazy6/eazy6.param` | 6 |

No separate download command is required. Calling the full builder or loading
one of these banks downloads any missing files. Use `overwrite=True` in the
Python loader when the local files should be refreshed from upstream.

Both standard EAZY banks are normalized to unit integrated template flux over
4000--8000 Angstrom. The fitted `log_c_scale` parameter carries the removed
overall amplitude, so this normalization convention does not restrict galaxy
brightness.

## Build a complete EAZY12 prior

The end-to-end entry point downloads missing inputs, fits every requested
HEALPix patch, combines the successful fits, records build provenance, and
trains the KDE artifacts:

```bash
python -m bedcosmo.num_visits.empirical.eazy.build_prior
```

Its principal outputs are written under:

```text
$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/
├── healpix/hp*/desi_eazy_empirical_weights.csv
├── desi_eazy_empirical_weights.csv
├── build_provenance.json
├── sed_prior_kde_native.joblib
├── sed_prior_kde_gaussianized.joblib
└── build.log
```

Useful bounded or partial builds include:

```bash
# Small pipeline test
python -m bedcosmo.num_visits.empirical.eazy.build_prior \
  --build-name empirical_prior/eazy12-test \
  --n-max 600

# Fit one patch without training the final KDE
python -m bedcosmo.num_visits.empirical.eazy.build_prior \
  --healpix 23040 \
  --skip-kde
```

## Build an EAZY6 prior

Select the six-component bank; the builder derives both its template path and
its `empirical_prior/eazy6` output directory:

```bash
python -m bedcosmo.num_visits.empirical.eazy.build_prior --template-source eazy6
```

At runtime, `template_source: eazy6` or `template_source: eazy12` in
`experiments/num_visits/prior_args_empirical.yaml` resolves the matching prior,
template bank, parameter count, and normalization settings.

DESI coadd flux is stored in units of
`10^-17 erg / (s cm^2 Angstrom)`. NumVisits therefore uses
`flux_unit_scale: 1.0e-17` when converting either EAZY/DESI fitted-prior source
to physical flux units.

## Individual tools

| Module or script | Purpose |
|---|---|
| `build_prior` | Run the complete fit, combine, provenance, and KDE workflow. |
| `fit_eazy_weights_to_desi` | Fit or plot a single DESI HEALPix patch. |
| `combine_healpix_weights` | Combine patch-level coefficient tables. |
| `compare_healpix_prior_params` | Compare coefficient distributions between patches. |
| `run_healpix_fits.sh` | Run patch fits sequentially with shell-configurable settings. |
| `run_healpix_diagnostic_plots.sh` | Regenerate plots from existing patch fits. |

The shell helpers are optional; `build_prior` is the recommended entry point
for a reproducible full build.

## Reduced EAZY banks

Subset searches, family discovery, spectral-feature diagnostics, and reduced
prior construction live under [`reduced/`](reduced/README.md). A generated bank
is stored beside its parent source, for example:

```text
spectral_templates/eazy12/reduced/eazy12_t1-t7.param
```

The reduced parameter file references the existing EAZY12 component files, so
the spectra themselves are not duplicated.

For shared KDE/flow runtime details, artifact snapshotting, and the broader
NumVisits empirical-prior architecture, see the parent
[empirical-prior README](../README.md).
