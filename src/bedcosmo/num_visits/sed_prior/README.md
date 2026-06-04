# Empirical galaxy SED prior (`sed_prior`)

Build and sample an empirical prior over galaxy SEDs for the `num_visits` BED experiment: fit EAZY templates to DESI spectra (NNLS), pool HEALPix patches, train a **smooth CLR KDE**, and sample through a GPU prior pool in `NumVisits` (`cosmo_model: empirical`).

**Production parameterization (KDE v3):** centered log-ratios \(f_k = \log a_k - \mathrm{mean}_j\log a_j\) for \(K=12\) templates, plus \(\log s\) and \(z\) — **14 features** (`f1`…`f12`, `log_c_scale`, `z`). Legacy artifacts used raw weights, logits, or masked support projection.

---

## Directory layout

| File | Role |
|------|------|
| `desi_get_dr_subset.py` | Download DESI DR1 coadd + redrock for selected HEALPix patches |
| `fit_eazy_weights_to_desi.py` | Per-galaxy NNLS template fit → weights CSV + fit diagnostics |
| `combine_healpix_weights.py` | Concatenate per-patch CSVs into one training table |
| `run_healpix_fits.sh` | Batch fits over the default 9 HEALPix patches |
| `compare_healpix_prior_params.py` | Cross-patch overlays of prior coordinates |
| `build_empirical_sed_prior_kde.py` | Train KDE + gaussianizer → `sed_prior_kde.joblib` |
| `diagnostic_plots.py` | Post-build diagnostics (CLR triangles, redshift, SED examples, mag leakage) |
| `diagnose_transform_input.py` | NumVisits `transform_input` triangle diagnostics |
| `prior_sampler.py` | GPU pool of KDE feature rows for training |
| `simplex.py` | Weight ↔ logit / CLR maps (numpy + torch) |
| `templates.py` | Load EAZY template bank |

**Typical paths**

| What | Path |
|------|------|
| EAZY templates | `~/data/num_visits/eazy/` |
| DESI tiny DR1 | `~/data/desi/tiny_dr1/` |
| Per-patch fits | `~/scratch/desi_eazy_hp{HEALPIX}/` or `$SCRATCH/desi_eazy_hp{HEALPIX}/` |
| **Production prior build** | `~/scratch/bedcosmo/desi_eazy_empirical_prior_full/` |
| Training KDE path | [`experiments/num_visits/prior_args_empirical.yaml`](../../../../experiments/num_visits/prior_args_empirical.yaml) |

**Notebook:** `experiments/num_visits/notebooks/empircal_prior.ipynb`

---

## Production prior: CLR + smooth KDE

### Coordinates

CSV fits store **`a1`…`aK`**, **`c1`…`cK`**, **`log_c_scale`**, **`z`**.

The KDE is trained on **CLR** features after a small simplex floor:

1. \(a_k \leftarrow (a_k + \varepsilon) / \sum_j (a_j + \varepsilon)\) with default \(\varepsilon = 10^{-5}\)
2. \(f_k = \log a_k - \mathrm{mean}_j \log a_j\) (row sums to zero; **no** reference template)
3. KDE in \((f_1,\ldots,f_K, \log s, z)\); decode weights via softmax on CLR

For \(K=12\):

| Feature | Meaning |
|---------|---------|
| `f1`…`f12` | Centered log-ratios (symmetric; 12 numbers, \(K{-}1\) DOF) |
| `log_c_scale` | \(\log s = \log\sum_k \|c_k\|\) |
| `z` | DESI redrock redshift |

Rest-frame SED: \(c_k = e^{\log s}\, a_k\), \(f_\mathrm{rest}(\lambda) \propto \sum_k c_k T_k(\lambda)\).

**NNLS** (\(c_k \ge 0\)): many training \(a_k\) are **exactly zero** (inactive templates). The smooth CLR prior keeps tiny positive KDE mass on inactive directions instead of hard support masks.

### End-to-end pipeline

```text
EAZY templates (~/data/num_visits/eazy/)
        ↓
desi_get_dr_subset.py
        ↓
fit_eazy_weights_to_desi.py     →  desi_eazy_hp{HEALPIX}/desi_eazy_empirical_weights.csv
        ↓
combine_healpix_weights.py      →  desi_eazy_empirical_prior_full/desi_eazy_empirical_weights.csv
        ↓
build_empirical_sed_prior_kde.py →  sed_prior_kde.joblib (+ KDE build triangles)
        ↓
diagnostic_plots.py all         →  diagnostics/{clr_triangle,redshift_histograms,...}/
        ↓
NumVisits (empirical)           →  GPU prior pool → SED → LSST magnitudes
```

**Selection cuts (fit):** `SPECTYPE == GALAXY`, `ZWARN == 0`, default **`z >= 0.01`** (`--z-min`, `--no-z-min` to disable).

**Quality cut:** default `chi2/dof <= 1.2` (`quality_pass`). Failed rows stay in the CSV / `dropped_fits.csv` but are excluded from KDE training.

---

## Step 0: DESI data (`desi_get_dr_subset.py`)

Downloads coadd + redrock FITS into a tree matching the fit script:

```text
~/data/desi/tiny_dr1/spectro/redux/iron/healpix/main/dark/{prefix}/{healpix}/
  coadd-main-dark-{healpix}.fits
  redrock-main-dark-{healpix}.fits
```

Flags: `--healpix ID ...`, `--top-n-healpix N`, `--skip-existing`. See `--help` for tile vs HEALPix modes.

---

## Step 1: Fit EAZY weights (`fit_eazy_weights_to_desi.py`)

### Spectral model

\[
f_\mathrm{DESI}(\lambda_\mathrm{obs})
\approx
\sum_k c_k \,
\frac{T_k(\lambda_\mathrm{obs}/(1+z))}{1+z},
\]

with DESI \(z\). Minimize weighted \(\chi^2\) on unmasked pixels (NNLS: \(c_k \ge 0\)).

### Normalization (`--coeff-norm l1`, default)

\(s = \sum_j |c_j|\), \(a_k = c_k/s\). For NNLS, \(a\) is on the simplex.

### Single HEALPix

```bash
cd src/bedcosmo/num_visits/sed_prior

python fit_eazy_weights_to_desi.py \
  --desi-dir ~/data/desi/tiny_dr1 \
  --healpix 23040 \
  --fit-method nnls \
  --z-min 0.01 \
  --outdir ~/scratch/desi_eazy_hp23040
```

Omit `--n-max` to fit **all** passing candidates. Use `--n-max 600` only for quick tests.

### Multi-patch batch

```bash
./run_healpix_fits.sh
```

| Env var | Default | Notes |
|---------|---------|--------|
| `DESI_DIR` | `~/data/desi/tiny_dr1` | |
| `SCRATCH` | `~/scratch/bedcosmo` | Per-patch `desi_eazy_hp*` under this base |
| `Z_MIN` | `0.01` | Redshift floor at candidate selection |
| `N_MAX` | *(unset)* | If set, subsample per patch (e.g. `600`) |
| `FORCE` | `0` | `FORCE=1` refits even if CSV exists |
| `SEED` | `7` | |

Default HEALPix: `23040 27257 27245 27259 27247 27256 27258 27344 26282`.

The script disables heavy triangle plots (`--no-triangle-plots`) for batch speed; use `--plot-only` on one patch for visuals.

### Combine patches

```bash
python combine_healpix_weights.py \
  --scratch-base ~/scratch \
  --out ~/scratch/bedcosmo/desi_eazy_empirical_prior_full/desi_eazy_empirical_weights.csv
```

Use the same `--scratch-base` as `SCRATCH` in `run_healpix_fits.sh` (often `~/scratch`, not `~/scratch/bedcosmo`).

### Compare patches

```bash
python compare_healpix_prior_params.py \
  --scratch-base ~/scratch \
  --outdir ~/scratch/healpix_prior_comparison
```

### Fit outputs

| File | Contents |
|------|----------|
| `desi_eazy_empirical_weights.csv` | `a*`, `c*`, `log_c_scale`, `z`, `chi2_dof`, `quality_pass`, … |
| `dropped_fits.csv` | Quality failures |
| `prior_params_triangle.png` | Training \((a_k,\log s,z)\), quality-pass |
| `coeffs_raw_triangle.png` | Raw \(c_k\) vs \(z\) |
| `spectrum_fit_examples*.png` | Spectrum + \(c_k\) bars |
| `chi2_dof_histogram.png` | \(\chi^2/\mathrm{dof}\) |

**Replot only:**

```bash
python fit_eazy_weights_to_desi.py --plot-only --fit-method nnls \
  --outdir ~/scratch/desi_eazy_hp23040 \
  --weights-csv ~/scratch/desi_eazy_hp23040/desi_eazy_empirical_weights.csv \
  --desi-dir ~/data/desi/tiny_dr1 --healpix 23040
```

### Key fit flags

| Flag | Default | Notes |
|------|---------|--------|
| `--fit-method` | `nnls` | `wls` for debugging only |
| `--z-min` | `0.01` | `--no-z-min` disables |
| `--max-chi2-dof` | `1.2` | `quality_pass` |
| `--n-max` | all | Subsample cap |
| `--coeff-norm` | `l1` | |

---

## Step 2: KDE prior (`build_empirical_sed_prior_kde.py`)

### Recommended mode (defaults)

| Setting | Value |
|---------|--------|
| `--parameterization` | `clr` |
| `--support-mode` | `smooth` (no random NNLS support mask) |
| `--simplex-smoothing-eps` | `1e-5` |
| `--bandwidth` | `0.3` (scaled space) |
| `--z-min` | `0.01` |
| `--gaussianizer-fit-source` | `kde` (100k reference draws) |
| `--gaussianizer-whitening` | `cholesky` |
| `--sample` | `20000` (post-save diagnostic triangles) |

Legacy **`--support-mode masked`** applies a random training-galaxy zero pattern after sampling; large LSST mag shifts — not recommended for production.

### Build

```bash
python build_empirical_sed_prior_kde.py \
  --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_full/desi_eazy_empirical_weights.csv \
  --out ~/scratch/bedcosmo/desi_eazy_empirical_prior_full/sed_prior_kde.joblib
```

**Conda:** use `bedcosmo` (needs `torch` for gaussianizer). Fits use `sedprior`.

### Artifacts

| File | Contents |
|------|----------|
| `sed_prior_kde.joblib` | KDE, scaler, `training_x`, gaussianizer, bounds, metadata |
| `sed_prior_kde.json` | Metadata summary |
| `kde_samples_*.png` | Diagnostic triangles when `--sample > 0` |
| `training_gaussianized_triangle.png` | Gaussianized training/coords |

### Python API

```python
from pathlib import Path

from bedcosmo.num_visits.sed_prior import (
    load_sed_prior_kde,
    sample_sed_prior,
    samples_to_coeffs,
)

artifact = load_sed_prior_kde(
    Path("~/scratch/bedcosmo/desi_eazy_empirical_prior_full/sed_prior_kde.joblib").expanduser()
)
x = sample_sed_prior(artifact, n_samples=5000, seed=0)  # (N, 14) CLR features
n = artifact["n_templates"]
a, log_s, z = samples_to_coeffs(x, n, parameterization="clr")
```

---

## Diagnostics (`diagnostic_plots.py`)

Not part of the build pipeline. All subcommands take a **prior build directory** and write under **`diagnostics/<name>/`** (override with `--outdir`).

```bash
python -m bedcosmo.num_visits.sed_prior.diagnostic_plots all \
  --prior-dir ~/scratch/bedcosmo/desi_eazy_empirical_prior_full
```

| Subcommand | Output subdir | What it checks |
|------------|---------------|----------------|
| `clr-triangle` | `clr_triangle/` | Low-weight template highlighting in KDE (and optional training) draws |
| `redshift-histograms` | `redshift_histograms/` | Redrock GALAXY vs STAR vs weights CSV; dashed line at KDE `z_min` |
| `sed-examples` | `sed_examples/` | NumVisits SEDs, LSST mags, weight heatmap, `empirical_seds.npz` |
| `mag-leakage` | `mag_leakage/` | Smooth KDE vs threshold-zeroed vs masked weights → \(\Delta m\) |
| `all` | all of the above | One-shot |

**Mag leakage (typical):** thresholding inactive templates (\(a_k \le 10^{-4}\)) changes LSST mags at the **sub-millimag** level; legacy masked support can shift by \(\sim 1\) mag.

Individual runs:

```bash
python -m bedcosmo.num_visits.sed_prior.diagnostic_plots clr-triangle \
  --prior-dir ~/scratch/bedcosmo/desi_eazy_empirical_prior_full --also-training
```

---

## `NumVisits` integration (`empirical`)

### Config

- **Parameters:** `f1`…`f12`, `log_c_scale`, `z` in [`models.yaml`](../../../../experiments/num_visits/models.yaml)
- **KDE path:** absolute path in [`prior_args_empirical.yaml`](../../../../experiments/num_visits/prior_args_empirical.yaml):

```yaml
prior_kde_path: "/home/ashandonay/scratch/bedcosmo/desi_eazy_empirical_prior_full/sed_prior_kde.joblib"
```

- **Training:** [`train_args.yaml`](../../../../experiments/num_visits/train_args.yaml) `empirical` block:

```yaml
transform_cosmo_params: [log_c_scale, z]   # bijector on these only
logit_flow_scale: 8.0                    # f_k → H*tanh(f/H) for the NF
```

`NumVisits` decodes CLR rows → simplex \(a_k\) → `_calculate_magnitudes`. Rebuild the KDE after changing parameterization or `feature_names`.

### `transform_input` diagnostics

```bash
python -m bedcosmo.num_visits.sed_prior.diagnose_transform_input \
  --kde-path ~/scratch/bedcosmo/desi_eazy_empirical_prior_full/sed_prior_kde.joblib \
  --outdir ~/scratch/bedcosmo/desi_eazy_empirical_prior_full/transform_diagnostic
```

Writes physical \((a_k, \log s, z)\) and post-transform Gaussian triangles.

### Submit training

```bash
./submit.sh train num_visits empirical
```

---

## Legacy / comparison notes

### Parameterizations

| Mode | Features | Status |
|------|----------|--------|
| **CLR + smooth** | `f1`…`fK`, `log_s`, `z` | **Production (v3)** |
| Logits | `f1`…`f_{K-1}`, `log_s`, `z` | Supported; not default |
| Raw weights | `a1`…`aK`, `log_s`, `z` | Legacy v1 |
| Masked support | post-sample zero mask | Legacy; distorts mags |

Use `diagnostic_plots sed-examples` for production-pipeline SED checks (KDE-sampled weights, not synthetic mixtures).

### Older scratch trees

| Directory | Role |
|-----------|------|
| `desi_eazy_empirical_prior_nnls` | Earlier 600/patch capped build |
| `desi_eazy_empirical_prior_full` | **Current** all-candidate multi-patch build |

---

## Quick reference

| Step | Setting |
|------|---------|
| Fit | **NNLS**, **L1** norm, **`z_min=0.01`**, fit **all** candidates (no `N_MAX`) |
| Combine | `desi_eazy_empirical_prior_full` |
| KDE | **CLR**, **smooth**, \(\varepsilon=10^{-5}\), bandwidth **0.3** |
| Training | `prior_kde_path` → `_full/sed_prior_kde.joblib` |
| Diagnostics | `diagnostic_plots all --prior-dir .../_full` |

---

## References

- EAZY: [gbrammer/eazy-photoz](https://github.com/gbrammer/eazy-photoz), default `templates/fsps_full/fsps_QSF_12_v3.param`
- DESI DR1 coadd + redrock under `~/data/desi/tiny_dr1`
