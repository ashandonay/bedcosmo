# Empirical galaxy SED prior (`sed_prior`)

This package builds and samples a **14-dimensional empirical prior** over galaxy spectral energy distributions for the `num_visits` BED experiment. It fits EAZY rest-frame templates to DESI HEALPix coadd spectra (NNLS), pools fits across patches, trains a masked Gaussian KDE on \((a_k, \log s, z)\), and provides utilities to draw prior samples and reconstruct template coefficients.

`NumVisits` in `../experiment.py` still defaults to a **blackbody** SED; wiring this prior into Pyro is the next integration step.

---

## Directory layout

| Script | Role |
|--------|------|
| `desi_get_dr_subset.py` | Download DESI DR1 spectra for selected HEALPix patches |
| `fit_eazy_weights_to_desi.py` | Fit template coefficients per galaxy; write weights CSV + diagnostics |
| `combine_healpix_weights.py` | Concatenate per-patch CSVs into one training table |
| `compare_healpix_prior_params.py` | Overlay / compare prior coordinates across HEALPix |
| `build_empirical_sed_prior_kde.py` | Train KDE, save `sed_prior_kde.joblib`, sample prior draws |
| `run_healpix_fits.sh` | Batch NNLS fits over the default HEALPix list |
| `generate_template_sed_examples.py` | Legacy: download EAZY templates, synthetic softmax mixtures (not production) |

**Data paths (typical):**

- EAZY templates: `~/data/num_visits/eazy/`
- DESI tiny DR1: `~/data/desi/tiny_dr1/`
- Scratch outputs: `~/scratch/bedcosmo/desi_eazy_hp{HEALPIX}/`, combined `~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/`

**Analysis notebook:** `experiments/num_visits/notebooks/empircal_prior.ipynb`

---

## Production prior: 14D coordinates

For \(K=12\) EAZY templates, each prior-quality galaxy is summarized by **14 numbers**:

| Index | Symbol | CSV / feature | Meaning |
|-------|--------|---------------|---------|
| 1–12 | \(a_1,\ldots,a_{12}\) | `a1` … `a12` | Template mixture **shape** (normalized weights) |
| 13 | \(\log s\) | `log_c_scale` | \(\log s = \log\sum_k \|c_k\|\) (overall amplitude) |
| 14 | \(z\) | `z` | Redshift (from DESI redrock for the fit; also a KDE feature) |

Raw fitted amplitudes are stored as `c1` … `cK`. **Reconstruction** (rest-frame SED):

\[
c_k = e^{\log s}\, a_k,
\qquad
f_\mathrm{rest}(\lambda) \propto \sum_k c_k\, T_k(\lambda).
\]

**NNLS** (recommended): \(c_k \ge 0\), so \(a_k \ge 0\) and \(\sum_k a_k = 1\). Many \(a_k\) are **exactly zero** (inactive templates). **WLS** allows signed \(c_k\); not used for the production prior.

### End-to-end pipeline

```text
EAZY templates (~/data/num_visits/eazy/)
        ↓
desi_get_dr_subset.py          →  local DESI coadd + redrock FITS
        ↓
fit_eazy_weights_to_desi.py    →  desi_eazy_empirical_weights.csv (per HEALPix)
        ↓
combine_healpix_weights.py     →  pooled CSV (optional, multi-patch)
        ↓
build_empirical_sed_prior_kde.py →  sed_prior_kde.joblib
        ↓
sample_sed_prior()               →  draws in ℝ¹⁴ → SED → LSST (future NumVisits)
```

**Quality cut:** default `chi2/dof ≤ 1.2` (`quality_pass=True`). Rows failing the cut remain in the CSV and `dropped_fits.csv` but are excluded from training triangles and KDE training.

---

## Step 0: DESI data (`desi_get_dr_subset.py`)

Downloads DESI DR1 coadd and redrock FITS for HEALPix patches into a local tree matching the layout expected by the fit script:

```text
~/data/desi/tiny_dr1/
  spectro/redux/iron/healpix/main/dark/230/23040/
    coadd-main-dark-23040.fits
    redrock-main-dark-23040.fits
```

Useful flags: `--top-n-healpix N`, `--healpix ID ...`, `--skip-existing`. See script `--help` for tile vs healpix modes.

---

## Step 1: Fit EAZY weights to DESI (`fit_eazy_weights_to_desi.py`)

### Spectral model

Observed-frame flux:

\[
f_\mathrm{DESI}(\lambda_\mathrm{obs})
\approx
\sum_{k=1}^{K}
c_k\,
\frac{T_k(\lambda_\mathrm{obs}/(1+z))}{1+z},
\]

with DESI redrock redshift \(z\). Template amplitudes \(c_k\) are fit by minimizing weighted \(\chi^2\) on unmasked pixels.

### Normalization (default `--coeff-norm l1`)

\[
s = \sum_{j=1}^{K} |c_j|,
\qquad
a_k = \frac{c_k}{s}.
\]

- For NNLS, \(c_k \ge 0\) so \(s = \sum_j c_j\) and \(a_k\) lie on the **simplex** (\(\sum_k a_k = 1\)).
- KDE is trained on \((a_1,\ldots,a_K, \log s, z)\).

Optional `--coeff-norm max` uses \(s = \max_j |c_j|\); the default pipeline assumes **l1**.

### WLS vs NNLS

| | NNLS (`--fit-method nnls`) | WLS (`--fit-method wls`) |
|---|---------------------------|--------------------------|
| Constraint | \(c_k \ge 0\) | none (signed \(c_k\)) |
| Prior geometry | simplex + sparse zeros | signed L1 shell |
| SED interpretation | physical positive mixtures | can subtract templates |
| Production prior | **yes** | debugging / comparison only |

For Pyro / BED, use **NNLS** and build the KDE with `--fit-method nnls`.

### Per-galaxy workflow

1. Load EAZY templates from `~/data/num_visits/eazy/`.
2. Read HEALPix coadd + redrock (`--desi-dir`, `--healpix`).
3. Select `SPECTYPE == GALAXY`, `ZWARN == 0`, targets present in coadd.
4. Fit \(c_k\) (WLS or NNLS); compute \(a_k\), \(s\), \(\log s\).
5. Write `desi_eazy_empirical_weights.csv` and diagnostic plots.

### Example: single HEALPix

```bash
cd src/bedcosmo/num_visits/sed_prior

python fit_eazy_weights_to_desi.py \
  --desi-dir ~/data/desi/tiny_dr1 \
  --healpix 23040 \
  --fit-method nnls \
  --n-max 600 \
  --outdir ~/scratch/bedcosmo/desi_eazy_hp23040
```

### Multi-patch batch fit

```bash
./run_healpix_fits.sh
# Env: DESI_DIR, SCRATCH (default ~/scratch/bedcosmo), N_MAX=600
```

Default HEALPix list: 23040, 27257, 27245, 27259, 27247, 27256, 27258, 27344, 26282.

### Combine patches

```bash
python combine_healpix_weights.py \
  --scratch-base ~/scratch/bedcosmo \
  --out ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv
```

### Compare patches

```bash
python compare_healpix_prior_params.py \
  --scratch-base ~/scratch/bedcosmo \
  --outdir ~/scratch/bedcosmo/healpix_prior_comparison
```

### Fit outputs

| File | Contents |
|------|----------|
| `desi_eazy_empirical_weights.csv` | All fits: `a1..aK`, `c1..cK`, `log_c_scale`, `z`, `chi2_dof`, `quality_pass`, … |
| `dropped_fits.csv` | Quality failures |
| `prior_params_triangle.png` | Training \((a_k, \log s, z)\), quality-pass only |
| `coeffs_raw_triangle.png` | Raw \(c_k\) vs \(z\) |
| `spectrum_fit_examples.png` / `_flux.png` | Random spectrum + \(c_k\) bar charts |
| `chi2_dof_histogram.png` | Fit quality |

**Replot from CSV (`--plot-only`):** loads an existing weights table and regenerates triangles and spectrum plots (must pass `--healpix` matching the loaded coadd for spectrum panels).

```bash
python fit_eazy_weights_to_desi.py --plot-only --fit-method nnls \
  --outdir ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls \
  --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv \
  --desi-dir ~/data/desi/tiny_dr1 --healpix 23040 --plot-n-examples 6
```

### Key fit CLI flags

| Flag | Default | Notes |
|------|---------|--------|
| `--fit-method` | `nnls` | `wls` or `nnls` |
| `--coeff-norm` | `l1` | `l1` or `max` for \(a_k\) |
| `--max-chi2-dof` | `1.2` | prior-quality cut |
| `--n-max` | all candidates | subsample for speed |
| `--plot-only` | off | replot from existing CSV |
| `--no-triangle-plots` | off | skip corner plots |
| `--plot-n-examples` | `6` | spectrum example count (`0` disables) |

---

## Step 2: KDE prior (`build_empirical_sed_prior_kde.py`)

Turns the pooled weights CSV into a **smooth, sampleable** prior on \((a_k, \log s, z)\).

### Why masked KDE (not naive Gaussian KDE alone)

Gaussian KDE smooths interior mass but cannot place density on **boundaries** \(a_k = 0\). NNLS training rows have many exact zeros (inactive templates). After each draw:

1. Sample \(\mathbf{x} = (a_1,\ldots,a_K, \log s, z)\) from a global KDE (bandwidth **0.3** in standardized space by default).
2. Clip each coordinate to training min/max.
3. **Support mask:** pick a random training galaxy; set \(a_k = 0\) wherever that galaxy had \(a_k = 0\).
4. **Renormalize** active templates so \(\sum_k a_k = 1\).

Disable the mask with `--no-support-mask` (not recommended for NNLS). Optional `--bandwidth-rule scott` or `silverman` instead of fixed `0.3`.

### Build and sample

```bash
python build_empirical_sed_prior_kde.py \
  --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv \
  --fit-method nnls \
  --bandwidth 0.3 \
  --sample 10000 --seed 7 --plot-kde-triangle \
  --out ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/sed_prior_kde.joblib
```

### Artifacts

| File | Contents |
|------|----------|
| `sed_prior_kde.joblib` | sklearn KDE + scaler, `training_x`, bounds, metadata |
| `sed_prior_kde.json` | Human-readable metadata |
| `kde_samples_triangle.png` | Optional (`--plot-kde-triangle` with `--sample > 0`) |

### Python API

```python
from pathlib import Path

from bedcosmo.num_visits.sed_prior import (
    coeffs_from_sample_row,
    load_sed_prior_kde,
    sample_sed_prior,
    samples_to_coeffs,
)

artifact = load_sed_prior_kde(
    Path("~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/sed_prior_kde.joblib").expanduser()
)

x = sample_sed_prior(artifact, n_samples=5000, seed=0)  # shape (5000, 14)

n = artifact["n_templates"]
a, log_s, z = samples_to_coeffs(x, n)
c = np.exp(log_s)[:, None] * a

# single galaxy
c_one = coeffs_from_sample_row(a[0], float(log_s[0]))
```

Or add `sed_prior` to `sys.path` and import from `build_empirical_sed_prior_kde` directly (as in the notebook).

### KDE CLI flags

| Flag | Default | Notes |
|------|---------|--------|
| `--fit-method` | `nnls` | enables simplex projection + default support mask |
| `--bandwidth` | `0.3` | scaled-space KDE bandwidth |
| `--sample` | `0` | test draws after save |
| `--plot-kde-triangle` | off | corner plot of KDE samples |
| `--no-support-mask` | off | skip zeroing inactive \(a_k\) |

---

## Diagnostics summary

| Plot | Source |
|------|--------|
| `prior_params_triangle.png` | Training coordinates (fit script) |
| `coeffs_raw_triangle.png` | Raw \(c_k\) vs \(z\) (fit script) |
| `kde_samples_triangle.png` | KDE draws (build script) |
| `spectrum_fit_examples*.png` | DESI vs model (fit script, needs coadd) |
| `compact_triangle_by_healpix.png` | Cross-patch comparison |

Sharp edges in training triangles are expected (simplex facets, exact NNLS zeros), not a KDE bug.

---

## Legacy: `generate_template_sed_examples.py`

**Not** the production prior. Useful for:

- First-time download of EAZY templates into `~/data/num_visits/eazy/`
- Quick plots of **synthetic** softmax mixtures (uniform \(z\), no DESI, no sparsity)

```bash
python generate_template_sed_examples.py --n-samples 12 --outdir outputs/sed_examples
```

| | Synthetic script | Production 14D prior |
|---|------------------|----------------------|
| Weights | Dirichlet / softmax | DESI NNLS \(a_k\) |
| Scale | shape only | \(\log s\) in prior |
| \(z\) | uniform box | DESI + KDE on training \(z\) |
| Sparsity | none | many \(a_k = 0\) |
| Output | `sample_seds.npz` | CSV + `sed_prior_kde.joblib` |

---

## Toward `NumVisits` (not yet wired)

Target Pyro pattern:

```python
x = sample_sed_prior(artifact, 1, seed=...)[0]   # a1..a12, log_c_scale, z
# pyro.sample("sed_prior", ...)  # Delta or dequantized empirical
# build rest-frame SED from a_k, log_s, z and template bank T_k
# then existing visit / noise / cosmology likelihood
```

The blackbody path in `experiment.py` samples temperature \(T\) instead of \((a_k, \log s, z)\).

---

## Quick reference: recommended production settings

| Step | Setting |
|------|---------|
| Fit method | **NNLS** |
| Normalization | **L1** (`--coeff-norm l1`) |
| Quality cut | `chi2/dof ≤ 1.2` |
| KDE bandwidth | **0.3** (scaled space) |
| Sampling | `sample_sed_prior()` with default support mask |

---

## References

- EAZY templates: [gbrammer/eazy-photoz](https://github.com/gbrammer/eazy-photoz), default `templates/fsps_full/fsps_QSF_12_v3.param`
- Brammer, van Dokkum & Coppi (2008), EAZY
- DESI DR1 coadd + redrock (local paths under `~/data/desi/tiny_dr1`)
