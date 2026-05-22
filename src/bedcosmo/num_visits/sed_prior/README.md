# Empirical galaxy SED prior (`sed_prior`)

This package builds and samples an empirical prior over galaxy SEDs for the `num_visits` BED experiment. It fits EAZY templates to DESI spectra (NNLS), pools fits, trains a masked Gaussian KDE, and provides GPU pool sampling for `NumVisits` (`eazy_kde`).

**Default KDE coordinates (v2):** \(K-1\) log-ratios \(f_k = \log(a_k/a_K)\) on the template simplex, plus \(\log s\) and \(z\) — **13 dimensions** for \(K=12\). Legacy v1 artifacts store \(a_1,\ldots,a_K\) directly (14D).

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

## Production prior: simplex logits + \(\log s\), \(z\)

CSV fits still store **`a1`…`aK`** (normalized weights). The KDE is trained on **`f1`…`f_{K-1}`** with \(f_k = \log(a_k/a_K)\) and reference template \(K\); weights recover as \(a = \mathrm{softmax}(f_1,\ldots,f_{K-1},0)\), so \(a_k\ge 0\) and \(\sum_k a_k = 1\) by construction.

For \(K=12\) templates, KDE features are **13 numbers**:

| Index | Symbol | Feature | Meaning |
|-------|--------|---------|---------|
| 1–11 | \(f_1,\ldots,f_{11}\) | `f1` … `f11` | Log-ratios vs template 12 |
| 12 | \(\log s\) | `log_c_scale` | \(\log s = \log\sum_k \|c_k\|\) |
| 13 | \(z\) | `z` | Redshift |

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
sample_sed_prior()               →  draws in ℝ¹³ (logits) → simplex \(a_k\) → SED → LSST
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

## `NumVisits` integration (`eazy_kde` model)

Training uses `cosmo_model: eazy_kde` with **13 cosmology parameters** (`f1`…`f11`, `log_c_scale`, `z`) in [`models.yaml`](../../../../experiments/num_visits/models.yaml).

Rebuild the KDE after changing parameterization:

```bash
python -m bedcosmo.num_visits.sed_prior.build_empirical_sed_prior_kde \
  --weights-csv ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv \
  --parameterization logits
```

- **`simplex.py`**: `weights_to_logits` / `logits_to_weights` (numpy + torch).
- **`prior_sampler.py`**: GPU pool of KDE rows in logit space.
- **`NumVisits`**: maps \(f_k\) → \(a_k\) via softmax before `_calculate_magnitudes_eazy`.

Legacy v1 joblib files (`a1`…`a12` features) must be rebuilt to match `models.yaml`.

Blackbody models (`bb`, `bb_temp`) are unchanged.

### `transform_input` and logit coordinates

Marginal CDF → Gaussian on every `f_k` is a poor match when logits are **multimodal**
(spike at KDE clip bounds plus an “active template” mode). The Gaussian triangle then shows
heavy tails at ±6.36 and distorted 2D panels.

**Recommended for `eazy_kde` training** (`train_args.yaml`):

```yaml
transform_cosmo_params: [log_c_scale, z]   # bijector only on these
logit_flow_scale: 8.0                    # f_k → H*tanh(f/H) for the NF
```

Retrain after changing this; old runs that bijector-transformed all 13 dims are not comparable.

### Diagnose `transform_input` (triangle plots)

```bash
python -m bedcosmo.num_visits.sed_prior.diagnose_transform_input \
  --kde-path ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/sed_prior_kde.joblib \
  --outdir ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/transform_diagnostic \
  --n-samples 8000 --cdf-samples 200000
```

Writes `triangle_physical_before_transform.png` (**$a_k$, $\\log s$, $z$**) and
`triangle_gaussian_after_transform.png`. Add `--also-logits-triangle` for $f_k$ marginals
(often multimodal — not the same as weight space).

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
