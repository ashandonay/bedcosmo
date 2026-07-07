# Empirical galaxy SED prior (`empirical`)

Build and sample an empirical prior over galaxy SEDs for the `num_visits` BED experiment: fit EAZY templates to DESI spectra (NNLS), pool HEALPix patches, train a **smooth ILR KDE**, and sample through a GPU prior pool in `NumVisits` (`cosmo_model: empirical`).

**Production parameterization:** isometric log-ratios (ILR) — the centered log-ratios \(f_k^{\mathrm{clr}} = \log a_k - \mathrm{mean}_j\log a_j\) expressed in an orthonormal basis of their sum-zero hyperplane, giving \(K-1\) full-rank coordinates for \(K=12\) templates, plus \(\log s\) and \(z\) — **13 features** (`f1`…`f11`, `log_c_scale`, `z`). ILR removes the exact CLR sum-zero constraint (which made the 14D differential entropy ill-posed); CLR remains the internal intermediate (`ilr = clr·V`, `a = softmax(clr)`) and a readable legacy stored format. Legacy artifacts used raw weights, logits, masked support projection, or 14D CLR.

---

### Prerequisites

- `conda activate bedcosmo` with editable install: `pip install -e ".[sed-prior]"`
- DESI HTTP credentials in `~/.desi_http_user` (`username:password`) for first-time DESI download
- `$SCRATCH` set or writes under `~/scratch/bedcosmo/`

### Data build

To generate the empirical prior data for BED training:
```bash
python -m bedcosmo.num_visits.empirical.build_prior
```

Default `--build-name empirical_prior` writes:

```text
$SCRATCH/bedcosmo/num_visits/empirical_prior/
  healpix/hp23040/desi_eazy_empirical_weights.csv
  healpix/hp27257/...
  desi_eazy_empirical_weights.csv
  sed_prior_kde.joblib
  sed_prior_y_kde.joblib
  sed_prior_kde.json
```

Shared inputs (reused across builds): `$SCRATCH/bedcosmo/desi/tiny_dr1/`, `$SCRATCH/bedcosmo/eazy/`.

Default HEALPix patches: `23040 27257 27245 27259 27247 27256 27258 27344 26282`.

### Resume / partial runs

Existing outputs are skipped unless `--force-desi` or `--force-fit` is set.

```bash
# KDE only (fits + combine already done)
python -m bedcosmo.num_visits.empirical.build_prior \
  --build-name empirical_prior \
  --skip-desi --skip-fit --skip-combine

# Single patch, no KDE
python -m bedcosmo.num_visits.empirical.build_prior \
  --healpix 23040 --skip-kde

# Refit all patches after changing fit settings
python -m bedcosmo.num_visits.empirical.build_prior --force-fit
```

### Key flags

| Flag | Default | Notes |
|------|---------|--------|
| `--build-name` | `empirical_prior` | Subdir under `num_visits/` |
| `--healpix` | 9 patches above | Override patch list |
| `--desi-dir` | `$SCRATCH/bedcosmo/desi/tiny_dr1` | Local DESI tree root |
| `--n-max` | all candidates | Subsample per patch (testing only) |
| `--z-min` | `0.01` | Redshift floor at candidate selection |
| `--max-chi2-dof` | `1.2` | Quality cut in fits and KDE table |
| `--force-desi` | off | Re-download DESI even if coadd exists |
| `--force-fit` | off | Re-fit even if per-patch CSV exists |
| `--skip-desi/fit/combine/kde` | off | Run subset of steps |
| `--kde-sample` | `20000` | Post-save KDE diagnostic triangles (`0` to skip) |

---

## Directory layout

| File | Role |
|------|------|
| `build_prior.py` | **Orchestrator:** DESI download → fits → combine → KDE (recommended entry point) |
| `paths.py` | Default scratch paths (`get_prior_kde_path`, `get_desi_data_dir`, …) |
| `desi_get_dr_subset.py` | Download DESI DR1 coadd + redrock for selected HEALPix patches |
| `fit_eazy_weights_to_desi.py` | Per-galaxy NNLS template fit → weights CSV + fit diagnostics |
| `combine_healpix_weights.py` | Concatenate per-patch CSVs into one training table |
| `fit_sed_prior_kde.py` | Train KDE + gaussianizer + y-KDE → `sed_prior_kde.joblib`, `sed_prior_y_kde.joblib` |
| `run_healpix_fits.sh` | Batch fits only (no combine/KDE; see orchestrator instead) |
| `run_healpix_diagnostic_plots.sh` | Per-patch `--plot-only` triangles + cross-patch comparison |
| `compare_healpix_prior_params.py` | Cross-patch overlays of prior coordinates |
| `diagnostic_plots.py` | Post-build KDE/NumVisits diagnostics |
| `diagnose_transform_input.py` | NumVisits `transform_input` triangle diagnostics |
| `sed_prior.py` | KDE prior: GPU pool, sampling, and log-density scoring |
| `simplex.py` | Weight ↔ CLR ↔ ILR maps + parameterization dispatch (numpy + torch) |
| `templates.py` | Load EAZY template bank |

**Typical paths** (under `$SCRATCH/bedcosmo`, or `~/scratch/bedcosmo` when `SCRATCH` is unset)

| What | Path |
|------|------|
| EAZY templates | `eazy/` (auto-downloaded on first fit) |
| DESI tiny DR1 | `desi/tiny_dr1/` |
| **Production prior build** | `num_visits/empirical_prior/` |
| Per-patch fits | `num_visits/empirical_prior/healpix/hp{HEALPIX}/` |
| Combined weights | `num_visits/empirical_prior/desi_eazy_empirical_weights.csv` |
| KDE artifact | `num_visits/empirical_prior/sed_prior_kde.joblib` |
| y-prior KDE (opt-in) | `num_visits/empirical_prior/sed_prior_y_kde.joblib` |
| Training config | [`prior_args_empirical.yaml`](../../../../experiments/num_visits/prior_args_empirical.yaml) (`prior_kde_source: null` → default scratch build at snapshot) |

**Notebook:** `experiments/num_visits/notebooks/empircal_prior.ipynb`

**Environment:** run all pipeline steps in the `bedcosmo` conda env (`pip install -e ".[sed-prior]"` for `fitsio`). Invoke scripts as modules, e.g. `python -m bedcosmo.num_visits.empirical.build_prior`.

---

## Production prior: ILR + smooth KDE

### Coordinates

CSV fits store **`a1`…`aK`**, **`c1`…`cK`**, **`log_c_scale`**, **`z`**.

The KDE is trained on **ILR** features after a small simplex floor:

1. \(a_k \leftarrow (a_k + \varepsilon) / \sum_j (a_j + \varepsilon)\) with default \(\varepsilon = 10^{-5}\)
2. \(f_k^{\mathrm{clr}} = \log a_k - \mathrm{mean}_j \log a_j\) (CLR; row sums to zero, **no** reference template)
3. Project CLR onto an orthonormal basis \(V\) of its sum-zero hyperplane: \(\mathbf{f} = \mathbf{f}^{\mathrm{clr}} V\), giving \(K-1\) full-rank ILR coords
4. KDE in \((f_1,\ldots,f_{K-1}, \log s, z)\); decode weights via \(\mathbf{f}^{\mathrm{clr}} = \mathbf{f} V^\top\), \(a = \mathrm{softmax}(\mathbf{f}^{\mathrm{clr}})\)

For \(K=12\):

| Feature | Meaning |
|---------|---------|
| `f1`…`f11` | Isometric log-ratios (\(K{-}1\) orthonormal coords; no sum constraint) |
| `log_c_scale` | \(\log s = \log\sum_k \|c_k\|\) |
| `z` | DESI redrock redshift |

Rest-frame SED: \(c_k = e^{\log s}\, a_k\), \(f_\mathrm{rest}(\lambda) \propto \sum_k c_k T_k(\lambda)\).

**Why ILR over CLR:** the 12 CLR coords sum to exactly zero, so the physical prior lives on a 13-manifold and its 14D differential entropy is ill-posed. ILR is CLR in an orthonormal basis of that hyperplane — 11 full-rank coords, `|det|=1`, so entropy is well-posed and basis-independent. The likelihood is unchanged (it only ever sees the reconstructed template weights). CLR remains the internal intermediate and a readable legacy stored format.

**NNLS** (\(c_k \ge 0\)): many training \(a_k\) are **exactly zero** (inactive templates). The smooth prior keeps tiny positive KDE mass on inactive directions instead of hard support masks.

### End-to-end pipeline

```text
build_prior.py  (one command; steps skip existing outputs)
  Step 1  ensure DESI coadd + redrock under desi/tiny_dr1/
  Step 2  fit_eazy_weights_to_desi.py  →  num_visits/<build>/healpix/hp*/desi_eazy_empirical_weights.csv
  Step 3  combine_healpix_weights.py   →  num_visits/<build>/desi_eazy_empirical_weights.csv
  Step 4  fit_sed_prior_kde.py → sed_prior_kde.joblib
        ↓
diagnostic_plots.py all         →  diagnostics/{clr_triangle,redshift_histograms,...}/
run_healpix_diagnostic_plots.sh →  per-patch fit triangles (optional; skipped during build)
        ↓
NumVisits (empirical)           →  GPU prior pool → SED → LSST magnitudes
```

**Selection cuts (fit):** `SPECTYPE == GALAXY`, `ZWARN == 0`, default **`z >= 0.01`** (`--z-min`, `--no-z-min` to disable).

**Quality cut:** default `chi2/dof <= 1.2` (`quality_pass`). Failed rows stay in the CSV / `dropped_fits.csv` but are excluded from KDE training.

**Scope:** default build uses **9 HEALPix patches** from DESI DR1 `tiny_dr1` (not the full survey sky). Omit `--n-max` to fit all quality-passing galaxies per patch.


---

## Step 0: DESI data (`desi_get_dr_subset.py`)

Usually handled automatically by `build_prior` (step 1) or `fit_eazy_weights_to_desi.py` (`--auto-download-desi`). Use this script directly for custom download layouts.

Downloads coadd + redrock FITS into a tree matching the fit script:

```text
$SCRATCH/bedcosmo/desi/tiny_dr1/spectro/redux/iron/healpix/main/dark/{prefix}/{healpix}/
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
python -m bedcosmo.num_visits.empirical.fit_eazy_weights_to_desi \
  --healpix 23040 \
  --build-name empirical_prior \
  --fit-method nnls \
  --z-min 0.01
```

Output defaults to `$SCRATCH/bedcosmo/num_visits/empirical_prior/healpix/hp23040/`. Omit `--n-max` to fit **all** passing candidates; use `--n-max 600` for quick tests.

### Multi-patch batch (fits only)

Prefer `build_prior` for the full pipeline. For fits alone:

```bash
./run_healpix_fits.sh
```

| Env var | Default | Notes |
|---------|---------|--------|
| `BUILD_NAME` | `empirical_prior` | Under `$SCRATCH/bedcosmo/num_visits/` |
| `N_MAX` | *(unset)* | If set, subsample per patch (e.g. `600`) |
| `FORCE` | `0` | `FORCE=1` refits even if CSV exists |
| `Z_MIN` | `0.01` | Redshift floor |
| `SEED` | `7` | |

DESI and EAZY paths use Python defaults (`$SCRATCH/bedcosmo/desi/tiny_dr1`, etc.).

### Combine patches

```bash
python -m bedcosmo.num_visits.empirical.combine_healpix_weights \
  --build-name empirical_prior
```

### Compare patches

```bash
python -m bedcosmo.num_visits.empirical.compare_healpix_prior_params \
  --build-name empirical_prior
```

Default output: `num_visits/empirical_prior/healpix_prior_comparison/`.

### Fit outputs

| File | Contents |
|------|----------|
| `desi_eazy_empirical_weights.csv` | `a*`, `c*`, `log_c_scale`, `z`, `chi2_dof`, `quality_pass`, … |
| `dropped_fits.csv` | Quality failures |
| `prior_params_triangle.png` | Training \((a_k,\log s,z)\), quality-pass |
| `coeffs_raw_triangle.png` | Raw \(c_k\) vs \(z\) |
| `spectrum_fit_examples*.png` | Spectrum + \(c_k\) bars |
| `chi2_dof_histogram.png` | \(\chi^2/\mathrm{dof}\) |

**Replot only** (per-patch triangles and spectrum examples):

```bash
python -m bedcosmo.num_visits.empirical.fit_eazy_weights_to_desi \
  --plot-only --healpix 23040 --build-name empirical_prior \
  --plot-n-examples 8 --plot-top-outliers 5
```

Or all patches: `./run_healpix_diagnostic_plots.sh`

### Key fit flags

| Flag | Default | Notes |
|------|---------|--------|
| `--fit-method` | `nnls` | `wls` for debugging only |
| `--z-min` | `0.01` | `--no-z-min` disables |
| `--max-chi2-dof` | `1.2` | `quality_pass` |
| `--n-max` | all | Subsample cap |
| `--coeff-norm` | `l1` | |

---

## Step 2: KDE prior (`fit_sed_prior_kde.py`)

### Recommended mode (defaults)

| Setting | Value |
|---------|--------|
| `--parameterization` | `ilr` |
| `--support-mode` | `smooth` (no random NNLS support mask) |
| `--simplex-smoothing-eps` | `1e-5` |
| `--bandwidth` | `0.3` (scaled space) |
| `--z-min` | `0.01` |
| `--gaussianizer-fit-source` | `kde` (100k reference draws) |
| `--gaussianizer-whitening` | `cholesky` |
| `--sample` | `20000` (post-save diagnostic triangles) |
| `--no-y-kde` | off (by default also writes `sed_prior_y_kde.joblib`) |
| `--y-kde-samples` | `50000` |

Legacy **`--support-mode masked`** applies a random training-galaxy zero pattern after sampling; large LSST mag shifts — not recommended for production.

### Build

Normally run via `build_prior` (step 4). Standalone:

```bash
python -m bedcosmo.num_visits.empirical.fit_sed_prior_kde \
  --build-name empirical_prior
```

Paths default from `paths.py` (`desi_eazy_empirical_weights.csv` and `sed_prior_kde.joblib` under the build directory). Requires `torch` (use `bedcosmo` env).

### Artifacts

| File | Contents |
|------|----------|
| `sed_prior_kde.joblib` | KDE, scaler, `training_x`, **NF gaussianizer** (`gaussianizer_state`), bounds, metadata |
| `sed_prior_y_kde.joblib` | Opt-in y-space prior KDE (fit beside KDE at build; not snapshotted into runs, not loaded by default — empirical EIG uses the N(0,I) shortcut) |
| `sed_prior_kde.json` | Metadata summary |
| `kde_samples_*.png` | Diagnostic triangles when `--sample > 0` |
| `training_gaussianized_triangle.png` | Gaussianized training/coords |

### Python API

```python
from bedcosmo.num_visits.empirical import (
    load_sed_prior_kde,
    sample_sed_prior,
    samples_to_coeffs,
)

from bedcosmo.num_visits.empirical.paths import get_prior_kde_path

artifact = load_sed_prior_kde(get_prior_kde_path())
x = sample_sed_prior(artifact, n_samples=5000, seed=0)  # (N, 13) ILR features
n = artifact["n_templates"]
a, log_s, z = samples_to_coeffs(x, n, parameterization="ilr")
```

---

## Diagnostics (`diagnostic_plots.py`)

Not part of the build pipeline. All subcommands take a **prior build directory** and write under **`diagnostics/<name>/`** (override with `--outdir`).

```bash
python -m bedcosmo.num_visits.empirical.diagnostic_plots all \
  --prior-dir $SCRATCH/bedcosmo/num_visits/empirical_prior
```

| Subcommand | Output subdir | What it checks |
|------------|---------------|----------------|
| `clr-triangle` | `clr_triangle/` | Low-weight template highlighting in CLR and Cholesky-whitened gaussianized KDE (and optional training) draws |
| `redshift-histograms` | `redshift_histograms/` | Redrock GALAXY vs STAR vs weights CSV; dashed line at KDE `z_min` |
| `sed-examples` | `sed_examples/` | NumVisits SEDs, LSST mags, weight heatmap, `empirical_seds.npz` |
| `mag-leakage` | `mag_leakage/` | Smooth KDE vs threshold-zeroed vs masked weights → \(\Delta m\) |
| `all` | all of the above | One-shot |

**Mag leakage (typical):** thresholding inactive templates (\(a_k \le 10^{-4}\)) changes LSST mags at the **sub-millimag** level; legacy masked support can shift by \(\sim 1\) mag.

Individual runs:

```bash
python -m bedcosmo.num_visits.empirical.diagnostic_plots clr-triangle \
  --prior-dir $SCRATCH/bedcosmo/num_visits/empirical_prior --also-training
```

---

## `NumVisits` integration (`empirical`)

### Config

- **Parameters:** `f1`…`f11`, `log_c_scale`, `z` in [`models.yaml`](../../../../experiments/num_visits/models.yaml)
- **KDE snapshot source:** [`prior_args_empirical.yaml`](../../../../experiments/num_visits/prior_args_empirical.yaml). Set `prior_kde_source: null` to snapshot from `get_prior_kde_path()`. Trained runs load `artifacts/empirical/sed_prior_kde.joblib` directly (like emulators).

```yaml
prior_kde_source: null
eazy_templates_dir: null   # defaults to $SCRATCH/bedcosmo/eazy/
```

Override with an absolute path when using a non-default `--build-name`.

- **Training:** [`train_args.yaml`](../../../../experiments/num_visits/train_args.yaml) `empirical` block:

```yaml
transform_input: true                    # transform all cosmo_params
input_transform_type: joint              # use build-prior gaussianizer joint block
logit_flow_scale: 8.0                    # unused when all params are joint-transformed
```

For empirical runs, **`param_bijector` is loaded from the KDE artifact** (`build_prior` gaussianizer), not rebuilt from GPU pool samples at train init. Checkpoints still store `bijector_state` for resume/eval. Eval never refits the bijector.

`NumVisits` decodes ILR rows → CLR → simplex \(a_k\) → `_calculate_magnitudes`. Rebuild the KDE after changing parameterization or `feature_names`.

### `transform_input` diagnostics

```bash
python -m bedcosmo.num_visits.empirical.diagnose_transform_input \
  --kde-path $SCRATCH/bedcosmo/num_visits/empirical_prior/sed_prior_kde.joblib
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
| **ILR + smooth** | `f1`…`f_{K-1}`, `log_s`, `z` | **Production** |
| CLR + smooth | `f1`…`fK`, `log_s`, `z` | Internal intermediate; buildable (`--parameterization clr`) for A/B |
| Logits / raw weights / masked support | — | Removed |

Use `diagnostic_plots sed-examples` for production-pipeline SED checks (KDE-sampled weights, not synthetic mixtures).

### Older scratch trees

Legacy layouts (`desi_eazy_hp*` at scratch root, `desi_eazy_empirical_prior_full`, etc.) are still discovered by `find_healpix_weights_csv()` for backward compatibility. New builds should use `num_visits/<build-name>/healpix/hp*/`.

---

## Quick reference

| Step | Command / setting |
|------|-------------------|
| **Full build** | `python -m bedcosmo.num_visits.empirical.build_prior` |
| Build name | `empirical_prior` (under `num_visits/`) |
| Fit | **NNLS**, **L1** norm, **`z_min=0.01`**, all candidates (no `--n-max`) |
| KDE | **ILR**, **smooth**, \(\varepsilon=10^{-5}\), bandwidth **0.3** |
| NF bijector | **`gaussianizer_state` in KDE artifact** (not rebuilt at train/eval) |
| y-prior | **`sed_prior_y_kde.joblib`** beside KDE (auto-fit at build; opt-in, not loaded by default) |
| Training | `prior_kde_source: null` at snapshot; runtime uses `artifacts/empirical/` |
| Fit diagnostics | `./run_healpix_diagnostic_plots.sh` |
| KDE diagnostics | `diagnostic_plots all --prior-dir .../empirical_prior` |

---

## References

- EAZY: [gbrammer/eazy-photoz](https://github.com/gbrammer/eazy-photoz), default `templates/fsps_full/fsps_QSF_12_v3.param`
- DESI DR1 coadd + redrock under `$SCRATCH/bedcosmo/desi/tiny_dr1`
