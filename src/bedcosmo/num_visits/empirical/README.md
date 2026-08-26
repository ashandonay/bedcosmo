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

Default `--build-name empirical_prior/eazy12` writes:

```text
$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/
  healpix/hp23040/desi_eazy_empirical_weights.csv
  healpix/hp27257/...
  desi_eazy_empirical_weights.csv
  sed_prior_kde_native.joblib
  sed_prior_kde_gaussianized.joblib
  sed_prior_kde_native.json
```

Shared inputs (reused across builds): `$SCRATCH/bedcosmo/desi/tiny_dr1/`, `$SCRATCH/bedcosmo/eazy/`.

`build_prior` also writes ``build.log`` and ``build_provenance.json`` into that
directory. The provenance records the template bank, template normalization
interval, spectral fitting and selection cuts, input paths, and KDE request.
Each HEALPix fit also has a ``fit_provenance.json`` beside its CSV. Combining
patches rejects incompatible template/normalization settings, and the build
provenance is embedded in the KDE artifact so NumVisits can validate and reuse
the exact normalization at runtime.

Default HEALPix patches: `23040 27257 27245 27259 27247 27256 27258 27344 26282`.

### Resume / partial runs

Existing outputs are skipped unless `--force-desi` or `--force-fit` is set.

```bash
# KDE only (fits + combine already done)
python -m bedcosmo.num_visits.empirical.build_prior \
  --build-name empirical_prior/eazy12 \
  --skip-desi --skip-fit --skip-combine

# Single patch, no KDE
python -m bedcosmo.num_visits.empirical.build_prior \
  --healpix 23040 --skip-kde

# Refit all patches after changing fit settings
python -m bedcosmo.num_visits.empirical.build_prior --force-fit

# Classic 6-template EAZY bank (separate scratch tree; 7D ILR prior)
python -m bedcosmo.num_visits.empirical.build_prior \
  --build-name empirical_prior/eazy6 \
  --template-param templates/eazy_v1.0.spectra.param
```

Train against that build with the same `cosmo_model: empirical`, overriding prior_args:

```bash
./submit.sh train num_visits empirical --prior-args-path prior_args_empirical_eazy6.yaml
```

See [`prior_args_empirical_eazy6.yaml`](../../../../experiments/num_visits/prior_args_empirical_eazy6.yaml).
Production default (`prior_args_empirical.yaml`, 12 templates) is unchanged.

### Key flags

| Flag | Default | Notes |
|------|---------|--------|
| `--build-name` | `empirical_prior/eazy12` | Relative path under `num_visits/` |
| `--template-param` | `templates/fsps_full/fsps_QSF_12_v3.param` | Template-bank listing (`.param`); use `templates/eazy_v1.0.spectra.param` for classic 6. |
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
| `fit_sed_prior_kde.py` | Train KDE + gaussianizer (+ offline y-KDE diagnostic) → `sed_prior_kde_native.joblib` |
| `prior_flow.py` | Train normalizing flow(s) over the prior (native + gaussianized) → `sed_prior_flow_*.pt` (the default `prior_source`) |
| `validate_prior_flow.py` | A/B the trained flow(s) against the KDE (panels + getdist triangles, both spaces) |
| `run_healpix_fits.sh` | Batch fits only (no combine/KDE; see orchestrator instead) |
| `run_healpix_diagnostic_plots.sh` | Per-patch `--plot-only` triangles + cross-patch comparison |
| `compare_healpix_prior_params.py` | Cross-patch overlays of prior coordinates |
| `discover_template_cohorts.py` | Exhaustive exact-N DESI refits and reduced-template cohort assignments |
| `plot_template_subset_examples.py` | Observed/full/reduced DESI fit galleries with individual template contributions |
| `diagnostic_plots.py` | Post-build KDE/NumVisits diagnostics |
| `diagnose_transform_input.py` | NumVisits `transform_input` triangle diagnostics |
| `sed_prior.py` | Empirical prior: GPU pool, sampling, log-density scoring, and flow attachment (`prior_source` {kde, flow}) |
| `simplex.py` | Weight ↔ CLR ↔ ILR maps + parameterization dispatch (numpy + torch) |
| `templates.py` | Load EAZY template bank |

**Typical paths** (under `$SCRATCH/bedcosmo`, or `~/scratch/bedcosmo` when `SCRATCH` is unset)

| What | Path |
|------|------|
| EAZY templates | `eazy/` (auto-downloaded on first fit) |
| DESI tiny DR1 | `desi/tiny_dr1/` |
| Empirical-prior variants | `num_visits/empirical_prior/{eazy12,eazy6,allow_zwarn,no_unstable}/` |
| **Production prior build** | `num_visits/empirical_prior/eazy12/` |
| Per-patch fits | `num_visits/empirical_prior/eazy12/healpix/hp{HEALPIX}/` |
| Combined weights | `num_visits/empirical_prior/eazy12/desi_eazy_empirical_weights.csv` |
| KDE artifact | `num_visits/empirical_prior/eazy12/sed_prior_kde_native.joblib` |
| gaussianized KDE (diagnostic) | `num_visits/empirical_prior/eazy12/sed_prior_kde_gaussianized.joblib` |
| Training config | [`prior_args_empirical.yaml`](../../../../experiments/num_visits/prior_args_empirical.yaml) (`prior_dir: null` → default scratch build at snapshot) |

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
  Step 4  fit_sed_prior_kde.py → sed_prior_kde_native.joblib
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
  --build-name empirical_prior/eazy12 \
  --fit-method nnls \
  --z-min 0.01
```

Output defaults to `$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/healpix/hp23040/`. Omit `--n-max` to fit **all** passing candidates; use `--n-max 600` for quick tests.

### Multi-patch batch (fits only)

Prefer `build_prior` for the full pipeline. For fits alone:

```bash
./run_healpix_fits.sh
```

| Env var | Default | Notes |
|---------|---------|--------|
| `BUILD_NAME` | `empirical_prior/eazy12` | Relative path under `$SCRATCH/bedcosmo/num_visits/` |
| `N_MAX` | *(unset)* | If set, subsample per patch (e.g. `600`) |
| `FORCE` | `0` | `FORCE=1` refits even if CSV exists |
| `Z_MIN` | `0.01` | Redshift floor |
| `SEED` | `7` | |

DESI and EAZY paths use Python defaults (`$SCRATCH/bedcosmo/desi/tiny_dr1`, etc.).

### Combine patches

```bash
python -m bedcosmo.num_visits.empirical.combine_healpix_weights \
  --build-name empirical_prior/eazy12
```

### Compare patches

```bash
python -m bedcosmo.num_visits.empirical.compare_healpix_prior_params \
  --build-name empirical_prior/eazy12
```

Default output: `num_visits/empirical_prior/eazy12/healpix_prior_comparison/`.

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
  --plot-only --healpix 23040 --build-name empirical_prior/eazy12 \
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
| `--no-gaussianized-kde` | off (by default also writes `sed_prior_kde_gaussianized.joblib`) |
| `--gaussianized-kde-samples` | `50000` |

Legacy **`--support-mode masked`** applies a random training-galaxy zero pattern after sampling; large LSST mag shifts — not recommended for production.

### Build

Normally run via `build_prior` (step 4). Standalone:

```bash
python -m bedcosmo.num_visits.empirical.fit_sed_prior_kde \
  --build-name empirical_prior/eazy12
```

Paths default from `paths.py` (`desi_eazy_empirical_weights.csv` and `sed_prior_kde_native.joblib` under the build directory). Requires `torch` (use `bedcosmo` env).

### Artifacts

| File | Contents |
|------|----------|
| `sed_prior_kde_native.joblib` | KDE, scaler, `training_x`, **NF gaussianizer** (`gaussianizer_state`), bounds, metadata |
| `sed_prior_kde_gaussianized.joblib` | Offline diagnostic KDE in gaussianized coords (auto-fit at build; not snapshotted / not used at runtime — EIG uses `sed_prior_flow_gaussianized.pt` or the N(0,I) shortcut) |
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

## Step 3: Prior normalizing flow (`prior_flow.py`)

The **default `prior_source` is `flow`**: the empirical prior is drawn from a trained
normalizing flow (zuko NSF) rather than the KDE plug-in. One flow is trained per space,
both fit to KDE draws and saved beside the KDE artifact:

- **`native`** — trained on native/ILR rows. Does two jobs at runtime: the prior-pool
  **sampler** and the prior **entropy** density (`H = -E[log p_flow]`, the flow's exact
  entropy — the same NF plug-in used for the posterior `H_post`, so `EIG = H_prior - H_post`
  uses one coherent estimator). For `transform_input: true` the entropy is the **Jacobian
  bridge** `H_y = H_native + E[log|det dT/dx|]` (native flow + the analytic gaussianizer
  Jacobian; no inverse needed).
- **`gaussianized`** — trained on KDE draws pushed through the production gaussianizer
  (`y = T(x)`). Currently a **density-only diagnostic** (validated by the A/B below); it is
  not consumed by the runtime entropy path.

Why a flow and not the KDE/`N(0,I)` shortcut: at 13D the KDE plug-in and kNN entropy are
biased, and the `N(0,I)` gaussianized shortcut overestimates `H_y` by ~13 nats. The flow's
plug-in entropy is exact up to MC error.

### Train

CPU-heavy (NSF). Trains `--space both` concurrently on one node (native + gaussianized,
half the cores each), ~10 min at production size.

```bash
# auto SLURM/local launcher (run from this dir)
./train_prior_flow.sh --space both

# or submit the SLURM job directly (CPU node, account desi)
sbatch scripts/slurm/train_prior_flow.sh --space both --n 100000 --epochs 400

# or run the module directly (cap threads on a login node)
python -m bedcosmo.num_visits.empirical.prior_flow --space both --threads 8
```

Reads the KDE from `get_prior_kde_path()` (override `--kde-path`) and writes beside it
(override `--out-dir`):

```text
$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/
  sed_prior_flow_native.pt              sed_prior_flow_native_train.log
  sed_prior_flow_gaussianized.pt        sed_prior_flow_gaussianized_train.log
  prior_flow_training_native.png        prior_flow_training_gaussianized.png
```

Each `*_train.log` holds that flow's full per-epoch eval-NLL history + hyperparameters
(one file per space, so the concurrent processes don't interleave). The best-eval history
is also stored inside the `.pt` (`meta["train"]`), and a convergence plot is written
automatically per space (`prior_flow_training_<space>.png`) beside each flow right after
it is saved by `plot_training_convergence` — so training a single space in its own job
never clobbers the other's plot. Regenerate one standalone by loading the `.pt` and
calling that function with its path.

### Validate (`validate_prior_flow.py`)

The gate before trusting `prior_source: flow`: the flow must reproduce the KDE. Run-free
and read-only. Two axes — `--space {native,gaussianized,both}` and `--plot {panel,triangle,both}`.

```bash
# everything (both spaces, panels + triangles), threads capped for a login node
python -m bedcosmo.num_visits.empirical.validate_prior_flow --threads 8

# just the native panel (fast)
python -m bedcosmo.num_visits.empirical.validate_prior_flow --space native --plot panel --threads 8
```

- **panel** — per-feature KS + covariance + entropy summary with a PASS/REVIEW verdict
  (native gates on feature KS, template-weight KS, cov Frobenius, and the flow NLL gap).
- **triangle** — overlaid getdist contour corner (KDE filled vs flow line). Uses fixed
  smoothing + auto boundary-range detection so sharp ILR edges (e.g. `f10`) don't produce
  blotchy contours.

Writes `validate_prior_flow_{native,gaussianized}{,_triangle}.png` beside the KDE (override
`--out-dir`). The native/`transform_input: false` path is validated end-to-end (A/B PASS;
eval `H_prior` matches the offline flow entropy).

The flows must exist in `prior_dir` beside the KDE (train them first). Runtime loads the
frozen copies from `artifacts/empirical/` after snapshot.

---

## Diagnostics (`diagnostic_plots.py`)

Not part of the build pipeline. All subcommands take a **prior build directory** and write under **`diagnostics/<name>/`** (override with `--outdir`).

```bash
python -m bedcosmo.num_visits.empirical.diagnostic_plots all \
  --prior-dir $SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12
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
  --prior-dir $SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12 --also-training
```

---

## `NumVisits` integration (`empirical`)

### Config

- **Parameters:** taken at runtime from the KDE artifact’s `feature_names` (not hardcoded from `models.yaml`). Production 12-template builds use `f1`…`f11`, `log_c_scale`, `z`; a 6-template prior_args points at a K=6 artifact and gets `f1`…`f5`, `log_c_scale`, `z` automatically.
- **Prior build dir:** [`prior_args_empirical.yaml`](../../../../experiments/num_visits/prior_args_empirical.yaml). Set `prior_dir: null` to use `$SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12`. That directory holds `sed_prior_kde_native.joblib` and (for `prior_source: flow`) the `sed_prior_flow_*.pt` files. Trained runs load the frozen copies from `artifacts/empirical/`.

```yaml
prior_dir: null            # null = default scratch empirical_prior/eazy12 build
template_dir: null         # defaults to $SCRATCH/bedcosmo/eazy/
prior_source: flow         # {flow (default), kde}; flow needs sed_prior_flow_*.pt in prior_dir
```

For the classic 6-template bank use [`prior_args_empirical_eazy6.yaml`](../../../../experiments/num_visits/prior_args_empirical_eazy6.yaml)
(`prior_dir` → `empirical_prior/eazy6`, `template_param: templates/eazy_v1.0.spectra.param`,
`prior_source: kde` until a flow is trained) with:

```bash
./submit.sh train num_visits empirical --prior-args-path prior_args_empirical_eazy6.yaml
```

Override `prior_dir` with an absolute path when using a non-default `--build-name`.
With `prior_source: flow` (the default), the trained flows are snapshotted into the run's
`artifacts/empirical/` alongside the KDE and drive the prior pool + entropy; set
`prior_source: kde` for the pre-flow KDE baseline (e.g. a flow-vs-KDE A/B). See
[Step 3](#step-3-prior-normalizing-flow-prior_flowpy) to train and validate the flows.

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
  --kde-path $SCRATCH/bedcosmo/num_visits/empirical_prior/eazy12/sed_prior_kde_native.joblib
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

To connect majority-T1 and majority-T7 subsets back to their actual DESI
coadds, run:

```bash
python experiments/num_visits/scripts/plot_eazy_dominant_cohort_traits.py
```

This writes a physical-traits figure plus a TARGETID/HEALPix-level CSV. The
measurements include narrow Dn4000 and local-continuum emission equivalent
widths for [O II], Hβ, [O III], and Hα. `--min-anchor-weight` controls cohort
purity (default: the dominant anchor must carry at least 50% of the fitted
weight).

### Discover fixed-size reduced-template cohorts

To refit every quality-passing DESI spectrum with every size-N subset of the
12-template bank:

```bash
python -m bedcosmo.num_visits.empirical.discover_template_cohorts \
  --n-templates 3 \
  --max-chi2-dof 1.2 \
  --max-delta-chi2-dof 0.05 \
  --max-color-rms 0.02 \
  --min-component-weight 0.01 \
  --min-cohort-size 100
```

The first run caches each spectrum's weighted 12-template sufficient
statistics; searches with other values of N reuse that cache. Outputs under
`<prior-dir>/reduced_template_cohorts/nN/` are:

| File | Contents |
|------|----------|
| `subset_summary.csv` | Independent total coverage, strictly exclusive coverage, and fit-quality distribution summaries for every subset |
| `subset_memberships.csv` | Every passing spectrum-subset pair, including that subset's reduced coefficients, scale, fit degradation, and LSST color RMS |
| `spectrum_assignments.csv` | Optional disjoint view that selects the best passing subset per spectrum |
| `subset_quality_matrices.npz` | Per-spectrum quality and pass mask for every candidate subset |
| `subset_discovery.png` | Largest independent cohorts with distributions of DESI fit degradation, absolute reduced-fit quality, and disjoint-assignment redshift |
| `discovery_parameters.json` | Exact thresholds and template settings |

`--min-component-weight` prevents an N-template cohort from being padded with
an unused template. Set it to zero to interpret N as "at most N"; leave it
positive when discovering physically distinct exact-N cohorts.
`exclusive_count` means spectra that pass exactly one candidate subset of the
requested N; it is distinct from `assigned_count`, which selects the best
passing subset even when several subsets pass.

To inspect what the templates contribute for explicit subsets or the largest
cohorts, generate example-fit galleries from a discovery directory:

```bash
python -m bedcosmo.num_visits.empirical.plot_template_subset_examples \
  --cohort-dir experiments/num_visits/plots/reduced_template_cohorts/eazy12/n3 \
  --templates T1+T7+T9 \
  --top-sets 3
```

For comparisons at fixed redshift, pass `--z-min` and `--z-max`. Add
`--distinct-across-subsets` when plotting several subsets to prevent the same
DESI target from appearing in more than one gallery.

Each gallery includes a representative member and one member rich in each
selected template. It overlays the observed DESI coadd, full-basis fit,
reduced fit, and each reduced template contribution; the side bars show the
normalized reduced coefficients. For EAZY6, also pass
`--build-name empirical_prior/eazy6 --template-param templates/eazy_v1.0.spectra.param`.
By default, every trace is divided by the same continuum estimated from the
DESI coadd with a 250-rest-Angstrom Gaussian and then smoothed by 8 observed
Angstroms for display. Defining the broad continuum scale in the rest frame
makes physical feature comparisons consistent across redshift while keeping
the additive template decomposition intact. The continuum estimate is weighted
by the DESI inverse variance so very noisy pixels cannot control the displayed
normalization. The wavelength axis defaults to
the observed frame. The continuum smoother reflects the
measured spectrum at its endpoints rather than extending a potentially noisy
last pixel. Values below 5% of the spectrum's 75th-percentile continuum or
below continuum S/N 3 are masked to prevent unstable display division. Use
`--no-continuum-normalize` for the
original flux view; there, the DESI measurements default to independent
inverse-variance-weighted 25-Angstrom bins with one-sigma error bars, and
`--display-bin-aa 0` shows every original pixel. The fits and quoted chi-square
values always use every original pixel, so none of these display transforms
affect cohort membership or fit quality. `--wavelength-frame rest` switches
the plotted coordinates, and `--shared-x` enables a shared union range.

### Build a prior from one reduced-template cohort

Turn the passing members of any discovered subset into a standalone empirical
prior with:

```bash
python -m bedcosmo.num_visits.empirical.build_reduced_template_prior \
  --cohort-dir experiments/num_visits/plots/reduced_template_cohorts/eazy12/n2 \
  --templates T1+T7
```

This writes a reduced EAZY `.param` file, a compatible coefficient table,
complete selection/template provenance, and the native and gaussianized KDEs.
For T1+T7 the prior has three features: one ILR shape coordinate, `log_c_scale`,
and `z`. Use `prior_args_empirical_eazy12_t1_t7.yaml` to load it in NumVisits.
The prior is conditioned on passing the cohort discovery thresholds; it does
not represent DESI galaxies that require other template directions.

### Older scratch trees

Legacy layouts (`desi_eazy_hp*` at scratch root, `desi_eazy_empirical_prior_full`, etc.) are still discovered by `find_healpix_weights_csv()` for backward compatibility. New builds should use `num_visits/<build-name>/healpix/hp*/`.

---

## Quick reference

| Step | Command / setting |
|------|-------------------|
| **Full build** | `python -m bedcosmo.num_visits.empirical.build_prior` |
| Build name | `empirical_prior/eazy12` (under `num_visits/`) |
| Fit | **NNLS**, **L1** norm, **`z_min=0.01`**, all candidates (no `--n-max`) |
| KDE | **ILR**, **smooth**, \(\varepsilon=10^{-5}\), bandwidth **0.3** |
| NF bijector | **`gaussianizer_state` in KDE artifact** (not rebuilt at train/eval) |
| y-KDE (offline) | **`sed_prior_kde_gaussianized.joblib`** beside KDE (diagnostic only; runtime uses prior flows) |
| **Prior flow** | `./train_prior_flow.sh --space both` → `sed_prior_flow_*.pt` beside KDE (default `prior_source: flow`) |
| Validate flow | `python -m bedcosmo.num_visits.empirical.validate_prior_flow --threads 8` |
| Training | `prior_dir: null` at snapshot; runtime uses `artifacts/empirical/` |
| Fit diagnostics | `./run_healpix_diagnostic_plots.sh` |
| KDE diagnostics | `diagnostic_plots all --prior-dir .../empirical_prior/eazy12` |

---

## References

- EAZY: [gbrammer/eazy-photoz](https://github.com/gbrammer/eazy-photoz), default `templates/fsps_full/fsps_QSF_12_v3.param`
- DESI DR1 coadd + redrock under `$SCRATCH/bedcosmo/desi/tiny_dr1`
