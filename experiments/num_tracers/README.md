# NumTracers Experiment

This experiment optimizes how a DESI-like spectroscopic survey divides its observing budget across target classes, maximizing the expected information gain (EIG) for cosmological parameters inferred from BAO distance measurements (`analysis: bao`) or from ShapeFit full-shape measurements (`analysis: shapefit`, see [ShapeFit analysis](#shapefit-analysis-analysis-shapefit)). The design variable is the fraction of total observations allocated to each tracer class; the goal is to find allocations more informative about cosmology than DESI's nominal split.

`analysis` takes the same values as desilike-emulator's `--analysis` and selects the tracer bins, the `emulators.yaml` subtree and the `models.yaml` block.

## Problem Description

DESI measures BAO distances in redshift bins, each dominated by a different target class:

| Tracer bin  | Target class | Role |
|-------------|--------------|------|
| BGS         | BGS          | Bright galaxies, lowest z |
| LRG1, LRG2  | LRG          | Luminous red galaxies |
| LRG3 + ELG1 | LRG, ELG     | Blended bin fed by *both* classes |
| ELG2        | ELG          | Emission-line galaxies |
| QSO         | QSO          | Quasars |
| Lya QSO     | QSO          | Lyman-alpha forest, highest z |

Observing more of a class shrinks the BAO error in its bins, but the total number of fibers is finite, so gains in one class are paid for elsewhere. The design problem has three components:

1. **Designs** — the per-class observation split: `d = {f_BGS, f_LRG, f_ELG, f_QSO}`
2. **Parameters** — cosmological parameters, e.g. `theta = {Om, H0*rd}` for `base`
3. **Features** — the BAO distance ratios `y = {DH/rd, DM/rd, DV/rd}` across bins (12 values for dr1, 13 for dr2)

Two facts drive everything downstream:

- **A design's *sum* is the total-observation budget.** The four entries are fractions of the nominal total (dr1: 8,151,393 observations; dr2: 18,605,668), so a design summing to 1.0 spends the nominal budget and one summing to 1.2 spends 20% more. Budget is therefore a *design axis*, not a separate setting.
- **Observed != passed.** The emulator and the noise model both want redshift-confirmed (*passed*) counts, obtained as `observed x efficiency` per tracer. `calc_passed` does this conversion, correctly blending the LRG3+ELG1 bin from both parent classes.

## Design Space (`design_args_*.yaml`)

| Field                | Type            | Description |
|----------------------|-----------------|-------------|
| `labels`             | list of strings | Target classes in the design, always `["BGS","LRG","ELG","QSO"]`. |
| `input_type`         | string          | `"variable"` builds a grid (or loads a file); `"nominal"` uses the single DESI nominal split. |
| `input_designs_path` | string or null  | **Absolute** path to a `.npy` of explicit designs, shape `(n_designs, 4)` or `(4,)`. When set, every grid field below is ignored — the file *is* the design pool. |
| `step`               | float or list   | Per-class grid spacing. |
| `lower`              | float or list   | Per-class lower bound (inclusive). |
| `upper`              | float or list   | Per-class upper bound (**exclusive**, via `np.arange`). |
| `sum_lower`          | float or null   | Minimum design sum, i.e. minimum budget. |
| `sum_upper`          | float or null   | Maximum design sum. Equal to `sum_lower` pins the budget. |

### Variant files

| File | Budget (sum) | Designs | Purpose |
|------|--------------|---------|---------|
| `design_args_dr1.yaml` | 1.0 (pinned) | 287 | Default dr1 grid; split-only optimization |
| `design_args_dr2.yaml` | 1.0 (pinned) | 209 | Same for dr2 bounds |
| `design_args_nominal.yaml` | 1.0 | 1 | The DESI nominal split, computed internally |
| `design_args_unconstrained.yaml` | 1.0 (pinned) | 560 | Coarse step, bounds opened to 1.0 |
| `design_args_dr1_budget.yaml` | [1.0, 1.2] | 2447 | Budget as an axis inside one flow |
| `design_args_nominal_p{00..25}.yaml` | 1.00 … 1.25 | 1 each | Uniformly scaled nominal, one file per budget |

## Creating design spaces

Use `bedcosmo.num_tracers.design` (the num_tracers counterpart of `bedcosmo.num_visits.design`). It writes the design array plus a matching `design_args_*.yaml` that points at it:

- `.npy` → `$SCRATCH/bedcosmo/num_tracers/designs/` (`--designs-dir`)
- `design_args_*.yaml` → this directory (`--out-dir`), where `--design-args-path` resolves
- `.png` → beside the `.npy` (`--plot` to relocate, `--no-plot` to skip)

The YAML header records the command that regenerates it.

Global flags (`--dataset`, `--out-dir`, `--designs-dir`, `--plot`, `--no-plot`) go **before** the subcommand; mode flags go after.

### `pool` — one multi-design pool (preferred)

Enumerates the per-class grid and keeps rows whose sum falls in `[sum_lower, sum_upper]`, so a single amortized flow measures the whole budget axis.

```bash
python -m bedcosmo.num_tracers.design pool \
    --sum-lower 1.0 --sum-upper 1.25 \
    --upper 0.07 0.42 0.58 0.40 \
    --include-scales 1.0 1.05 1.10 1.15 1.20 1.25 \
    --n-target 1000 --name budget_1p0_1p25
```

| Flag | Effect |
|------|--------|
| `--sum-lower/--sum-upper` | Budget range spanned by the pool |
| `--step/--lower/--upper` | Per-class grid, 4 values each in `[BGS, LRG, ELG, QSO]` order |
| `--include-scales` | Force-add uniformly scaled nominal designs as within-pool reference points (they fall between grid nodes) |
| `--n-target`, `--seed` | Cap pool size by seeded random subsample; pinned rows always survive |
| `--name` | Filename for the `.npy` (`.npy` suffix optional). Omit for a date-stamped default |
| `--yaml` | YAML file name in `--out-dir`, e.g. `design_args_budget.yaml` (default `design_args_<name>.yaml`); refuses to overwrite an existing file |

The generator reproduces the experiment's own grid exactly: the defaults give **287** designs at sum 1.0 and **2447** over [1.0, 1.2], matching `design_args_dr1.yaml` and `design_args_dr1_budget.yaml`.

### `scaled` — one single-design pool per budget

```bash
python -m bedcosmo.num_tracers.design scaled --scales 1.00 1.05 1.10 1.15 1.20 1.25
```

Writes `nominal_scaled_pNN.npy` + `design_args_nominal_pNN.yaml` per scale — the nominal split times `s`, so the split is fixed and only the budget moves. Each file feeds one training run, contributing one point to `plotting.compare_increasing_design`.

### Choosing between them

**Pool size sets the training data.** The batch is `n_particles_per_device x n_designs` (`lexpand` in `pyro_oed_src.py`), so a 1-design pool sees far less data per step than a grid pool at the same `total_steps`. Because the NF EIG estimator is a variational *lower* bound, an underfit flow reads *low* — fit error masquerades as a design effect.

Measured on identical settings (emulator mode, 200k steps, npd=100):

| Run | Pool | Pairs/step | Nominal-design EIG |
|-----|------|------------|--------------------|
| `f7ac6d99` | 287 designs | 114,800 | **8.591** |
| `176650a8` | 1 design (same nominal vector) | 400 | **8.393** |

Same design, bit-identical prior entropy — the 0.198 bit gap is entirely posterior fit. So:

- **Never compare EIG across separately-trained flows.** Within one flow (nominal 8.591 → optimal 8.628 = +0.037 bits) is trustworthy; across flows is not.
- `eigs_std` (~0.012) is the sampling error for a *fixed* flow. It does not capture flow-to-flow variability (~0.2 bits here), so error bars badly understate cross-run uncertainty.
- To measure a budget trend, put the whole axis in **one** pool rather than one run per point.
- Match a proven footprint: 287 designs x npd 100 = 28,700/device. `pool` prints the `--n-particles-per-device` that hits it.

### Watch the budget skew

The per-class `upper` values cap the reachable sum. With the `design_args_dr1.yaml` bounds the maximum is **1.23**, and the grid is heavily bottom-weighted:

| Budget bin | Designs |
|------------|---------|
| [1.00, 1.05) | 1217 |
| [1.05, 1.10) | 729 |
| [1.10, 1.15) | 387 |
| [1.15, 1.20) | 129 |
| [1.20, 1.25) | 16 |

A pool built this way is well-fit at low budget and underfit at high budget, which mimics saturation that isn't physical. Raise `--upper` alongside `--sum-upper` (e.g. `--upper 0.07 0.42 0.58 0.40` lifts the ceiling to 1.42 and puts ~9% of designs above 1.20), then use `--n-target` to bring the size back down. Note this widens the design space beyond `design_args_dr1.yaml`, so results are not directly comparable to runs on the default grid.

Also note `--upper` is exclusive via `np.arange`, and float accumulation makes that unreliable at the boundary — set it slightly above the value you want rather than exactly on it.

## Parameters (`prior_args_*.yaml`)

Each entry under `parameters` defines a `distribution` (type + bounds), optional `multiplier`, `plot` ranges, and a `latex` label. `constraints` prune the prior:

- `valid_densities` — `Om + Ok` within `[0, 1]`
- `high_z_matter_dom` — `w0 + wa <= 0`

| File | Use |
|------|-----|
| `prior_args_hrdrag.yaml` | Default; `Om`, `Ok`, `w0`, `wa`, `hrdrag` |
| `prior_args.yaml` | Base prior set |
| `prior_args_small.yaml` | Narrowed ranges for fast tests |
| `prior_args_shapefit.yaml` | `analysis: shapefit`; omega-basis cosmology matching the emulators' training box, plus the `omega_m_domain` constraint |
| `prior_args_posterior*.yaml` | Posterior-informed priors, per model/dataset |

## Likelihood Model

`pyro_model` samples parameters from the prior, computes BAO means, and draws features from a multivariate normal. Means are identical in both modes — `D_H/rd`, `D_M/rd`, `D_V/rd` at each bin's effective redshift. The modes differ only in the **covariance**:

### `likelihood_mode: scaling`

Rescales the reference DESI covariance by a shot-noise idealization:

```
factor_i = sqrt(nominal_passed_ratio_i / passed_ratio_i)
```

`passed_ratio` already carries both the split and the budget (`calc_passed` is linear in the design), so a uniform scale `s` gives `sigma -> 1/sqrt(s)`. Do **not** divide by the design sum again — that double-counts the budget and yields `1/s`. It is a no-op at sum 1.0, so it only ever affected free-budget designs. Regression tests: `tests/test_num_tracers_sigma_scaling.py`.

With `vary_lya_qso: false` (default) the Lya QSO rows are pinned at their nominal error.

### `likelihood_mode: emulator`

`_build_emulator_covariance` evaluates per-bin neural emulators (checkpoints in `emulators.yaml`) on the passed tracer counts and the sampled cosmology, giving a learned BAO forecast rather than an idealization. Dense bins saturate well short of `1/sqrt(N)`: at 1.2x nominal the measured sigma ratios are BGS 0.955, LRG1 0.966, LRG2 0.961, LRG3_ELG1 0.944, ELG2 0.906, QSO 0.895, versus `1/sqrt(1.2) = 0.913`. This mode never calls `sigma_scaling_factor`.

Emulator bins map to data rows as `BGS, LRG1, LRG2, LRG3_ELG1 -> "LRG3+ELG1", ELG2, QSO, Lya_QSO`. A `null` checkpoint falls back to the fixed nominal covariance.

`emulators.yaml` is keyed `<analysis>: <dataset>: <cosmo_model>: <quantity>: <tracer_bin>`, mirroring `$SCRATCH/bedcosmo/num_tracers/emulator/{analysis}/models/{dataset}/{cosmo_model}/`. For bao, `<quantity>` is the forecast space (`config` | `fourier`) and `emulator_space` picks one per run. At submission the checkpoints are copied into the run's `artifacts/emulators/<tracer_bin>.pt`.

## ShapeFit analysis (`analysis: shapefit`)

The data vector is DESI's ShapeFit compression per tracer bin, `y = [qiso, qap, f_sigmar, m]` for each of `BGS, LRG1, LRG2, LRG3, ELG2, QSO`, giving 24 values. Each bin has two desilike-emulator networks, both taking `(N_tracers, omega_cdm, omega_b, h, ln10A_s, n_s)`:

| Quantity | Predicts | Role |
|----------|----------|------|
| `mean`  | `qiso, qap, f_sigmar, m` | Likelihood mean. It depends on `N_tracers` because z_eff(N) moves the compressed parameters. |
| `covar` | 4 `sigma_*` and 6 `rho_*` | That bin's 4x4 covariance block. |

Both are listed under `shapefit: <dataset>: <cosmo_model>: {mean, covar}` in `emulators.yaml` (currently `dr1/base`, `v4`). Every bin needs both checkpoints, because there is no DESI covariance to fall back to. They are snapshotted to `artifacts/emulators/{mean,covar}/<tracer_bin>.pt`.

- **Tracer bins** come from desilike-emulator's `tracers_for("shapefit")`. The 0.8–1.1 bin is **LRG3 only**, not BAO's LRG3+ELG1, because ELG1 is excluded from DESI's full-shape analysis. A bin's passed `N_tracers` sums its `components` in `desi_tracers.csv` (design class fraction × observed share × efficiency × nominal total), as desilike-emulator's `util.ntracers` does. At the nominal design it reproduces `util.ntracers` exactly (LRG3: 859,822).
- **Covariance** is block-diagonal across bins. The six `rho` are predicted independently, so each 4x4 correlation is projected to its nearest positive-definite matrix by flooring the eigenvalues at 1e-6 and restoring the unit diagonal. With the v4 emulators this changes about 5e-5 of LRG2 prior draws and nothing else.
- **Prior** (`prior_args_shapefit.yaml`) matches the emulators' training distributions. The `omega_m_domain` constraint redraws samples until `(omega_cdm + omega_b + omega_nu)/h^2` lies in [0.01, 0.99], the emulators' domain. Nuisance parameters are marginalized inside the emulators, so they are not sampled.
- **Nominal data** (`central_val`, the flow's nominal context) is the mean emulator at the DESI template fiducial (AbacusSummit c000) and the nominal design.
- **Not applicable**: `likelihood_mode` must be `emulator`. `vary_z_eff`, `apply_desi_syst` and `emulator_sqrtn_ref` are rejected, and `emulator_space`, `include_D_M`, `include_D_V` and `vary_lya_qso` are unused.
- **DESI reference posterior**: `load_nominal_samples` reads DESI's DR1 ShapeFit-alone chain (`desi-shapefit-all-nolya` with the BBN and ns10 priors, the same tracers and priors as this forecast) from `~/data/desi/shapefit_dr1/mcmc_samples/base.npy`, converted from the public DR1 full-shape VAC as recorded in `PROVENANCE.md` beside it. It is DESI's posterior for the real data, so it is centred on the measured cosmology, not the fiducial.

## Cosmology Models (`models.yaml`)

Keyed by `<analysis>: <cosmo_model>`, giving `parameters`, `latex_labels`, and optional `constraints`.

| Model | Parameters |
|-------|------------|
| `base` | `Om`, `hrdrag` |
| `base_omegak` | `+ Ok` (constraint: `valid_densities`) |
| `base_w` | `+ w0` |
| `base_w_wa` | `+ w0`, `wa` (constraint: `high_z_matter_dom`) |
| `base_omegak_w_wa` | all five (both constraints) |

The `shapefit` analysis has one model, `base`: `omega_cdm`, `omega_b`, `h`, `ln10A_s`, `n_s` (constraint: `omega_m_domain`).

## Running

```bash
# Default dr1 grid
./submit.sh train num_tracers base

# A generated pool, sized to it
./submit.sh train num_tracers base \
    --design-args-path design_args_budget_1p0_1p25.yaml \
    --n-particles-per-device 29

# One budget point, scaling mode
./submit.sh train num_tracers base \
    --mlflow-exp uniform_increase_base_scaling \
    --likelihood-mode scaling \
    --design-args-path design_args_nominal_p25.yaml \
    --n-particles-per-device 5000 --time 05:00 --queue regular
```

```bash
# ShapeFit, from the base entry
./submit.sh train num_tracers base \
    --analysis shapefit --prior-args-path prior_args_shapefit.yaml \
    --mlflow-exp shapefit_base
```

ShapeFit runs from the `base` entry rather than an entry of its own. `train.py` reads its defaults from `train_args.yaml[cosmo_model]`, so a separate entry with `cosmo_model: base` would inherit `base`'s values for any key it set to null or false.

Any `train_args.yaml` key is overridable on the command line (`--likelihood-mode`, `--design-args-path`, `--n-particles-per-device`, …).
