# VariableRedshift Experiment

This experiment chooses the redshifts at which to measure BAO distances so that the measurements carry the most expected information gain (EIG) about cosmological parameters. It is a small, fast toy problem compared with `num_tracers` and `num_visits`. It is useful for testing the training and evaluation pipeline, and the grid calculation can check it against brute-force EIG.

## Problem Description

A BAO measurement at redshift `z` constrains the Hubble distance `D_H(z)/r_d` and, optionally, the transverse comoving distance `D_M(z)/r_d`. Different cosmological parameters leave their strongest imprint at different redshifts, so where you observe matters.

1. **Designs**: the redshifts observed, `d = {z_1, ..., z_n}` (the number of redshifts is set by `labels` in the design args)
2. **Parameters**: cosmological parameters, e.g. `theta = {Om, hrdrag}` for `base`
3. **Features**: `y = {D_H/r_d}` at each redshift, plus `D_M/r_d` when `include_D_M: true`

The code is `src/bedcosmo/variable_redshift/experiment.py` (`VariableRedshift`). Distances come from `CosmologyMixin.D_H_func` / `D_M_func` in `src/bedcosmo/cosmology.py`, which include radiation and massive neutrinos.

## Design Space (`design_args.yaml`)

| Field | Type | Description |
|---|---|---|
| `labels` | list of strings | One label per observed redshift, e.g. `["z_1", "z_2"]`. The length sets `n_redshifts`. |
| `input_type` | string | `"variable"` builds a grid. |
| `input_designs_path` | string or null | **Absolute** path to a `.npy` of explicit designs, shape `(n_designs, n_redshifts)`. When set, the grid fields are ignored. |
| `step` | float | Grid spacing in `z`. |
| `lower` | float | Lowest redshift in the grid. |
| `upper` | float | Highest redshift. **Inclusive** here (`arange(lower, upper + step, step)`), unlike `num_tracers`. |

With more than one redshift, the grid is the Cartesian product over labels, **deduplicated by permutation**. `(z_1, z_2) = (0.5, 1.5)` and `(1.5, 0.5)` are the same design, so each design is stored sorted.

The **nominal design** is the midpoint of `[lower, upper]` for one redshift, or `n_redshifts` evenly spaced points from `lower` to `upper` otherwise. Posterior plots and `central=True` data use it, with distances evaluated at the Planck18 fiducial (`PLANCK18_FIDUCIAL`, overridable with `central_params`). Central values, including overrides such as `--central-param-hrdrag`, are in reported units with the multiplier applied, like posterior samples. For example, `hrdrag` is `H_0 r_d` in km/s (Planck18: 9907.9), whichever prior file is used.

## Parameters (`prior_args*.yaml`)

All priors are uniform. Each entry under `parameters` has a `distribution` (`lower`, `upper`), an optional `multiplier`, `plot` ranges and a `latex` label. The `plot` range (physical units) is the posterior plot window and its outlier fence: samples outside it are drawn as edge `x` markers and counted in the legend. The prior range is in sampling units, and the physical value is `multiplier × sample`. The distance functions apply `hrdrag_multiplier`, and posterior samples are reported in physical units.

Constraints are enforced by sampling the affected pair jointly from `ConstrainedUniform2D`:

- `valid_densities`: `0 < Om + Ok < 1`
- `high_z_matter_dom`: `w0 + wa < 0`

| File | Use |
|---|---|
| `prior_args_hrdrag.yaml` | Default in `train_args.yaml`. `hrdrag` sampled in `[0.1, 10]` with multiplier `10000` |
| `prior_args.yaml` | `hrdrag` sampled in `[10, 1000]` with multiplier `100` |

Both files give the same physical `H_0 r_d` range (1,000 to 100,000 km/s) and differ only in the scale the flow sees.

A `prior_flow` key (absolute path to a trained flow checkpoint) replaces the uniform prior with a previous run's posterior.

## Likelihood Model

`pyro_model` samples parameters, computes `D_H/r_d` (and `D_M/r_d`) at each design redshift, and draws features from a multivariate normal with a **diagonal** covariance. The error model depends on whether an error table exists:

- **Error table present** (`~/data/variable_error.csv`, columns `z`, `DH_errors`, `DM_errors`): errors are **fractional**. They are interpolated in `z`, multiplied by `error_scale`, and multiplied by the predicted distance.
- **No table:** errors are **constant absolute** values `sigma_D_H` / `sigma_D_M`, and `error_scale` has no effect.

The table lives outside the repository. Whether it exists on the machine running the job silently changes the likelihood. `variable_error.ipynb` in `notebooks/` shows how it was made.

Two code paths always use the constant `sigma_D_H` / `sigma_D_M`, even when the table exists:

- `sample_data(..., central=True)`
- `unnorm_lfunc`, the likelihood used by the grid calculation. It also reads only the **first** design label, so grid EIG supports one redshift only.

Keep this in mind when comparing flow and grid EIG.

## Cosmology Models (`models.yaml`)

| Model | Parameters |
|---|---|
| `base_om` | `Om` |
| `base` | `Om`, `hrdrag` |
| `base_omegak` | `+ Ok` (constraint: `valid_densities`) |
| `base_w` | `+ w0` |
| `base_w_wa` | `+ w0`, `wa` (constraint: `high_z_matter_dom`) |
| `base_omegak_w_wa` | all five (both constraints) |

Parameters a model leaves out are held at their fiducial values (`Ok = 0`, `w0 = -1`, `wa = 0`).

## Running

```bash
# Two-redshift design grid, default prior
./submit.sh train variable_redshift base

# Evolving dark energy, shorter run
./submit.sh train variable_redshift base_w_wa --total-steps 50000
```

For brute-force grid EIG, see "Grid EIG Calculation" in the top-level `README.md`. The one-redshift limit above applies.

`notebooks/` holds exploratory analyses: brute-force EIG (`brute_force.ipynb`), flow training diagnostics (`nf_train_analysis.ipynb`), and Gaussian-process variants (`gp_hubble_dist_*.ipynb`).
