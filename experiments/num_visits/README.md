# NumVisits Experiment

This experiment optimizes the per-filter visit allocation for the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST), maximizing the expected information gain (EIG) for photometric redshift inference. The design variable is the number of visits in each LSST broadband filter; the goal is to find allocations that are more informative about galaxy redshift than the baseline survey strategy.

## Problem Description

LSST images the southern sky through six broadband filters spanning the optical to near-infrared:

| Filter | Wavelength range (nm) | Nominal visits (10 yr) |
|--------|-----------------------|------------------------|
| u      | 320 -- 400            | 70                     |
| g      | 400 -- 550            | 100                    |
| r      | 550 -- 700            | 230                    |
| i      | 700 -- 820            | 230                    |
| z      | 820 -- 920            | 200                    |
| y      | 950 -- 1050           | 200                    |

By observing each galaxy in multiple filters we measure its broadband colors (magnitude differences), which encode the redshift through the shifting of spectral features across the filter set. More visits in a given band reduce photometric noise, but the total number of visits is constrained by the survey schedule.

The Bayesian experimental design problem has three components:

1. **Designs** -- the number of visits per filter: `d = {N_u, N_g, N_r, N_i, N_z, N_y}`
2. **Parameters** -- the redshift of the source galaxy: `theta = {z}`
3. **Features** -- the observed apparent magnitudes: `y = {m_u, m_g, m_r, m_i, m_z, m_y}`

## Design Space (`design_args.yaml`)

The design configuration controls which filters are varied and how the grid of candidate designs is built.

| Field                | Type              | Description |
|----------------------|-------------------|-------------|
| `labels`             | list of strings   | Filter bands included in the design (e.g. `["u","g","r","i","z","y"]`). Filters not listed are held at their nominal values. |
| `input_type`         | string            | `"variable"` builds a grid over the bands; `"nominal"` uses a single fixed design equal to the fiducial visit counts. |
| `input_designs_path` | string or null    | Absolute path to a `.npy` file of explicit design points (shape `(n_designs, n_filters)`). When set, the grid parameters below are ignored. |
| `step`               | float or list     | Grid spacing for each filter. A scalar applies to all filters; a list sets per-filter spacing. |
| `lower`              | float or list     | Lower bound on visits for each filter (scalar or per-filter list). |
| `upper`              | float or list     | Upper bound on visits for each filter (scalar or per-filter list). |
| `sum_lower`          | float or null     | Minimum total visits across all varied filters. Defaults to the nominal total if null. |
| `sum_upper`          | float or null     | Maximum total visits across all varied filters. Defaults to the nominal total if null. |

The `sum_lower` / `sum_upper` constraint prunes the Cartesian grid so that only designs whose visit counts sum to a value within `[sum_lower, sum_upper]` are retained. Setting both to the same value (e.g. `1030`) enforces a fixed total-visit budget.

### Variant files

- **`design_args_2d.yaml`** -- varies only `u` and `g` (2 filters), with `sum_lower = sum_upper = 170`.
- **`design_args_3d.yaml`** -- varies `u`, `g`, and `r` (3 filters), with `sum_lower = sum_upper = 400`.

These are useful for low-dimensional visualization or faster exploratory runs.

## Parameters (`prior_args.yaml`)

Prior configuration files define the parameter space and their prior distributions.

### Structure

```yaml
parameters:
  <name>:
    distribution:
      type: "<distribution_type>"
      # distribution-specific fields
    plot:
      lower: <float>
      upper: <float>
    latex: "<LaTeX label>"

constraints: {}

prior_flow_path: null   # optional
```

Each parameter entry contains:

- **`distribution`** -- specifies the prior type and its hyperparameters:
  - `uniform`: requires `lower` and `upper`.
  - `gamma`: requires `shape` (concentration) and `z_0` (scale; internally converted to rate = 1/z_0). Models the LSST-style redshift distribution `p(z) ~ (z/z_0)^(shape-1) * exp(-z/z_0)`.
  - `gaussian`: requires `loc` (mean) and `scale` (standard deviation).
- **`plot`** -- plotting range (`lower`, `upper`) used for posterior visualization.
- **`latex`** -- LaTeX string for axis labels.

**`constraints`** is a placeholder dictionary (currently unused).

**`prior_flow_path`** (optional) -- absolute path to a trained normalizing-flow checkpoint. When set, the flow is loaded and used as the prior instead of the analytic distribution, enabling sequential experimental design where the posterior from a previous round becomes the prior for the next.

### Prior files

- **`prior_args_uniform.yaml`** -- `z ~ Uniform(0.1, 3.0)`.
- **`prior_args_gamma.yaml`** -- `z ~ Gamma(shape=3.0, z_0=0.3)`, a more realistic galaxy redshift distribution.

## Likelihood Model

The forward model is implemented in `src/bedcosmo/num_visits/experiment.py` and consists of three stages.

### 1. Magnitude calculation (`_calculate_magnitudes`)

A blackbody spectral energy distribution (SED) at a fixed temperature (default 5000 K) is assumed for all galaxies. The source luminosity is set by normalizing the blackbody to a fixed bolometric luminosity of 10^9 L_sun via an effective radius `R_eff`. Given a redshift `z`:

1. Each wavelength in a common grid is shifted to the rest frame: `lambda_rest = lambda_obs / (1 + z)`.
2. The rest-frame blackbody surface flux `F(lambda_rest, T)` is computed and scaled to a luminosity `L = 4 * pi * R_eff^2 * F`.
3. The observed flux is `f = L / [(1 + z) * 4 * pi * D_L(z)^2]`, where `D_L` is the luminosity distance (Planck18 flat LCDM: H0 = 67.4 km/s/Mpc, Omega_m = 0.315).
4. The photon flux in each LSST filter is obtained by integrating `f * T_b(lambda) * lambda / (hc)` over the filter transmission curve `T_b` (loaded via `speclite`).
5. Fluxes are converted to AB magnitudes using the LSST photometric zeropoints from SMTN-002.

### 2. Magnitude errors (`_magnitude_errors`)

Photometric uncertainties are derived from a matched-filter signal-to-noise calculation on a simulated postage stamp:

1. A Gaussian galaxy profile (sigma = 2 arcsec) convolved with a Gaussian PSF (FWHM = 0.67 arcsec) is rendered onto a pixel grid (pixel scale 0.2 arcsec, stamp size 31x31) using GalSim.
2. Source counts per pixel are computed from the model magnitude, the zeropoint, and the total exposure time (`N_visits * t_visit`, where `t_visit = 30 s` by default: 2 exposures of 15 s each).
3. Noise variance per pixel sums four terms:
   - **Sky background**: `sbar * N_visits * t_visit * (pixel_scale)^2`, where `sbar` is the dark-sky brightness in counts/s/arcsec^2.
   - **Source Poisson noise**: the pixel counts themselves.
   - **Dark current**: `dark_current * N_visits * t_visit` (default 0.2 e-/s).
   - **Read noise**: `read_noise^2 * N_visits * n_exp` (default 8.8 e- per read, 2 reads per visit).
4. A matched-filter SNR is computed: `SNR = sum(pixel^2) / sqrt(sum(pixel^2 * variance))`, summing over pixels above a detection threshold.
5. The magnitude error is `sigma_m = (2.5 / ln 10) / SNR`, with an optional systematic floor `sigma_sys` added in quadrature. Sources with SNR too low for a meaningful error are assigned `sigma_m = 10` (effectively undetectable).

### 3. Generative model (`pyro_model`)

The Pyro probabilistic model ties the components together:

1. Sample `z` from the prior (analytic distribution or prior flow).
2. Compute mean magnitudes `mu = _calculate_magnitudes(z)` for each filter.
3. Compute per-filter magnitude errors `sigma = _magnitude_errors(mu, N_visits)`.
4. Build a diagonal covariance matrix `C = diag(sigma^2)`.
5. Sample observed magnitudes `y ~ MultivariateNormal(mu, C)`.

## Cosmology Models (`models.yaml`)

Defines which parameters belong to each named model variant. Currently there is a single model:

```yaml
base:
  parameters: ["z"]
  latex_labels: ["z"]
```

Additional model variants (e.g. adding galaxy SED parameters) can be added here.
