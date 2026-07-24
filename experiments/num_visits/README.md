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
- **`prior_args_bbt.yaml`** -- the gamma `z` prior plus a broad `T ~ Uniform(4000, 10000)` K (for `bbt`). Band normalization lets `T` span a wide color range without cool sources dropping out of the LSST bands.
- **`prior_args_bbtm.yaml`** -- the gamma `z` prior plus `T ~ Uniform(4000, 10000)` K and `M_ref ~ Uniform(-22, -20)` (for `bbtm`). `M_ref` is the rest-frame AB absolute magnitude at `ref_wavelength`; under `norm_mode: band` the SED amplitude is pinned to it, so brightness (`M_ref`) and color (`T`) are decoupled and cooling does not dim the source in-band. The range straddles a bright super-L* galaxy (-22) and a fainter one (-20); `M* ~ -21` for the local luminosity function.
- **`prior_args_empirical.yaml`** -- the empirical SED prior over the 13D ILR features (`f1..f11`, `log_c_scale`, `z`); `prior_source` selects `{kde, flow}` and **defaults to `flow`**. See the [Cosmology Models](#cosmology-models-modelsyaml) section (`empirical`) for details.

## Likelihood Model

The forward model is implemented in `src/bedcosmo/num_visits/experiment.py` and consists of three stages. The **signal SED** (§1) has two variants selected by `--cosmo-model`: a blackbody (`bb`, `bbt`) or a data-driven EAZY template mixture (`empirical`, §1b). The **noise model** (§2) and the diagonal-Gaussian sampling (§3) are shared by both.

### 1. Magnitude calculation — blackbody (`_observed_spectral_flux`, `_calculate_magnitudes`)

Every galaxy is modeled as a single blackbody at temperature `T`. This is a deliberate **toy SED** — it lacks the 4000 Å break, nebular emission lines, and dust of a real galaxy (use the `empirical` model, §1b, for a data-driven alternative) — but it gives a smooth, differentiable map from `(z, T, brightness)` to LSST magnitudes for exercising the design pipeline.

A blackbody's **shape** is fixed by `T`; its **amplitude** (how luminous the source is) needs a separate convention, selected by `norm_mode`.

#### SED normalization (`norm_mode`)

**`band` (default) — anchor an in-band absolute magnitude.** Used by `bbt` and `bbtm`. The rest-frame spectral luminosity at a reference wavelength `ref_wavelength` (default 5000 Å, rest-frame optical) is pinned to a fixed rest-frame **AB absolute magnitude** `M_ref`:

1. `M_ref → L_lambda(lambda_ref)` via the AB definition (`m_AB = -2.5 log10(f_nu) - 48.6`, "absolute" referenced to 10 pc).
2. Scale the blackbody to hit that anchor: `4*pi*R^2 = L_lambda(lambda_ref) / B(lambda_ref, T)`. **`T` enters the normalization here** — it sets how much to scale the curve so its value at `lambda_ref` equals the fixed in-band luminosity.
3. The spectral luminosity at every wavelength is then `L_lambda(lambda) = 4*pi*R^2 * B(lambda, T)`.

This **decouples brightness from color**: `M_ref` controls how bright the source is *in the band LSST sees*, and `T` controls the SED shape, independently. A cooler source is **not** dimmed out of the optical the way a fixed-bolometric source is — that dimming was an artifact of holding the wavelength-integrated total fixed while the blackbody peak moved into the rest-IR.

`L_bol` and the emitting radius are **derived diagnostics, not inputs**. `L_bol = 4*pi*R^2 * sigma*T^4 = L_ref * BC(T)` (a bolometric correction that floats with `T`), and *neither appears on the flux path* — the observed magnitudes depend only on `L_lambda` at the filter wavelengths, never on the bolometric total.

`M_ref` is a **single variable** whose status is decided by whether a model lists it in `models.yaml`:

- **`bbt`** does not list it → `M_ref` is a fixed experiment constant (set in `train_args`, default `-21 ≈ M*`). Only `z` and `T` are sampled; every galaxy shares the same in-band brightness.
- **`bbtm`** lists it → `M_ref` is a sampled per-galaxy parameter with a prior. A prior on `M_ref` *is* a luminosity-function prior; giving it realistic width breaks the zero-scatter standard candle, so colors (hence the per-band visit allocation) carry real weight.

**`bolometric` (legacy) — anchor the total luminosity.** Used by `bb`. Normalize to a fixed bolometric luminosity `l_bol` (default `1e9` L_sun) and back-solve the radius from Stefan-Boltzmann, `R_eff = sqrt(l_bol / (4*pi*sigma*T^4))`, making the source a zero-scatter standard candle whose SNR scale is set by `l_bol`. Here `T` and `l_bol` are experiment-level scalars (`bb` samples only `z`).

#### From SED to magnitudes

Given the normalized rest-frame SED and a redshift `z`:

1. Shift the common wavelength grid to the rest frame: `lambda_rest = lambda_obs / (1 + z)`.
2. Evaluate the spectral luminosity `L_lambda(lambda_rest) = 4*pi*R^2 * B(lambda_rest, T)` (per the normalization above).
3. Observed flux: `f = L_lambda / [(1 + z) * 4*pi * D_L(z)^2]`, where `D_L` is the luminosity distance (Planck18 flat LCDM: H0 = 67.4 km/s/Mpc, Omega_m = 0.315). Distance dimming (`1/D_L^2`) is what carries redshift information through *brightness*; colors carry it through the wavelength shift.
4. The photon flux in each LSST filter is obtained by integrating `f * T_b(lambda) * lambda / (hc)` over the filter transmission curve `T_b` (loaded via `speclite`).
5. Fluxes are converted to AB magnitudes using the LSST photometric zeropoints from SMTN-002.

### 1b. Magnitude calculation — empirical EAZY template mixture (`_observed_spectral_flux`, `_calculate_magnitudes`)

The `empirical` cosmo-model replaces the blackbody with a **data-driven mixture of 12 EAZY templates** fit to DESI galaxies (see [`empirical/README.md`](../../src/bedcosmo/num_visits/empirical/README.md)). The rest-frame SED is a non-negative weighted sum of the template spectra; the parameters are the mixture weights (on the simplex), an overall amplitude, and the redshift.

Parameters (13D, **ILR** parameterization; `K = 12` templates):

- `f1`…`f11` — isometric-log-ratio (ILR) coordinates of the 12 simplex weights. `a = ilr_to_weights_torch(f1..f11)` maps the 11 ILR coordinates to 12 non-negative weights that sum to 1 (`bedcosmo.num_visits.empirical.simplex`).
- `log_c_scale` — log overall amplitude. Raw per-template coefficients are `c_k = exp(log_c_scale) * a_k` (`_coeffs_from_a_log_s`).
- `z` — redshift.

Given these, the observed-frame flux density is computed by `_observed_spectral_flux`:

1. Shift the common wavelength grid to the rest frame: `lambda_rest = lambda_obs / (1 + z)`.
2. Interpolate each of the 12 templates onto `lambda_rest` and take the weighted sum, with the `(1 + z)` bandpass factor: `flux(lambda_obs) = (1 / (1 + z)) * sum_k c_k * T_k(lambda_rest)` (a single batched `einsum` over templates).
3. Convert to AB magnitudes with the same filter integration + zeropoints as steps 4–5 of §1 (`_calculate_magnitudes`).

Unlike the blackbody model, absolute brightness is a **free amplitude** (`log_c_scale`) rather than a luminosity-distance normalization: redshift enters only through the wavelength shift and the `1/(1 + z)` factor — i.e. through **colors**, not a distance modulus. Photometric redshift is therefore constrained by how spectral features move across the six filters, marginalizing over template shape (`f1..f11`) and amplitude (`log_c_scale`).

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

The Pyro probabilistic model ties the components together. The mean magnitudes are built differently per cosmo-model, then a shared noise model is applied.

For the blackbody models (`bb`, `bbt`, `bbtm`):

1. Sample `z` (plus `T` for `bbt`, plus `T` and `M_ref` for `bbtm`) from the prior (analytic distribution or prior flow).
2. Compute mean magnitudes `mu = _calculate_magnitudes(z)` for each filter (§1).

For the `empirical` model:

1. Draw a feature row `(f1..f11, log_c_scale, z)` from the empirical SED prior (`sed_prior.sample_batch`; KDE or flow per [`prior_args_empirical.yaml`](prior_args_empirical.yaml)) and record each parameter as a `Delta` sample site.
2. Decode to physical SED quantities `(a, log_c_scale, z)` and compute mean magnitudes `mu` via the template mixture (§1b).

Both variants then share the same noise model:

3. Compute per-filter magnitude errors `sigma = _magnitude_errors(mu, N_visits)` (§2).
4. Build a **diagonal** covariance matrix `C = diag(sigma^2)` — the six band magnitudes are independent given the SED.
5. Sample observed magnitudes `y ~ MultivariateNormal(mu, C)`.

## Cosmology Models (`models.yaml`)

Defines which parameters belong to each named training variant. `train_args.yaml` selects a block via `--cosmo-model`.

| Model | Parameters | SED forward model | Prior |
|-------|------------|-------------------|-------|
| `bb` | `z` | Blackbody (fixed T, bolometric) | Analytic gamma on `z` |
| `bbt` | `z`, `T` | Blackbody (band-normalized, fixed `M_ref`) | Analytic gamma + uniform `T` |
| `bbtm` | `z`, `T`, `M_ref` | Blackbody (band-normalized) | Analytic gamma + uniform `T` + uniform `M_ref` |
| `empirical` | `f1`…`f11`, `log_c_scale`, `z` (13D; ILR simplex coords, see §1b) | EAZY template mixture | Empirical SED prior — KDE or normalizing flow (`artifacts/empirical/`; rebuild with `--parameterization ilr`) |

Example:

```yaml
empirical:
  parameters: [f1, ..., f11, log_c_scale, z]  # ILR coords → a1..a12 weights
  latex_labels: ["$a_1$", ..., "$z$"]
```

For `empirical`, the prior is selected by `prior_source` in [`prior_args_empirical.yaml`](prior_args_empirical.yaml) — a strict enum `{kde, flow}` that **defaults to `flow`**. Both are fit to the DESI/EAZY template weights by `src/bedcosmo/num_visits/empirical/`:

- **`flow` (default)** — a trained normalizing flow over the ILR features. Empirical runs **require** the trained flow checkpoints (`sed_prior_flow_native.pt`, plus `sed_prior_flow_gaussianized.pt` when `transform_input=True`) in `$SCRATCH/bedcosmo/num_visits/empirical_prior/`; they are snapshotted into each run's `artifacts/empirical/`. Train them (~10 min, CPU, both at once) with `./scripts/train_prior_flow.sh --space both`.
- **`kde`** — a masked KDE (`sed_prior_kde_native.joblib`); the legacy/fallback path.

Either way, training draws from a GPU-resident pool of prior samples (drawn from the flow, or KDE samples) and integrates the template SEDs on the GPU. See [`empirical/README.md`](../../src/bedcosmo/num_visits/empirical/README.md).

## Marginal EIG over parameter subsets (`eval_args.yaml`)

In addition to the joint EIG over all cosmological parameters, evaluation can
compute the **marginal EIG** over a chosen subset `S` of parameters,
marginalizing over the rest. For the `empirical` model this is useful to ask,
e.g., how informative a design is about `log_c_scale` and `z` alone while
marginalizing the SED mixture weights `f_i`.

For a subset `S`:

```
EIG_S(d) = H[p(θ_S)] − E_y[ H[ q(θ_S | y, d) ] ]
```

Because the trained guide `q(θ|y,d)` is a joint normalizing flow and the
empirical prior is a joint density (normalizing flow by default, or KDE),
neither marginal is available in closed form.
Both entropies are estimated from samples in **physical** parameter space with a
k-nearest-neighbor (Kozachenko–Leonenko) estimator (`src/bedcosmo/entropy.py`).
The marginal posterior entropy is a nested Monte Carlo estimate: outer samples
`y ~ p(y|d)` and, for each, `K` guide samples whose subset marginal entropy is
estimated and averaged.

For the empirical prior (flow or KDE), prior rows are drawn **without replacement**
from the GPU sample pool (up to pool size) so k-NN entropy is not biased by
duplicate pool rows. The prior sample count tracks `marginal_inner_samples * marginal_outer_y`
(with a floor of 4096), matching the posterior MC depth rather than a fixed 20k.
Per design, inner posterior samples are drawn for each outer ``y ~ p(y|d)``;
k-NN entropy is computed per outer ``y`` (on ``K`` samples) and averaged over
``M`` outers, matching ``E_y[H(q(theta_S|y,d))]``. Do not pool ``K×M`` rows
into one k-NN call — that estimates mixture entropy ``H[q(theta_S|d)]``, which
is larger and biases marginal EIG low (often negative).

YAML fields (under a cosmo-model block):

| Field | Meaning | Default |
|-------|---------|---------|
| `marginal_eig_subsets` | List of subsets; each inner list is the param names to keep (e.g. `[[log_c_scale, z]]`). Empty/absent disables the feature. | none |
| `marginal_outer_y` | Outer `y ~ p(y|d)` samples per design | 8 |
| `marginal_inner_samples` | Guide samples `K` per outer `y` | 200 |
| `marginal_knn_k` | Neighbor rank `k` for the k-NN estimator | 3 |

CLI override (manual eval, or `--eval-`-prefixed for auto-eval):

```bash
./submit.sh eval num_visits <run_id> --marginal-eig-subsets "log_c_scale,z"
# multiple subsets: semicolon-separated groups, or a JSON list-of-lists
./submit.sh eval num_visits <run_id> --marginal-eig-subsets "log_c_scale,z; f1,f2"
```

Results are written into the same `eig_data` JSON under
`step_{N}["marginal"][subset_id]` (where `subset_id = "+".join(params)`), with
per-design `eigs_avg`/`eigs_std`, the nominal-design value, and the marginal
prior entropy. Evaluation also produces `eig_designs_marginal_<subset_id>` (EIG
vs design for the subset, over all input designs) and
`posterior_marginal_<subset_id>` (triangle plot restricted to the subset).
