# Num-tracers neural emulators (BAO & ShapeFit)

This package builds PyTorch regressors that emulate **desilike** outputs: Fisher covariance elements for BAO / ShapeFit observables (and, for ShapeFit only, mean ShapeFit parameters). Training data are `.npz` grids of inputs `x` and targets `y`; scripts live next to per-analysis subpackages `bao/` and `shapefit/`.

## Requirements

- **`desilike`** (and its dependencies, including JAX where Fisher workers use it).
- **`SCRATCH`** environment variable for default data and MLflow paths (`get_default_save_path`, training logs). If it is unset, pass explicit `--save-path` to prep scripts and `--save-path` / absolute paths for `eval_nn.py` as needed.

Default layout (when `SCRATCH` is set):

- Training data: `$SCRATCH/bedcosmo/num_tracers/emulator/training_data/{analysis}/{cosmo_model}/{quantity}/v{N}/`
- MLflow runs: `file:$SCRATCH/bedcosmo/num_tracers/emulator/mlruns`

Run scripts from this directory (`emulator/`) so imports such as `util` → `bao` / `shapefit` resolve correctly:

```bash
cd src/bedcosmo/num_tracers/emulator
```

---

## Analysis directories (`bao/`, `shapefit/`)

| Directory   | Role |
|------------|------|
| **`bao/`** | BAO Fisher covariance emulator: inputs include survey / cosmology parameters (e.g. `N_tracers`, `Om`, `hrdrag`, …); targets are upper-triangular BAO covariance elements `cov_DH_over_rd_*`, `cov_DM_over_rd_*`. Only **`covar`** training is supported (`--quantity covar`). |
| **`shapefit/`** | ShapeFit emulators: **`prep_covar.py`** → Fisher covariance for `(qiso, qap, f_sigmar, m)`; **`prep_mean.py`** → mean ShapeFit parameters from `ShapeFitPowerSpectrumExtractor` (no `N_tracers` in prior). |

Shared per-analysis files:

| File | Purpose |
|------|---------|
| **`model_config.yaml`** | Named blocks of NN and optimizer hyperparameters. The key you pass as `--nn-model` (or the default derived from `--cosmo-model`) selects one block. |
| **`model.py`** | PyTorch module registered in `util.ARCHITECTURE_REGISTRY` (currently **`resnet`**: SiLU residual MLP — `ResNetRegressor` for BAO, `base_regressor` for ShapeFit). |
| **`prep_covar.py`** | CLI to sample priors, run Fisher forecasts in parallel (optional `--workers`), and write `train.npz` / `test.npz` under `v{N}/`. Exposes `run_fisher`, `TARGET_NAMES`, `DEFAULT_PRIORS` for `util.get_pipeline` (used by **`eval_nn.py`**). |
| **`run_prep_covar.sh`** | Optional shell wrapper for batch jobs. |
| **`run_single_fisher.py`** | Debugging / one-off Fisher runs. |

ShapeFit-only:

| File | Purpose |
|------|---------|
| **`prep_mean.py`** | Samples cosmological priors, runs the ShapeFit extractor, saves mean targets `qiso`, `qap`, `f_sigmar`, `m`. Used for `--quantity mean` training and evaluation. |

---

## `model_config.yaml`

Top-level keys are arbitrary labels (e.g. `base`, `base_scaled`, `base_omegak_w_wa`). Training selects a block via:

1. **`--nn-model <key>`** if set, else  
2. **`--cosmo-model`** as the key, else  
3. If that key is missing, **`--nn-model` was not set**, and the file contains a **`default`** key, that block is used.

For **ShapeFit**, `shapefit/model_config.yaml` currently defines only **`base`**. If you train with `--cosmo-model base_w_wa`, either add a `base_w_wa` block to the YAML or pass **`--nn-model base`** to reuse the shared hyperparameters.

Typical fields (all consumed by `train_nn.py` where applicable):

- **`architecture`**: must exist in `util.ARCHITECTURE_REGISTRY` for that analysis (e.g. `resnet`).
- **`hidden_dim`**, **`n_hidden`**, **`expand`**, **`dropout`**: passed into the regressor.
- **`lr`**, **`final_lr_frac`**, **`weight_decay`**, **`batch_size`**
- **`warmup_fraction`**, **`scheduler_type`** (`constant`, `cosine`, `linear`, `exponential`, `lambda`), **`lr_restarts`**, **`lr_gamma`** (for the lambda-style schedule)
- **`grad_clip`**: global norm clipping (default in code if omitted: `1.0`)

BAO’s YAML may use keys that do not match `cosmo-model` names (e.g. `base_scaled`); use **`--nn-model`** explicitly in that case.

---

## `prep_covar.py` (BAO and ShapeFit)

Both scripts share most flags. They write **`{prefix}train.npz`** / **`{prefix}test.npz`** with arrays **`x`**, **`y`**, **`param_names`**, **`target_names`**. Prefix is empty unless **`--name`** is set (e.g. `LRG2` → `LRG2_train.npz`).

| Argument | Description |
|----------|-------------|
| `--n-samples` | Number of accepted prior samples (default `5000`). |
| `--batch-size` | LHS batch size for generation loops. |
| `--seed` | RNG seed. |
| `--test-size` | Fraction held out for `test.npz` (default `0.2`). |
| `--save-path` | Root directory; default from `get_default_save_path(analysis=..., quantity="covar", cosmo_model=...)`. |
| `--sigma-clip` | Truncation for normal priors in LHS (default `4.0`). |
| `--verbose-every` | Progress print interval. |
| `--workers` | Parallel worker processes (`1` = serial). |
| `--zrange Z_MIN Z_MAX` | Footprint redshift bin (default `1.2 1.4`). |
| `--z-eff` | Optional effective redshift (else midpoint of `zrange`). |
| `--name` | Tracer prefix for output filenames. |
| `--version` | Force `v{N}`; else auto-increment. |
| `--ntracers-range LOW HIGH` | Override `N_tracers` uniform bounds. |
| `--priors-json` | Full prior dict as JSON (must be self-consistent with varied parameters). |
| `--cosmo-model` | **BAO**: `base`, `base_w`, `base_w_wa`, `base_omegak`, `base_omegak_w_wa` (default `base_omegak_w_wa`). **ShapeFit**: `base`, `base_w_wa` (default `base`). |

**BAO only:** `--tracer-bin` selects a key in `tracers.yaml` for reconstruction / nuisance defaults (default `LRG2`).

BAO additionally applies **constraint** filters from `CONSTRAINTS` when the relevant parameters are both varied.

ShapeFit `prep_covar` can pass a **`checkpoint_fn`** to save intermediate datasets during long runs.

---

## `prep_mean.py` (ShapeFit only)

| Argument | Description |
|----------|-------------|
| `--n-samples` | Default `10000`. |
| `--batch-size` | Default `256`. |
| `--seed`, `--test-size`, `--save-path`, `--sigma-clip`, `--verbose-every` | Same idea as `prep_covar`. |
| `--priors-json` | Optional full prior override. |
| `--cosmo-model` | `base` or `base_w_wa` (default `base`). |

Outputs go to `training_data/shapefit/{cosmo_model}/mean/v{N}/`.

---

## Top-level components

| File | Role |
|------|------|
| **`util.py`** | `build_model`, `get_default_save_path`, `get_pipeline` (loads `prep_covar` / `prep_mean` for ground truth), `save_dataset`, LHS sampling, tracer bins for **`--tracer-bin`**. |
| **`train_nn.py`** | Loads YAML + `.npz` data, standardizes inputs/targets, trains with MLflow logging, saves checkpoints under the run’s artifacts. |
| **`eval_nn.py`** | Loads a checkpoint, draws LHS parameters, compares NN to `get_pipeline` ground truth, writes diagnostic plots. |
| **`scale_data.py`** | Post-processes a directory of `.npz` files: multiplies `y` by user-defined factors of input variables; writes a sibling directory and **`scale_info.json`** (used at train/eval time to track scaling). |
| **`test_cov_scaling.py`** | Tests for the scale expression language. |
| **`run_pipeline.py`** | **Legacy**: docstring and paths point to old `prep_shapefit_data.py` / different CLI; prefer explicit prep → train → eval. |
| **`notebooks/`** | Exploration (training, errors, shapefit extractor). |

---

## `scale_data.py`

Positional usage (not `argparse`):

```bash
python scale_data.py <data_dir> <expr1> [expr2 ...]
```

Each **expression** is an infix formula using `+ - * / ^ **`, parentheses, `exp(...)`, `log(...)`, and variables among: **`N_tracers`**, **`Om`**, **`Ok`**, **`w0`**, **`wa`**, **`hrdrag`**. Variables must appear in the dataset’s `param_names`.

Output directory: `<data_dir>_<suffix>_scaled` where `suffix` encodes the expressions. A **`scale_info.json`** records `scale_expressions` and `source_dir`. **`train_nn.py`** copies `scale_expressions` into the checkpoint so **`eval_nn.py`** can invert scaling when comparing to the true Fisher / extractor.

---

## `train_nn.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--analysis` | `shapefit` | `shapefit` or `bao`. |
| `--quantity` | `mean` | `mean` or `covar` (must match data and available pipelines; BAO only supports `covar`). |
| `--cosmo-model` | `base` | `base`, `base_w`, `base_w_wa`, `base_omegak`, `base_omegak_w_wa` — selects subdirectory under `training_data/...` and default YAML key. |
| `--nn-model` | *(see above)* | YAML key in `{analysis}/model_config.yaml`. |
| `--data-dir` | `latest` | Absolute path to a data folder, or a folder name under the cosmo/quantity root (e.g. `v3`, or a scaled dir), or `latest` → highest `v{N}`. |
| `--epochs` | `10000` | |
| `--patience` | `0` | Early stopping on test loss (`0` = off). |
| `--seed` | `0` | |
| `--mlflow-exp` | `default` | MLflow experiment name. |
| `--run-name` | `None` | Optional run name. |
| `--log-normalize` | off | Symlog targets before z-score. |
| `--eval-atol`, `--eval-rtol` | `2e-3` | Passed to post-training eval. |
| `--tracer-bin` | `None` | One of `BGS`, `LRG1`, `LRG2`, `LRG3_ELG1`, `ELG2`, `QSO`, `Lya_QSO`: loads `{name}_train.npz` / `{name}_test.npz` if present and scopes eval priors / redshift. |

Training uses the first available CUDA device; evaluation inside `eval_nn.py` uses **`cuda:1`** if CUDA is available, else CPU.

---

## `eval_nn.py`

Exactly one of **`--run-id`**, **`--run-dir`**, or **`--model-path`** must identify a checkpoint (`model_best.pt` preferred, then `model.pt`).

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-id` | `None` | MLflow run ID; resolves artifacts under `file:$SCRATCH/bedcosmo/num_tracers/emulator/mlruns`. |
| `--run-dir` | `None` | Directory containing `checkpoints/model_best.pt` or `model.pt`. |
| `--model-path` | `None` | Direct path to `.pt`. |
| `--n-samples` | `200` | LHS comparison samples. |
| `--seed` | `42` | |
| `--hist-xlims` | `None` | JSON dict mapping target name to `[lo, hi]` for histograms. |
| `--rtol`, `--atol` | `2e-3` | `numpy.allclose` tolerances for outlier reporting. |
| `--log-scale` | off | Log/symlog axes on triangle plots. |
| `--analysis` | `shapefit` | Must match the trained model. |
| `--quantity` | `covar` | `covar` or `mean`. |
| `--save-path` | auto | Plot output directory. |
| `--tracer-bin` | `None` | Same choices as training; aligns priors and `zrange` / `z_eff` with DESI bin definitions. |

---

## Typical workflows

**1. BAO covariance emulator**

```bash
python bao/prep_covar.py --cosmo-model base --n-samples 10000 --workers 8
# optional: python scale_data.py $SCRATCH/.../bao/base/covar/v1 'log(N_tracers)' '1/exp(Om)'
python train_nn.py --analysis bao --quantity covar --cosmo-model base --nn-model base_scaled --data-dir latest
python eval_nn.py --run-id <mlflow_run_id> --analysis bao --quantity covar
```

**2. ShapeFit covariance emulator**

```bash
python shapefit/prep_covar.py --cosmo-model base_w_wa --name LRG2 --zrange 0.6 0.8 --z-eff 0.706
python train_nn.py --analysis shapefit --quantity covar --cosmo-model base_w_wa --data-dir latest --tracer-bin LRG2
python eval_nn.py --run-dir <path_to_run_artifacts> --analysis shapefit --quantity covar --tracer-bin LRG2
```

**3. ShapeFit mean parameters**

```bash
python shapefit/prep_mean.py --cosmo-model base
python train_nn.py --analysis shapefit --quantity mean --data-dir latest
python eval_nn.py --model-path /path/to/model.pt --analysis shapefit --quantity mean
```

---

## `.npz` layout

- **`x`**: `(n_samples, n_params)` float32  
- **`y`**: `(n_samples, n_targets)` float32  
- **`param_names`**, **`target_names`**: string arrays  

After **`scale_data.py`**, the scaled directory also contains **`scale_info.json`**.
