# bedcosmo

Bayesian Experimental Design for Cosmology

## Overview

bedcosmo provides a framework for optimizing cosmological surveys for parameter inference. The goal is to estimate the Expected Information Gain (EIG) by training normalizing flows to approximate densities with variational inference, enabling data-driven optimization of galaxy redshift surveys like LSST/DESI.

The core approach:
1. Train a conditional normalizing flow to approximate the posterior distribution p(θ|x, d) where θ are cosmological parameters, x is observed data, and d is the experimental design
2. Use the trained flow to compute EIG across different designs
3. Identify optimal survey configurations that maximize information about cosmological parameters

## Installation

Dependencies are defined in `pyproject.toml`. Install the package in editable mode with dev extras (tests, linting, notebooks):

```bash
git clone https://github.com/ashandonay/bedcosmo.git
cd bedcosmo
pip install -e ".[dev]"
```

This installs all runtime and development dependencies (PyTorch, JAX, MLflow, cobaya, etc.) from PyPI.

### Optional: Conda environment

If you use conda, create an env with Python 3.10 and install the project (deps come from `pyproject.toml`):

```bash
conda create -n bedcosmo python=3.10 libstdcxx-ng -y
conda activate bedcosmo
pip install -e ".[dev]"
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA 12.4+ (for GPU training)

## Quick Start

### Training a Model

```bash
./submit.sh train num_tracers base
```

This trains a neural flow for the `base` cosmology model (Omega_m, H_0rd parameters) on the `num_tracers` experiment. The script auto-detects whether SLURM is available (submits via `sbatch`) or runs locally (via `torchrun`).

```bash
# SLURM with custom time and queue
./submit.sh train num_tracers base --time 04:00 --queue regular

# Separate train/eval time and queue
./submit.sh train num_tracers base --train-time 04:00 --eval-time 00:30

# Force local execution
./submit.sh train num_tracers base --local --gpus 1

# Debug mode (uses SLURM debug queue / 'debug' MLflow experiment, no auto-eval)
./submit.sh train num_tracers base --debug
```

### Resuming/Restarting Training

```bash
# Resume from a checkpoint (continues the same run)
./submit.sh resume num_tracers <run_id> <step>

# Restart from a checkpoint (creates a new run)
./submit.sh restart num_tracers <run_id> <step>
```

### Auto-Evaluation

Evaluation runs automatically after train/restart/resume jobs complete. The eval job extracts the run ID from the training log and loads defaults from `eval_args.yaml`.

Pass eval-specific arguments with the `--eval-` prefix:

```bash
# Train with grid eval afterwards
./submit.sh train num_tracers base --eval-grid --eval-grid-param-pts 2000 --eval-grid-feature-pts 500

# Custom eval time for SLURM
./submit.sh train num_tracers base --eval-time 01:00 --eval-n-evals 5

# Disable auto-eval
./submit.sh train num_tracers base --no-eval
```

Auto-eval is disabled by default in `--debug` mode. Use `--auto-eval` to force it on.

For SLURM, the eval job is submitted as a dependent job (`afterany`) and checks if training completed successfully before running.

### Manual Evaluation

To evaluate a specific run manually:

```bash
./submit.sh eval num_tracers <run_id>
```

The `run_id` is the MLflow run ID printed when training starts (or find it in the MLflow UI).

## Experiments

### num_tracers

Optimizes the allocation of observing time across DESI tracer types (BGS, LRG, ELG, QSO) for BAO measurements.

**Design space:** Fractional allocation to each tracer type

### variable_redshift

Optimizes redshift-dependent survey strategy for measuring cosmological distance parameters.

**Design space:** Redshift of observations

### num_visits

Optimizes the number of visits per field for photometric surveys such as LSST.

## Cosmology Models

Each experiment supports multiple cosmological parameter spaces:

| Model | Parameters | Description |
|-------|------------|-------------|
| `base` | Ωm, H₀rd | Flat ΛCDM with matter density and sound horizon |
| `base_omegak` | Ωm, Ωk, H₀rd | Non-flat ΛCDM with curvature |
| `base_w` | Ωm, w₀, H₀rd | Flat wCDM with constant dark energy EoS |
| `base_w_wa` | Ωm, w₀, wa, H₀rd | Flat w₀waCDM with evolving dark energy |
| `base_omegak_w_wa` | Ωm, Ωk, w₀, wa, H₀rd | Full model with curvature and evolving dark energy |

## Configuration

Training is configured via YAML files in each experiment directory:

- `train_args.yaml` - Training hyperparameters per cosmology model
- `models.yaml` - Parameter definitions and constraints
- `prior_args.yaml` - Prior distributions for cosmological parameters
- `design_args.yaml` - Design space specifications

### Overriding Configuration

CLI arguments override YAML defaults. Unprefixed args are assumed to be for training. Use `--train-` or `--eval-` prefixes to be explicit:

```bash
# These are equivalent for training args
./submit.sh train num_tracers base --initial-lr 0.0001 --total-steps 300000
./submit.sh train num_tracers base --train-initial-lr 0.0001 --train-total-steps 300000

# Mix train and eval args
./submit.sh train num_tracers base --train-initial-lr 0.0001 --eval-grid --eval-grid-param-pts 2000
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `total_steps` | Total training iterations | 200000 |
| `initial_lr` | Initial learning rate | 0.0005 |
| `final_lr` | Final learning rate (cosine decay) | 0.00001 |
| `n_transforms` | Number of flow transforms | 4-6 |
| `cond_hidden_size` | Conditioning network hidden size | 128-512 |
| `n_particles_per_device` | Batch size per GPU | 30-100 |
| `checkpoint_step_freq` | Steps between checkpoints | 2000 |

## MLflow Tracking

All experiments are tracked with MLflow. Runs are stored at:
```
$SCRATCH/bedcosmo/{cosmo_exp}/mlruns/
```

To view the MLflow UI:
```bash
cd $SCRATCH/bedcosmo/num_tracers
mlflow ui --port 5000
```

Then open `http://localhost:5000` in your browser.

### Logged Artifacts

- Model checkpoints (`checkpoint_rank0_step*.pt`)
- Training metrics (loss, learning rate)
- Hyperparameters
- EIG evaluation results (JSON)

## Adding a New Experiment

1. Create a new directory: `my_experiment/`

2. Implement the experiment class in `my_experiment.py`:
```python
class MyExperiment:
    def __init__(self, cosmo_exp, design):
        # Initialize cosmology and design
        pass

    def likelihood(self, theta, design):
        # Return log-likelihood
        pass
```

3. Create configuration files:
   - `train_args.yaml` - Training parameters
   - `models.yaml` - Cosmology model definitions
   - `prior_args.yaml` - Prior distributions
   - `design_args.yaml` - Design space

4. Train: `./submit.sh train my_experiment base`

## Development

### Running Tests

```bash
pytest                      # All tests with coverage
pytest -m "not slow"        # Skip slow tests
pytest tests/test_util.py   # Specific test file
```

### Code Quality

```bash
black .                     # Format code
ruff check .                # Lint
ruff check --fix .          # Auto-fix lint issues
mypy .                      # Type checking
```

## Troubleshooting

### Job Fails Immediately

Check output logs in `$SCRATCH/bedcosmo/{cosmo_exp}/logs/`

### Out of Memory

Reduce `n_particles_per_device` in the training args or via CLI:
```bash
./submit.sh train num_tracers base --n-particles-per-device 20
```

### MLflow Connection Issues

Ensure the tracking URI is set correctly. The framework uses file-based tracking:
```python
mlflow.set_tracking_uri(f"file:{storage_path}/mlruns")
```

### Resuming from Wrong Checkpoint

Use `--restart-checkpoint` to specify an exact checkpoint file:
```bash
./submit.sh restart num_tracers <run_id> --restart-checkpoint /path/to/checkpoint.pt
```

## Grid EIG Calculation

The grid calculator (`bedcosmo.grid_calc`) computes Expected Information Gain by brute-force enumeration over discretized parameter, feature, and design grids using https://github.com/dkirkby/bayesdesign. This serves as a ground-truth reference for validating the neural flow approach.

### Running

```bash
# Via SLURM
sbatch scripts/slurm/grid_calc.sh

# Directly
python -m bedcosmo.grid_calc num_visits \
    --design-args-path design_args_2d.yaml \
    --prior-args-path prior_args_uniform.yaml \
    --param-pts 1000 --feature-pts 500
```

Key CLI flags:

| Flag | Description |
|------|-------------|
| `--design-args-path` | Design space config (e.g. `design_args_2d.yaml`) |
| `--prior-args-path` | Prior config (e.g. `prior_args_uniform.yaml`). Defaults to the path in `train_args.yaml` |
| `--use-experiment-prior` | Use the experiment's built-in prior instead of a YAML file |
| `--param-pts` | Points per parameter axis (default 75) |
| `--feature-pts` | Points per feature axis (default 35) |
| `--adaptive-features` | Concentrate feature grid points around detectable features |
| `--feature-range NAME:LO,HI` | Override feature axis bounds (e.g. `--feature-range u:-10,60`) |
| `--no-plots` | Skip plot generation |

Outputs are saved to `$SCRATCH/bedcosmo/{cosmo_exp}/grid_calc/{timestamp}/` and include `eig_data_grid.json`, posterior and marginal plots, and the feature grid diagnostic.

### How the Grids Are Built

**Parameter grid:** For each cosmological parameter, the axis spans the prior support (or the 1%–99% quantile range for unbounded priors) with `param-pts` evenly spaced points.

**Design grid:** Taken directly from the experiment's `designs_grid` attribute, which is configured by the design args YAML.

**Feature grid:** Bounds are determined in priority order:

1. **Explicit ranges** (`--feature-range`): used directly if provided for all features.
2. **Auto-inferred:** The experiment's forward model predicts features (e.g. magnitudes) across the parameter grid. Feature errors are computed at the worst-case (minimum-visit) design. The axis spans `feature ± 4σ` (with σ capped at 3.0 to prevent blow-up from non-detections), plus 10% padding on each side.

With `--adaptive-features`, 60% of the point budget is placed in a dense region around detectable features (error < 3) and the remaining 40% spans the full range, then the two grids are merged.

### Prior PDF

The prior PDF is evaluated as the product of independent marginals over the parameter grid. Sources (in priority order): a `prior_args` YAML file, the experiment's built-in prior (`--use-experiment-prior`), or uniform (if neither is given).

