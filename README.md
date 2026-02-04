# bedcosmo

Bayesian Experimental Design for Cosmology

## Overview

bedcosmo provides a framework for optimizing cosmological surveys for parameter inference. The goal is to estimate the Expected Information Gain (EIG) by training normalizing flows to approximate densities with variational inference, enabling data-driven optimization of galaxy redshift surveys like LSST/DESI.

The core approach:
1. Train a conditional normalizing flow to approximate the posterior distribution p(θ|x, d) where θ are cosmological parameters, x is observed data, and d is the experimental design
2. Use the trained flow to compute EIG across different designs
3. Identify optimal survey configurations that maximize information about cosmological parameters

## Installation

### Create a conda environment

```bash
conda env create -f environment.yaml
conda activate bedcosmo
```

### Install the package

```bash
git clone https://github.com/ashandonay/bedcosmo.git
cd bedcosmo
pip install -e ".[dev]"
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA 12.4+ (for GPU training)

## Quick Start

### Training a Model

```bash
./scripts/local/submit.sh train num_tracers base --gpus 1
```

This trains a neural flow for the `base` cosmology model (Ωm, H₀rd parameters) on the `num_tracers` experiment.

For HPC clusters with SLURM:

```bash
./scripts/slurm/submit.sh train num_tracers base
```

### Resuming/Restarting Training

```bash
# Resume from a checkpoint (continues the same run)
./scripts/local/submit.sh resume num_tracers <run_id> <step>

# Restart from a checkpoint (creates a new run)
./scripts/local/submit.sh restart num_tracers <run_id> <step>
```

### Evaluating a Model

After training completes, evaluate the model to compute EIG:

```bash
./scripts/local/submit.sh eval num_tracers <run_id>
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

CLI arguments override YAML defaults:

```bash
./scripts/local/submit.sh train num_tracers base --initial_lr 0.0001 --total_steps 300000
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

4. Train: `./scripts/local/submit.sh train my_experiment base`

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
./scripts/local/submit.sh train num_tracers base --n_particles_per_device 20
```

### MLflow Connection Issues

Ensure the tracking URI is set correctly. The framework uses file-based tracking:
```python
mlflow.set_tracking_uri(f"file:{storage_path}/mlruns")
```

### Resuming from Wrong Checkpoint

Use `--restart_checkpoint` to specify an exact checkpoint file:
```bash
./scripts/local/submit.sh restart num_tracers <run_id> --restart_checkpoint /path/to/checkpoint.pt
```

## Citation

If you use this code, please cite:
```
[Citation to be added]
```

## License

MIT License
