# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bedcosmo is a Bayesian Experimental Design framework for cosmology. It uses neural flows (normalizing flows) to optimize experimental designs for cosmological parameter inference, targeting galaxy redshift surveys like LSST. The project runs on NERSC HPC with SLURM job submission and uses MLflow for experiment tracking.

## Project Structure

```
bedcosmo/
├── src/bedcosmo/           # Source code (installable package)
│   ├── train.py            # Trainer class for distributed training
│   ├── evaluate.py         # Evaluator class for model evaluation
│   ├── util.py             # Bijector, helper functions, config loading
│   ├── plotting.py         # Visualization classes
│   ├── pyro_oed_src.py     # OED loss functions
│   ├── num_tracers/        # NumTracers experiment class
│   ├── variable_redshift/  # VariableRedshift experiment class
│   └── num_visits/         # NumVisits experiment class
├── experiments/            # Configuration files and notebooks
│   ├── num_tracers/        # YAML configs, cobaya/, cosmopower/, notebooks/
│   ├── variable_redshift/  # YAML configs, notebooks/
│   └── num_visits/         # YAML configs
├── submit.sh               # Training script
├── scripts/
│   └── slurm/              # SLURM job scripts (train.sh, eval.sh)
├── tests/                  # Test files
└── pyproject.toml          # Package configuration
```

## Key Commands

### Development
```bash
pip install -e .                # Install package in editable mode
pytest                          # Run all tests with coverage
pytest -m "not slow"            # Skip slow tests
black .                         # Format code (100 char line length)
ruff check .                    # Lint
ruff check --fix .              # Auto-fix lint issues
mypy .                          # Type check
```

### Job Submission
```bash
# Training (auto-detects SLURM on NERSC, local elsewhere)
./submit.sh train num_tracers base
./submit.sh train num_tracers base_w_wa --initial_lr 0.0001

# SLURM with custom time and queue
./submit.sh train num_tracers base --time 04:00 --queue regular
./submit.sh train num_tracers base --time 02:00 --queue shared --nodes 2

# Force local execution on a SLURM system
./submit.sh train num_tracers base --local --gpus 2

# Debug mode (logs to 'debug' MLflow experiment)
./submit.sh train num_tracers base --debug

# Evaluation
./submit.sh eval num_tracers <run_id>
./submit.sh eval num_tracers <run_id> --eig_file_path <path_to_json>

# Resume/Restart training
./submit.sh resume num_tracers <run_id> <step>
./submit.sh restart num_tracers <run_id> <step>

# Optional flags: --debug, --log_usage, --profile
```

## Architecture

### Core Pipeline (src/bedcosmo/)
- **train.py**: `Trainer` class handles distributed training (DDP), neural flow learning, checkpointing, MLflow logging
- **evaluate.py**: `Evaluator` class loads trained models, computes Expected Information Gain (EIG), generates posterior samples
- **plotting.py**: `BasePlotter`, `RunPlotter`, `ComparisonPlotter` for visualization
- **util.py**: `Bijector` class for parameter space normalization (CDF-based), `get_experiment_config_path()` for config loading
- **pyro_oed_src.py**: OED loss functions (`nf_loss`), `LikelihoodDataset`

### Experiment Code (src/bedcosmo/{experiment}/)
Each experiment subpackage contains:
- `experiment.py`: Main class defining likelihood/model (e.g., NumTracers, VariableRedshift)
- `__init__.py`: Re-exports the main class

### Configuration Files (experiments/{experiment}/)
Each experiment directory contains:
- `train_args.yaml`: Training configuration per cosmology model
- `eval_args.yaml`: Evaluation configuration
- `models.yaml`: Parameter definitions and constraints
- `prior_args_*.yaml`: Prior specifications
- `design_args_*.yaml`: Design space specifications
- `notebooks/`: Jupyter notebooks for analysis
- Additional directories (e.g., `cobaya/`, `cosmopower/`) for external tool configs

### Configuration Priority
1. Hardcoded defaults in code
2. YAML config files (lowest priority)
3. CLI arguments (highest priority, override YAML)

### Storage Paths
- Package code: `src/bedcosmo/`
- Experiment configs: `experiments/{cosmo_exp}/`
- MLflow runs and checkpoints: `$SCRATCH/bedcosmo/{cosmo_exp}`

## Common Patterns

### Imports
```python
# Top-level imports
from bedcosmo import NumTracers, VariableRedshift, NumVisits
from bedcosmo import init_experiment, auto_seed, Bijector

# Subpackage imports
from bedcosmo.num_tracers import NumTracers
from bedcosmo.train import Trainer
from bedcosmo.util import get_experiment_config_path
```

### Config Path Resolution
```python
from bedcosmo.util import get_experiment_config_path

# Get path to experiment config file
config_path = get_experiment_config_path('num_tracers', 'train_args.yaml')

# Environment variable override (optional)
# Set BED_COSMO_EXPERIMENTS to use custom experiments directory
```

### MLflow Integration
```python
mlflow.set_tracking_uri(f"file:{storage_path}/mlruns")
mlflow.start_run(run_name=...)
mlflow.log_params({...})
mlflow.log_metrics({...})
```

### Distributed Training
Uses PyTorch DDP with environment variables: `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`

### Random Seeding
Use `auto_seed(seed)` from util.py to seed torch, numpy, pyro, and random consistently.

## Dependencies

Core: torch, pyro, jax, mlflow, numpy, scipy, matplotlib, getdist, astropy, bayesdesign, pydantic, pyyaml
