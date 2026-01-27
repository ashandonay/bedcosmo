# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BED_cosmo is a Bayesian Experimental Design framework for cosmology. It uses neural flows (normalizing flows) to optimize experimental designs for cosmological parameter inference, targeting galaxy redshift surveys like LSST. The project runs on NERSC HPC with SLURM job submission and uses MLflow for experiment tracking.

## Key Commands

### Development
```bash
pytest                          # Run all tests with coverage
pytest -m "not slow"            # Skip slow tests
black .                         # Format code (100 char line length)
ruff check .                    # Lint
ruff check --fix .              # Auto-fix lint issues
mypy .                          # Type check
```

### Job Submission (NERSC/SLURM)
```bash
# Training
./scripts/slurm/submit.sh train num_tracers base
./scripts/slurm/submit.sh train num_tracers base_w_wa --initial_lr 0.0001

# Evaluation
./scripts/slurm/submit.sh eval num_tracers <run_id>
./scripts/slurm/submit.sh eval num_tracers <run_id> --eig_file_path <path_to_json>

# Resume/Restart training
./scripts/slurm/submit.sh resume num_tracers <run_id> <step>
./scripts/slurm/submit.sh restart num_tracers <run_id> <step>

# Available cosmology models: base, base_omegak, base_w, base_w_wa, base_omegak_w_wa
# Optional flags: --log_usage, --profile
```

## Architecture

### Core Pipeline
- **train.py**: `Trainer` class handles distributed training (DDP), neural flow learning, checkpointing, MLflow logging
- **evaluate.py**: `Evaluator` class loads trained models, computes Expected Information Gain (EIG), generates posterior samples
- **plotting.py**: `BasePlotter`, `RunPlotter`, `ComparisonPlotter` for visualization
- **util.py**: `Bijector` class for parameter space normalization (CDF-based), helper functions
- **pyro_oed_src.py**: OED loss functions (`nf_loss`), `LikelihoodDataset`

### Experiment Modules
Each experiment module (`num_tracers/`, `variable_redshift/`, `num_visits/`) contains:
- `{experiment}.py`: Main class defining likelihood/model
- `train_args.yaml`: Training configuration per cosmology model
- `models.yaml`: Parameter definitions and constraints
- `prior_args_*.yaml`: Prior specifications
- `design_args_*.yaml`: Design space specifications

### Configuration Priority
1. Hardcoded defaults in code
2. YAML config files (lowest priority)
3. CLI arguments (highest priority, override YAML)

### Storage Paths
- Code/configs: `$HOME/bed/BED_cosmo`
- MLflow runs and checkpoints: `$SCRATCH/bed/BED_cosmo/{cosmo_exp}`

## Common Patterns

### Path Management
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_abs = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir_abs)
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
