# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-03

### Added

- **BaseExperiment abstract base class** (`src/bedcosmo/base.py`)
  - Defines common interface for all experiment classes
  - Abstract methods: `__init__`, `init_designs`, `init_prior`, `pyro_model`, `sample_valid_parameters`
  - Shared implementations: `params_to_unconstrained`, `params_from_unconstrained`, `get_guide_samples`, `sample_data`, `sample_params_from_data_samples`
  - Prior flow sampling methods with caching support: `_sample_prior_flow`, `_sample_prior_flow_direct`, `_init_prior_flow_cache`, `clear_prior_flow_cache`

- **CosmologyMixin class** (`src/bedcosmo/cosmology.py`)
  - Shared cosmology distance calculations extracted from experiment classes
  - Methods: `D_H_func`, `D_M_func`, `D_V_func`, `_E_of_z`
  - Helper functions: `_interp1`, `_infer_plate_shape`, `_cumsimpson`
  - `_NeutrinoTableCache` for efficient neutrino density evolution calculations

- **Local job submission scripts** (`scripts/local/`)
  - Non-SLURM scripts for running jobs directly (renamed from `scripts/entropy/`)
  - Same interface as SLURM scripts: `submit.sh train|eval|resume|restart`

- **New subpackages in num_tracers**
  - `src/bedcosmo/num_tracers/cobaya/` - Contains `plot_cosmo_chains.py`
  - `src/bedcosmo/num_tracers/cosmopower/` - Contains `prep_data.py`, `train_cp.py`

### Changed

- **NumTracers** now inherits from `BaseExperiment` and `CosmologyMixin`
  - Removed duplicate cosmology methods (now inherited)
  - Removed duplicate prior flow methods (now inherited from BaseExperiment)

- **VariableRedshift** now inherits from `BaseExperiment` and `CosmologyMixin`
  - Removed duplicate cosmology methods and `_NeutrinoTableCache` (now in CosmologyMixin)
  - Removed duplicate helper functions

- **NumVisits** now inherits from `BaseExperiment` and `CosmologyMixin`
  - Removed duplicate `_cumsimpson` function
  - Replaced `_sample_from_prior_flow` with base class `_sample_prior_flow`
  - Added `sample_valid_parameters` stub to satisfy BaseExperiment interface

- **Package exports** (`src/bedcosmo/__init__.py`)
  - Now exports `BaseExperiment` and `CosmologyMixin`

- **Documentation updates**
  - `README.md`: Added local script examples, updated all `submit.sh` paths to use full paths
  - `CLAUDE.md`: Added `scripts/local/` to project structure, added local job submission section

### Removed

- Duplicate code across experiment classes:
  - Cosmology distance functions
  - Neutrino table caching
  - Prior flow sampling methods
  - Helper functions (`_interp1`, `_cumsimpson`, `_infer_plate_shape`)

### Fixed

- Consistent prior flow attribute names across all experiments:
  - `prior_flow`, `prior_flow_nominal_context`, `prior_flow_transform_input`
  - `prior_flow_batch_size`, `prior_flow_cache_size`, `prior_flow_use_cache`
  - `_prior_flow_cache`, `_prior_flow_cache_idx`
