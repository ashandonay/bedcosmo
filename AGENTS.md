# AGENTS.md

Guidance for coding agents working in this repository.

## What this project is

bedcosmo is a Bayesian Experimental Design (BED) framework for cosmology and astronomical surveys. For each candidate survey design `d`, it scores how much the data would teach us about the parameters `θ` using the **Expected Information Gain (EIG)**. Here `y` is the simulated data. The workflow:

1. An **experiment** defines a prior `p(θ)`, a forward model and likelihood `p(y | θ, d)`, and a pool of candidate designs.
2. A **conditional normalizing flow** `q(θ | y, d)` is trained (variationally, in PyTorch/Pyro with DDP) to approximate the posterior over all designs at once.
3. The trained flow is **evaluated** to estimate EIG per design, which picks out the most informative designs.
4. Optionally, a brute-force **grid calculation** computes EIG on discretized grids as a ground-truth check for low-dimensional problems.

The code runs mainly on NERSC (SLURM, GPU nodes). Runs, checkpoints and artifacts are tracked with MLflow under `$SCRATCH/bedcosmo/{cosmo_exp}/`.

## Where to find details: read the READMEs

This file covers the big picture only. **Before you debug or change experiment-specific code, read the README for that area.** Each README documents its likelihood model, design space, prior configuration and YAML fields:

| README | Covers |
|---|---|
| `README.md` | Installation, quick start, cosmology models, MLflow, grid EIG calculation (`grid_calc`) and how its grids are built |
| `experiments/num_tracers/README.md` | DESI tracer-allocation experiment: BAO likelihood, emulator vs. scaling likelihood modes, design and prior YAML fields |
| `experiments/num_visits/README.md` | LSST visits-per-filter experiment: photometric forward model, SED priors, design and prior YAML fields |
| `src/bedcosmo/num_visits/empirical/README.md` | Empirical galaxy SED prior pipeline: DESI + EAZY template fits, ILR coordinates, KDE and prior-flow builds, provenance |
| `src/bedcosmo/num_visits/empirical/reduced/README.md` | Reduced EAZY template bases: subset discovery, diagnostics, reduced-prior builds |

`variable_redshift` has no README yet. For it, read `experiments/variable_redshift/*.yaml` and `src/bedcosmo/variable_redshift/experiment.py` directly.

If you change behavior that a README documents, update that README in the same change.

## Repository layout

```
src/bedcosmo/            Installable package (pip install -e .)
  base.py                BaseExperiment: the abstract interface every experiment implements
  cosmology.py           CosmologyMixin: background cosmology distances (D_H, D_M, D_V, ...)
  custom_dist.py         Custom Pyro/torch distributions used as priors
  transform.py           Bijector: maps physical parameters to ~Gaussian flow coordinates
  pyro_oed_src.py        OED losses (nf_loss etc.) and LikelihoodDataset
  train.py               Trainer: DDP training loop, checkpointing, MLflow logging
  evaluate.py            Evaluator: loads a trained run, computes EIG and posteriors, makes plots
  grid_calc.py           GridCalculation: brute-force grid EIG (bayesdesign), overlays with NF results
  entropy.py             Sample-based entropy estimators (kNN, KDE, Gaussian) for marginal EIG
  plotting.py            BasePlotter / RunPlotter / ComparisonPlotter
  profiling.py           Opt-in timing instrumentation (--profile)
  util.py                Shared helpers: config paths, seeding, init_experiment, init_nf, load_model, MLflow queries
  num_tracers/           DESI tracer-allocation experiment (+ emulator, CMB likelihood, cobaya/cosmopower tooling)
  num_visits/            LSST visit-allocation experiment (+ empirical/ SED prior pipeline)
  variable_redshift/     Redshift-dependent survey experiment
experiments/{cosmo_exp}/ Per-experiment configs (YAML), analysis scripts/, notebooks/
submit.sh                Single entry point for every job type (train/eval/resume/restart/grid)
scripts/slurm/           SLURM batch scripts that submit.sh dispatches to
scripts/create_run.py    Pre-creates the MLflow run and snapshots config at submit time
scripts/truncate_metrics.py  Trims MLflow metrics back to the resume step
tests/                   pytest suite
```

### Core pipeline (`src/bedcosmo/`)

- **Experiments (`base.py`, `cosmology.py`, `{experiment}/experiment.py`).** Each experiment subclasses `BaseExperiment`. It implements `init_designs`, `init_prior`, `sample_parameters` and `pyro_model`, the generative model `θ → y` for a batch of designs. Cosmological experiments also mix in `CosmologyMixin` for distance calculations. `util.init_experiment(...)` builds an experiment from run args, so train, eval and grid all construct experiments the same way. An experiment owns its own physics. The pipeline modules stay experiment-agnostic.
- **Input transform (`transform.py`).** `Bijector` Gaussianizes flow inputs. It uses per-dimension empirical CDFs plus normal scores (`marginal`), with optional joint Cholesky whitening (`joint`). It is fit from prior samples or a fixed reference matrix. Its state is serialized into checkpoints.
- **Losses (`pyro_oed_src.py`).** EIG estimators and the flow training loss. This path works in **nats**. `entropy.py` estimators return **bits**, and results are converted once, at the reporting edge.
- **Training (`train.py`).** `Trainer` sets up DDP, builds the experiment and the flow (`util.init_nf`), then runs the optimization loop. It writes checkpoints to the MLflow run's `artifacts/checkpoints/` and logs metrics and params. It either attaches to a run pre-created by `submit.sh` or, for restarts and resumes, loads one.
- **Evaluation (`evaluate.py`).** `Evaluator` loads a run and checkpoint and computes EIG across the design pool, including optional marginal EIG over parameter subsets. It writes `eig_data*.json` and plots into the run's artifacts. It can overlay grid results.
- **Grid (`grid_calc.py`).** Brute-force EIG on discretized parameter, feature and design grids. It is CPU-oriented and is the reference for validating the flow at low dimension.
- **Plotting (`plotting.py`).** All figure generation, shared by eval and the notebooks.
- **Profiling (`profiling.py`).** `@profile_method`, `profile_loop`, `profile_section` and `ProfileTimerGroup`. They do nothing unless `--profile` is set, and report on rank 0 only.

### Configuration (`experiments/{cosmo_exp}/`)

- `train_args.yaml` / `eval_args.yaml`: default CLI args, keyed by **cosmology model** (e.g. `base`, `base_w_wa`, or `empirical` for num_visits).
- `models.yaml`: parameter definitions per cosmology model.
- `prior_args*.yaml`: prior specifications. `design_args*.yaml`: design-pool specifications. Train args pick one of each through `--prior-args-path` / `--design-args-path`.
- Resolve paths with `bedcosmo.util.get_experiment_config_path(cosmo_exp, name)`. The `BED_COSMO_EXPERIMENTS` environment variable overrides the experiments directory.

Precedence: code defaults < YAML < CLI.

## Running jobs: `submit.sh`

All jobs go through `./submit.sh <job_type> <cosmo_exp> [model_or_run_id] [step] [--args...]`. The script decides how to run:

- **Mode:** SLURM (`sbatch` of a script in `scripts/slurm/`) if `sbatch` exists, otherwise local (`torchrun --nproc_per_node=<gpus> -m bedcosmo.<module>`). Force a mode with `--local` or `--slurm`.
- **Arg resolution:** for `train`, it reads the `cosmo_model` block of `train_args.yaml`. For `eval`, it reads the block of `eval_args.yaml`. YAML keys become `--kebab-case` flags, and CLI flags replace them. The resolved argv is **frozen at submission**.
- **Logs:** `$SCRATCH/bedcosmo/{cosmo_exp}/logs/{jobid}_{jobname}.log`.

### Job types

| Job | Command | What happens |
|---|---|---|
| **train** | `./submit.sh train num_tracers base` | `scripts/create_run.py` pre-creates the MLflow run, tags it `queued`, and **snapshots** the referenced prior/design YAMLs and data files (emulator `.pt`, SED KDE, prior flow) into its artifacts. Edits made while the job waits in the queue therefore can't leak in. The job attaches with `--attach-run-id`. Runs `bedcosmo.train` via `scripts/slurm/train.sh`. |
| **eval** | `./submit.sh eval num_tracers <run_id>` | Runs `bedcosmo.evaluate` on a finished run with `eval_args.yaml` defaults. `cosmo_model` is inferred from MLflow. `--grid` also sends a sibling grid job. `--marginal` runs only marginal EIG. |
| **resume** | `./submit.sh resume num_tracers <run_id> <step>` | Continues the **same** run from the checkpoint at `<step>`, restoring optimizer, scheduler and RNG state. First runs `truncate_metrics.py` so metrics logged after `<step>` are dropped. |
| **restart** | `./submit.sh restart num_tracers <run_id> <step>` | Starts a **new** run from `<run_id>`'s weights at `<step>` (or `--restart-checkpoint <file>`). Copies the old run's prior/design args. Use it to branch with changed hyperparameters. The optimizer starts fresh by default. `--restart-optimizer` loads the checkpoint's optimizer state instead, with the learning rate reset to `initial_lr`. |
| **grid** | `./submit.sh grid num_visits --param-pts 1000 --feature-pts 500` | Runs `bedcosmo.grid_calc` as one process, CPU by default (`--node-type gpu` for GPU nodes). Takes no YAML defaults: everything comes from the CLI. |

### Auto-eval

After `train`, `resume` or `restart`, an eval job is chained automatically. On SLURM it is submitted with `--dependency=afterany:<train_job>`. `eval.sh` reads the training log, skips if training did not finish, and pulls the run_id from the `MLFlow Run Info:` line. Locally it runs after training exits.

- `--no-eval` disables it. `--debug` also disables it (it logs to the `debug` MLflow experiment and uses the debug queue). `--auto-eval` turns it back on.
- `--eval-time` sets the eval job's time limit.

### Argument prefixes

- Unprefixed `--foo` goes to the primary job (training for train/resume/restart).
- `--train-foo` / `--eval-foo` send the flag to that stage explicitly. For eval with `--grid`, `--grid-foo` goes only to the grid job.
- `--prior-<field> <value>` overrides one field of the prior YAML, applied before the snapshot (e.g. `--prior-template-source eazy6`). `--prior-args-path` and `--prior-flow-path` are ordinary train flags, not field overrides.
- SLURM flags: `--time`, `--queue`, `--nodes`, `--gpus`, `--exclude`, and the per-stage `--train-time`, `--train-queue`, `--train-nodes`, `--eval-time`, `--grid-time`. Other flags: `--profile`, `--log-usage`.

Run `./submit.sh` with no arguments for the full usage text. `submit.sh` is the source of truth for flags.

### Job-safety rules

- **Don't edit `src/` while submitted jobs are queued or running** unless the user says it is safe. Jobs import the package from the shared filesystem when the process starts, so a queued job picks up whatever is on disk at that moment. A half-finished refactor can silently corrupt results. The config snapshot protects YAML and data files, not code.
- Don't submit, cancel or resume jobs unless the user asks. They use allocation hours and write to shared MLflow state.
- For quick checks, use `--debug` (its own MLflow experiment, no auto-eval) or `--local`.

## Development

```bash
conda activate bedcosmo            # pipeline environment
pip install -e ".[dev]"
pytest                             # full suite with coverage
pytest -m "not slow"               # skip slow tests
pytest tests/test_foo.py -k name   # targeted
black .                            # format (100-char lines)
ruff check .                       # lint (ruff check --fix . to auto-fix)
```

Import style: `from bedcosmo import NumVisits, init_experiment, auto_seed, Bijector` (top-level names load lazily). Otherwise import from the defining module, e.g. `from bedcosmo.transform import Bijector`.

Seed with `util.auto_seed(seed, rank)`. It seeds torch, numpy, pyro and random together.

## Code practices

The goal is **the simplest, most elegant implementation that stays readable.** In practice:

- **Solve the problem in front of you.** Don't add speculative options, configuration knobs, or abstractions "for later". Three similar lines are better than a premature helper.
- **No needless wrappers.** Don't write functions that only rename or forward to another call, or classes that exist to hold one function. Call the underlying thing directly. Add an abstraction only when it removes real duplication or makes the code clearly easier to read.
- **No backward compatibility unless asked.** When you rename or restructure something, update every caller and delete the old path. Don't leave aliases, deprecated shims, fallback branches for old formats, or `# legacy` code unless the user asks for them. The same goes for re-exports kept "for old imports".
- **Delete dead code.** Remove unused functions, parameters, flags and imports that your change leaves behind. Don't comment code out. Git keeps the history.
- **Fail loudly.** Don't wrap code in broad `try/except`, return silent defaults, or add `getattr(..., None)` guards against states that shouldn't happen. A clear error beats a quietly wrong EIG. Validate at boundaries (CLI, YAML, file loading) and trust internal invariants.
- **Match the surrounding code.** Follow the naming, idioms, comment density and structure of the file you're in. Put logic in the layer that owns it: experiment physics in the experiment class, generic pipeline logic in `train.py` / `evaluate.py` / `util.py`.
- **Comments explain why, not what.** Record non-obvious reasoning, units (nats vs. bits), shapes and conventions. Don't narrate the code.
- **Keep changes focused.** Don't reformat or refactor unrelated code in the same change. Flag it instead.
- **Keep docs in sync.** If a change alters YAML fields, CLI flags or documented behavior, update the relevant README (and this file, if the big picture changed).
- **Test what you change.** Add or update tests in `tests/` for new behavior, and run the relevant tests before calling the work done. Report failures honestly.
