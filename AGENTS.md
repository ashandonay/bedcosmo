# AGENTS.md

bedcosmo is a Bayesian Experimental Design framework for cosmology and astronomical surveys. A conditional normalizing flow `q(θ | y, d)` is trained across a pool of survey designs `d`. The trained flow then scores each design by its Expected Information Gain (EIG). Jobs run on NERSC through SLURM, and everything is tracked in MLflow under `$SCRATCH/bedcosmo/{cosmo_exp}/`.

## Rules

**Always**
- Read the README for the area you're touching before changing it (see [Where details live](#where-details-live)).
- Run the relevant tests before calling work done.
- Update the matching README when you change YAML fields, CLI flags or documented behavior.

**Ask first**
- Before submitting, cancelling, resuming or restarting jobs. They cost allocation hours and write to shared MLflow state. For quick checks, use `--debug` or `--local`.
- Before editing `src/` while jobs are queued or running. Jobs import the package from the shared filesystem when the process starts, so a queued job picks up whatever is on disk then. A half-finished edit can silently corrupt results.

**Never**
- Compare EIG values across different runs. Compare designs only within one trained flow.
- Mix units. The loss path in `pyro_oed_src.py` works in nats. Estimators in `entropy.py` return bits. Convert once, at reporting, with `nats_to_bits` / `bits_to_nats`.

## Commands

```bash
conda activate bedcosmo && pip install -e ".[dev]"
pytest -m "not slow"                  # quick suite; `pytest tests/test_x.py -k name` for one test
black . && ruff check --fix .         # format + lint
./submit.sh                           # prints full job usage (submit.sh is the source of truth for flags)
./submit.sh train num_visits empirical --debug --local
```

## Where details live

| Read | For |
|---|---|
| `README.md` | Install, quick start, `submit.sh` usage (train/eval/resume/restart, auto-eval, `--train-`/`--eval-` prefixes), MLflow, grid EIG |
| `experiments/num_tracers/README.md` | DESI tracer allocation: BAO likelihood, emulator vs. scaling modes, YAML fields |
| `experiments/num_visits/README.md` | LSST visits per filter: photometric forward model, SED priors, YAML fields |
| `src/bedcosmo/num_visits/empirical/README.md` | Empirical SED prior build: DESI + EAZY fits, ILR coordinates, KDE / prior flow, provenance |
| `src/bedcosmo/num_visits/empirical/reduced/README.md` | Reduced EAZY template bases |

`variable_redshift` has no README. Read its YAMLs and `experiment.py` directly.

## Architecture

- **Experiments** (`src/bedcosmo/{num_tracers,num_visits,variable_redshift}/experiment.py`) subclass `BaseExperiment` (`base.py`), and cosmological ones also use `CosmologyMixin` (`cosmology.py`). Each one defines its designs, prior, parameter sampling and `pyro_model`. Experiment-specific physics lives here and nowhere else.
- **`util.init_experiment` / `util.init_nf`** build the experiment and the flow from run args. Train, eval and grid all construct them through these functions.
- **`train.py` (`Trainer`)**: DDP training loop, checkpoints in the run's `artifacts/checkpoints/`, MLflow logging.
- **`evaluate.py` (`Evaluator`)**: loads a run, computes joint and marginal EIG over the design pool, writes `eig_data*.json` and plots.
- **`grid_calc.py`**: brute-force grid EIG, the ground-truth reference for low-dimensional problems.
- **`transform.py` (`Bijector`)**: Gaussianizes flow inputs. Its state is stored in checkpoints.
- **`plotting.py`, `profiling.py`**: figures, and timing instrumentation that only runs with `--profile`.
- **Config** (`experiments/{cosmo_exp}/`): `train_args.yaml` and `eval_args.yaml` are keyed by cosmology model, and they name a `prior_args*.yaml` and a `design_args*.yaml`. Precedence: code defaults < YAML < CLI.

## How jobs behave (beyond the README)

- `submit.sh` flattens the model's YAML block into `--kebab-case` flags at submission time, and CLI flags replace them. The argv is frozen from then on.
- **train** runs `scripts/create_run.py` first. It pre-creates the MLflow run and snapshots the referenced prior/design YAMLs and data files (emulators, SED KDE, prior flow) into the run's artifacts. The job attaches with `--attach-run-id`. This protects config and data, **not code**.
- **resume** continues the same run and calls `scripts/truncate_metrics.py` to drop metrics logged after the resume step.
- **restart** creates a new run from the old weights and copies the old run's prior/design args. The optimizer starts fresh, and `--restart-optimizer` loads the checkpoint's optimizer state instead, with the learning rate reset to `initial_lr`.
- **Auto-eval** on SLURM is a job that depends on the training job (`afterany`). `scripts/slurm/eval.sh` checks the training log for a line ending in `completed.` and reads the run_id from `MLFlow Run Info:`. Don't change those log lines without updating `eval.sh`.
- `--prior-<field> <v>` overrides one field of the prior YAML before the snapshot. `eval --grid` also launches a sibling grid job, and `--grid-<arg>` sends args to that job only. `eval --marginal` runs only the marginal EIG.
- Logs: `$SCRATCH/bedcosmo/{cosmo_exp}/logs/{jobid}_{jobname}.log`.

## Code practices

Aim for the simplest, most elegant implementation that stays readable.

- **Solve the task in front of you.** Don't add speculative options or abstractions "for later". Repeating a few similar lines is better than an early helper.
- **Call things directly.** Don't write functions that only rename or forward another call. Add an abstraction only when it removes real duplication.
  ```python
  # Don't
  def get_prior_samples(experiment, n):
      return experiment.sample_parameters((n,))
  # Do: call experiment.sample_parameters((n,)) at the call site
  ```
- **No backward compatibility unless asked.** When you rename or restructure, update every caller and delete the old path. Leave no aliases, shims, re-exports, or fallback branches for old formats.
- **Delete what your change leaves unused**: functions, parameters, flags, imports. Don't comment code out.
- **Fail loudly.** Don't use broad `try/except`, silent defaults, or `getattr(..., None)` guards for states that shouldn't happen. Raise a clear error instead. Validate at boundaries (CLI, YAML, file loading) and trust internal invariants.
- **Match the file you're in**: naming, idioms, comment density. Comments should explain why, units and shapes, not narrate the code.
- **Keep changes focused.** If you notice unrelated cleanup, mention it instead of doing it in the same change.

## Definition of done

- Relevant tests pass, and new behavior has a test in `tests/`.
- The READMEs and this file match the new behavior.
- No dead code, compatibility shims or debug prints remain.
- Report anything you couldn't verify, such as a job you didn't run, and say why.
