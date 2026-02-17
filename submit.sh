#!/bin/bash

# Unified submit script: auto-detects SLURM availability and dispatches accordingly.
# Usage: ./submit.sh <job_type> [cosmo_exp] [id] [arguments...]

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure SCRATCH is set (used for MLflow storage, logs, etc.)
if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH="$HOME/scratch"
    echo "SCRATCH was unset; set to: $SCRATCH"
fi

# Check if job type is provided
if [ $# -eq 0 ]; then
    echo "Error: Job type is required"
    echo ""
    echo "Usage: ./submit.sh <job_type> [cosmo_exp] [model_or_id] [arguments...]"
    echo ""
    echo "Available job types:"
    echo "  train                - Training job"
    echo "  eval                 - Evaluation job"
    echo "  resume               - Resume training job"
    echo "  restart              - Restart training job"
    echo ""
    echo "Available cosmo_models:"
    echo "  base, base_omegak, base_w, base_w_wa, base_omegak_w_wa"
    echo ""
    echo "The script will load default parameters from <cosmo_exp>/train_args.yaml"
    echo "for the specified cosmo_model. Command-line arguments will override YAML defaults."
    echo ""
    echo "Execution mode options:"
    echo "  --local              - Force local execution (torchrun) even if SLURM is available"
    echo "  --slurm              - Force SLURM submission even if sbatch is not detected"
    echo "  (default: auto-detect based on sbatch availability)"
    echo ""
    echo "SLURM options:"
    echo "  --time <HH:MM[:SS]> - SLURM time limit (default: 00:30)"
    echo "  --queue <qos>       - SLURM QOS/queue (default: regular)"
    echo "  --nodes <N>         - Number of nodes (default: 1)"
    echo "  --exclude <nodes>   - Exclude specific SLURM nodes"
    echo ""
    echo "Local options:"
    echo "  --gpus <N>          - GPUs / torchrun processes (default: auto-detect)"
    echo "  --omp-threads <N>   - Set OMP_NUM_THREADS"
    echo ""
    echo "Common options:"
    echo "  --gpus <N>          - SLURM: overrides gpus-per-node; Local: sets nproc_per_node"
    echo "  --debug             - Use debug queue + 'debug' MLflow experiment (for train/restart)"
    echo "  --no-eval           - Disable auto-eval (on by default, off for --debug)"
    echo "  --eval-<arg> <val>  - Pass args to auto-eval (e.g. --eval-grid --eval-grid-param-pts 2000)"
    echo "  --eval-time <HH:MM> - SLURM time limit for auto-eval job (default: 00:30)"
    echo "  --log-usage, --profile, --restart-optimizer"
    echo ""
    echo "Examples:"
    echo "  ./submit.sh train num_tracers base"
    echo "  ./submit.sh train num_tracers base --time 04:00 --queue regular"
    echo "  ./submit.sh train num_tracers base --local --gpus 2"
    echo "  ./submit.sh eval num_tracers abc123"
    echo "  ./submit.sh resume num_tracers abc123 5000 --time 01:00 --queue regular"
    echo "  ./submit.sh restart num_tracers abc123 10000"
    echo "  ./submit.sh train num_tracers base_w_wa --initial-lr 0.0001 --log-usage"
    exit 1
fi

JOB_TYPE=$1
shift  # Remove job type from arguments

# ──────────────────────────────────────────────────────────────────────
# Mode detection (auto, --local, or --slurm)
# ──────────────────────────────────────────────────────────────────────
FORCE_LOCAL=false
FORCE_SLURM=false

# ──────────────────────────────────────────────────────────────────────
# Initialize variables
# ──────────────────────────────────────────────────────────────────────
DEFAULT_COSMO_EXP="num_tracers"
COSMO_EXP=""
COSMO_MODEL=""
RUN_ID=""
RESUME_ID=""
RESTART_ID=""
RESUME_STEP=""
RESTART_STEP=""
RESTART_CHECKPOINT=""
LOG_USAGE=false
PROFILE=false
DEBUG=false
RESTART_OPTIMIZER=false
AUTO_EVAL=true
AUTO_EVAL_SET_BY_USER=false
EVAL_TIME="00:30"
EVAL_EXTRA_ARGS=()  # Args prefixed with --eval_ to pass to auto-eval job

# New unified options
SLURM_TIME="00:30"
SLURM_QUEUE="regular"
SLURM_NODES="1"

# Local-specific options
GPUS=""
GPUS_SET_BY_USER=false
QUEUE_SET_BY_USER=false
OMP_THREADS=""

declare -A CLI_ARGS  # Associative array for command-line arguments
POSITIONAL_ARGS=()
SBATCH_ARGS=()

# ──────────────────────────────────────────────────────────────────────
# Parse named arguments
# ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            FORCE_LOCAL=true
            shift 1
            ;;
        --slurm)
            FORCE_SLURM=true
            shift 1
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --queue)
            SLURM_QUEUE="$2"
            QUEUE_SET_BY_USER=true
            shift 2
            ;;
        --nodes)
            SLURM_NODES="$2"
            shift 2
            ;;
        --cosmo-exp)
            COSMO_EXP="$2"
            shift 2
            ;;
        --cosmo-model)
            COSMO_MODEL="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --resume-id)
            RESUME_ID="$2"
            shift 2
            ;;
        --restart-id)
            RESTART_ID="$2"
            shift 2
            ;;
        --resume-step)
            RESUME_STEP="$2"
            shift 2
            ;;
        --restart-step)
            RESTART_STEP="$2"
            shift 2
            ;;
        --restart-checkpoint)
            RESTART_CHECKPOINT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            GPUS_SET_BY_USER=true
            shift 2
            ;;
        --omp-threads)
            OMP_THREADS="$2"
            shift 2
            ;;
        --log-usage)
            LOG_USAGE=true
            shift 1
            ;;
        --profile)
            PROFILE=true
            shift 1
            ;;
        --debug)
            DEBUG=true
            shift 1
            ;;
        --restart-optimizer)
            RESTART_OPTIMIZER=true
            shift 1
            ;;
        --auto-eval)
            AUTO_EVAL=true
            AUTO_EVAL_SET_BY_USER=true
            shift 1
            ;;
        --no-eval)
            AUTO_EVAL=false
            shift 1
            ;;
        --eval-time)
            EVAL_TIME="$2"
            shift 2
            ;;
        --exclude)
            SBATCH_ARGS+=("--exclude=$2")
            shift 2
            ;;
        --train-time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --train-queue)
            SLURM_QUEUE="$2"
            QUEUE_SET_BY_USER=true
            shift 2
            ;;
        --train-nodes)
            SLURM_NODES="$2"
            shift 2
            ;;
        --train-*)
            # Train-specific args: strip train- prefix and treat as CLI arg
            train_key="${1#--train-}"
            if [[ $# -gt 1 ]] && [[ ! $2 =~ ^-- ]]; then
                CLI_ARGS["$train_key"]="$2"
                shift 2
            else
                CLI_ARGS["$train_key"]="true"
                shift 1
            fi
            ;;
        --eval-*)
            # Eval-specific args: strip eval- prefix and collect for auto-eval
            eval_key="${1#--eval-}"
            if [[ $# -gt 1 ]] && [[ ! $2 =~ ^-- ]]; then
                EVAL_EXTRA_ARGS+=("--$eval_key" "$2")
                shift 2
            else
                EVAL_EXTRA_ARGS+=("--$eval_key")
                shift 1
            fi
            ;;
        --*)
            # Store CLI argument (will override YAML defaults, assumed to be for train)
            key="${1#--}"
            # Check if next arg is a value or another flag
            if [[ $# -gt 1 ]] && [[ ! $2 =~ ^-- ]]; then
                CLI_ARGS["$key"]="$2"
                shift 2
            else
                # Boolean flag
                CLI_ARGS["$key"]="true"
                shift 1
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift 1
            ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────
# Resolve execution mode
# ──────────────────────────────────────────────────────────────────────
if [ "$FORCE_LOCAL" = true ] && [ "$FORCE_SLURM" = true ]; then
    echo "Error: --local and --slurm are mutually exclusive"
    exit 1
fi

if [ "$FORCE_LOCAL" = true ]; then
    EXECUTION_MODE="local"
elif [ "$FORCE_SLURM" = true ]; then
    EXECUTION_MODE="slurm"
elif command -v sbatch &> /dev/null; then
    EXECUTION_MODE="slurm"
else
    EXECUTION_MODE="local"
fi

# Normalize --time: append :00 if only HH:MM given
if [[ "$SLURM_TIME" =~ ^[0-9]+:[0-9]+$ ]]; then
    SLURM_TIME="${SLURM_TIME}:00"
fi
if [[ "$EVAL_TIME" =~ ^[0-9]+:[0-9]+$ ]]; then
    EVAL_TIME="${EVAL_TIME}:00"
fi

# --debug implies SLURM debug queue (unless --queue was explicitly set)
if [ "$DEBUG" = true ] && [ "$QUEUE_SET_BY_USER" = false ]; then
    SLURM_QUEUE="debug"
fi

# --debug disables auto-eval (unless --auto-eval was explicitly passed)
if [ "$DEBUG" = true ] && [ "$AUTO_EVAL_SET_BY_USER" = false ]; then
    AUTO_EVAL=false
fi

echo "Execution mode: $EXECUTION_MODE"

# ──────────────────────────────────────────────────────────────────────
# Positional argument fallback logic
# ──────────────────────────────────────────────────────────────────────
if [ -z "$COSMO_EXP" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    COSMO_EXP="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
fi

case $JOB_TYPE in
    eval)
        if [ -z "$RUN_ID" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
            RUN_ID="${POSITIONAL_ARGS[0]}"
            POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        fi
        ;;
    resume)
        if [ -z "$RESUME_ID" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
            RESUME_ID="${POSITIONAL_ARGS[0]}"
            POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        fi
        if [ -z "$RESUME_STEP" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
            RESUME_STEP="${POSITIONAL_ARGS[0]}"
            POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        fi
        ;;
    restart)
        if [ -z "$RESTART_ID" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
            RESTART_ID="${POSITIONAL_ARGS[0]}"
            POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        fi
        if [ -z "$RESTART_STEP" ] && [ -z "$RESTART_CHECKPOINT" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
            candidate="${POSITIONAL_ARGS[0]}"
            if [[ "$candidate" == *.pt ]] || [[ "$candidate" == *checkpoint* ]] || [[ "$candidate" == */* ]]; then
                RESTART_CHECKPOINT="$candidate"
            else
                RESTART_STEP="$candidate"
            fi
            POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
        fi
        ;;
    *)
        # No additional positional handling required
        ;;
esac

# Allow positional cosmo_model for job types that require it
if [ -z "$COSMO_MODEL" ] && [ "$JOB_TYPE" != "eval" ] && [ "$JOB_TYPE" != "resume" ] && [ "$JOB_TYPE" != "restart" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    COSMO_MODEL="${POSITIONAL_ARGS[0]}"
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
fi

if [ -z "$COSMO_EXP" ]; then
    COSMO_EXP="$DEFAULT_COSMO_EXP"
fi

if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    echo "Error: Unexpected positional arguments: ${POSITIONAL_ARGS[*]}"
    echo "Usage: ./submit.sh $JOB_TYPE [cosmo_exp] [id] [additional args...]"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Validate required arguments
# ──────────────────────────────────────────────────────────────────────
if [ -z "$COSMO_EXP" ]; then
    echo "Error: cosmo_exp is required"
    echo "Usage: ./submit.sh $JOB_TYPE [cosmo_exp] [cosmo_model] [additional args...]"
    exit 1
fi

# cosmo_model is NOT required for eval/resume/restart (inferred from MLflow run)
if [ -z "$COSMO_MODEL" ] && [ "$JOB_TYPE" != "eval" ] && [ "$JOB_TYPE" != "resume" ] && [ "$JOB_TYPE" != "restart" ]; then
    echo "Error: --cosmo-model is required for $JOB_TYPE jobs"
    echo "Usage: ./submit.sh $JOB_TYPE [cosmo_exp] [cosmo_model] [additional args...]"
    echo "Available models: base, base_omegak, base_w, base_w_wa, base_omegak_w_wa"
    echo ""
    echo "Note: cosmo_model can be provided positionally after cosmo_exp."
    echo "Note: --cosmo-model is not needed for 'eval', 'resume', or 'restart' jobs (inferred from MLflow run)"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Activate conda environment (needed for MLflow and YAML parsing)
# ──────────────────────────────────────────────────────────────────────
if [ "$EXECUTION_MODE" = "slurm" ]; then
    module load conda 2>/dev/null || true
fi

if [ "$CONDA_DEFAULT_ENV" = "bedcosmo" ]; then
    echo "Using existing conda environment: bedcosmo"
else
    # Source conda.sh to enable conda activate
    if [ -n "$CONDA_PREFIX" ] && [[ "$CONDA_PREFIX" == *"/envs/"* ]]; then
        CONDA_BASE="${CONDA_PREFIX%/envs/*}"
    else
        CONDA_BASE=$(conda info --base 2>/dev/null)
    fi
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook 2>/dev/null)" || {
            echo "Error: conda not found. Please activate the bedcosmo env first: conda activate bedcosmo"
            exit 1
        }
    fi
    conda activate bedcosmo
    echo "Activated conda environment: bedcosmo"
fi

# Prefer conda env libraries (e.g. libstdc++) to avoid CXXABI / version conflicts with system libs
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ──────────────────────────────────────────────────────────────────────
# For eval jobs, infer cosmo_model from MLflow run
# ──────────────────────────────────────────────────────────────────────
if [ "$JOB_TYPE" = "eval" ] && [ -z "$COSMO_MODEL" ]; then
    echo "Inferring cosmo_model from MLflow run ${RUN_ID}..."
    TEMP_STDERR=$(mktemp)
    set +e
    COSMO_MODEL=$(timeout 30 python3 -c "
import mlflow
import os
import sys
import warnings
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore', category=FutureWarning)

mlflow.set_tracking_uri('file:' + os.environ['SCRATCH'] + '/bedcosmo/${COSMO_EXP}/mlruns')
client = MlflowClient()
try:
    run = client.get_run('${RUN_ID}')
    cosmo_model = run.data.params.get('cosmo_model', 'base')
    print(cosmo_model, flush=True)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>"$TEMP_STDERR")
    PYTHON_EXIT_CODE=$?
    STDERR_CONTENT=$(cat "$TEMP_STDERR" 2>/dev/null)
    rm -f "$TEMP_STDERR"
    set -e

    if [ -n "$STDERR_CONTENT" ]; then
        echo "Warning from MLflow (non-fatal):" >&2
        echo "$STDERR_CONTENT" >&2
    fi

    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        if [ $PYTHON_EXIT_CODE -eq 124 ]; then
            echo "Error: Timeout while getting cosmo_model from run ${RUN_ID} (command took >30 seconds)"
        else
            echo "Error: Failed to get cosmo_model from run ${RUN_ID} (exit code: $PYTHON_EXIT_CODE)"
        fi
        if [ -n "$COSMO_MODEL" ]; then
            echo "Partial output: $COSMO_MODEL"
        fi
        exit 1
    fi

    COSMO_MODEL=$(echo "$COSMO_MODEL" | tail -n 1 | xargs)

    if [ -z "$COSMO_MODEL" ]; then
        echo "Error: Could not extract cosmo_model from MLflow run ${RUN_ID}"
        exit 1
    fi

    echo "Inferred cosmo_model: $COSMO_MODEL"
    echo ""
fi

# ──────────────────────────────────────────────────────────────────────
# Load YAML config
# ──────────────────────────────────────────────────────────────────────
if [ "$JOB_TYPE" = "resume" ] || [ "$JOB_TYPE" = "restart" ]; then
    echo "$JOB_TYPE job: Skipping YAML config (parameters from MLflow or command-line)"
    YAML_ARGS=()
    CONFIG_FILE=""
elif [ "$JOB_TYPE" = "eval" ]; then
    CONFIG_FILE="$PROJECT_ROOT/experiments/${COSMO_EXP}/eval_args.yaml"
else
    CONFIG_FILE="$PROJECT_ROOT/experiments/${COSMO_EXP}/train_args.yaml"
fi

if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE"
    echo "Proceeding without default configuration..."
    YAML_ARGS=()
elif [ -n "$CONFIG_FILE" ]; then
    echo "Loading config from: $CONFIG_FILE"
    if [ -n "$COSMO_MODEL" ]; then
        echo "Using model: $COSMO_MODEL"
    else
        echo "Warning: COSMO_MODEL is empty, this may cause issues"
    fi

    YAML_ARGS=()
    YAML_ARG_COUNT=0

    TEMP_YAML_STDERR=$(mktemp)
    YAML_OUTPUT=$(python3 -c "
import yaml
import sys
import json
import warnings

warnings.filterwarnings('ignore')

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)

    if '$COSMO_MODEL' not in config:
        print(f'ERROR: Model $COSMO_MODEL not found in config file', file=sys.stderr)
        print(f'Available models: {list(config.keys())}', file=sys.stderr)
        sys.exit(1)

    model_config = config['$COSMO_MODEL']

    print(json.dumps(model_config))
except Exception as e:
    print(f'ERROR: Failed to parse YAML: {e}', file=sys.stderr)
    sys.exit(1)
" 2>"$TEMP_YAML_STDERR")
    YAML_EXIT_CODE=$?
    YAML_STDERR=$(cat "$TEMP_YAML_STDERR" 2>/dev/null)
    rm -f "$TEMP_YAML_STDERR"

    if [ $YAML_EXIT_CODE -ne 0 ]; then
        echo "Error: Failed to load YAML config for model '$COSMO_MODEL'"
        if [ -n "$YAML_STDERR" ]; then
            echo "$YAML_STDERR"
        fi
        if [ -n "$YAML_OUTPUT" ]; then
            echo "Output: $YAML_OUTPUT"
        fi
        exit 1
    fi

    while IFS="=" read -r key value; do
        # Convert underscores to hyphens for CLI flags (POSIX convention)
        key="${key//_/-}"

        # Skip if this key was provided via CLI (CLI overrides YAML)
        if [[ -v CLI_ARGS["$key"] ]]; then
            continue
        fi

        # Skip null/None values
        # Exception: for eval jobs, include fixed_design even if null
        if [ "$value" = "null" ] || [ "$value" = "None" ] || [ -z "$value" ]; then
            if [ "$JOB_TYPE" = "eval" ] && [ "$key" = "fixed_design" ]; then
                YAML_ARGS+=("--$key" "null")
                YAML_ARG_COUNT=$((YAML_ARG_COUNT + 1))
            fi
            continue
        fi

        if [ "$value" = "true" ]; then
            YAML_ARGS+=("--$key")
            YAML_ARG_COUNT=$((YAML_ARG_COUNT + 1))
        elif [ "$value" = "false" ]; then
            continue
        elif [[ "$value" == \[*\] ]]; then
            YAML_ARGS+=("--$key" "$value")
            YAML_ARG_COUNT=$((YAML_ARG_COUNT + 1))
        else
            YAML_ARGS+=("--$key" "$value")
            YAML_ARG_COUNT=$((YAML_ARG_COUNT + 1))
        fi
    done < <(echo "$YAML_OUTPUT" | python3 -c "
import json
import sys

data = json.load(sys.stdin)
for key, value in data.items():
    if value is None:
        print(f'{key}=null')
    elif isinstance(value, list):
        formatted = json.dumps(value)
        print(f'{key}={formatted}')
    elif isinstance(value, bool):
        print(f'{key}={str(value).lower()}')
    else:
        print(f'{key}={value}')
")

    echo "Config loaded successfully ($YAML_ARG_COUNT arguments from YAML)"
fi

# ──────────────────────────────────────────────────────────────────────
# Add CLI arguments (these override YAML)
# ──────────────────────────────────────────────────────────────────────
for key in "${!CLI_ARGS[@]}"; do
    value="${CLI_ARGS[$key]}"
    if [ "$value" = "true" ]; then
        YAML_ARGS+=("--$key")
    elif [ "$value" != "false" ]; then
        YAML_ARGS+=("--$key" "$value")
    fi
done

# When --debug is set, filter out mlflow_exp from YAML_ARGS (we'll add --mlflow-exp debug later)
# For eval jobs, filter out cosmo_model from YAML_ARGS (evaluate.py doesn't accept it)
if [ "$DEBUG" = true ] || [ "$JOB_TYPE" = "eval" ]; then
    FILTERED_ARGS=()
    skip_next=false
    for ((i=0; i<${#YAML_ARGS[@]}; i++)); do
        if [ "$skip_next" = true ]; then
            skip_next=false
            continue
        fi
        if [ "$DEBUG" = true ] && [ "${YAML_ARGS[$i]}" = "--mlflow-exp" ]; then
            skip_next=true
            continue
        fi
        if [ "$JOB_TYPE" = "eval" ] && [ "${YAML_ARGS[$i]}" = "--cosmo-model" ]; then
            skip_next=true
            continue
        fi
        FILTERED_ARGS+=("${YAML_ARGS[$i]}")
    done
    YAML_ARGS=("${FILTERED_ARGS[@]}")
fi

# ──────────────────────────────────────────────────────────────────────
# Validate and build arguments based on job type
# ──────────────────────────────────────────────────────────────────────
case $JOB_TYPE in
    train)
        SLURM_SCRIPT_FILE="train.sh"
        PYTHON_MODULE="bedcosmo.train"
        FINAL_ARGS=("--cosmo-exp" "$COSMO_EXP" "${YAML_ARGS[@]}")

        if [ "$DEBUG" = true ]; then
            FINAL_ARGS+=("--mlflow-exp" "debug")
        fi
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log-usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;

    eval)
        SLURM_SCRIPT_FILE="eval.sh"
        PYTHON_MODULE="bedcosmo.evaluate"
        if [ -z "$RUN_ID" ]; then
            echo "Error: run_id is required for eval"
            echo "Usage: ./submit.sh eval [cosmo_exp] [run_id] [additional args...]"
            exit 1
        fi
        FINAL_ARGS=("--cosmo-exp" "$COSMO_EXP" "--run-id" "$RUN_ID" "${YAML_ARGS[@]}")
        echo ""
        echo "Eval job configuration:"
        echo "  Cosmo experiment: $COSMO_EXP"
        echo "  Run ID: $RUN_ID"
        if [ -n "$COSMO_MODEL" ]; then
            echo "  Cosmo model: $COSMO_MODEL"
        fi
        if [ ${#YAML_ARGS[@]} -gt 0 ]; then
            echo "  Eval params from YAML: ${#YAML_ARGS[@]} arguments"
        else
            echo "  Eval params from YAML: none (using defaults or command-line only)"
        fi
        ;;

    resume)
        SLURM_SCRIPT_FILE="train.sh"
        PYTHON_MODULE="bedcosmo.train"
        if [ -z "$RESUME_ID" ]; then
            echo "Error: resume_id is required for resume"
            echo "Usage: ./submit.sh resume [cosmo_exp] [resume_id] [resume_step] [additional args...]"
            exit 1
        fi
        if [ -z "$RESUME_STEP" ]; then
            echo "Error: resume_step is required for resume"
            echo "Usage: ./submit.sh resume [cosmo_exp] [resume_id] [resume_step] [additional args...]"
            exit 1
        fi

        FINAL_ARGS=("--cosmo-exp" "$COSMO_EXP" "--resume-id" "$RESUME_ID" "--resume-step" "$RESUME_STEP" "${YAML_ARGS[@]}")

        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log-usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;

    restart)
        SLURM_SCRIPT_FILE="train.sh"
        PYTHON_MODULE="bedcosmo.train"
        if [ -z "$RESTART_ID" ]; then
            echo "Error: restart_id is required for restart"
            echo "Usage: ./submit.sh restart [cosmo_exp] [restart_id] [restart_step|restart_checkpoint] [additional args...]"
            exit 1
        fi
        if [ -z "$RESTART_STEP" ] && [ -z "$RESTART_CHECKPOINT" ]; then
            echo "Error: Either --restart-step or --restart-checkpoint is required for restart"
            echo "Usage: ./submit.sh restart [cosmo_exp] [restart_id] [restart_step|restart_checkpoint] [additional args...]"
            exit 1
        fi

        FINAL_ARGS=("--cosmo-exp" "$COSMO_EXP" "--restart-id" "$RESTART_ID")

        if [ -n "$RESTART_STEP" ]; then
            FINAL_ARGS+=("--restart-step" "$RESTART_STEP")
        fi
        if [ -n "$RESTART_CHECKPOINT" ]; then
            FINAL_ARGS+=("--restart-checkpoint" "$RESTART_CHECKPOINT")
        fi
        if [ "$RESTART_OPTIMIZER" = true ]; then
            FINAL_ARGS+=("--restart-optimizer")
        fi
        FINAL_ARGS+=("${YAML_ARGS[@]}")

        if [ "$DEBUG" = true ]; then
            FINAL_ARGS+=("--mlflow-exp" "debug")
        fi
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log-usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;

    *)
        echo "Error: Unknown job type: $JOB_TYPE"
        echo ""
        echo "Available job types: train, eval, resume, restart"
        exit 1
        ;;
esac

# Log/job name: use job type so logs are labeled train, eval, resume, restart (or just "debug" when --debug)
LOG_NAME="$JOB_TYPE"
if [ "$DEBUG" = true ]; then
    LOG_NAME="debug"
fi

# ──────────────────────────────────────────────────────────────────────
# Print job summary
# ──────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Submitting $JOB_TYPE job ($EXECUTION_MODE mode)"
if [ "$EXECUTION_MODE" = "slurm" ]; then
    echo "Script: $SLURM_SCRIPT_FILE"
    echo "Time:   $SLURM_TIME"
    echo "Queue:  $SLURM_QUEUE"
    echo "Nodes:  $SLURM_NODES"
fi
echo "Cosmo experiment: $COSMO_EXP"
echo "Cosmo model: $COSMO_MODEL"
if [ "$EXECUTION_MODE" = "local" ]; then
    echo "GPUs: ${GPUS:-auto}"
    echo "OMP threads: ${OMP_THREADS:-not set}"
fi
if [ -n "$RUN_ID" ]; then
    echo "Run ID: $RUN_ID"
fi
if [ -n "$RESUME_ID" ]; then
    echo "Resume ID: $RESUME_ID"
fi
if [ -n "$RESTART_ID" ]; then
    echo "Restart ID: $RESTART_ID"
fi
if [ -n "$RESUME_STEP" ]; then
    echo "Resume step: $RESUME_STEP"
fi
if [ -n "$RESTART_STEP" ]; then
    echo "Restart step: $RESTART_STEP"
fi
echo ""

# Count actual number of arguments
arg_count=0
i=0
while [ $i -lt ${#FINAL_ARGS[@]} ]; do
    arg="${FINAL_ARGS[$i]}"
    if [[ "$arg" == --* ]]; then
        arg_count=$((arg_count + 1))
        if [ $((i+1)) -lt ${#FINAL_ARGS[@]} ]; then
            next="${FINAL_ARGS[$((i+1))]}"
            if [[ "$next" != --* ]]; then
                i=$((i+2))
            else
                i=$((i+1))
            fi
        else
            i=$((i+1))
        fi
    else
        i=$((i+1))
    fi
done

echo "Final arguments ($arg_count args):"
i=0
while [ $i -lt ${#FINAL_ARGS[@]} ]; do
    arg="${FINAL_ARGS[$i]}"
    if [[ "$arg" == --* ]] && [ $((i+1)) -lt ${#FINAL_ARGS[@]} ]; then
        next="${FINAL_ARGS[$((i+1))]}"
        if [[ "$next" != --* ]]; then
            printf '  %s %s\n' "$arg" "$next"
            i=$((i+2))
        else
            printf '  %s\n' "$arg"
            i=$((i+1))
        fi
    else
        printf '  %s\n' "$arg"
        i=$((i+1))
    fi
done
echo "=========================================="
echo ""

# ──────────────────────────────────────────────────────────────────────
# Build CLI overrides string for logging in SLURM job scripts
# ──────────────────────────────────────────────────────────────────────
CLI_OVERRIDES_STR=""
for key in "${!CLI_ARGS[@]}"; do
    value="${CLI_ARGS[$key]}"
    if [ "$value" = "true" ]; then
        CLI_OVERRIDES_STR+="--$key "
    elif [ "$value" != "false" ]; then
        CLI_OVERRIDES_STR+="--$key $value "
    fi
done
if [ "$LOG_USAGE" = true ]; then
    CLI_OVERRIDES_STR+="--log-usage "
fi
if [ "$PROFILE" = true ]; then
    CLI_OVERRIDES_STR+="--profile "
fi
if [ "$DEBUG" = true ]; then
    CLI_OVERRIDES_STR+="--debug "
fi
if [ "$RESTART_OPTIMIZER" = true ]; then
    CLI_OVERRIDES_STR+="--restart-optimizer "
fi
CLI_OVERRIDES_STR="${CLI_OVERRIDES_STR% }"
export BED_CLI_OVERRIDES="$CLI_OVERRIDES_STR"

# ======================================================================
# EXECUTION
# ======================================================================

# For resume jobs, run metrics truncation before launching (both modes)
if [ "$JOB_TYPE" = "resume" ]; then
    TRUNCATE_SCRIPT="$PROJECT_ROOT/scripts/truncate_metrics.py"
    if [[ -f "$TRUNCATE_SCRIPT" ]]; then
        echo "=========================================="
        echo "Pre-step: Truncating metrics to resume step..."
        echo "=========================================="
        set +e
        python3 "$TRUNCATE_SCRIPT" --run-id "$RESUME_ID" --resume-step "$RESUME_STEP" --cosmo-exp "$COSMO_EXP"
        TRUNCATE_EXIT=$?
        set -e

        if [[ $TRUNCATE_EXIT -eq 0 ]]; then
            echo "Metrics truncation completed successfully!"
        else
            echo "Error: Metrics truncation failed (exit code: $TRUNCATE_EXIT). Aborting resume."
            exit 1
        fi
        echo ""
    else
        echo "Warning: Truncate script not found at $TRUNCATE_SCRIPT"
        echo "Proceeding without metrics truncation..."
    fi
fi

if [ "$EXECUTION_MODE" = "slurm" ]; then
    # ──────────────────────────────────────────────────────────────────
    # SLURM path: submit via sbatch
    # ──────────────────────────────────────────────────────────────────
    SLURM_SCRIPT_DIR="$PROJECT_ROOT/scripts/slurm"

    if [ ! -f "$SLURM_SCRIPT_DIR/$SLURM_SCRIPT_FILE" ]; then
        echo "Error: Script file not found: $SLURM_SCRIPT_DIR/$SLURM_SCRIPT_FILE"
        exit 1
    fi

    # Build sbatch flags (these override #SBATCH directives in the script)
    SBATCH_ARGS+=("--job-name=$LOG_NAME" "--time=$SLURM_TIME" "--qos=$SLURM_QUEUE" "--nodes=$SLURM_NODES")
    if [ "$GPUS_SET_BY_USER" = true ]; then
        SBATCH_ARGS+=("--gpus-per-node=$GPUS")
    fi
    if [ "$DEBUG" = true ]; then
        SBATCH_ARGS+=("--mail-type=NONE")
    fi

    TEMP_SBATCH_OUTPUT=$(mktemp)
    set +e
    sbatch "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT_DIR/$SLURM_SCRIPT_FILE" "${FINAL_ARGS[@]}" > "$TEMP_SBATCH_OUTPUT" 2>&1
    SBATCH_EXIT_CODE=$?
    set -e
    SBATCH_OUTPUT=$(cat "$TEMP_SBATCH_OUTPUT")
    rm -f "$TEMP_SBATCH_OUTPUT"

    if [ $SBATCH_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "ERROR: Failed to submit job to SLURM"
        echo "=========================================="
        echo ""
        echo "Exit code: $SBATCH_EXIT_CODE"
        echo ""
        echo "SLURM error message:"
        echo "----------------------------------------"
        echo "$SBATCH_OUTPUT"
        echo "----------------------------------------"
        echo ""
        echo "Please check:"
        echo "  1. SLURM is available and accessible"
        echo "  2. All required arguments are valid"
        echo "  3. The script file exists: $SLURM_SCRIPT_DIR/$SLURM_SCRIPT_FILE"
        echo "  4. Your QOS/account limits (time, nodes, etc.)"
        if [ "$JOB_TYPE" = "eval" ]; then
            echo "  5. The MLflow run ID exists: $RUN_ID"
            echo "  6. The cosmo_exp directory exists: $PROJECT_ROOT/$COSMO_EXP"
        fi
        echo ""
        exit 1
    fi

    echo ""
    echo "Job submitted successfully!"
    echo "$SBATCH_OUTPUT"
    echo ""
    echo "Check status with: squeue -u $USER"

    # ──────────────────────────────────────────────────────────────
    # Auto-eval: submit dependent eval job after training completes
    # ──────────────────────────────────────────────────────────────
    if [ "$AUTO_EVAL" = true ] && [[ "$JOB_TYPE" == "train" || "$JOB_TYPE" == "restart" || "$JOB_TYPE" == "resume" ]]; then
        TRAIN_JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+$')
        if [ -z "$TRAIN_JOB_ID" ]; then
            echo "Warning: Could not parse train job ID from sbatch output. Skipping auto-eval."
        else
            echo ""
            echo "==========================================="
            echo "Submitting dependent eval job (auto_eval)"
            echo "==========================================="
            echo "  Dependency: afterok:$TRAIN_JOB_ID"
            if [ ${#EVAL_EXTRA_ARGS[@]} -gt 0 ]; then
                echo "  Eval args: ${EVAL_EXTRA_ARGS[*]}"
            fi

            EVAL_SCRIPT="$PROJECT_ROOT/scripts/slurm/eval.sh"
            EVAL_SBATCH_ARGS=("--dependency=afterany:$TRAIN_JOB_ID" "--job-name=eval" "--time=$EVAL_TIME" "--qos=$SLURM_QUEUE" "--nodes=1")
            EVAL_FINAL_ARGS=("--cosmo-exp" "$COSMO_EXP" "--train-job-id" "$TRAIN_JOB_ID" "${EVAL_EXTRA_ARGS[@]}")

            # Load eval_args.yaml for the cosmo_model if available
            EVAL_CONFIG_FILE="$PROJECT_ROOT/experiments/${COSMO_EXP}/eval_args.yaml"
            if [ -n "$COSMO_MODEL" ] && [ -f "$EVAL_CONFIG_FILE" ]; then
                EVAL_YAML_OUTPUT=$(python3 -c "
import yaml, json, sys, warnings
warnings.filterwarnings('ignore')
try:
    with open('$EVAL_CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    if '$COSMO_MODEL' in config:
        model_config = config['$COSMO_MODEL']
        # Remove cosmo_model (inferred from run) and keys we're overriding
        for k in ['cosmo_model']:
            model_config.pop(k, None)
        print(json.dumps(model_config))
    else:
        print('{}')
except Exception as e:
    print('{}', file=sys.stdout)
    print(f'Warning: {e}', file=sys.stderr)
" 2>/dev/null)

                while IFS="=" read -r key value; do
                    # Convert underscores to hyphens for CLI flags (POSIX convention)
                    key="${key//_/-}"

                    # Skip keys that were explicitly provided via --eval-* args
                    skip_key=false
                    for ((ei=0; ei<${#EVAL_EXTRA_ARGS[@]}; ei++)); do
                        if [[ "${EVAL_EXTRA_ARGS[$ei]}" == "--$key" ]]; then
                            skip_key=true
                            break
                        fi
                    done
                    if [ "$skip_key" = true ]; then
                        continue
                    fi
                    if [ "$value" = "null" ] || [ "$value" = "None" ] || [ -z "$value" ]; then
                        continue
                    fi
                    if [ "$value" = "true" ]; then
                        EVAL_FINAL_ARGS+=("--$key")
                    elif [ "$value" = "false" ]; then
                        continue
                    else
                        EVAL_FINAL_ARGS+=("--$key" "$value")
                    fi
                done < <(echo "$EVAL_YAML_OUTPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for key, value in data.items():
    if value is None:
        print(f'{key}=null')
    elif isinstance(value, list):
        formatted = json.dumps(value)
        print(f'{key}={formatted}')
    elif isinstance(value, bool):
        print(f'{key}={str(value).lower()}')
    else:
        print(f'{key}={value}')
" 2>/dev/null)
            fi

            TEMP_EVAL_OUTPUT=$(mktemp)
            set +e
            sbatch "${EVAL_SBATCH_ARGS[@]}" "$EVAL_SCRIPT" "${EVAL_FINAL_ARGS[@]}" > "$TEMP_EVAL_OUTPUT" 2>&1
            EVAL_EXIT_CODE=$?
            set -e
            EVAL_OUTPUT=$(cat "$TEMP_EVAL_OUTPUT")
            rm -f "$TEMP_EVAL_OUTPUT"

            if [ $EVAL_EXIT_CODE -ne 0 ]; then
                echo "Warning: Failed to submit auto-eval job (exit code: $EVAL_EXIT_CODE)"
                echo "$EVAL_OUTPUT"
            else
                echo "$EVAL_OUTPUT"
                echo ""
                echo "Eval job will run after train job $TRAIN_JOB_ID completes successfully."
            fi
        fi
    fi

else
    # ──────────────────────────────────────────────────────────────────
    # Local path: run directly via torchrun
    # ──────────────────────────────────────────────────────────────────

    # Detect available GPUs and set or cap GPUS
    AVAILABLE_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) || AVAILABLE_GPUS=0
    [ -z "$AVAILABLE_GPUS" ] && AVAILABLE_GPUS=0
    if [ "$GPUS_SET_BY_USER" = false ]; then
        if [ "$JOB_TYPE" = "eval" ]; then
            GPUS=1
            echo "Eval job: using 1 GPU (override with --gpus if needed)"
        elif [ "$AVAILABLE_GPUS" -le 0 ]; then
            GPUS=1
            echo "No GPUs detected; using 1 process (CPU). Training may fail if code requires CUDA."
        else
            GPUS=$AVAILABLE_GPUS
            echo "No --gpus set; using $GPUS GPU(s) (detected $AVAILABLE_GPUS available)"
        fi
    else
        if [ "$AVAILABLE_GPUS" -le 0 ]; then
            GPUS=1
            echo "Warning: no GPUs available; capping to 1 process. Training may fail if code requires CUDA."
        elif [ "$GPUS" -gt "$AVAILABLE_GPUS" ]; then
            echo "Warning: --gpus $GPUS exceeds available ($AVAILABLE_GPUS); capping to $AVAILABLE_GPUS"
            GPUS=$AVAILABLE_GPUS
        fi
    fi

    # Set OMP_NUM_THREADS if specified
    if [ -n "$OMP_THREADS" ]; then
        export OMP_NUM_THREADS=$OMP_THREADS
    fi

    # Change to project root
    cd "$PROJECT_ROOT"

    # Setup logging
    SCRATCH_DIR="${SCRATCH:-$HOME/scratch}"
    LOG_BASE_DIR="${SCRATCH_DIR}/bedcosmo/${COSMO_EXP}/logs"
    mkdir -p "$LOG_BASE_DIR"
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_FILE="${LOG_BASE_DIR}/${TIMESTAMP}_${LOG_NAME}.log"

    echo "Logging output to: $LOG_FILE"
    echo ""
    echo "Executing: torchrun --nproc_per_node=$GPUS -m $PYTHON_MODULE [${#FINAL_ARGS[@]} arguments]"
    echo ""
    torchrun --nproc_per_node=$GPUS -m "$PYTHON_MODULE" "${FINAL_ARGS[@]}" > "$LOG_FILE" 2>&1
    TRAIN_EXIT_CODE=$?

    # ──────────────────────────────────────────────────────────────
    # Auto-eval: run eval after training completes (local mode)
    # ──────────────────────────────────────────────────────────────
    if [ "$AUTO_EVAL" = true ] && [[ "$JOB_TYPE" == "train" || "$JOB_TYPE" == "restart" || "$JOB_TYPE" == "resume" ]] && [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "==========================================="
        echo "Training complete. Running auto-eval..."
        echo "==========================================="

        # Extract run_id from training log
        AUTO_RUN_ID=$(grep "MLFlow Run Info:" "$LOG_FILE" | head -n 1 | awk -F'/' '{print $NF}')
        if [ -z "$AUTO_RUN_ID" ]; then
            echo "Error: Could not extract run_id from training log $LOG_FILE"
            echo "Skipping auto-eval."
        else
            echo "Extracted run_id: $AUTO_RUN_ID"

            EVAL_LOG_FILE="${LOG_BASE_DIR}/${TIMESTAMP}_eval.log"
            EVAL_ARGS=("--cosmo-exp" "$COSMO_EXP" "--run-id" "$AUTO_RUN_ID" "${EVAL_EXTRA_ARGS[@]}")

            # Load eval_args.yaml for the cosmo_model if available
            EVAL_CONFIG_FILE="$PROJECT_ROOT/experiments/${COSMO_EXP}/eval_args.yaml"
            if [ -n "$COSMO_MODEL" ] && [ -f "$EVAL_CONFIG_FILE" ]; then
                EVAL_YAML_OUTPUT=$(python3 -c "
import yaml, json, sys, warnings
warnings.filterwarnings('ignore')
try:
    with open('$EVAL_CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    if '$COSMO_MODEL' in config:
        model_config = config['$COSMO_MODEL']
        for k in ['cosmo_model']:
            model_config.pop(k, None)
        print(json.dumps(model_config))
    else:
        print('{}')
except Exception as e:
    print('{}', file=sys.stdout)
" 2>/dev/null)

                while IFS="=" read -r key value; do
                    if [[ "$key" == "grid" || "$key" == "grid_param_pts" || "$key" == "grid_feature_pts" ]]; then
                        continue
                    fi
                    if [ "$value" = "null" ] || [ "$value" = "None" ] || [ -z "$value" ]; then
                        continue
                    fi
                    if [ "$value" = "true" ]; then
                        EVAL_ARGS+=("--$key")
                    elif [ "$value" = "false" ]; then
                        continue
                    else
                        EVAL_ARGS+=("--$key" "$value")
                    fi
                done < <(echo "$EVAL_YAML_OUTPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for key, value in data.items():
    if value is None:
        print(f'{key}=null')
    elif isinstance(value, list):
        formatted = json.dumps(value)
        print(f'{key}={formatted}')
    elif isinstance(value, bool):
        print(f'{key}={str(value).lower()}')
    else:
        print(f'{key}={value}')
" 2>/dev/null)
            fi

            echo "Eval arguments: ${EVAL_ARGS[*]}"
            echo "Logging eval output to: $EVAL_LOG_FILE"
            echo ""
            torchrun --nproc_per_node=1 -m bedcosmo.evaluate "${EVAL_ARGS[@]}" > "$EVAL_LOG_FILE" 2>&1
        fi
    elif [ "$AUTO_EVAL" = true ] && [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "Training failed (exit code: $TRAIN_EXIT_CODE). Skipping auto-eval."
    fi
fi
