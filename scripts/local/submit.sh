#!/bin/bash

# Helper script to validate arguments, load config defaults, and run training/eval jobs locally via torchrun
# Usage: ./submit.sh <job_type> [cosmo_exp] [id] [arguments...]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if job type is provided
if [ $# -eq 0 ]; then
    echo "Error: Job type is required"
    echo ""
    echo "Usage: ./submit.sh <job_type> [cosmo_exp] [model_or_id] [arguments...]"
    echo ""
    echo "Available job types:"
    echo "  train                - Training job"
    echo "  eval                 - Evaluation job"
    echo "  debug                - Debug training job"
    echo "  resume               - Resume training job"
    echo "  restart              - Restart training job"
    echo ""
    echo "Available cosmo_models:"
    echo "  base, base_omegak, base_w, base_w_wa, base_omegak_w_wa"
    echo ""
    echo "The script will load default parameters from <cosmo_exp>/train_args.yaml"
    echo "for the specified cosmo_model. Command-line arguments will override YAML defaults."
    echo ""
    echo "Examples:"
    echo "  ./submit.sh train num_tracers base"
    echo "  ./submit.sh train --cosmo_exp num_tracers --cosmo_model base_w_wa --initial_lr 0.0001 --gpus 2"
    echo "  ./submit.sh debug num_tracers --cosmo_model base --profile"
    echo "  ./submit.sh eval num_tracers abc123"
    echo "  ./submit.sh resume num_tracers abc123 5000"
    echo "  ./submit.sh restart num_tracers abc123 10000"
    echo ""
    echo "Optional flags: --gpus <N> (default: 2), --omp_threads <N> (default: 32)"
    echo "                --log_usage, --profile, --restart_optimizer"
    echo "Note: 'eval', 'resume', and 'restart' jobs do not require --cosmo_model (inferred from MLflow run)"
    exit 1
fi

JOB_TYPE=$1
shift  # Remove job type from arguments

# Initialize variables
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
RESTART_OPTIMIZER=false
GPUS=2  # Default GPU count
OMP_THREADS=32  # Default OMP threads

declare -A CLI_ARGS  # Associative array for command-line arguments
POSITIONAL_ARGS=()

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cosmo_exp)
            COSMO_EXP="$2"
            shift 2
            ;;
        --cosmo_model)
            COSMO_MODEL="$2"
            shift 2
            ;;
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        --resume_id)
            RESUME_ID="$2"
            shift 2
            ;;
        --restart_id)
            RESTART_ID="$2"
            shift 2
            ;;
        --resume_step)
            RESUME_STEP="$2"
            shift 2
            ;;
        --restart_step)
            RESTART_STEP="$2"
            shift 2
            ;;
        --restart_checkpoint)
            RESTART_CHECKPOINT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --omp_threads)
            OMP_THREADS="$2"
            shift 2
            ;;
        --log_usage)
            LOG_USAGE=true
            shift 1
            ;;
        --profile)
            PROFILE=true
            shift 1
            ;;
        --restart_optimizer)
            RESTART_OPTIMIZER=true
            shift 1
            ;;
        --*)
            # Store CLI argument (will override YAML defaults)
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

# Fill positional fallbacks where appropriate
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

# Validate required arguments
if [ -z "$COSMO_EXP" ]; then
    echo "Error: cosmo_exp is required"
    echo "Usage: ./submit.sh $JOB_TYPE [cosmo_exp] [cosmo_model] [additional args...]"
    exit 1
fi

# cosmo_model is NOT required for eval/resume/restart (inferred from MLflow run)
if [ -z "$COSMO_MODEL" ] && [ "$JOB_TYPE" != "eval" ] && [ "$JOB_TYPE" != "resume" ] && [ "$JOB_TYPE" != "restart" ]; then
    echo "Error: --cosmo_model is required for $JOB_TYPE jobs"
    echo "Usage: ./submit.sh $JOB_TYPE [cosmo_exp] [cosmo_model] [additional args...]"
    echo "Available models: base, base_omegak, base_w, base_w_wa, base_omegak_w_wa"
    exit 1
fi

# Activate conda environment early (needed for MLflow and YAML parsing)
if [ -f "$HOME/.bashrc" ] && grep -q "conda" "$HOME/.bashrc"; then
    source "$HOME/.bashrc"
fi

if command -v conda &> /dev/null; then
    # Get conda base and activate
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate bed-cosmo
        echo "Activated conda environment: bed-cosmo"
    else
        echo "Warning: Could not find conda.sh, trying alternative activation..."
        eval "$(conda shell.bash hook)"
        conda activate bed-cosmo
    fi
else
    echo "Error: conda not found. Please ensure conda is installed and in PATH."
    exit 1
fi

# For eval jobs, infer cosmo_model from MLflow run
if [ "$JOB_TYPE" = "eval" ] && [ -z "$COSMO_MODEL" ]; then
    echo "Inferring cosmo_model from MLflow run ${RUN_ID}..."
    COSMO_MODEL=$(python3 -c "
import mlflow
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from mlflow.tracking import MlflowClient

# Use SCRATCH if available, otherwise use local directory
scratch_path = os.environ.get('SCRATCH', os.path.expanduser('~') + '/scratch')
mlflow.set_tracking_uri('file:' + scratch_path + '/bedcosmo/${COSMO_EXP}/mlruns')
client = MlflowClient()
try:
    run = client.get_run('${RUN_ID}')
    cosmo_model = run.data.params.get('cosmo_model', 'base')
    print(cosmo_model)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    exit(1)
" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "Error: Failed to get cosmo_model from run ${RUN_ID}"
        echo "$COSMO_MODEL"
        exit 1
    fi
    echo "Inferred cosmo_model: $COSMO_MODEL"
fi

# Load YAML config file for this cosmo_exp and cosmo_model (skip for resume/restart)
if [ "$JOB_TYPE" = "resume" ] || [ "$JOB_TYPE" = "restart" ]; then
    echo "$JOB_TYPE job: Skipping YAML config (parameters from MLflow or command-line)"
    YAML_ARGS=()
    CONFIG_FILE=""
elif [ "$JOB_TYPE" = "eval" ]; then
    # Use eval_args.yaml for eval jobs
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
    echo "Using model: $COSMO_MODEL"
    
    # Parse YAML and extract config for specified model
    YAML_ARGS=()
    
    # Create a Python script to extract the model config
    YAML_OUTPUT=$(python3 -c "
import yaml
import sys
import json

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    if '$COSMO_MODEL' not in config:
        print(f'ERROR: Model $COSMO_MODEL not found in config file', file=sys.stderr)
        print(f'Available models: {list(config.keys())}', file=sys.stderr)
        sys.exit(1)
    
    model_config = config['$COSMO_MODEL']
    
    # Output as JSON for easier parsing in bash
    print(json.dumps(model_config))
except Exception as e:
    print(f'ERROR: Failed to parse YAML: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

    # Check if Python command succeeded
    if [ $? -ne 0 ]; then
        echo "$YAML_OUTPUT"
        exit 1
    fi
    
    # Parse the JSON output and build argument list
    while IFS="=" read -r key value; do
        # Skip if this key was provided via CLI (CLI overrides YAML)
        if [[ -v CLI_ARGS["$key"] ]]; then
            continue
        fi
        
        # Skip null/None values (explicitly check for both JSON null and Python None string)
        # Exception: for eval jobs, include fixed_design even if null (evaluate.py accepts it)
        if [ "$value" = "null" ] || [ "$value" = "None" ] || [ -z "$value" ]; then
            # For eval jobs, pass fixed_design even if null (evaluate.py will parse "null" as None)
            if [ "$JOB_TYPE" = "eval" ] && [ "$key" = "fixed_design" ]; then
                YAML_ARGS+=("--$key" "null")
            fi
            continue
        fi
        
        # Handle boolean values
        if [ "$value" = "true" ]; then
            YAML_ARGS+=("--$key")
        elif [ "$value" = "false" ]; then
            # Skip false boolean flags
            continue
        # Handle arrays (keep as JSON-like format)
        elif [[ "$value" == \[*\] ]]; then
            YAML_ARGS+=("--$key" "$value")
        # Handle numbers and strings
        else
            YAML_ARGS+=("--$key" "$value")
        fi
    done < <(echo "$YAML_OUTPUT" | python3 -c "
import json
import sys

data = json.load(sys.stdin)
for key, value in data.items():
    if value is None:
        # Explicitly output 'null' for None values (JSON null)
        print(f'{key}=null')
    elif isinstance(value, list):
        # Format lists as JSON-like strings
        formatted = '[' + ', '.join(str(v) for v in value) + ']'
        print(f'{key}={formatted}')
    elif isinstance(value, bool):
        # Output lowercase boolean for consistency
        print(f'{key}={str(value).lower()}')
    else:
        # Output as-is (numbers and strings)
        print(f'{key}={value}')
")
fi

# Add CLI arguments (these override YAML)
for key in "${!CLI_ARGS[@]}"; do
    value="${CLI_ARGS[$key]}"
    # Handle boolean flags
    if [ "$value" = "true" ]; then
        YAML_ARGS+=("--$key")
    elif [ "$value" != "false" ]; then
        YAML_ARGS+=("--$key" "$value")
    fi
done

# For debug jobs, filter out mlflow_exp from YAML_ARGS (debug.sh sets it to "debug")
# For eval jobs, filter out cosmo_model from YAML_ARGS (evaluate.py doesn't accept it)
if [ "$JOB_TYPE" = "debug" ] || [ "$JOB_TYPE" = "eval" ]; then
    FILTERED_ARGS=()
    skip_next=false
    for ((i=0; i<${#YAML_ARGS[@]}; i++)); do
        if [ "$skip_next" = true ]; then
            skip_next=false
            continue
        fi
        # Filter mlflow_exp for debug, cosmo_model for eval
        if [ "$JOB_TYPE" = "debug" ] && [ "${YAML_ARGS[$i]}" = "--mlflow_exp" ]; then
            skip_next=true
            continue
        fi
        if [ "$JOB_TYPE" = "eval" ] && [ "${YAML_ARGS[$i]}" = "--cosmo_model" ]; then
            skip_next=true
            continue
        fi
        FILTERED_ARGS+=("${YAML_ARGS[$i]}")
    done
    YAML_ARGS=("${FILTERED_ARGS[@]}")
fi

# Set environment variables
export OMP_NUM_THREADS=$OMP_THREADS

# Validate and build command based on job type
case $JOB_TYPE in
    train|debug)
        PYTHON_MODULE="bedcosmo.train"
        FINAL_ARGS=("--cosmo_exp" "$COSMO_EXP" "--cosmo_model" "$COSMO_MODEL" "${YAML_ARGS[@]}")

        # For debug jobs, set mlflow_exp to "debug"
        if [ "$JOB_TYPE" = "debug" ]; then
            FINAL_ARGS+=("--mlflow_exp" "debug")
        fi
        # Add optional boolean flags
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log_usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;
    
    eval)
        PYTHON_MODULE="bedcosmo.evaluate"
        if [ -z "$RUN_ID" ]; then
            echo "Error: run_id is required for eval"
            echo "Usage: ./submit.sh eval [cosmo_exp] [run_id] [additional args...]"
            exit 1
        fi
        FINAL_ARGS=("--cosmo_exp" "$COSMO_EXP" "--run_id" "$RUN_ID" "${YAML_ARGS[@]}")
        ;;
    
    resume)
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
        
        # Run metrics truncation script first
        TRUNCATE_SCRIPT="$PROJECT_ROOT/scripts/truncate_metrics.py"
        if [[ -f "$TRUNCATE_SCRIPT" ]]; then
            echo "=========================================="
            echo "Step 1: Truncating metrics to resume step..."
            echo "=========================================="
            python3 "$TRUNCATE_SCRIPT" --run_id "$RESUME_ID" --resume_step "$RESUME_STEP"
            
            if [[ $? -eq 0 ]]; then
                echo "Metrics truncation completed successfully!"
            else
                echo "Warning: Metrics truncation failed, but continuing with training resume..."
            fi
            echo ""
        fi
        
        echo "=========================================="
        echo "Step 2: Resuming training..."
        echo "=========================================="
        
        FINAL_ARGS=("--resume_id" "$RESUME_ID" "--resume_step" "$RESUME_STEP" "${YAML_ARGS[@]}")
        # Add optional boolean flags
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log_usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;
    
    restart)
        PYTHON_MODULE="bedcosmo.train"
        if [ -z "$RESTART_ID" ]; then
            echo "Error: restart_id is required for restart"
            echo "Usage: ./submit.sh restart [cosmo_exp] [restart_id] [restart_step|restart_checkpoint] [additional args...]"
            exit 1
        fi
        # Either restart_step OR restart_checkpoint must be specified
        if [ -z "$RESTART_STEP" ] && [ -z "$RESTART_CHECKPOINT" ]; then
            echo "Error: Either --restart_step or --restart_checkpoint is required for restart"
            echo "Usage: ./submit.sh restart [cosmo_exp] [restart_id] [restart_step|restart_checkpoint] [additional args...]"
            exit 1
        fi
        FINAL_ARGS=("--restart_id" "$RESTART_ID")
        if [ -n "$RESTART_STEP" ]; then
            FINAL_ARGS+=("--restart_step" "$RESTART_STEP")
        fi
        if [ -n "$RESTART_CHECKPOINT" ]; then
            FINAL_ARGS+=("--restart_checkpoint" "$RESTART_CHECKPOINT")
        fi
        if [ "$RESTART_OPTIMIZER" = true ]; then
            FINAL_ARGS+=("--restart_optimizer")
        fi
        FINAL_ARGS+=("${YAML_ARGS[@]}")
        # Add optional boolean flags
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log_usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;
    
    *)
        echo "Error: Unknown job type: $JOB_TYPE"
        echo ""
        echo "Available job types: train, eval, debug, resume, restart"
        exit 1
        ;;
esac

# Print job information
echo "=========================================="
echo "Running $JOB_TYPE job"
echo "Cosmo experiment: $COSMO_EXP"
echo "Cosmo model: $COSMO_MODEL"
echo "GPUs: $GPUS"
echo "OMP threads: $OMP_THREADS"
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
echo "Final arguments (${#FINAL_ARGS[@]} args):"
# Print arguments in pairs for readability
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

# Change to project root to run the command
cd "$PROJECT_ROOT"

# Setup logging directory and file
SCRATCH_DIR="${SCRATCH:-$HOME/scratch}"
LOG_BASE_DIR="${SCRATCH_DIR}/bedcosmo/${COSMO_EXP}/logs"
mkdir -p "$LOG_BASE_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_BASE_DIR}/${TIMESTAMP}_${JOB_TYPE}.log"

echo "Logging output to: $LOG_FILE"
echo ""

# Execute the command
echo "Executing: torchrun --nproc_per_node=$GPUS -m $PYTHON_MODULE [${#FINAL_ARGS[@]} arguments]"
echo "Output will be logged to: $LOG_FILE"
echo ""
# Redirect both stdout and stderr to log file only (no terminal output)
torchrun --nproc_per_node=$GPUS -m "$PYTHON_MODULE" "${FINAL_ARGS[@]}" > "$LOG_FILE" 2>&1
