#!/bin/bash

# Helper script to validate arguments, load config defaults, and submit SLURM jobs
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
    echo "  ./submit.sh train --cosmo_exp num_tracers --cosmo_model base_w_wa --initial_lr 0.0001 --log_usage"
    echo "  ./submit.sh debug num_tracers --cosmo_model base --profile"
    echo "  ./submit.sh eval num_tracers abc123"
    echo "  ./submit.sh resume num_tracers abc123 5000 --log_usage"
    echo "  ./submit.sh restart num_tracers abc123 10000"
    echo "  ./submit.sh restart num_tracers abc123 checkpoint_rank0_step10000.pt --restart_optimizer"
    echo ""
    echo "Optional flags: --log_usage, --profile (for train/debug/resume/restart)"
    echo "Restart-specific: --restart_checkpoint, --restart_optimizer"
    echo "Note: 'eval', 'resume', and 'restart' jobs do not require --cosmo_model (inferred from MLflow run)"
    echo "Note: 'restart' requires either --restart_step OR --restart_checkpoint"
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
declare -A CLI_ARGS  # Associative array for command-line arguments
POSITIONAL_ARGS=()
SBATCH_ARGS=()

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
        --exclude)
            SBATCH_ARGS+=("--exclude=$2")
            shift 2
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
    echo ""
    echo "Note: cosmo_model can be provided positionally after cosmo_exp."
    echo "Note: --cosmo_model is not needed for 'eval', 'resume', or 'restart' jobs (inferred from MLflow run)"
    exit 1
fi

# Activate conda environment early (needed for MLflow and YAML parsing on login node)
# This is necessary because we use Python with mlflow and yaml modules before submitting to SLURM
module load conda 2>/dev/null || true
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate bed-cosmo

# For eval jobs, infer cosmo_model from MLflow run
if [ "$JOB_TYPE" = "eval" ] && [ -z "$COSMO_MODEL" ]; then
    echo "Inferring cosmo_model from MLflow run ${RUN_ID}..."
    COSMO_MODEL=$(python3 -c "
import mlflow
import os
import sys
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('file:' + os.environ['SCRATCH'] + '/bed/BED_cosmo/${COSMO_EXP}/mlruns')
client = MlflowClient()
try:
    run = client.get_run('${RUN_ID}')
    cosmo_model = run.data.params.get('cosmo_model', 'base')
    print(cosmo_model)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    exit(1)
" 2>&1)
    
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
    CONFIG_FILE="$PROJECT_ROOT/${COSMO_EXP}/eval_args.yaml"
else
    CONFIG_FILE="$PROJECT_ROOT/${COSMO_EXP}/train_args.yaml"
fi

if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE"
    echo "Proceeding without default configuration..."
    YAML_ARGS=()
elif [ -n "$CONFIG_FILE" ]; then
    echo "Loading config from: $CONFIG_FILE"
    echo "Using model: $COSMO_MODEL"
    
    # Parse YAML and extract config for specified model
    # Using Python to parse YAML reliably
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
    # Use a more robust approach: read all key-value pairs into YAML_ARGS array
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
        
        # Handle boolean values (now consistently lowercase from Python)
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

# Validate and build arguments based on job type
case $JOB_TYPE in
    train|debug|variable_reshift_debug)
        SCRIPT_FILE="${JOB_TYPE}.sh"
        FINAL_ARGS=("--cosmo_exp" "$COSMO_EXP" "${YAML_ARGS[@]}")
        # Add optional boolean flags
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log_usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;
    
    eval)
        SCRIPT_FILE="eval.sh"
        if [ -z "$RUN_ID" ]; then
            echo "Error: run_id is required for eval"
            echo "Usage: ./submit.sh eval [cosmo_exp] [run_id] [additional args...]"
            exit 1
        fi
        FINAL_ARGS=("--cosmo_exp" "$COSMO_EXP" "--run_id" "$RUN_ID" "${YAML_ARGS[@]}")
        ;;
    
    resume)
        SCRIPT_FILE="resume.sh"
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
        FINAL_ARGS=("--cosmo_exp" "$COSMO_EXP" "--resume_id" "$RESUME_ID" "--resume_step" "$RESUME_STEP" "${YAML_ARGS[@]}")
        # Add optional boolean flags
        if [ "$LOG_USAGE" = true ]; then
            FINAL_ARGS+=("--log_usage")
        fi
        if [ "$PROFILE" = true ]; then
            FINAL_ARGS+=("--profile")
        fi
        ;;
    
    restart)
        SCRIPT_FILE="restart.sh"
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
        FINAL_ARGS=("--cosmo_exp" "$COSMO_EXP" "--restart_id" "$RESTART_ID")
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
        echo "Available job types: train, eval, debug, resume, restart, variable_reshift_debug"
        exit 1
        ;;
esac

# Verify script file exists
if [ ! -f "$SCRIPT_DIR/$SCRIPT_FILE" ]; then
    echo "Error: Script file not found: $SCRIPT_DIR/$SCRIPT_FILE"
    exit 1
fi

# Submit the job
echo "=========================================="
echo "Submitting $JOB_TYPE job to SLURM"
echo "Script: $SCRIPT_FILE"
echo "Cosmo experiment: $COSMO_EXP"
echo "Cosmo model: $COSMO_MODEL"
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

# Count actual number of arguments (each --key counts as 1, regardless of whether it has a value)
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

sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/$SCRIPT_FILE" "${FINAL_ARGS[@]}"

echo ""
echo "Job submitted successfully!"
echo "Check status with: squeue -u $USER"
