#!/bin/bash

# Shared formatting for CLI overrides passed through submit.sh.
print_bed_cli_overrides() {
    local empty_description="${1:-none - using YAML defaults}"

    if [ -n "${BED_CLI_OVERRIDES:-}" ]; then
        echo "CLI Overrides (non-default values):"
        echo "------------------------------------"
        set -- $BED_CLI_OVERRIDES
        while [ $# -gt 0 ]; do
            if [[ "$1" == --* ]]; then
                if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                    echo "  $1 $2"
                    shift 2
                else
                    echo "  $1"
                    shift 1
                fi
            else
                shift 1
            fi
        done
    else
        echo "CLI Overrides: ($empty_description)"
    fi
}

# Jobs log: one append-only file per experiment, $SCRATCH/bedcosmo/{cosmo_exp}/jobs.log,
# whose path submit.sh exports as BED_JOBS_LOG. Each job gets a QUEUED entry at
# submission (submit.sh command, optional --note, path of its full log inside the
# MLflow run), a STARTED entry and one end entry (COMPLETED / FAILED / STOPPED / SKIPPED).

# Usage: jobs_log_append EVENT SUMMARY [DETAIL_LINE...]
# The entry is written with a single printf so concurrent jobs don't interleave lines.
jobs_log_append() {
    : "${BED_JOBS_LOG:?BED_JOBS_LOG is not set (submit.sh sets it)}"
    local entry line
    entry=$(printf '[%s] %-9s %s' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2")
    shift 2
    for line in "$@"; do
        entry+=$'\n'"    $line"
    done
    mkdir -p "$(dirname "$BED_JOBS_LOG")"
    printf '%s\n' "$entry" >> "$BED_JOBS_LOG"
}

# Path of an MLflow run's directory (file store: mlruns/<exp_id>/<run_id>).
# Usage: mlflow_run_dir COSMO_EXP RUN_ID
mlflow_run_dir() {
    local matches=("${SCRATCH}/bedcosmo/$1/mlruns/"*/"$2")
    if [ ! -d "${matches[0]}" ]; then
        echo "Error: MLflow run $2 not found under ${SCRATCH}/bedcosmo/$1/mlruns" >&2
        return 1
    fi
    echo "${matches[0]}"
}

# Record a job's STARTED entry now and its end entry when the calling script exits.
# BED_JOB_SUMMARY (set by submit.sh) labels both entries.
# SIGTERM (scancel or the SLURM time limit) and SIGINT only set a flag, so the script
# carries on to the EXIT trap once the running step returns. A job killed outright
# (SIGKILL, node failure) never writes an end entry and stays STARTED.
# Set JOB_SKIPPED="<reason>" before exiting to record SKIPPED instead of COMPLETED.
# Under SLURM the STARTED entry also names the node, for matching against sacct.
# Usage: jobs_log_track JOB_ID [DETAIL_LINE...]
jobs_log_track() {
    local job_id=$1 summary="$BED_JOB_SUMMARY job=$1"
    shift
    JOB_START=$(date +%s)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        summary+=" node=$(hostname)"
    fi
    jobs_log_append STARTED "$summary" "$@"
    trap 'JOB_STOPPED=1' TERM INT
    trap "_jobs_log_finish \$? $job_id" EXIT
}

_jobs_log_finish() {
    local code=$1 status details=()
    if [ -n "${JOB_SKIPPED:-}" ]; then
        status=SKIPPED
        details+=("reason: $JOB_SKIPPED")
    elif [ -n "${JOB_STOPPED:-}" ]; then
        status=STOPPED
    elif [ "$code" -eq 0 ]; then
        status=COMPLETED
    else
        status=FAILED
    fi
    jobs_log_append "$status" "$BED_JOB_SUMMARY job=$2 exit=$code elapsed=$(_jobs_log_duration)" "${details[@]}"
}

# Time since jobs_log_track as 45s, 12m or 4h18m.
_jobs_log_duration() {
    local s=$(( $(date +%s) - JOB_START ))
    if [ "$s" -lt 60 ]; then
        echo "${s}s"
    elif [ "$s" -lt 3600 ]; then
        echo "$((s / 60))m"
    else
        printf '%dh%02dm\n' $((s / 3600)) $((s % 3600 / 60))
    fi
}
