#!/bin/bash
# Unified launcher for empirical SED prior-flow training.
#
# Auto-detects SLURM vs local exactly like submit.sh: uses sbatch when available
# (override with --local / --slurm). On SLURM it submits
# scripts/slurm/train_prior_flow.sh; locally it caps threads and runs the module
# directly. All other args pass through to the training module.
#
# Usage (run from this directory, src/bedcosmo/num_visits/empirical/):
#   ./train_prior_flow.sh --space both --n 100000 --epochs 400
#   ./train_prior_flow.sh --time 00:30:00 --queue debug --space native --n 50000
#   ./train_prior_flow.sh --local --threads 8 --space native --n 20000
#
# Module flags: --space {native,gaussianized,both} --n --epochs --hidden --transforms
#               --bins --batch-size --lr --seed --out-dir --kde-path --threads

set -euo pipefail

# This launcher lives at src/bedcosmo/num_visits/empirical/; repo root is 4 up.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="auto"
SB_TIME=""
SB_QUEUE=""
PASS_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --local) MODE="local"; shift ;;
        --slurm) MODE="slurm"; shift ;;
        --time) SB_TIME="$2"; shift 2 ;;
        --queue) SB_QUEUE="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

if [ "$MODE" = "auto" ]; then
    if command -v sbatch &>/dev/null; then MODE="slurm"; else MODE="local"; fi
fi

if [ "$MODE" = "slurm" ]; then
    SB_FLAGS=()
    [ -n "$SB_TIME" ] && SB_FLAGS+=("-t" "$SB_TIME")
    [ -n "$SB_QUEUE" ] && SB_FLAGS+=("-q" "$SB_QUEUE")
    echo "[launch] SLURM: sbatch ${SB_FLAGS[*]:-} scripts/slurm/train_prior_flow.sh ${PASS_ARGS[*]:-}"
    exec sbatch "${SB_FLAGS[@]}" scripts/slurm/train_prior_flow.sh "${PASS_ARGS[@]}"
fi

# Local execution -- also the body of the SLURM job, which re-invokes this launcher
# with --local on the compute node (so the run logic lives in exactly one place).
# THREADS defaults to 8 to avoid thrashing a shared login node; the SLURM path
# forwards the full-node count via --threads. When --space is "both" the native and
# gaussianized flows are independent, so we train them CONCURRENTLY, each pinned to
# half the threads -- a single small NSF can't saturate a many-core node, so this
# halves wall time without a second node.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

MOD="bedcosmo.num_visits.empirical.prior_flow"
THREADS=8
SPACE="both"
BASE_ARGS=()  # PASS_ARGS minus --space/--threads (re-added explicitly per process)
i=0
while [ "$i" -lt "${#PASS_ARGS[@]}" ]; do
    case "${PASS_ARGS[$i]}" in
        --threads) THREADS="${PASS_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        --space) SPACE="${PASS_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        *) BASE_ARGS+=("${PASS_ARGS[$i]}"); i=$((i + 1)) ;;
    esac
done

if [ "$SPACE" != "both" ]; then
    echo "[launch] local: space=$SPACE threads=$THREADS  args: ${BASE_ARGS[*]:-}"
    export OMP_NUM_THREADS="$THREADS"
    exec python -u -m "$MOD" "${BASE_ARGS[@]}" --space "$SPACE" --threads "$THREADS"
fi

HALF=$((THREADS / 2))
if [ "$HALF" -lt 1 ]; then HALF=1; fi
echo "[launch] local: --space both -> training native + gaussianized concurrently" \
    "(${HALF} threads each)"
OMP_NUM_THREADS="$HALF" python -u -m "$MOD" "${BASE_ARGS[@]}" \
    --space native --threads "$HALF" &
pid_native=$!
OMP_NUM_THREADS="$HALF" python -u -m "$MOD" "${BASE_ARGS[@]}" \
    --space gaussianized --threads "$HALF" &
pid_gauss=$!

rc=0
wait "$pid_native" || rc=$?
wait "$pid_gauss" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[prior-flow] a training process exited non-zero (rc=$rc)" >&2
fi
exit "$rc"
