#!/usr/bin/env bash
# Train the two excluded tasks (dm1, anti) under the paper's exact protocol.
#
# One worker per network seed; within a seed the two phases run in order,
# because phase 2 loads the checkpoint phase 1 writes. Each phase trains both
# tasks. All suites are resumable (`resume: true` skips completed run_ids), so
# re-running this script continues rather than recomputes.
#
# Checkpoints use a `modcog_dm1anti` prefix, distinct from `modcog_revised8`,
# so the paper's 24 networks cannot be touched.
#
# Usage:
#   scripts/run_tanh_h512_modcog_dm1anti_baseline.sh [PYTHON] [THREADS]

set -uo pipefail

PYTHON="${1:-python3}"
THREADS="${2:-3}"

SEEDS=(0 1 2)
PHASE1_STEM="train_tanh_h512_modcog_dm1anti_6k_seqbest_no_l2"
PHASE2_STEM="continue_tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2"
LOG_DIR="results/dm1anti_logs"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TORCH_NUM_THREADS="${THREADS}"

mkdir -p "${LOG_DIR}"

echo "generating configs..."
"${PYTHON}" scripts/generate_tanh_h512_modcog_dm1anti_baseline_suite.py || exit 1

echo
echo "training ${#SEEDS[@]} seeds in parallel (${THREADS} threads each), two phases per seed"
declare -a PIDS=() LABELS=()
for seed in "${SEEDS[@]}"; do
  (
    set -o pipefail
    "${PYTHON}" -u -m pruning_benchmark --mode suite \
      --config "configs/${PHASE1_STEM}_seed${seed}.json" > "${LOG_DIR}/seed${seed}_phase1.log" 2>&1 || exit 1
    "${PYTHON}" -u -m pruning_benchmark --mode suite \
      --config "configs/${PHASE2_STEM}_seed${seed}.json" > "${LOG_DIR}/seed${seed}_phase2.log" 2>&1 || exit 1
  ) &
  PIDS+=("$!"); LABELS+=("seed${seed}")
  echo "  start seed${seed}"
done

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  done  ${LABELS[$i]}"
  else
    echo "  FAILED ${LABELS[$i]} -- see ${LOG_DIR}/${LABELS[$i]}_phase*.log" >&2
    status=1
  fi
done

echo
if (( status == 0 )); then
  echo "all seeds complete; summarizing"
  "${PYTHON}" scripts/summarize_tanh_h512_modcog_dm1anti_baseline.py
else
  echo "one or more seeds failed; rerun this script to resume" >&2
fi
exit "${status}"
