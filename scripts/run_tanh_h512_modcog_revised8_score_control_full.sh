#!/usr/bin/env bash
# Run the full uninformative-score / gain-matched control suite, one worker per task.
#
# Each task has its own config and its own output CSV, so the workers never
# contend for a file and each is independently resumable (`resume: true` skips
# run_ids already present in that task's CSV).  Re-running this script after an
# interruption resumes rather than recomputes.
#
# Thread budget: MEASURED on an 11-core machine, 8 workers over the 8 tasks:
#   THREADS=1  ->  21.0 runs/min, load ~10
#   THREADS=2  ->  20.3 runs/min, load ~20
# Throughput is flat, so the suite is system/bandwidth-bound rather than
# core-count-bound, and the extra threads only inflate load.  THREADS=1 is the
# default because it leaves the machine usable at identical throughput.  Note
# that torch still exceeds one core per worker despite these env vars, so
# JOBS*THREADS understates real core usage.
#
# Usage:
#   scripts/run_..._score_control_full.sh [PYTHON] [JOBS] [THREADS]

set -uo pipefail

PYTHON="${1:-python3}"
MAX_JOBS="${2:-8}"
THREADS="${3:-1}"

STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80"
CONFIG_DIR="configs/score_control_full"
LOG_DIR="results/${STEM}/logs"

TASKS=(
  ctxdlydm2intseq
  ctxdlydm1intseq
  dlydm1intseq
  dlydm2intseq
  multidlydmintseq
  dm1seqr
  dm2seql
  dmsintseq
)

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TORCH_NUM_THREADS="${THREADS}"

mkdir -p "${LOG_DIR}"

echo "generating per-task configs..."
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_score_control_full_suite.py --validate-inputs || exit 1

echo
echo "launching up to ${MAX_JOBS} workers x ${THREADS} threads over ${#TASKS[@]} tasks"
declare -a PIDS=()
declare -a PIDTASK=()

for task in "${TASKS[@]}"; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 5; done
  cfg="${CONFIG_DIR}/${STEM}_${task}.json"
  log="${LOG_DIR}/${task}.log"
  echo "  start ${task}  (log: ${log})"
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "${cfg}" > "${log}" 2>&1 &
  PIDS+=("$!")
  PIDTASK+=("${task}")
done

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  done  ${PIDTASK[$i]}"
  else
    echo "  FAILED ${PIDTASK[$i]} -- see ${LOG_DIR}/${PIDTASK[$i]}.log" >&2
    status=1
  fi
done

echo
if (( status == 0 )); then
  echo "all tasks complete; merging and summarizing"
  "${PYTHON}" scripts/summarize_tanh_h512_modcog_revised8_score_control_full.py
else
  echo "one or more tasks failed; rerun this script to resume" >&2
fi
exit "${status}"
