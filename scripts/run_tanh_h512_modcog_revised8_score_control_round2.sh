#!/usr/bin/env bash
# Run the round-2 reviewer control suite, one worker per task.
#
# Same execution model as the round-1 script: each task has its own config and
# its own output CSV, so workers never contend for a file and each is
# independently resumable (`resume: true` skips run_ids already present).
# Re-running after an interruption resumes rather than recomputes.
#
# THREADS=1 by default -- round 1 measured 21.0 runs/min at THREADS=1 and
# 20.3 at THREADS=2 on an 11-core machine, i.e. the suite is bandwidth-bound,
# so extra threads buy nothing and only inflate load.
#
# SHARD=1 splits each task four ways by sparsity, giving 32 units of work
# instead of 8 so that JOBS can exceed the number of tasks.  Use it only on a
# machine with more than eight usable cores, and measure: the unsharded layout
# already drew ~760% CPU at load ~10.4 on the 11-core (5P + 6E) development
# machine, where more workers mostly add queuing.  Do not switch layouts
# midway through a set of results -- `resume` keys on the individual output
# CSV, so a shard cannot resume an unsharded run or vice versa.
#
# Usage:
#   scripts/run_..._score_control_round2.sh [PYTHON] [JOBS] [THREADS] [SHARD]

set -uo pipefail

PYTHON="${1:-python3}"
MAX_JOBS="${2:-8}"
THREADS="${3:-1}"
SHARD="${4:-0}"
SPARSITIES=(p50 p60 p70 p80)

STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_round2_p50_80"
CONFIG_DIR="configs/score_control_round2"
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

GEN_ARGS=(--validate-inputs)
UNITS=("${TASKS[@]}")
if (( SHARD == 1 )); then
  GEN_ARGS+=(--shard-by-sparsity)
  UNITS=()
  for task in "${TASKS[@]}"; do
    for pct in "${SPARSITIES[@]}"; do UNITS+=("${task}_${pct}"); done
  done
fi

echo "generating configs..."
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_score_control_round2_suite.py "${GEN_ARGS[@]}" || exit 1

echo
echo "launching up to ${MAX_JOBS} workers x ${THREADS} threads over ${#UNITS[@]} units (SHARD=${SHARD})"
declare -a PIDS=()
declare -a PIDTASK=()

for task in "${UNITS[@]}"; do
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
  "${PYTHON}" scripts/summarize_tanh_h512_modcog_revised8_score_control_round2.py
else
  echo "one or more tasks failed; rerun this script to resume" >&2
fi
exit "${status}"
