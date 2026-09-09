#!/usr/bin/env bash
# Run the two peer-review revision suites, one worker per (suite, task) config.
#
# Each config has its own output CSV, so workers never contend for a file and
# each is independently resumable (`resume: true` skips run_ids already
# present). Re-running this script after an interruption resumes rather than
# recomputes.
#
# Thread budget: measured on an 11-core machine, 8 workers over the task
# configs, THREADS=1 gave 21.0 runs/min at load ~10 and THREADS=2 gave 20.3 at
# load ~20 -- throughput is flat, so 1 keeps the machine usable. Note torch
# still exceeds one core per worker despite these env vars.
#
# Usage:
#   scripts/run_tanh_h512_modcog_revised8_revision_suites.sh [PYTHON] [JOBS] [THREADS]

set -uo pipefail

PYTHON="${1:-python3}"
MAX_JOBS="${2:-8}"
THREADS="${3:-1}"

LOW_STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40"
NOISE_STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80"

TASKS=(
  ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
  multidlydmintseq dm1seqr dm2seql dmsintseq
)

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TORCH_NUM_THREADS="${THREADS}"

echo "generating per-task configs..."
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_revision_suites.py --validate-inputs || exit 1

declare -a JOBS=()
for stem in "${LOW_STEM}" "${NOISE_STEM}"; do
  mkdir -p "results/${stem}/logs"
  for task in "${TASKS[@]}"; do
    JOBS+=("${stem}:${task}")
  done
done

echo
echo "launching ${#JOBS[@]} configs, up to ${MAX_JOBS} workers x ${THREADS} threads"
declare -a PIDS=() LABELS=()
for job in "${JOBS[@]}"; do
  stem="${job%%:*}"; task="${job##*:}"
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 5; done
  cfg="configs/${stem}/${stem}_${task}.json"
  log="results/${stem}/logs/${task}.log"
  echo "  start ${stem##*seqbest_} / ${task}"
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "${cfg}" > "${log}" 2>&1 &
  PIDS+=("$!"); LABELS+=("${stem##*seqbest_}/${task}")
done

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  done  ${LABELS[$i]}"
  else
    echo "  FAILED ${LABELS[$i]}" >&2
    status=1
  fi
done

echo
if (( status == 0 )); then
  echo "all configs complete; summarizing"
  "${PYTHON}" scripts/summarize_tanh_h512_modcog_revised8_revision_suites.py
else
  echo "one or more configs failed; rerun this script to resume" >&2
fi
exit "${status}"
