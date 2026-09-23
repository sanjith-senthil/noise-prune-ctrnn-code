#!/usr/bin/env bash
# H=512 6k-checkpoint baseline control: same width, less training, lower
# unpruned accuracy. Tests the baseline-accuracy explanation of the H=1024 drop
# inside one architecture. See the generator's docstring.
#
# Deliberately low JOBS by default: this is meant to run alongside a larger
# H=1024 suite without starving it.
#
# Usage: scripts/run_tanh_h512_modcog_revised8_baseline_control.sh [PY] [JOBS] [THREADS]

set -uo pipefail
PYTHON="${1:-python3}"
MAX_JOBS="${2:-2}"
THREADS="${3:-1}"

STEM="task_preservation_tanh_h512_modcog_revised8_6k_seqbest_baseline_control_p50_80"
CONFIG_DIR="configs/${STEM}"
LOG_DIR="results/${STEM}/logs"
TASKS=(ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
       multidlydmintseq dm1seqr dm2seql dmsintseq)
SPARSITIES=(p50 p60 p70 p80)

export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}" TORCH_NUM_THREADS="${THREADS}"

mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/*.log
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_baseline_control_suite.py \
  --shard-by-sparsity || exit 1

UNITS=(); for t in "${TASKS[@]}"; do for p in "${SPARSITIES[@]}"; do UNITS+=("${t}_${p}"); done; done
echo "launching up to ${MAX_JOBS} workers over ${#UNITS[@]} units"
declare -a PIDS=() PIDTASK=()
for unit in "${UNITS[@]}"; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 10; done
  "${PYTHON}" -u -m pruning_benchmark --mode suite \
    --config "${CONFIG_DIR}/${STEM}_${unit}.json" > "${LOG_DIR}/${unit}.log" 2>&1 &
  PIDS+=("$!"); PIDTASK+=("${unit}")
done
status=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "  FAILED ${PIDTASK[$i]} -- see ${LOG_DIR}/${PIDTASK[$i]}.log" >&2; status=1
  fi
done
(( status == 0 )) && echo "all ${#UNITS[@]} units complete" || echo "some units failed" >&2
exit "${status}"
