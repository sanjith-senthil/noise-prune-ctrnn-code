#!/usr/bin/env bash
# Pruning on dm1 and anti, the two excluded tasks. 438 runs.
# See generate_tanh_h512_modcog_dm1anti_pruning_suite.py.
set -uo pipefail
if [[ "${DM1ANTI_CAFFEINATED:-0}" != "1" ]]; then
  export DM1ANTI_CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"
fi
PYTHON="${1:-python3}"; MAX_JOBS="${2:-8}"; THREADS="${3:-1}"
STEM="task_preservation_tanh_h512_modcog_dm1anti_12k_seqbest_pruning_p10_80"
LOG_DIR="results/${STEM}/logs"
TASKS=(dm1 anti)
UNITSUFFIX=(baseline p10 p20 p30 p40 p50 p60 p70 p80)
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}" \
       NUMEXPR_NUM_THREADS="${THREADS}"
mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/*.log
"${PYTHON}" scripts/generate_tanh_h512_modcog_dm1anti_pruning_suite.py || exit 1
# The baselines first and alone: every pruned run divides by the unpruned
# accuracy from the same network, and the eval/score batch caches are written
# on first use. Running them concurrently would have several workers racing to
# create the same cache file.
for t in "${TASKS[@]}"; do
  "${PYTHON}" -u -m pruning_benchmark --mode suite \
      --config "configs/${STEM}/${STEM}_${t}_baseline.json" \
      > "${LOG_DIR}/${t}_baseline.log" 2>&1 || { echo "FAILED ${t}_baseline" >&2; exit 1; }
done
echo "baselines done $(date '+%H:%M:%S')"
UNITS=(); for t in "${TASKS[@]}"; do for p in "${UNITSUFFIX[@]:1}"; do UNITS+=("${t}_${p}"); done; done
echo "dm1anti pruning: ${#UNITS[@]} units, ${MAX_JOBS} workers, started $(date '+%H:%M:%S')"
start=$(date +%s); declare -a PIDS=() PIDUNIT=()
for unit in "${UNITS[@]}"; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 10; done
  "${PYTHON}" -u -m pruning_benchmark --mode suite \
      --config "configs/${STEM}/${STEM}_${unit}.json" > "${LOG_DIR}/${unit}.log" 2>&1 &
  PIDS+=("$!"); PIDUNIT+=("${unit}")
done
failed=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "  FAILED ${PIDUNIT[$i]}" >&2; failed=$((failed+1)); }
done
echo "dm1anti pruning done in $(( ($(date +%s)-start)/60 ))m; units failed: ${failed}/${#UNITS[@]}"
exit $(( failed > 0 ))
