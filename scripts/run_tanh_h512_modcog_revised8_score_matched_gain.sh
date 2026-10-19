#!/usr/bin/env bash
# The noise score with a global rescale, completing the score x rescale 2x2.
# 288 runs. See generate_tanh_h512_modcog_revised8_score_matched_gain_suite.py.
set -uo pipefail
if [[ "${SCOREMATCH_CAFFEINATED:-0}" != "1" ]]; then
  export SCOREMATCH_CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"
fi
PYTHON="${1:-python3}"; MAX_JOBS="${2:-8}"; THREADS="${3:-1}"
STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_matched_gain_p10_80"
LOG_DIR="results/${STEM}/logs"
TASKS=(ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
       multidlydmintseq dm1seqr dm2seql dmsintseq)
SPARSITIES=(p10 p20 p30 p40 p50 p60 p70 p80)
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}" \
       NUMEXPR_NUM_THREADS="${THREADS}"
mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/*.log
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_score_matched_gain_suite.py || exit 1
UNITS=(); for t in "${TASKS[@]}"; do for p in "${SPARSITIES[@]}"; do
  [[ -f "configs/${STEM}/${STEM}_${t}_${p}.json" ]] && UNITS+=("${t}_${p}")
done; done
echo "score-matched gain: ${#UNITS[@]} units, ${MAX_JOBS} workers, started $(date '+%H:%M:%S')"
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
echo "score-matched gain done in $(( ($(date +%s)-start)/60 ))m; units failed: ${failed}/${#UNITS[@]}"
exit $(( failed > 0 ))
