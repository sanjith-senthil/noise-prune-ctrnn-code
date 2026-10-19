#!/usr/bin/env bash
# The cap-percentile sweep under the corrected normalisation. 2,592 runs,
# ~3 h at 8 workers. See generate_modcog_revised8_capsweep_suite.py.
# Resumable per run; runs under caffeinate so an idle machine cannot stop it.
set -uo pipefail
if [[ "${CAPSWEEP_CAFFEINATED:-0}" != "1" ]]; then
  export CAPSWEEP_CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"
fi
PYTHON="${1:-python3}"; MAX_JOBS="${2:-8}"; THREADS="${3:-1}"; SCORE="${4:-noise}"
# "noise" keeps the original stem and run_ids so the existing 2,592 rows resume;
# "magnitude" is the symmetric sweep, which lets each score pick its own best
# percentile instead of holding the noise score's choice fixed for both.
if [[ "${SCORE}" == "magnitude" ]]; then TAG="capsweep_mag_p50_80"; else TAG="capsweep_p50_80"; fi
STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_${TAG}"
LOG_DIR="results/${STEM}/logs"
TASKS=(ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
       multidlydmintseq dm1seqr dm2seql dmsintseq)
SPARSITIES=(p50 p60 p70 p80)
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}" \
       NUMEXPR_NUM_THREADS="${THREADS}"
mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/*.log
"${PYTHON}" scripts/generate_modcog_revised8_capsweep_suite.py --score "${SCORE}" || exit 1
UNITS=(); for t in "${TASKS[@]}"; do for p in "${SPARSITIES[@]}"; do UNITS+=("${t}_${p}"); done; done
echo "capsweep[${SCORE}]: ${#UNITS[@]} units, ${MAX_JOBS} workers, started $(date '+%H:%M:%S')"
start=$(date +%s)
declare -a PIDS=() PIDUNIT=()
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
echo "capsweep done in $(( ($(date +%s)-start)/60 ))m; units failed: ${failed}/${#UNITS[@]}"
exit $(( failed > 0 ))
