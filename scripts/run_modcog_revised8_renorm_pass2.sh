#!/usr/bin/env bash
# Second pass of the renormalisation re-run.
#
#   H = 1024, arm-set "rest": the three sampling arms the first pass left out
#   (capped rescale, magnitude rescale, capped magnitude rescale), so the size
#   figure can show every arm on one normalisation instead of dropping three.
#
#   H = 512, optionally: the two capped arms again, if the recalibrated constant
#   differs from the one the first pass used. Pass "512 1024" to do both.
#
# The cap constant is supplied on the command line rather than baked in, because
# it is recalibrated from the q50 of the corrected 1/p distribution and the whole
# point of this pass is that the number may have moved.
#
# Usage:
#   scripts/run_modcog_revised8_renorm_pass2.sh PYTHON CAP [JOBS] [THREADS] [SIZES]
set -uo pipefail
if [[ "${PASS2_CAFFEINATED:-0}" != "1" ]]; then
  export PASS2_CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"
fi
PYTHON="${1:?python}"; CAP="${2:?cap value}"; MAX_JOBS="${3:-8}"
THREADS="${4:-1}"; SIZES="${5:-1024}"
TASKS=(ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
       multidlydmintseq dm1seqr dm2seql dmsintseq)
SPARSITIES=(p10 p20 p30 p40 p50 p60 p70 p80)
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}" \
       NUMEXPR_NUM_THREADS="${THREADS}"
overall=0; t0=$(date +%s)
for HIDDEN in ${SIZES}; do
  # Every pass after the first needs its OWN stem. The harness resumes on
  # run_id alone and the run_id carries no cap value, so reusing the primary
  # stem would skip all 1,152 runs and leave the configs advertising a cap the
  # rows were never produced with. H=512 here means "re-run the capped arms at
  # a new cap", which is a different stem, not the primary one.
  if [[ "${HIDDEN}" == "1024" ]]; then ARMSET="rest"; TAG="rest"; else ARMSET="first"; TAG="recap"; fi
  STEM="task_preservation_tanh_h${HIDDEN}_modcog_revised8_12k_seqbest_renorm_${TAG}_p10_80"
  LOG_DIR="results/${STEM}/logs"; mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/*.log
  echo "=== H=${HIDDEN} arm-set=${ARMSET} cap=${CAP} ==="
  "${PYTHON}" scripts/generate_modcog_revised8_renorm_suite.py \
      --hidden-size "${HIDDEN}" --arm-set "${ARMSET}" --cap-value "${CAP}" \
      --stem-tag "${TAG}" || exit 1
  UNITS=(); for t in "${TASKS[@]}"; do for p in "${SPARSITIES[@]}"; do UNITS+=("${t}_${p}"); done; done
  echo "  ${#UNITS[@]} units, ${MAX_JOBS} workers, started $(date '+%H:%M:%S')"
  s=$(date +%s); declare -a PIDS=() PIDUNIT=()
  for unit in "${UNITS[@]}"; do
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 10; done
    "${PYTHON}" -u -m pruning_benchmark --mode suite \
        --config "configs/${STEM}/${STEM}_${unit}.json" > "${LOG_DIR}/${unit}.log" 2>&1 &
    PIDS+=("$!"); PIDUNIT+=("${unit}")
  done
  failed=0
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "  FAILED ${PIDUNIT[$i]}" >&2; failed=$((failed+1)); overall=1; }
  done
  unset PIDS PIDUNIT
  echo "  H=${HIDDEN} done in $(( ($(date +%s)-s)/60 ))m; units failed: ${failed}/${#UNITS[@]}"
done
echo "pass 2 complete in $(( ($(date +%s)-t0)/60 ))m"
exit "${overall}"
