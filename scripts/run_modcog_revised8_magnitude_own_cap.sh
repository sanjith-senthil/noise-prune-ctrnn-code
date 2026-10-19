#!/usr/bin/env bash
# The magnitude arm at ITS OWN constant cap, so the two scores are compared on
# equal footing.
#
# Figure 5b caps both scores at 3.695. That number is the mean over 50-80% of
# each network's median candidate-edge 1/p, measured on the NOISE score. Applying
# the identical construction to the MAGNITUDE score gives 3.584. Comparing
# noise-prune at 3.695 against magnitude at 3.584 holds the *rule* fixed and lets
# each score set its own constant, which is the matched comparison; comparing
# both at 3.695 instead holds the *number* fixed and asks a different question.
#
# Arm-set "capped" runs both cap-dependent arms, so this pass also produces
# noise-prune at 3.584. That is deliberate: with the existing 3.695 rows it
# completes a 2x2 over {score} x {cap}, which separates the effect of the score
# from the effect of the constant instead of confounding them.
#
# ITS OWN STEM. The harness resumes on run_id alone and the run_id carries no cap
# value, so writing this into the "recap" stem -- which already holds the 3.695
# rows -- would skip all 1,152 runs while rewriting the configs to advertise a cap
# the rows were never produced with.
#
# Usage:
#   scripts/run_modcog_revised8_magnitude_own_cap.sh PYTHON [CAP] [JOBS] [THREADS]
set -uo pipefail
if [[ "${MAGCAP_CAFFEINATED:-0}" != "1" ]]; then
  export MAGCAP_CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"
fi
PYTHON="${1:?python}"; CAP="${2:-3.584}"; MAX_JOBS="${3:-8}"; THREADS="${4:-1}"
# The stem carries the cap, because the harness resumes on run_id alone and the
# run_id does not. Two caps sharing a stem means the second pass skips every run
# while rewriting the configs to advertise a cap the rows never used.
TAG="${5:-magcap}"
HIDDEN=512; ARMSET="capped"
TASKS=(ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
       multidlydmintseq dm1seqr dm2seql dmsintseq)
SPARSITIES=(p10 p20 p30 p40 p50 p60 p70 p80)
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}" \
       NUMEXPR_NUM_THREADS="${THREADS}"
STEM="task_preservation_tanh_h${HIDDEN}_modcog_revised8_12k_seqbest_renorm_${TAG}_p10_80"
LOG_DIR="results/${STEM}/logs"; mkdir -p "${LOG_DIR}"; rm -f "${LOG_DIR}"/*.log
echo "=== H=${HIDDEN} arm-set=${ARMSET} cap=${CAP} stem=${STEM} ==="
"${PYTHON}" scripts/generate_modcog_revised8_renorm_suite.py \
    --hidden-size "${HIDDEN}" --arm-set "${ARMSET}" --cap-value "${CAP}" \
    --stem-tag "${TAG}" || exit 1
UNITS=(); for t in "${TASKS[@]}"; do for p in "${SPARSITIES[@]}"; do UNITS+=("${t}_${p}"); done; done
echo "  ${#UNITS[@]} units, ${MAX_JOBS} workers, started $(date '+%H:%M:%S')"
t0=$(date +%s); declare -a PIDS=() PIDUNIT=()
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
echo "done in $(( ($(date +%s)-t0)/60 ))m; units failed: ${failed}/${#UNITS[@]}"
exit $(( failed > 0 ))
