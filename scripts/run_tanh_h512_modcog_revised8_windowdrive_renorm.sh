#!/usr/bin/env bash
# The covariance window x drive 2x2, re-run under the corrected probability
# normalisation over 10-80%. The original suite predates the fix and records
# prob_normalize_mode = clip throughout. Resumable; runs already in an output
# CSV are skipped by run_id.
set -uo pipefail
PYTHON="${1:-python3}"; MAX_JOBS="${2:-8}"; THREADS="${3:-1}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}" TORCH_NUM_THREADS="${THREADS}"
STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_windowdrive_renorm_p10_80"
CFG="configs/${STEM}"; LOG="results/${STEM}/logs"; mkdir -p "${LOG}"
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_window_taskinput_suite.py \
  --full-range --shard-by-sparsity || exit 1
pids=(); names=()
for f in "${CFG}"/*.json; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 5; done
  b=$(basename "$f" .json)
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "$f" > "${LOG}/${b}.log" 2>&1 &
  pids+=("$!"); names+=("$b")
done
bad=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED ${names[$i]}" >&2; bad=1; }; done
(( bad == 0 )) && echo "windowdrive renorm suite complete" || echo "re-run to resume" >&2
