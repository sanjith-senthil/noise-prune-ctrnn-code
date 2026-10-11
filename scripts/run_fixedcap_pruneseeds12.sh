#!/usr/bin/env bash
# Pruning seeds 1-2 for the constant-factor capped arms at H=512, so the arm
# carries the same three draws as every other stochastic arm in the figures.
# Resumable: keys on run_id, appends per run.
set -uo pipefail
PYTHON="${1:-python3}"; MAX_JOBS="${2:-12}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fixedcapfull_p50_80"
CFG="configs/${STEM}_pruneseeds12"; LOG="results/${STEM}/logs"; mkdir -p "${LOG}"
"${PYTHON}" scripts/generate_modcog_revised8_fixedcap_pruneseeds_suite.py || exit 1
pids=(); names=()
for f in "${CFG}"/*.json; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 5; done
  b=$(basename "$f" .json)
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "$f" > "${LOG}/${b}.log" 2>&1 &
  pids+=("$!"); names+=("$b")
done
bad=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED ${names[$i]}" >&2; bad=1; }; done
(( bad == 0 )) && echo "pruning seeds 1-2 complete" || echo "re-run to resume" >&2
