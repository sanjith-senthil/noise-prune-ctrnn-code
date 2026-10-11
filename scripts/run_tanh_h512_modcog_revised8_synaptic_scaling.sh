#!/usr/bin/env bash
# Gain-restoration rules applied to the noise-prune mask: synaptic scaling
# (row-wise, postsynaptic), its presynaptic mirror (column-wise), and one global
# factor. Resumable; arms already on disk are skipped by run_id.
set -uo pipefail
PYTHON="${1:-python3}"; MAX_JOBS="${2:-10}"; THREADS="${3:-1}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}" VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}" TORCH_NUM_THREADS="${THREADS}"
STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_synaptic_p50_80"
CFG="configs/${STEM}"; LOG="results/${STEM}/logs"; mkdir -p "${LOG}"
"${PYTHON}" scripts/generate_tanh_h512_modcog_revised8_synaptic_scaling_suite.py \
  --shard-by-sparsity || exit 1
pids=(); names=()
for f in "${CFG}"/*.json; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 5; done
  b=$(basename "$f" .json)
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "$f" > "${LOG}/${b}.log" 2>&1 &
  pids+=("$!"); names+=("$b")
done
bad=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED ${names[$i]}" >&2; bad=1; }; done
(( bad == 0 )) && echo "synaptic suite complete" || echo "re-run to resume" >&2
