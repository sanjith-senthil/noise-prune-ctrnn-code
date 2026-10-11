#!/usr/bin/env bash
# Constant-factor cap at 10-40% sparsity, so Figure 1 spans the paper's full
# range with one fixed factor. Resumable; 50-80% is already on disk and skipped.
set -uo pipefail
PYTHON="${1:-python3}"; MAX_JOBS="${2:-12}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
STEM="task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fixedcapfull_p50_80"
CFG="configs/${STEM}"; LOG="results/${STEM}/logs"; mkdir -p "${LOG}"
TASKS=(ctxdlydm2intseq ctxdlydm1intseq dlydm1intseq dlydm2intseq
       multidlydmintseq dm1seqr dm2seql dmsintseq)
pids=(); names=()
for p in p10 p20 p30 p40; do for t in "${TASKS[@]}"; do
  f="${CFG}/${STEM}_${t}_${p}.json"; [ -f "$f" ] || continue
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 5; done
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "$f" \
    > "${LOG}/${t}_${p}.log" 2>&1 &
  pids+=("$!"); names+=("${t}_${p}")
done; done
bad=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED ${names[$i]}" >&2; bad=1; }; done
(( bad == 0 )) && echo "low-sparsity cap complete" || echo "re-run to resume" >&2
