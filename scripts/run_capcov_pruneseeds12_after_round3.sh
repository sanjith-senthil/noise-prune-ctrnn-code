#!/usr/bin/env bash
# Adds pruning seeds 1 and 2 to the H=1024 capped-covariance magnitude arms.
# Waits for any running suite to finish first so the two never compete.
#
# Resumable: the harness keys on run_id and appends per run, so re-running this
# picks up wherever it stopped and never repeats finished work.
set -uo pipefail
PYTHON="${1:-python3}"
MAX_JOBS="${2:-12}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1

echo "waiting for the current suites to finish..."
while pgrep -f "run_revision_round3_all.sh" > /dev/null; do sleep 60; done
while pgrep -f "pruning_benchmark --mode suite" > /dev/null; do sleep 30; done
echo "clear at $(date +%H:%M:%S); starting"

STEM="task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_capcov_p50_80"
CFG="configs/${STEM}_pruneseeds12"
LOG="results/${STEM}/logs"
mkdir -p "${LOG}"
"${PYTHON}" scripts/generate_modcog_revised8_capcov_pruneseeds_suite.py || exit 1

pids=(); names=()
for f in "${CFG}"/*.json; do
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 10; done
  b=$(basename "$f" .json)
  "${PYTHON}" -u -m pruning_benchmark --mode suite --config "$f" > "${LOG}/${b}.log" 2>&1 &
  pids+=("$!"); names+=("$b")
done
bad=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "  FAILED ${names[$i]}" >&2; bad=1; }
done
(( bad == 0 )) && echo "pruning seeds 1-2 complete" || echo "some units failed; re-run to resume" >&2
