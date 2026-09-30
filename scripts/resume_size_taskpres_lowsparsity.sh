#!/usr/bin/env bash
# Resume the H=1024 10-40% task-preservation suite after a sleep, crash or reboot.
#
# Safe to run at any time, including while the suite is already running: each
# unit resumes per run_id from its own CSV, so completed runs are skipped and
# nothing is recomputed or duplicated. If the suite is already running this
# just reports and exits rather than starting a second copy.
#
#   scripts/resume_size_taskpres_lowsparsity.sh            # resume
#   scripts/resume_size_taskpres_lowsparsity.sh --status   # report only
#
# Run it from the release_code_staging directory.

set -uo pipefail

HIDDEN=1024
TOTAL=1464
STEM="task_preservation_tanh_h${HIDDEN}_modcog_revised8_12k_seqbest_revised_task_only_p10_40"
DIR="results/${STEM}"
PYTHON="${PYTHON:-/Users/sanjithsenthil/Efficient RNN Project/.venv/bin/python}"

done_rows() {
  cat "${DIR}"/*.csv 2>/dev/null | grep -c "taskpreslow" || echo 0
}
error_rows() {
  "${PYTHON}" - "$DIR" <<'PY' 2>/dev/null || echo 0
import csv, glob, sys
n = 0
for f in glob.glob(f"{sys.argv[1]}/*.csv"):
    try:
        n += sum(1 for r in csv.DictReader(open(f)) if str(r.get("error") or "").strip())
    except Exception:
        pass
print(n)
PY
}

N=$(done_rows); E=$(error_rows)
echo "suite ${STEM}"
echo "  completed runs : ${N} / ${TOTAL}"
echo "  error rows     : ${E}  (these are retried automatically on resume)"

RUNNING=$(pgrep -f "pruning_benchmark --mode suite" | wc -l | tr -d ' ')
echo "  live workers   : ${RUNNING}"

if [[ "${1:-}" == "--status" ]]; then
  exit 0
fi

if (( RUNNING > 0 )); then
  echo
  echo "Already running -- nothing to do. Re-run with --status to check progress."
  exit 0
fi

if (( N >= TOTAL )) && (( E == 0 )); then
  echo
  echo "Suite is complete."
  exit 0
fi

echo
echo "Resuming under caffeinate (prevents idle, disk and system sleep while it runs)."
exec caffeinate -dimsu bash scripts/run_modcog_revised8_size_taskpres_lowsparsity.sh \
  "${PYTHON}" "${HIDDEN}" 8 1
