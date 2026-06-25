#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_tanh_h512_modcog_revised8_lnp_snp_q50_capped_rescale_3seed_suite.py --validate-inputs
MPLCONFIGDIR=/tmp/mpl python3 -m pruning_benchmark --mode suite \
  --config configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80.json
python3 scripts/summarize_tanh_h512_modcog_revised8_lnp_snp_q50_capped_rescale_3seed.py
