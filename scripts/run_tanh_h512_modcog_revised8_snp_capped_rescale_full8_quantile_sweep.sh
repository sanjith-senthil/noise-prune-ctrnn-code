#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_tanh_h512_modcog_revised8_snp_capped_rescale_full8_quantile_sweep.py --validate-inputs
python3 -m pruning_benchmark --mode suite \
  --config configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_snp_capped_rescale_q50_q75_q90_full8_p50_80.json
