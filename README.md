# Mod-Cog CTRNN Pruning Code

This repository is the curated public code release for the paper's CTRNN
task-preservation pruning experiments on Mod-Cog tasks.

## Scope

Included pruning methods:
- L-NP mask and L-NP rescale
- S-NP mask and S-NP rescale
- capped L-NP/S-NP rescale
- magnitude pruning
- random pruning
- OBS-compensated pruning

Included experiment scope:
- 8 Mod-Cog tasks
- 3 trained-network seeds per task
- CTRNN hidden size 512
- tanh activation, dt = 10 ms, tau = 100 ms
- no recurrent self-connections
- 12,000 training steps with sequence-accuracy checkpoint selection
- pruning levels 50%, 60%, 70%, and 80%

Excluded from this curated release:
- OP-NP
- trajectory-preservation experiments
- JSE analyses
- scaling-claim analyses
- alternate RNN architectures
- exploratory pruning baselines not used in the manuscript

The original full working repository remains recoverable through the archive tag
created before cleanup. This release is intended to be the clean public code
surface.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The release vendors the Mod-Cog task definitions and a minimal NeuroGym
compatibility subset. NeuroGym is included only because the Mod-Cog tasks are
implemented as NeuroGym-style `TrialEnv` classes; generic NeuroGym benchmark
tasks are not part of this release.

## Main Commands

Generate/run the main task-preservation suite:

```bash
python3 scripts/generate_tanh_h512_modcog_revised8_revised_task_only_suite.py
python3 -m pruning_benchmark --mode suite --config configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_revised_task_only_p50_80.json
```

Generate/run the three-seed capped-rescale quantile completion:

```bash
python3 scripts/generate_tanh_h512_modcog_revised8_lnp_snp_capped_rescale_missing_quantiles_pruneseeds1_2_suite.py
python3 -m pruning_benchmark --mode suite --config configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_q90_pruneseeds1_2_full8_p50_80.json
```

Validate a colocated official artifact directory:

```bash
python3 scripts/validate_revised_paper_artifacts.py
```

The data release should contain the frozen CSVs, checkpoints, fixed evaluation
batches, and manifest referenced by the manuscript.
