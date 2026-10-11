# Mod-Cog CTRNN Pruning Code

This repository is the curated public code release for the paper's CTRNN
task-preservation pruning experiments on Mod-Cog tasks.

## The companion data deposit

This repository is the code only. The trained networks and the result tables are
in the data deposit:

  https://github.com/sanjith-senthil/noise-prune-ctrnn-data

**Every config here loads checkpoints by a path relative to the working
directory**, for example
`checkpoints/tanh_h512_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed0/modcog_dm1seqr_seed0.pt`.
So before running anything, place or symlink the data deposit's `checkpoints/`
directory at the root of this repository:

```bash
git clone https://github.com/sanjith-senthil/noise-prune-ctrnn-data
ln -s ../noise-prune-ctrnn-data/checkpoints checkpoints
```

A few summarising scripts also look for `../data_release_staging/...` as a
sibling directory; clone the two repositories side by side if you need those.

The fixed evaluation and pruning-score batches are NOT in either deposit. They
do not need to be: the runner builds and caches them deterministically from the
seeds recorded in every config, so the first run of a suite regenerates exactly
the batches the published results used.

Licensed under the MIT License; see LICENSE. The data deposit is CC BY 4.0.

## Scope

Included pruning methods:
- noise-prune mask (the noise score with no rescaling)
- noise-prune sample-and-rescale
- noise-prune capped rescale, and the same cap on the magnitude score
- magnitude sample-and-rescale, and deterministic single-factor rescales
- magnitude pruning
- random pruning
- OBS-compensated pruning
- probability controls: shuffled and uniform retention probabilities

The Lyapunov-based variant (`noise_prune*`, called L-NP or V-NP in older code
and CSVs) is still implemented and runnable, but it is not reported in the
revised manuscript.

Included experiment scope:
- 8 Mod-Cog tasks
- 3 trained-network seeds per task
- CTRNN hidden sizes 512 and 1024
- tanh activation, dt = 10 ms, tau = 100 ms, so the leak alpha = dt/tau = 0.1
- no recurrent self-connections
- 12,000 training steps with sequence-accuracy checkpoint selection
- pruning levels 10% through 80% in steps of 10

Excluded from this curated release:
- OP-NP
- trajectory-preservation experiments
- JSE analyses
- alternate RNN architectures

## The probability normalisation (read this before reproducing anything)

The sampling rule chooses a retention probability per connection. Until
2026-10-09 it scaled its unnormalised scores by `K = M / sum(q)`, where `M` is
the requested number of retained connections, and then clipped to 1. Clipping
discards the probability mass pushed above 1, so the network kept fewer
connections than requested: at a requested 10% sparsity on H=512 it removed
33% of the recurrent weights. The shortfall falls monotonically with sparsity
and is 0.7% at 80%.

`prob_normalize="bisect"`, now the default everywhere, instead solves for the
single `K` with `sum(min(1, K q)) = M`, so the expected retained count equals
the budget. Measured over 4,032 pruned networks, the realized sparsity is
within 0.04 percentage points of the request at every level, which is at least
as accurate as deterministic top-k.

Set `sim_np_prob_normalize: "clip"` in a config to reproduce results published
before that date. The mode and the solved `K` are recorded on every run.
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

Generate and run the main suite, all sampling arms over 10-80% at both sizes:

```bash
scripts/run_modcog_revised8_renorm.sh .venv/bin/python 8 1 "512 1024"
```

Re-measure the amplification cap, then re-run the cap-dependent arms at the
value it reports:

```bash
scripts/run_modcog_revised8_capq50.sh .venv/bin/python 8 1
scripts/run_modcog_revised8_renorm_pass2.sh .venv/bin/python 3.695 8 1 "512 1024"
```

Any pass after the first needs its own `--stem-tag`: the harness resumes on
`run_id` alone and the `run_id` carries no cap value, so a second pass written
to the first pass's stem would skip every run while rewriting the configs.

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
