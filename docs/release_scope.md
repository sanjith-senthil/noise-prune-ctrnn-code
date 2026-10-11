# Release Scope

This code release is scoped to the final manuscript analyses:

- CTRNN training on the revised 8-task Mod-Cog battery.
- Main task-preservation pruning suites at 10% to 80% sparsity, in steps of 10,
  at CTRNN hidden sizes 512 and 1024.
- noise-prune mask and sample-and-rescale, the capped rescale on both the
  noise and magnitude scores, magnitude sample-and-rescale, the deterministic
  single-factor rescales, the shuffled and uniform probability controls, and
  capped L-NP/S-NP
  rescale, random pruning, magnitude pruning, and OBS-compensated pruning.
- Capped-rescale quantile suites and analysis scripts.
- S-NP robustness checks cited in the manuscript.
- Statistical scripts for the task-retention and capped-rescale comparison
  families.

Excluded legacy work:

- OP-NP.
- trajectory-preservation experiments.
- JSE analyses.
- scaling-claim analyses.
- alternate RNN architectures.
- exploratory pruning baselines not used in the manuscript.

The full internal work history remains available from the archived pre-cleanup
repository tag. This public release is intentionally smaller and paper-facing.

## Naming and Configuration Notes

- The matched-diagonal flag recorded in some released rescale configurations is
  nullified by the no-self-connection constraint and is inactive in all reported
  runs.
- Some legacy code paths and configuration names label linearized noise-prune as
  `V-NP`. In this release, `V-NP` is identical to `L-NP` as defined in the
  manuscript.
