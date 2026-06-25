# Release Scope

This code release is scoped to the final manuscript analyses:

- CTRNN training on the revised 8-task Mod-Cog battery.
- Main task-preservation pruning suites at 50%, 60%, 70%, and 80% sparsity.
- L-NP mask, L-NP rescale, S-NP mask, S-NP rescale, capped L-NP/S-NP
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
