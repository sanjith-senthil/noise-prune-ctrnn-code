# Mod-Cog Default Sequence Lengths

For Mod-Cog tasks, the pipeline now derives `T` from the environment timing
so the full trial (including the decision period) is covered by default.

Policy:
- If `ng_T` is provided, it is used.
- Otherwise, compute `T = ceil(sum(timing_ms) / dt) + safety_steps`.
- Timing callables (e.g., random fixation) are sampled a few times and the max is used.

This prevents truncating trials (e.g., running `T=600` when the task timing is ~1900 ms),
which can make last-step evaluation appear feedforward-solvable.

To see the current defaults for all tasks, run:
```bash
python3 scripts/assess_modcog_laststep.py --tasks all --max_tasks 1
```

The computed `T` values are recorded in run metadata via `task_meta`.

## Target Masking and Loss Averaging

Mod-Cog labels use class `0` for fixation. The training and evaluation
pipeline remaps fixation labels from `0` to `-1`, so the network is neither
trained nor scored on fixation periods. Cross-entropy loss uses
`ignore_index=-1`.

When sequence labels are used, logits and targets are flattened over the full
`(timestep, batch element)` grid before loss and sequence-accuracy computation.
The loss is therefore averaged over each remaining valid prediction entry
individually. This gives every scored prediction equal weight; averaging first
within each timestep would overweight timesteps that contain only a few valid
entries after fixation masking.
