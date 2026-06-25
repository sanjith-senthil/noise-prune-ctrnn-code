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
