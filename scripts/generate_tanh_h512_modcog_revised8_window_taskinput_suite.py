#!/usr/bin/env python3
"""The 2x2: covariance window (end-step vs whole trial) x drive (task+noise vs noise only).

Self-contained at one pruning seed so all four cells are paired within trained
network rather than joined across suites. The end-step pair reproduces the
`noise_only` suite's setting (which has it at three pruning seeds) and so doubles
as a regression check.

The whole-trial arms use conditional centering. Trajectory-mean centering over a
whole trial folds the time-varying mean of the trajectory into the covariance,
which is a different object rather than a longer-window version of the same one;
conditional centering subtracts the matched noise-free trajectory and leaves the
noise-induced deviation, which is what the end-step arm is also measuring. They
also use a fixed 98-rollout budget (the published effective count) so the two
windows cost the same simulation, rather than the sample cap giving a whole-trial
window only 3 rollouts.

4 arms x 8 tasks x 3 network seeds x 3 sparsities = 288 runs, n = 24.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_windowdrive_p50_70"
CONFIG_DIR = Path("configs/window_drive")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7)
PRUNING_SEED = 0
ROLLOUTS = 98
STRAT = "simulation_noise_prune_rescale"

# label, burn_in, centering, zero_task_input, num_rollouts
ARMS = (
    ("endstep_taskplusnoise",   300, "trajectory_mean", False, None),
    ("endstep_noiseonly",       300, "trajectory_mean", True,  None),
    ("wholetrial_taskplusnoise",  0, "conditional",     False, ROLLOUTS),
    ("wholetrial_noiseonly",      0, "conditional",     True,  ROLLOUTS),
)


def shard(a): return "" if a is None else f"_p{int(a*100)}"
def cfg(t, a=None): return CONFIG_DIR / f"{SUITE_STEM}_{t}{shard(a)}.json"
def out(t, a=None): return OUTPUT_DIR / f"{t}{shard(a)}.csv"


def make(task, ng_t, amounts=AMOUNTS, baseline=True, sh=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = _rs._common(task, ng_t, seed)
        if baseline:
            runs.append({"run_id": f"wd_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for (label, burn, cen, zero, nroll) in ARMS:
            for amount in amounts:
                r = {"run_id": f"wd_{task}_netseed{seed}_{label}_p{int(amount*100)}_pruneseed{PRUNING_SEED}",
                     "strategy": STRAT, "amount": amount,
                     "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                     "noise_rng_seed": PRUNING_SEED,
                     "sim_np_burn_in_steps": burn, "sim_np_centering": cen,
                     "sim_np_zero_task_input": zero,
                     "score_batch_seed": 100_000 + PRUNING_SEED,
                     "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED), **common}
                if nroll is not None: r["sim_np_num_rollouts"] = nroll
                runs.append(r)
    return {"run_id": f"{SUITE_STEM}_{task}{shard(sh)}", "output_csv": str(out(task, sh)),
            "defaults": _rs.defaults_block(), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--shard-by-sparsity", action="store_true")
    a = ap.parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tot = n = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in AMOUNTS] if a.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            sh = amounts[0] if a.shard_by_sparsity else None
            s = make(task, ng_t, amounts, baseline=(i == 0), sh=sh)
            cfg(task, sh).write_text(json.dumps(s, indent=2) + "\n"); tot += len(s["runs"]); n += 1
    print(f"wrote {n} configs; total runs: {tot}  ({len(ARMS)} arms)")


if __name__ == "__main__":
    main()
