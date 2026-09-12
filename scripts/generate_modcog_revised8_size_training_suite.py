#!/usr/bin/env python3
"""Train the Mod-Cog eight at a different hidden size, under the paper's exact protocol.

Reviewer 1, comment 4 asks for a second network size, noting that H = 512 sits far
outside the regime where the original spectral guarantee holds and suggesting
H = 128 and H = 1024 as a reasonable pair. This trains a matched set at the
requested size so the pruning comparisons can be repeated without any change of
protocol -- the point being to add a size axis, not a second experimental design.

Why the protocol transfers unchanged
------------------------------------
The recurrent matrix is initialised with ``kaiming_uniform_`` and
``recurrent_init_std=None``, so entries have std ~ sqrt(2/H) and the initial
spectral radius rho ~ std*sqrt(H) is **H-invariant**. Measured at seed 0:
rho = 1.475 (H=128), 1.443 (H=512), 1.452 (H=1024). A network at a new size
therefore starts in the same dynamical regime, and learning rate, step count,
batch size and gradient clipping need no retuning. That is what makes the size
comparison clean rather than a confound between size and hyperparameters.

Protocol (identical to the paper's 24 networks)
-----------------------------------------------
Phase 1  6,000 steps at lr 1.2e-3
Phase 2  6,000 further steps at lr 6e-4, resuming from the phase-1 checkpoint
Both     batch 256, clip 1.0, no L2, validation every 500 steps on 64 batches at
         eval_seed 0, checkpoint selected by max validation sequence accuracy with
         cross-entropy as tie-break. Per-task ng_T follows the paper: 40 for the
         five ctx/dly/multi tasks, 30 for dm1seqr and dm2seql, 38 for dmsintseq.

Layout
------
One config per (task, network seed) holding **both** phases in order, so a single
worker carries one network from scratch to 12,000 steps and the 24 units are
independently parallel and independently resumable. Per-run keys override the
defaults block, which is how the two phases carry different lr and checkpoint
paths inside one config.

Safety
------
``guard_paths`` refuses to emit any config whose checkpoint paths could collide
with the paper's H=512 networks. The paper's 24 checkpoints are the source of
truth for every frozen number in the manuscript and must not be writable from
here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = (
    ("ctxdlydm2intseq", "modcog:ctxdlydm2intseq", 40),
    ("ctxdlydm1intseq", "modcog:ctxdlydm1intseq", 40),
    ("dlydm1intseq", "modcog:dlydm1intseq", 40),
    ("dlydm2intseq", "modcog:dlydm2intseq", 40),
    ("multidlydmintseq", "modcog:multidlydmintseq", 40),
    ("dm1seqr", "modcog:dm1seqr", 30),
    ("dm2seql", "modcog:dm2seql", 30),
    ("dmsintseq", "modcog:dmsintseq", 38),
)
NETWORK_SEEDS = (0, 1, 2)
PHASE1_STEPS, PHASE1_LR = 6000, 0.0012
PHASE2_STEPS, PHASE2_LR = 6000, 0.0006
PROTECTED_SUBSTRINGS = ("h512",)


def phase1_dir(h: int, seed: int) -> str:
    return f"checkpoints/tanh_h{h}_modcog_revised8_6k_seqbest_no_l2_seed{seed}"


def phase2_dir(h: int, seed: int) -> str:
    return f"checkpoints/tanh_h{h}_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{seed}"


def suite_stem(h: int) -> str:
    return f"train_tanh_h{h}_modcog_revised8_to12k_seqbest_no_l2"


def guard_paths(h: int, paths) -> None:
    """Refuse to write anywhere that could collide with the paper's H=512 networks."""
    if h == 512:
        raise SystemExit(
            "Refusing to target H=512: those are the paper's 24 frozen networks. "
            "This script exists to add a *different* size.")
    bad = [p for p in paths if any(s in p for s in PROTECTED_SUBSTRINGS)]
    if bad:
        raise SystemExit(f"Refusing to write protected paths: {sorted(set(bad))}")


def defaults_block(h: int) -> dict:
    """The paper's training defaults, with hidden_size as the only change."""
    return {
        "hidden_size": h,
        "train_steps": PHASE1_STEPS,
        "ft_steps": 0,
        "last_only": False,
        "eval_last_only": False,
        "device": "cpu",
        "movement_batches": 20,
        "model_type": "ctrnn",
        "activation": "tanh",
        "ng_T": 0,
        "ng_B": 256,
        "eval_sample_batches": 64,
        "eval_seed": 0,
        "reset_results": False,
        "resume": True,
        "lr": PHASE1_LR,
        "clip": 1.0,
        "train_progress": True,
        "train_progress_every": 100,
        "train_select_best": True,
        "train_val_interval": 500,
        "train_best_metric": "acc_sequence",
        "train_best_metric_mode": "max",
        "train_best_tie_metric": "loss",
        "train_best_tie_metric_mode": "min",
        "no_self_connections": True,
        "use_dale": False,
    }


def make_unit(h: int, task_label: str, task: str, ng_t: int, seed: int) -> dict:
    ck1 = f"{phase1_dir(h, seed)}/modcog_{task_label}_seed{seed}.pt"
    ck2 = f"{phase2_dir(h, seed)}/modcog_{task_label}_seed{seed}.pt"
    guard_paths(h, [ck1, ck2])
    common = {
        "strategy": "none", "amount": 0.0, "no_prune": True,
        "seed": seed, "task": task, "ng_T": ng_t, "recurrent_l2_lambda": 0.0,
    }
    return {
        "run_id": f"{suite_stem(h)}_{task_label}_seed{seed}",
        "output_csv": f"results/{suite_stem(h)}/{task_label}_seed{seed}.csv",
        "defaults": defaults_block(h),
        "runs": [
            {   # phase 1 -- from scratch
                "run_id": f"train_modcog_{task_label}_tanh_h{h}_6k_seqbest_no_l2_seed{seed}",
                "train_steps": PHASE1_STEPS, "lr": PHASE1_LR,
                "save_model_path": ck1, **common,
            },
            {   # phase 2 -- resume and anneal
                "run_id": (f"continue_modcog_{task_label}_tanh_h{h}_to12k_lr0006"
                           f"_seqbest_no_l2_seed{seed}"),
                "train_steps": PHASE2_STEPS, "lr": PHASE2_LR,
                "load_model_path": ck1, "save_model_path": ck2, **common,
            },
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    h = args.hidden_size

    cfg_dir = Path(f"configs/{suite_stem(h)}")
    out_dir = Path(f"results/{suite_stem(h)}")
    units = [(t, task, ng, s) for s in NETWORK_SEEDS for t, task, ng in TASKS]
    if args.dry_run:
        print(f"would write {len(units)} configs to {cfg_dir}")
        for t, _task, _ng, s in units[:3]:
            print(f"  {cfg_dir}/{suite_stem(h)}_{t}_seed{s}.json")
        print("  ...")
        return

    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, task, ng, s in units:
        u = make_unit(h, t, task, ng, s)
        (cfg_dir / f"{suite_stem(h)}_{t}_seed{s}.json").write_text(json.dumps(u, indent=2) + "\n")
    print(f"wrote {len(units)} configs to {cfg_dir}")
    print(f"  H = {h}; {len(TASKS)} tasks x {len(NETWORK_SEEDS)} seeds = {len(units)} networks")
    print(f"  each unit: {PHASE1_STEPS} steps @ lr {PHASE1_LR} -> "
          f"{PHASE2_STEPS} steps @ lr {PHASE2_LR} (12,000 total)")
    print(f"  phase-1 checkpoints -> {phase1_dir(h, 0)}/ (and seeds 1,2)")
    print(f"  phase-2 checkpoints -> {phase2_dir(h, 0)}/ (and seeds 1,2)")
    print("  protocol identical to the paper's H=512 set apart from hidden_size")


if __name__ == "__main__":
    main()
