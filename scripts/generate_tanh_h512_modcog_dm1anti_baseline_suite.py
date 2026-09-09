#!/usr/bin/env python3
"""Generate training suites for the two excluded tasks, dm1 and anti.

Reviewer 1 (minor point 1) notes that ll. 92-95 and 259-261 exclude dm1 and
anti as "uninformative for pruning comparisons" -- because they "either do not
require recurrence or become trivially solvable as long as networks are not too
small" -- without supporting data.  This suite trains those two tasks under the
*exact* protocol used for the paper's eight, so their unpruned accuracy can be
reported and the exclusion documented rather than asserted.

Protocol (identical to the revised-8 suites):
    H = 512, tanh, no self-connections, Adam, batch 256, grad-clip 1.0,
    no recurrent L2, two phases of 6,000 steps at lr 1.2e-3 then 6e-4,
    validation every 500 steps on 64 batches (eval_seed 0), checkpoint selected
    by max validation sequence accuracy with CE loss as tie-break.

Sequence length follows the convention the paper's own tasks use: T is the
environment timing rounded up to whole steps, plus a 10-step safety margin.
Verified against the released tasks -- dm1seqr and dm2seql derive to 20 and use
ng_T = 30; dmsintseq derives to 28 and uses 38; the intseq tasks derive to 30
and use 40.  Applying the same rule:
    dm1  : 1200 ms / 100 ms dt = 12 -> ng_T = 22
    anti : 2000 ms / 100 ms dt = 20 -> ng_T = 30

Checkpoints and result CSVs use a `modcog_dm1anti` prefix, deliberately
distinct from `modcog_revised8`, so nothing belonging to the paper's 24
networks can be overwritten.

WHAT THIS DOES AND DOES NOT SHOW: unpruned accuracy establishes that these
tasks are learned to (near) ceiling, supporting "trivially solvable".  It does
*not* by itself establish that pruning comparisons on them are uninformative --
that would need a pruning suite on these checkpoints, which this script does not
generate.  Scope any claim accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIG_ROOT = Path("configs")
RESULT_ROOT = Path("results")

# (task label, modcog id, ng_T)  -- see module docstring for the derivation
TASKS = (
    ("dm1", "modcog:dm1", 22),
    ("anti", "modcog:anti", 30),
)
SEEDS = (0, 1, 2)

PHASE1_STEPS = 6000
PHASE1_LR = 0.0012
PHASE2_STEPS = 6000
PHASE2_LR = 0.0006

PHASE1_STEM = "train_tanh_h512_modcog_dm1anti_6k_seqbest_no_l2"
PHASE2_STEM = "continue_tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2"
PHASE1_CKPT_DIR = "checkpoints/tanh_h512_modcog_dm1anti_6k_seqbest_no_l2_seed{seed}"
PHASE2_CKPT_DIR = "checkpoints/tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2_seed{seed}"


def defaults_block(steps: int, lr: float) -> dict:
    return {
        "hidden_size": 512,
        "train_steps": steps,
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
        "lr": lr,
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


def phase1_suite(seed: int) -> dict:
    ckpt = PHASE1_CKPT_DIR.format(seed=seed)
    runs = [{
        "run_id": f"train_modcog_{label}_tanh_h512_6k_seqbest_no_l2_seed{seed}",
        "strategy": "none", "amount": 0.0, "no_prune": True, "seed": seed,
        "task": task, "ng_T": ng_t, "recurrent_l2_lambda": 0.0,
        "save_model_path": f"{ckpt}/modcog_{label}_seed{seed}.pt",
    } for label, task, ng_t in TASKS]
    return {"run_id": f"{PHASE1_STEM}_seed{seed}",
            "output_csv": str(RESULT_ROOT / f"{PHASE1_STEM}_seed{seed}.csv"),
            "defaults": defaults_block(PHASE1_STEPS, PHASE1_LR), "runs": runs}


def phase2_suite(seed: int) -> dict:
    src = PHASE1_CKPT_DIR.format(seed=seed)
    dst = PHASE2_CKPT_DIR.format(seed=seed)
    runs = [{
        "run_id": f"continue_modcog_{label}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{seed}",
        "strategy": "none", "amount": 0.0, "no_prune": True, "seed": seed,
        "task": task, "ng_T": ng_t, "recurrent_l2_lambda": 0.0,
        "load_model_path": f"{src}/modcog_{label}_seed{seed}.pt",
        "save_model_path": f"{dst}/modcog_{label}_seed{seed}.pt",
    } for label, task, ng_t in TASKS]
    return {"run_id": f"{PHASE2_STEM}_seed{seed}",
            "output_csv": str(RESULT_ROOT / f"{PHASE2_STEM}_seed{seed}.csv"),
            "defaults": defaults_block(PHASE2_STEPS, PHASE2_LR), "runs": runs}


def guard_against_clobber() -> None:
    """Refuse to run if any target path could belong to the paper's networks."""
    bad = []
    for seed in SEEDS:
        for d in (PHASE1_CKPT_DIR.format(seed=seed), PHASE2_CKPT_DIR.format(seed=seed)):
            if "revised8" in d:
                bad.append(d)
    for stem in (PHASE1_STEM, PHASE2_STEM):
        if "revised8" in stem:
            bad.append(stem)
    if bad:
        raise SystemExit(f"refusing to generate: paths collide with the paper's suites: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan without writing configs")
    args = ap.parse_args()
    guard_against_clobber()

    total = 0
    for seed in SEEDS:
        for suite in (phase1_suite(seed), phase2_suite(seed)):
            path = CONFIG_ROOT / f"{suite['run_id']}.json"
            total += len(suite["runs"])
            if args.dry_run:
                print(f"  would write {path}  ({len(suite['runs'])} runs)")
                continue
            CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(suite, indent=2) + "\n")
            print(f"  wrote {path}  ({len(suite['runs'])} runs)")
    print(f"\ntasks: {[t[0] for t in TASKS]}  seeds: {list(SEEDS)}")
    print(f"total training runs: {total} "
          f"({len(TASKS)} tasks x {len(SEEDS)} seeds x 2 phases)")
    print(f"checkpoints -> {PHASE2_CKPT_DIR.format(seed='{0,1,2}')}")


if __name__ == "__main__":
    main()
