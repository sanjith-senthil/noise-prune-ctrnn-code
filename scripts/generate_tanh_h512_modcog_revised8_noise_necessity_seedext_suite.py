#!/usr/bin/env python3
"""Extend the noise-necessity ablation to three pruning seeds (rescale variant).

Reviewer 2, comment 4 asks whether the pruning scores reflect task-driven or
noise-induced activity, and specifically for "covariance estimated without added
noise".  The existing ``noise_necessity`` suite answers that -- its
``sigma_factor = 0.001`` arm is task input with the injected noise effectively
off -- but it ran **one** pruning seed per cell rather than the three the
official task-preservation protocol averages.  The analysis unit was still the
trained network (n = 24), so the estimates are unbiased; they are simply noisier
than the paper's, which costs power exactly where the effects are small.

That matters for one claim in particular.  "sigma_nat is an interior optimum"
rests on differences of 0.01-0.06, and under a paired test on the single-seed
data it holds for the mask variant at 60-80% but **not** at 50%, and not at any
sparsity for the rescale variant.  Three seeds per cell is what makes that
claim testable at the standard the rest of the paper uses.

Scope
-----
Rescale variant only (``simulation_noise_prune_rescale``).  The mask variant is
the deterministic ablation; the rescale variant is the method the paper actually
proposes, so it is the one whose noise dependence has to be established at full
standard.  All five sigma factors, all four sparsities, 8 tasks x 3 network
seeds, pruning seeds **1 and 2** -- seed 0 already exists in the original suite
and is merged by the summarizer rather than recomputed.

Regression arm
--------------
``sf1p0`` is additionally re-run at pruning seed **0**, where the original suite
already has it.  It must reproduce those rows exactly; that is the check that
the extension sits on the same footing as the suite it is being merged into.
Without it the merge would be an assumption rather than a verified fact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_seedext_p50_80"
SOURCE_STEM = _rs.NOISE_STEM
CONFIG_DIR = Path("configs/noise_necessity_seedext")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM

TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
AMOUNTS = _rs.NOISE_AMOUNTS
FACTORS = _rs.NOISE_FACTORS
NEW_PRUNING_SEEDS = (1, 2)
REGRESSION_SEED = 0
REGRESSION_FACTOR = 1.0
STRATEGY = "simulation_noise_prune_rescale"
LABEL = "snp_rescale"


def tag(factor: float) -> str:
    return f"sf{str(factor).replace('.', 'p')}"


def shard_suffix(amount) -> str:
    return "" if amount is None else f"_p{int(amount * 100)}"


def config_path(task: str, amount=None) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task}{shard_suffix(amount)}.json"


def output_csv(task: str, amount=None) -> Path:
    return OUTPUT_DIR / f"{task}{shard_suffix(amount)}.csv"


def validate_inputs() -> None:
    missing = []
    for seed in NETWORK_SEEDS:
        for task, _t, _n in TASKS:
            need = [_rs.checkpoint_path(task, seed), _rs.eval_batches_path(task)]
            need += [_rs.score_batches_path(task, s)
                     for s in (*NEW_PRUNING_SEEDS, REGRESSION_SEED)]
            missing += [p for p in need if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def make_task_suite(task: str, ng_t: int, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = _rs._common(task, ng_t, seed)
        if include_baseline:
            runs.append({"run_id": f"noisenecext_{task}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **common})
        plan = [(f, s) for f in FACTORS for s in NEW_PRUNING_SEEDS]
        plan.append((REGRESSION_FACTOR, REGRESSION_SEED))
        for factor, pseed in plan:
            for amount in amounts:
                runs.append({
                    "run_id": (f"noisenecext_{task}_netseed{seed}_{LABEL}_{tag(factor)}_"
                               f"p{int(amount * 100)}_pruneseed{pseed}"),
                    "strategy": STRATEGY, "amount": amount,
                    "seed": pseed, "pruning_seed": pseed, "noise_rng_seed": pseed,
                    "sim_np_sigma_factor": factor,
                    "score_batch_seed": 100_000 + pseed,
                    "score_batches_path": _rs.score_batches_path(task, pseed),
                    **common,
                })
    return {"run_id": f"{SUITE_STEM}_{task}{shard_suffix(shard)}",
            "output_csv": str(output_csv(task, shard)),
            "defaults": _rs.defaults_block(), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--shard-by-sparsity", action="store_true")
    args = ap.parse_args()
    if args.validate_inputs:
        validate_inputs()
    # defaults come from the source generator itself, so they cannot drift
    print(f"defaults inherited from {SOURCE_STEM}'s generator "
          f"({len(_rs.defaults_block())} keys)")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task, _t, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            suite = make_task_suite(task, ng_t, amounts=amounts,
                                    include_baseline=(i == 0), shard=shard)
            config_path(task, shard).write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {CONFIG_DIR}")
    print(f"total runs: {total}  "
          f"({len(FACTORS)} factors x {len(NEW_PRUNING_SEEDS)} new seeds + regression arm)")
    print(f"merged with {SOURCE_STEM} seed 0 -> 3 pruning seeds per cell, n = 24 networks")


if __name__ == "__main__":
    main()
