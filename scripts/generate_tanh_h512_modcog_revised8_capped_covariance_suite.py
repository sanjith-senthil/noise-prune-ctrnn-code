#!/usr/bin/env python3
"""Does capping the rescale give the covariance term its job back?

THE ARGUMENT.  Uncapped, ``E[W_hat_ij] = w_ij`` for *any* retention score --
the probability cancels out of the expectation -- so the score cannot touch the
systematic part of the pruning error and can only move variance, where
magnitude-proportional retention is already the optimum.  That is why the
covariance is worth +0.128/+0.168/+0.095 when the mask is deterministic and only
+0.015/+0.020/+0.012 once survivors are rescaled (n = 24).

Capping breaks the cancellation.  With amplification capped at ``c``,

    E[W_hat_ij] = w_ij * min(1, p_ij * c)

which is biased low on exactly the edges whose retention probability falls below
``1/c``.  The score now decides *which* connections get under-restored -- a
first-order job.  So the prediction is that **the covariance's contribution grows
as the cap tightens**, and the contrast that measures it is

    capped full score   vs   capped magnitude-only score (covariance dropped)

at matched cap quantile.  The uncapped pair is the q = 1 end of the same curve
and doubles as the regression anchor: `snp_rescale` must reproduce the frozen
task-preservation suite and `snp_magnitude` the covariance-ablation suite.

Why this matters beyond the statistic: uncapped rescaling needs amplification
factors with a mean of 29.6x on surviving small edges at 80% sparsity and a
maximum around 345x, which is not defensible as a synaptic mechanism.  Capping is
required anyway.  If the covariance earns more under the cap, then it earns more
in precisely the regime the method has to operate in.

uncapped and q50 x 2 arms x 8 tasks x 3 network seeds x 3 sparsities, one
pruning seed.  n = 24 trained networks.
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

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_cappedcov_p50_70"
CONFIG_DIR = Path("configs/capped_cov")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM

TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
# The 60% screen established the gradient: the covariance contribution rises
# +0.020 -> +0.053 -> +0.091 -> +0.151 from uncapped through q70, q50, q30, and
# q50 is the useful point -- highest absolute retention (0.886, above uncapped's
# 0.810) with the covariance worth 4.6x its uncapped value. This run confirms q50
# across the range. 60% re-runs are skipped by `resume`.
AMOUNTS = (0.5, 0.6, 0.7)
PRUNING_SEED = 0

FULL_UNCAPPED = "simulation_noise_prune_rescale"
MAG_UNCAPPED = "simulation_noise_prune_magnitude_rescale"
FULL_CAPPED = "simulation_noise_prune_capped_rescale"
MAG_CAPPED = "simulation_noise_prune_capped_magnitude_rescale"

# (tag, cap_quantile or None for uncapped)
CAPS = (
    ("uncapped", None),   # paired reference, and the regression anchor
    ("q50", 0.50),
)
# (suffix, strategy_uncapped, strategy_capped)
ARMS = (
    ("full", FULL_UNCAPPED, FULL_CAPPED),
    ("mag",  MAG_UNCAPPED,  MAG_CAPPED),
)

SIM = {FULL_UNCAPPED, MAG_UNCAPPED, FULL_CAPPED, MAG_CAPPED}


def shard_suffix(a): return "" if a is None else f"_p{int(a * 100)}"
def config_path(t, a=None): return CONFIG_DIR / f"{SUITE_STEM}_{t}{shard_suffix(a)}.json"
def output_csv(t, a=None): return OUTPUT_DIR / f"{t}{shard_suffix(a)}.csv"


def validate_inputs() -> None:
    missing = []
    for seed in NETWORK_SEEDS:
        for task, _t, _n in TASKS:
            need = [_rs.checkpoint_path(task, seed), _rs.eval_batches_path(task),
                    _rs.score_batches_path(task, PRUNING_SEED)]
            missing += [p for p in need if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def make_task_suite(task, ng_t, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = _rs._common(task, ng_t, seed)
        if include_baseline:
            runs.append({"run_id": f"capcov_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for cap_tag, cap_q in CAPS:
            for suffix, strat_unc, strat_cap in ARMS:
                strategy = strat_unc if cap_q is None else strat_cap
                label = f"{cap_tag}_{suffix}"
                for amount in amounts:
                    run = {
                        "run_id": (f"capcov_{task}_netseed{seed}_{label}_"
                                   f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                        "strategy": strategy, "amount": amount,
                        "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                        "noise_rng_seed": PRUNING_SEED,
                        "prob_control_seed": PRUNING_SEED,
                        "score_batch_seed": 100_000 + PRUNING_SEED,
                        "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED),
                        **common,
                    }
                    if cap_q is not None:
                        run["rescale_cap_mode"] = "quantile"
                        run["rescale_cap_quantile"] = cap_q
                    runs.append(run)
    return {"run_id": f"{SUITE_STEM}_{task}{shard_suffix(shard)}",
            "output_csv": str(output_csv(task, shard)),
            "defaults": _rs.defaults_block(), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--shard-by-sparsity", action="store_true")
    a = ap.parse_args()
    if a.validate_inputs:
        validate_inputs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in AMOUNTS] if a.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if a.shard_by_sparsity else None
            suite = make_task_suite(task, ng_t, amounts=amounts,
                                    include_baseline=(i == 0), shard=shard)
            config_path(task, shard).write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {CONFIG_DIR}")
    print(f"total runs: {total}   {len(CAPS)} cap levels x {len(ARMS)} arms")


if __name__ == "__main__":
    main()
