#!/usr/bin/env python3
"""Is the H=1024 drop a *size* effect or a *baseline-accuracy* effect?

The H=1024 checkpoints train to 0.974 mean unpruned sequence accuracy against
H=512's 0.676 under an identical recipe, and within BOTH sizes the networks with
the higher unpruned accuracy retain the least
(Spearman(baseline, retention) = -0.52 to -0.75 at H=512 and -0.38 to -0.78 at
H=1024, S-NP rescale). A rank ANCOVA over all 48 networks finds no residual size
term once baseline accuracy is in the model (p = 0.23 / 0.72 / 0.91 / 0.98 at
50/60/70/80% sparsity). So the cross-size retention comparison is confounded.

That is correlational. This suite makes it causal *within one size*: the same
eight tasks, the same width, the same eval and score batches, the same pruning
protocol -- only the amount of training differs. The 6k-step H=512 checkpoints
sit at 0.487-0.818 unpruned against the 12k checkpoints' 0.582-0.892.

The prediction, if baseline accuracy is what drives retention, is that the
*less* trained H=512 networks retain MORE, task for task, despite being the same
architecture. If instead retention is a property of the architecture, the two
training lengths should retain alike.

Prediction fails => the size confound argument is wrong and the H=1024 drop
needs another explanation. Either way the answer is cheap: 312 H=512 runs.

Pruning seed 0 only, so each cell pairs against pruning seed 0 of the frozen
12k task-preservation artifact.
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

SUITE_STEM = ("task_preservation_tanh_h512_modcog_revised8_6k_seqbest"
              "_baseline_control_p50_80")
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7, 0.8)
PRUNING_SEED = 0

# (label, strategy, needs_score_batches)
ARMS = (
    ("snp_rescale", "simulation_noise_prune_rescale", True),
    ("snp_mask", "simulation_noise_prune_mask_only", True),
    ("magnitude", "l1_unstructured", False),
)


def checkpoint_path(task: str, seed: int) -> str:
    return (f"checkpoints/tanh_h512_modcog_revised8_6k_seqbest_no_l2_seed{seed}/"
            f"modcog_{task}_seed{seed}.pt")


def make_unit(task, ng_t, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = {
            "task": f"modcog:{task}", "ng_T": ng_t,
            "load_model_path": checkpoint_path(task, seed),
            "source_model_label": f"{task}_tanh_h512_6k_seqbest_no_l2_seed{seed}",
            "source_run_id": f"train_modcog_{task}_tanh_h512_6k_seqbest_no_l2_seed{seed}",
            "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000, "eval_batches_path": _rs.eval_batches_path(task),
        }
        if include_baseline:
            runs.append({"run_id": f"base6k_{task}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **common})
        for label, strategy, needs_score in ARMS:
            for amount in amounts:
                r = {"run_id": (f"base6k_{task}_netseed{seed}_{label}_"
                                f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                     "strategy": strategy, "amount": amount,
                     "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                     "noise_rng_seed": PRUNING_SEED, **common}
                if needs_score:
                    r["score_batch_seed"] = 100_000 + PRUNING_SEED
                    r["score_batches_path"] = _rs.score_batches_path(task, PRUNING_SEED)
                runs.append(r)
    return {"run_id": f"{SUITE_STEM}_{task}{_shard(shard)}",
            "output_csv": f"results/{SUITE_STEM}/{task}{_shard(shard)}.csv",
            "defaults": _rs.defaults_block(), "runs": runs}


def _shard(a) -> str:
    return "" if a is None else f"_p{int(a * 100)}"


def validate() -> None:
    missing = [p for seed in NETWORK_SEEDS for task, _t, _n in TASKS
               for p in (checkpoint_path(task, seed), _rs.eval_batches_path(task),
                         _rs.score_batches_path(task, PRUNING_SEED))
               if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-by-sparsity", action="store_true")
    args = ap.parse_args()
    validate()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task, _t, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            u = make_unit(task, ng_t, amounts=amounts,
                          include_baseline=(i == 0), shard=shard)
            (CONFIG_DIR / f"{SUITE_STEM}_{task}{_shard(shard)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {CONFIG_DIR}; total runs {total}")


if __name__ == "__main__":
    main()
