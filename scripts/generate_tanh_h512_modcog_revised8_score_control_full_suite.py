#!/usr/bin/env python3
"""Generate the full uninformative-score / gain-matched control suite.

Publication-grade version of the pilot: all 8 Mod-Cog tasks x 3 trained-network
seeds x 4 sparsity levels, three pruning seeds for every stochastic arm.  This
matches the analysis unit of the main task-preservation suite (n = 24 trained
networks after averaging pruning-seed replicates), so the control comparisons
can be reported on the same footing as Figs. 1-2.

Arms (11)
---------
reference            noise_prune, simulation_noise_prune_rescale
uninformative score  {noise_prune, simulation_noise_prune}_{shuffled,uniform}_rescale
gain-matched         l1_unstructured_gain (inv_density and match_l1),
                     random_unstructured_gain (inv_density)
uncorrected baseline l1_unstructured, random_unstructured

One config is emitted PER TASK, each with its own ``output_csv``.  The suite
harness rewrites the whole CSV on every append, so parallel workers must never
share an output file; per-task configs make the suite safely parallel and
independently resumable.  Merge the per-task CSVs with
``summarize_tanh_h512_modcog_revised8_score_control_full.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80"
CONFIG_DIR = Path("configs/score_control_full")
OUTPUT_DIR = Path(f"results/{SUITE_STEM}")
BATCH_CACHE_DIR = Path("results/fixed_batches/tanh_h512_modcog_revised8_taskpres")

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
PRUNING_SEEDS = (0, 1, 2)
AMOUNTS = (0.5, 0.6, 0.7, 0.8)

# (label, strategy, pruning seeds, extra per-run options)
METHODS = (
    ("lnp_rescale", "noise_prune", PRUNING_SEEDS, {}),
    ("snp_rescale", "simulation_noise_prune_rescale", PRUNING_SEEDS, {}),
    ("lnp_shuffled", "noise_prune_shuffled_rescale", PRUNING_SEEDS, {}),
    ("lnp_uniform", "noise_prune_uniform_rescale", PRUNING_SEEDS, {}),
    ("snp_shuffled", "simulation_noise_prune_shuffled_rescale", PRUNING_SEEDS, {}),
    ("snp_uniform", "simulation_noise_prune_uniform_rescale", PRUNING_SEEDS, {}),
    ("magnitude", "l1_unstructured", (0,), {}),
    ("magnitude_gain_invdensity", "l1_unstructured_gain", (0,), {"gain_mode": "inv_density"}),
    ("magnitude_gain_matchl1", "l1_unstructured_gain", (0,), {"gain_mode": "match_l1"}),
    ("random", "random_unstructured", PRUNING_SEEDS, {}),
    ("random_gain_invdensity", "random_unstructured_gain", PRUNING_SEEDS, {"gain_mode": "inv_density"}),
)

SNP_STRATEGIES = {
    "simulation_noise_prune_rescale",
    "simulation_noise_prune_shuffled_rescale",
    "simulation_noise_prune_uniform_rescale",
}
CONTROL_STRATEGIES = {
    "noise_prune_shuffled_rescale",
    "noise_prune_uniform_rescale",
    "simulation_noise_prune_shuffled_rescale",
    "simulation_noise_prune_uniform_rescale",
}
PROB_CONTROL_SEED_BASE = 900_000

REQUIRED_DEFAULT_KEYS = {
    "hidden_size",
    "train_steps",
    "ft_steps",
    "last_only",
    "device",
    "movement_batches",
    "ng_T",
    "ng_B",
    "eval_sample_batches",
}


def checkpoint_path(task_label: str, network_seed: int) -> str:
    return (
        f"checkpoints/tanh_h512_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{network_seed}/"
        f"modcog_{task_label}_seed{network_seed}.pt"
    )


def eval_batches_path(task_label: str) -> str:
    return str(BATCH_CACHE_DIR / "eval" / f"{task_label}_eval_seed200000.pt")


def score_batches_path(task_label: str, pruning_seed: int) -> str:
    return str(BATCH_CACHE_DIR / "score" / f"{task_label}_score_seed{100_000 + pruning_seed}.pt")


def config_path(task_label: str) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task_label}.json"


def output_csv(task_label: str) -> Path:
    return OUTPUT_DIR / f"{task_label}.csv"


def validate_inputs() -> None:
    missing = []
    for network_seed in NETWORK_SEEDS:
        for task_label, _task, _ng_t in TASKS:
            required = [checkpoint_path(task_label, network_seed), eval_batches_path(task_label)]
            required.extend(score_batches_path(task_label, seed) for seed in PRUNING_SEEDS)
            missing.extend(path for path in required if not Path(path).exists())
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def validate_default_schema(defaults: dict) -> None:
    missing = sorted(REQUIRED_DEFAULT_KEYS - set(defaults))
    if missing:
        raise ValueError(f"Suite defaults missing required keys: {', '.join(missing)}")
    if defaults["eval_sample_batches"] <= 0:
        raise ValueError("eval_sample_batches must be > 0 for deterministic evaluation in suites.")


def defaults_block() -> dict:
    return {
        "reset_results": False,
        "resume": True,
        "model_type": "ctrnn",
        "hidden_size": 512,
        "activation": "tanh",
        "train_steps": 0,
        "ft_steps": 0,
        "skip_training": True,
        "last_only": False,
        "eval_last_only": False,
        "device": "cpu",
        "movement_batches": 20,
        "ng_T": 0,
        "ng_B": 256,
        "eval_sample_batches": 128,
        "eval_steps_pre0": 100,
        "eval_steps_pre": 100,
        "eval_steps_post0": 100,
        "eval_steps_post": 100,
        "noise_sigma": 1.0,
        "noise_eps": 0.3,
        "noise_leak_shift": 0.0,
        "noise_matched_diagonal": False,
        "sim_np_sigma": None,
        "sim_np_sigma_source": "natural_voltage",
        "sim_np_observable_space": "rate",
        "sim_np_inject_space": "rate",
        "sim_np_centering": "trajectory_mean",
        "sim_np_max_samples": 25000,
        "sim_np_burn_in_steps": 300,
    }


def make_task_suite(task_label: str, task: str, ng_t: int) -> dict:
    runs = []
    for network_seed in NETWORK_SEEDS:
        common = {
            "task": task,
            "ng_T": ng_t,
            "load_model_path": checkpoint_path(task_label, network_seed),
            "source_model_label": f"{task_label}_tanh_h512_12k_lr0006_seqbest_no_l2_seed{network_seed}",
            "source_run_id": f"continue_modcog_{task_label}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{network_seed}",
            "source_network_seed": network_seed,
            "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000,
            "eval_batches_path": eval_batches_path(task_label),
        }
        runs.append({
            "run_id": f"ctrl_full_{task_label}_netseed{network_seed}_baseline",
            "strategy": "none",
            "amount": 0.0,
            "no_prune": True,
            "seed": network_seed,
            **common,
        })
        for label, strategy, seeds, extra in METHODS:
            for amount in AMOUNTS:
                for pruning_seed in seeds:
                    run = {
                        "run_id": (
                            f"ctrl_full_{task_label}_netseed{network_seed}_{label}_"
                            f"p{int(amount * 100)}_pruneseed{pruning_seed}"
                        ),
                        "strategy": strategy,
                        "amount": amount,
                        "seed": pruning_seed,
                        "pruning_seed": pruning_seed,
                        "noise_rng_seed": pruning_seed,
                        **common,
                        **extra,
                    }
                    if strategy in SNP_STRATEGIES:
                        run["score_batch_seed"] = 100_000 + pruning_seed
                        run["score_batches_path"] = score_batches_path(task_label, pruning_seed)
                    if strategy in CONTROL_STRATEGIES:
                        run["prob_control_seed"] = PROB_CONTROL_SEED_BASE + pruning_seed
                    runs.append(run)
    return {
        "run_id": f"{SUITE_STEM}_{task_label}",
        "output_csv": str(output_csv(task_label)),
        "defaults": defaults_block(),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-inputs", action="store_true")
    args = parser.parse_args()
    if args.validate_inputs:
        validate_inputs()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for task_label, task, ng_t in TASKS:
        suite = make_task_suite(task_label, task, ng_t)
        validate_default_schema(suite["defaults"])
        config_path(task_label).write_text(json.dumps(suite, indent=2) + "\n")
        total += len(suite["runs"])
        print(f"  {task_label:<20}{len(suite['runs']):>6} runs -> {config_path(task_label)}")

    per_task = total // len(TASKS)
    print(f"\nwrote {len(TASKS)} per-task configs to {CONFIG_DIR}")
    print(f"outputs -> {OUTPUT_DIR}/<task>.csv")
    print(f"total runs: {total} ({per_task} per task)")
    print(f"expected analysis units: {len(TASKS) * len(NETWORK_SEEDS)} trained networks "
          f"x {len(AMOUNTS)} sparsities")


if __name__ == "__main__":
    main()
