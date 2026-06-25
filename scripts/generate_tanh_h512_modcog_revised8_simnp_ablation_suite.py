#!/usr/bin/env python3
"""Generate final-suite-compatible S-NP implementation ablation.

This suite reruns the two S-NP implementation contrasts needed for the paper
on the final H512 revised-8 checkpoints and fixed task-preservation batches:

    centering: rate/trajectory_mean vs rate/conditional
    score space: rate/trajectory_mean vs voltage/trajectory_mean

It intentionally uses one pruning seed per cell to keep the ablation focused
and tractable. The main official task-preservation suite already provides
multi-seed estimates for the selected rate/trajectory-mean configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIG_PATH = Path(
    "configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_simnp_ablation_p50_80_fixedbatches.json"
)
OUTPUT_CSV = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_simnp_ablation_p50_80_fixedbatches.csv"
)
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
AMOUNTS = (0.5, 0.6, 0.7, 0.8)
PRUNING_SEED = 0
ABLATIONS = (
    ("rate_trajmean", "rate", "rate", "trajectory_mean"),
    ("rate_conditional", "rate", "rate", "conditional"),
    ("voltage_trajmean", "voltage", "voltage", "trajectory_mean"),
)


def checkpoint_path(task_label: str, network_seed: int) -> str:
    return (
        f"checkpoints/tanh_h512_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{network_seed}/"
        f"modcog_{task_label}_seed{network_seed}.pt"
    )


def source_label(task_label: str, network_seed: int) -> str:
    return f"{task_label}_tanh_h512_12k_lr0006_seqbest_no_l2_seed{network_seed}"


def source_run_id(task_label: str, network_seed: int) -> str:
    return f"continue_modcog_{task_label}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{network_seed}"


def amount_tag(amount: float) -> str:
    return f"p{int(round(amount * 100.0))}"


def eval_batches_path(task_label: str) -> str:
    return str(BATCH_CACHE_DIR / "eval" / f"{task_label}_eval_seed200000.pt")


def score_batches_path(task_label: str, pruning_seed: int) -> str:
    return str(BATCH_CACHE_DIR / "score" / f"{task_label}_score_seed{100_000 + int(pruning_seed)}.pt")


def validate_checkpoints() -> None:
    missing = [
        checkpoint_path(task_label, network_seed)
        for network_seed in NETWORK_SEEDS
        for task_label, _task_name, _ng_t in TASKS
        if not Path(checkpoint_path(task_label, network_seed)).exists()
    ]
    if missing:
        raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))


def defaults() -> dict:
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
        "noise_matched_diagonal": True,
        "sim_np_sigma_source": "natural_voltage",
        "sim_np_max_samples": 25000,
        "sim_np_burn_in_steps": 300,
    }


def make_suite() -> dict:
    runs = []
    for network_seed in NETWORK_SEEDS:
        for task_label, task_name, ng_t in TASKS:
            common = {
                "task": task_name,
                "ng_T": ng_t,
                "load_model_path": checkpoint_path(task_label, network_seed),
                "source_model_label": source_label(task_label, network_seed),
                "source_run_id": source_run_id(task_label, network_seed),
                "source_network_seed": network_seed,
                "source_recurrent_l2_lambda": 0.0,
                "eval_seed": 200_000,
                "eval_batches_path": eval_batches_path(task_label),
            }
            runs.append(
                {
                    "run_id": f"simnp_ablation_final_{task_label}_netseed{network_seed}_baseline",
                    "strategy": "none",
                    "amount": 0.0,
                    "no_prune": True,
                    "seed": network_seed,
                    **common,
                }
            )
            for ablation_label, observable_space, inject_space, centering in ABLATIONS:
                for amount in AMOUNTS:
                    runs.append(
                        {
                            "run_id": (
                                f"simnp_ablation_final_{task_label}_netseed{network_seed}_"
                                f"{ablation_label}_{amount_tag(amount)}_pruneseed{PRUNING_SEED}"
                            ),
                            "strategy": "simulation_noise_prune_mask_only",
                            "amount": amount,
                            "seed": PRUNING_SEED,
                            "pruning_seed": PRUNING_SEED,
                            "noise_rng_seed": PRUNING_SEED,
                            "score_batch_seed": 100_000 + PRUNING_SEED,
                            "score_batches_path": score_batches_path(task_label, PRUNING_SEED),
                            "sim_np_observable_space": observable_space,
                            "sim_np_inject_space": inject_space,
                            "sim_np_centering": centering,
                            **common,
                        }
                    )
    return {
        "run_id": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_simnp_ablation_p50_80_fixedbatches",
        "output_csv": str(OUTPUT_CSV),
        "defaults": defaults(),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.validate_checkpoints:
        validate_checkpoints()

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(make_suite(), indent=2) + "\n")
    print(f"wrote {CONFIG_PATH}")
    print(f"task output will be {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
