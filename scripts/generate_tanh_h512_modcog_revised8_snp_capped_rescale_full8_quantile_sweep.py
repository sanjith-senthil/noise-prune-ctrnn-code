#!/usr/bin/env python3
"""Generate full 8-task S-NP capped-rescale q50/q75/q90 task suite.

This is the step-4 official-suite extension: all 8 revised Mod-Cog tasks,
3 trained H=512 networks per task, pruning amounts 50/60/70/80%, one pruning
seed, and S-NP rescale compared against q50/q75/q90 capped S-NP rescale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
PRUNING_SEED = 0
AMOUNTS = (0.5, 0.6, 0.7, 0.8)
CAP_QUANTILES = (0.50, 0.75, 0.90)

SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_snp_capped_rescale_q50_q75_q90_full8_p50_80"
CONFIG_PATH = Path(f"configs/{SUITE_ID}.json")
OUTPUT_CSV = Path(f"results/{SUITE_ID}.csv")


def checkpoint_path(task_label: str, network_seed: int) -> str:
    return (
        f"checkpoints/tanh_h512_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{network_seed}/"
        f"modcog_{task_label}_seed{network_seed}.pt"
    )


def source_label(task_label: str, network_seed: int) -> str:
    return f"{task_label}_tanh_h512_12k_lr0006_seqbest_no_l2_seed{network_seed}"


def source_run_id(task_label: str, network_seed: int) -> str:
    return f"continue_modcog_{task_label}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{network_seed}"


def eval_batches_path(task_label: str) -> str:
    return str(BATCH_CACHE_DIR / "eval" / f"{task_label}_eval_seed200000.pt")


def score_batches_path(task_label: str) -> str:
    return str(BATCH_CACHE_DIR / "score" / f"{task_label}_score_seed100000.pt")


def amount_tag(amount: float) -> str:
    return f"p{int(round(amount * 100.0))}"


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(quantile * 100.0)):02d}"


def validate_inputs() -> None:
    missing = []
    for network_seed in NETWORK_SEEDS:
        for task_label, _task, _ng_t in TASKS:
            for path in (
                checkpoint_path(task_label, network_seed),
                eval_batches_path(task_label),
                score_batches_path(task_label),
            ):
                if not Path(path).exists():
                    missing.append(path)
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))


def make_suite() -> dict:
    defaults = {
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
        "sim_np_observable_space": "rate",
        "sim_np_inject_space": "rate",
        "sim_np_centering": "trajectory_mean",
        "sim_np_max_samples": 25000,
        "sim_np_burn_in_steps": 300,
        "rescale_cap_mode": "quantile",
        "rescale_cap_value": None,
    }

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
                    "run_id": f"snp_cap_full8_{task_label}_netseed{network_seed}_baseline",
                    "strategy": "none",
                    "amount": 0.0,
                    "no_prune": True,
                    "seed": network_seed,
                    **common,
                }
            )
            for amount in AMOUNTS:
                runs.append(
                    {
                        "run_id": (
                            f"snp_cap_full8_{task_label}_netseed{network_seed}_"
                            f"snp_rescale_{amount_tag(amount)}_pruneseed{PRUNING_SEED}"
                        ),
                        "strategy": "simulation_noise_prune_rescale",
                        "amount": amount,
                        "seed": PRUNING_SEED,
                        "pruning_seed": PRUNING_SEED,
                        "noise_rng_seed": PRUNING_SEED,
                        "score_batch_seed": 100_000,
                        "score_batches_path": score_batches_path(task_label),
                        **common,
                    }
                )
                for quantile in CAP_QUANTILES:
                    qtag = quantile_tag(quantile)
                    runs.append(
                        {
                            "run_id": (
                                f"snp_cap_full8_{task_label}_netseed{network_seed}_"
                                f"snp_capped_rescale_{qtag}_{amount_tag(amount)}_pruneseed{PRUNING_SEED}"
                            ),
                            "strategy": "simulation_noise_prune_capped_rescale",
                            "amount": amount,
                            "seed": PRUNING_SEED,
                            "pruning_seed": PRUNING_SEED,
                            "noise_rng_seed": PRUNING_SEED,
                            "score_batch_seed": 100_000,
                            "score_batches_path": score_batches_path(task_label),
                            "rescale_cap_quantile": quantile,
                            **common,
                        }
                    )

    return {
        "run_id": SUITE_ID,
        "output_csv": str(OUTPUT_CSV),
        "defaults": defaults,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-inputs", action="store_true")
    args = parser.parse_args()
    if args.validate_inputs:
        validate_inputs()

    suite = make_suite()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"wrote {CONFIG_PATH}")
    print(f"output will be {OUTPUT_CSV}")
    print(f"runs: {len(suite['runs'])}")


if __name__ == "__main__":
    main()
