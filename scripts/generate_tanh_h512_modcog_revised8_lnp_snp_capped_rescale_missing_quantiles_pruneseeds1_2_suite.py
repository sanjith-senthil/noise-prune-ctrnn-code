#!/usr/bin/env python3
"""Generate pruning-seed 1/2 continuation suite for the cap-percentile curve.

Existing completed data cover q50 for both L-NP/S-NP with pruning seeds
{0,1,2}, and cover the non-q50 quantiles with pruning seed 0. This suite fills
the missing pruning seeds {1,2} for q10/q20/q30/q40/q60/q70/q80/q90 so the
cap-percentile curve can be summarized with three pruning seeds per cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITE_ID = (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_q90_"
    "pruneseeds1_2_full8_p50_80"
)
CONFIG_PATH = Path(f"configs/{SUITE_ID}.json")
OUTPUT_CSV = Path(f"results/{SUITE_ID}.csv")
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
PRUNING_SEEDS = (1, 2)
AMOUNTS = (0.5, 0.6, 0.7, 0.8)
QUANTILES = (0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90)
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


def amount_tag(amount: float) -> str:
    return f"p{int(round(amount * 100.0))}"


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(quantile * 100.0)):02d}"


def source_label(task_label: str, network_seed: int) -> str:
    return f"{task_label}_tanh_h512_12k_lr0006_seqbest_no_l2_seed{network_seed}"


def source_run_id(task_label: str, network_seed: int) -> str:
    return f"continue_modcog_{task_label}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{network_seed}"


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
        "noise_matched_diagonal": False,
        "sim_np_sigma": None,
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
        for task_label, task, ng_t in TASKS:
            common = {
                "task": task,
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
                    "run_id": f"cap_curve_ps12_{task_label}_netseed{network_seed}_baseline",
                    "strategy": "none",
                    "amount": 0.0,
                    "no_prune": True,
                    "seed": network_seed,
                    **common,
                }
            )
            method_specs = (
                ("lnp_capped_rescale", "noise_prune_capped_rescale"),
                ("snp_capped_rescale", "simulation_noise_prune_capped_rescale"),
            )
            for label, strategy in method_specs:
                for quantile in QUANTILES:
                    qtag = quantile_tag(quantile)
                    for amount in AMOUNTS:
                        for pruning_seed in PRUNING_SEEDS:
                            run = {
                                "run_id": (
                                    f"cap_curve_ps12_{task_label}_netseed{network_seed}_"
                                    f"{label}_{qtag}_{amount_tag(amount)}_pruneseed{pruning_seed}"
                                ),
                                "strategy": strategy,
                                "amount": amount,
                                "seed": pruning_seed,
                                "pruning_seed": pruning_seed,
                                "noise_rng_seed": pruning_seed,
                                "rescale_cap_quantile": quantile,
                                **common,
                            }
                            if strategy == "simulation_noise_prune_capped_rescale":
                                run["score_batch_seed"] = 100_000 + pruning_seed
                                run["score_batches_path"] = score_batches_path(task_label, pruning_seed)
                            runs.append(run)

    return {"run_id": SUITE_ID, "output_csv": str(OUTPUT_CSV), "defaults": defaults, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-inputs", action="store_true")
    args = parser.parse_args()
    if args.validate_inputs:
        validate_inputs()
    suite = make_suite()
    validate_default_schema(suite["defaults"])
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(suite, indent=2) + "\n")
    print(f"wrote {CONFIG_PATH}")
    print(f"output will be {OUTPUT_CSV}")
    print(f"runs: {len(suite['runs'])}")
    print("expected completed rows: 3096 = 24 baselines + 3072 capped rows")


if __name__ == "__main__":
    main()
