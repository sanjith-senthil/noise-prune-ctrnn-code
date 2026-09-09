#!/usr/bin/env python3
"""Generate the uninformative-score / gain-matched control pilot suite.

This suite answers the reviewer question of whether noise-prune's benefit is
attributable to the covariance-derived retention probabilities, or merely to
stochastic masking plus gain restoration.

Arms
----
reference
    ``noise_prune`` (L-NP rescale) and ``simulation_noise_prune_rescale``
    (S-NP rescale), re-run inside this suite so every comparison is
    apples-to-apples and so the frozen results are regression-checked.

uninformative score
    ``*_shuffled_rescale`` permutes the retention probabilities across edges.
    Expected density and the amplification distribution are preserved exactly;
    only the score-to-edge assignment is destroyed.
    ``*_uniform_rescale`` replaces every probability by the mean, giving
    stochastic masking plus a single constant gain at the same expected
    density.

gain-matched baseline
    ``l1_unstructured_gain`` and ``random_unstructured_gain`` rescale survivors
    to matched recurrent gain.  ``inv_density`` multiplies by
    ``1 / retained density`` (the literal correction, exact for an unbiased
    mask); ``match_l1`` rescales so the retained absolute weight sum equals the
    unpruned value.  The two coincide for random pruning but not for magnitude,
    whose survivors are already the large weights.

Scope is deliberately reduced relative to the main suite: network seed 0 only
and the two most discriminating sparsity levels.  Run the full grid only if the
pilot shows a covariance increment worth measuring precisely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_pilot_p70_80"
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
NETWORK_SEEDS = (0,)
PRUNING_SEEDS = (0, 1, 2)
AMOUNTS = (0.7, 0.8)

# (label, strategy, pruning seeds, extra per-run options)
METHODS = (
    # references
    ("lnp_rescale", "noise_prune", PRUNING_SEEDS, {}),
    ("snp_rescale", "simulation_noise_prune_rescale", PRUNING_SEEDS, {}),
    # uninformative-score controls
    ("lnp_shuffled", "noise_prune_shuffled_rescale", PRUNING_SEEDS, {}),
    ("lnp_uniform", "noise_prune_uniform_rescale", PRUNING_SEEDS, {}),
    ("snp_shuffled", "simulation_noise_prune_shuffled_rescale", PRUNING_SEEDS, {}),
    ("snp_uniform", "simulation_noise_prune_uniform_rescale", PRUNING_SEEDS, {}),
    # gain-matched baselines
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
    }
    runs = []
    for network_seed in NETWORK_SEEDS:
        for task_label, task, ng_t in TASKS:
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
                "run_id": f"ctrl_pilot_{task_label}_netseed{network_seed}_baseline",
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
                                f"ctrl_pilot_{task_label}_netseed{network_seed}_{label}_"
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
    counts: dict[str, int] = {}
    for run in suite["runs"]:
        counts[run["strategy"]] = counts.get(run["strategy"], 0) + 1
    print(f"wrote {CONFIG_PATH}")
    print(f"output will be {OUTPUT_CSV}")
    print(f"runs: {len(suite['runs'])}")
    for strategy in sorted(counts):
        print(f"  {strategy:<44}{counts[strategy]:>5}")


if __name__ == "__main__":
    main()
