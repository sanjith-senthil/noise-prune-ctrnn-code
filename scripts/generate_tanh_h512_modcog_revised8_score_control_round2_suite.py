#!/usr/bin/env python3
"""Generate the second round of reviewer-requested pruning controls.

Round 1 (``..._score_control_full_...``) covered four of the six controls the
reviewer proposed: sample-and-rescale with uniform probabilities, the same with
probabilities shuffled across edges, and magnitude pruning followed by either a
``1 / retained density`` or a constant-L1 rescale.  This suite adds the two that
were missing.

New arms
--------
``magnitude_gain_spectral``
    Magnitude pruning followed by the scalar that restores ``rho(W_rec)`` to its
    unpruned value.  The reviewer asked for magnitude "at matched expected
    recurrent gain", and for a recurrent network the spectral radius is the
    operative sense of gain -- it is what decides whether activity expands or
    contracts.  Round 1's two conventions are count- and mass-based readings of
    the same phrase, and for magnitude pruning they are not equivalent to it:
    magnitude discards the small weights, which carry most of the L1 mass but
    little of the spectrum, so ``inv_density`` and ``match_l1`` overshoot the
    original gain several-fold instead of restoring it.  See the module
    docstring of ``pruning/score_controls.py`` for the measured ratios.

``lnp_magnitude`` / ``snp_magnitude``
    Noise-prune with the covariance factor dropped, i.e. retention
    probabilities proportional to ``|w_ij|`` alone rather than to
    ``K |w_ij| (C_ii + C_jj -/+ 2 C_ij)``, at the same expected density,
    followed by the usual ``1 / p`` rescale.  Round 1's shuffle and uniform
    controls remove *all* per-edge information; this one removes only the
    covariance term, so the contrast against ``lnp_rescale`` / ``snp_rescale``
    attributes the effect to the covariance specifically.

``random_gain_spectral``
    Not requested, but it is the control that licenses the argument above: for
    an unbiased mask the three gain conventions should coincide, so this arm
    shows the convention only matters where the mask is magnitude-biased.

Regression arms
---------------
``magnitude``, ``lnp_rescale`` and ``snp_rescale`` are re-run here at network
seed 0 / pruning seed 0 purely so their values can be checked against the frozen
round-1 tables.  ``magnitude`` is deterministic and must reproduce exactly; the
two stochastic arms consume the same seeds and should also reproduce exactly.
The summarizer performs that check and fails loudly if it does not hold.

Scope
-----
One trained-network seed (0) and one pruning seed (0), across all eight Mod-Cog
tasks and four sparsities -- enough to read the effect quickly.  It is *not* the
n = 24 analysis unit of the main suite, so results from it are descriptive and
must not be quoted with the paper's inferential statistics.

As in round 1, one config is emitted per task with its own ``output_csv``: the
suite harness rewrites the whole CSV on every append, so parallel workers must
never share an output file.

Sharding
--------
``--shard-by-sparsity`` splits each task's config four ways, one per sparsity,
giving 32 configs instead of 8 and so allowing more than eight concurrent
workers.  It exists because the per-task layout caps parallelism at the number
of tasks, which leaves cores idle on a machine with more than eight of them.
Whether that actually helps is a property of the machine, not of the suite: on
the 11-core (5 performance + 6 efficiency) development machine the eight-worker
layout already draws ~760% CPU at load ~10.4, and round 1 measured throughput
flat between one and two threads per worker, so it is bandwidth-bound and extra
workers mainly add queuing.  Measure before assuming a gain.

The unpruned baseline run is emitted only in each task's lowest-sparsity shard,
since it does not depend on sparsity and the summarizer resolves baselines by
``source_model_label`` across the merged frame.  Sharded and unsharded configs
write into the same output directory with distinct file names, so the two
layouts must not be mixed within one set of results -- a shard writes only the
rows it owns, and ``resume`` keys on the individual output CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_round2_p50_80"
CONFIG_DIR = Path("configs/score_control_round2")
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
NETWORK_SEEDS = (0,)
PRUNING_SEEDS = (0,)
AMOUNTS = (0.5, 0.6, 0.7, 0.8)

# (label, strategy, extra per-run options).  Every arm uses pruning seed 0.
METHODS = (
    # the two controls this round exists to add
    ("magnitude_gain_spectral", "l1_unstructured_gain", {"gain_mode": "match_spectral_radius"}),
    # The other two defensible readings of "matched recurrent gain". rho(J_lin) is the
    # spectral radius of the network's actual state-transition operator (1-a)I + aW --
    # the operator that iterates, and already the paper's `post_rec_linear_rho`. match_l2
    # is the mean-field reading (g^2 = N Var(w)). Measured rather than interpolated
    # because the gain response is sharp enough that a 7% difference in factor moves
    # retention substantially.
    ("magnitude_gain_jlin", "l1_unstructured_gain", {"gain_mode": "match_rho_jlin"}),
    ("magnitude_gain_l2", "l1_unstructured_gain", {"gain_mode": "match_l2"}),
    # Synaptic scaling: per-neuron restoration of total absolute input weight. The only
    # rescale here that is a *local* rule -- each row's factor depends solely on the
    # weights that neuron receives -- so it is the biologically motivated baseline and
    # the one a homeostatic-plasticity objection points at.
    ("magnitude_gain_rowl1", "l1_unstructured_gain", {"gain_mode": "match_l1_rowwise"}),
    ("lnp_magnitude", "noise_prune_magnitude_rescale", {}),
    ("snp_magnitude", "simulation_noise_prune_magnitude_rescale", {}),
    # licenses the "convention only matters for a biased mask" argument
    ("random_gain_spectral", "random_unstructured_gain", {"gain_mode": "match_spectral_radius"}),
    # regression arms: must reproduce the frozen round-1 values exactly
    ("magnitude", "l1_unstructured", {}),
    ("lnp_rescale", "noise_prune", {}),
    ("snp_rescale", "simulation_noise_prune_rescale", {}),
)

SNP_STRATEGIES = {
    "simulation_noise_prune_rescale",
    "simulation_noise_prune_magnitude_rescale",
}
CONTROL_STRATEGIES = {
    "noise_prune_magnitude_rescale",
    "simulation_noise_prune_magnitude_rescale",
}
# Matches round 1 so a shuffle-seeded arm would draw the same stream; the
# magnitude control is deterministic and ignores it, but the field is carried
# for uniformity of the run records.
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


def shard_suffix(amount: float | None) -> str:
    return "" if amount is None else f"_p{int(amount * 100)}"


def config_path(task_label: str, amount: float | None = None) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task_label}{shard_suffix(amount)}.json"


def output_csv(task_label: str, amount: float | None = None) -> Path:
    return OUTPUT_DIR / f"{task_label}{shard_suffix(amount)}.csv"


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
    """Identical to the round-1 suite, so the two are directly comparable."""
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


def make_task_suite(
    task_label: str, task: str, ng_t: int, amounts: tuple[float, ...] = AMOUNTS,
    include_baseline: bool = True, shard: float | None = None,
) -> dict:
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
        if include_baseline:
            runs.append({
                "run_id": f"ctrl_r2_{task_label}_netseed{network_seed}_baseline",
                "strategy": "none",
                "amount": 0.0,
                "no_prune": True,
                "seed": network_seed,
                **common,
            })
        for label, strategy, extra in METHODS:
            for amount in amounts:
                for pruning_seed in PRUNING_SEEDS:
                    run = {
                        "run_id": (
                            f"ctrl_r2_{task_label}_netseed{network_seed}_{label}_"
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
        "run_id": f"{SUITE_STEM}_{task_label}{shard_suffix(shard)}",
        "output_csv": str(output_csv(task_label, shard)),
        "defaults": defaults_block(),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--shard-by-sparsity", action="store_true",
                        help="emit one config per (task, sparsity) instead of per task, "
                             "so more than len(TASKS) workers can run concurrently")
    args = parser.parse_args()
    if args.validate_inputs:
        validate_inputs()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    n_configs = 0
    for task_label, task, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            suite = make_task_suite(
                task_label, task, ng_t, amounts=amounts,
                include_baseline=(i == 0), shard=shard,
            )
            validate_default_schema(suite["defaults"])
            path = config_path(task_label, shard)
            path.write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"])
            n_configs += 1
            print(f"  {task_label + shard_suffix(shard):<24}{len(suite['runs']):>6} runs -> {path}")

    per_task = total // len(TASKS)
    print(f"\nwrote {n_configs} configs to {CONFIG_DIR}")
    print(f"outputs -> {OUTPUT_DIR}/<task>.csv")
    print(f"total runs: {total} ({per_task} per task)")
    print(f"scope: {len(NETWORK_SEEDS)} network seed x {len(PRUNING_SEEDS)} pruning seed "
          f"x {len(TASKS)} tasks x {len(AMOUNTS)} sparsities -- descriptive, not the n = 24 unit")


if __name__ == "__main__":
    main()
