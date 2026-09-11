#!/usr/bin/env python3
"""Promote the covariance-ablation control to the official suite standard.

Reviewer 1 asked for five controls.  Four were already run at the full standard
in ``..._score_control_full_p50_80``:

  1. sample-and-rescale with uniform p          -> lnp_uniform / snp_uniform
  2. sample-and-rescale with p shuffled         -> lnp_shuffled / snp_shuffled
  3. magnitude, rescaled to constant sum |w|    -> magnitude_gain_matchl1
  4. magnitude, rescaled by 1/(retained density)-> magnitude_gain_invdensity
     (with random_gain_invdensity, the other half of the reviewer's ask)

The fifth was only ever run at one network seed.  This suite brings it to the
same standard.

  5. probabilities proportional to |w| with the covariance factor dropped,
     then the usual sample-and-rescale -> lnp_magnitude / snp_magnitude

What arm 5 is
-------------
NOT magnitude pruning.  Noise-prune's retention probability is a product,
``p_ij = K |w_ij| (C_ii + C_jj -/+ 2 C_ij)``; this arm deletes the covariance
factor and keeps everything else, so ``p_ij`` is proportional to ``|w_ij|``
alone, renormalised to the same expected density.  Edges are still drawn by
independent Bernoulli trials and survivors are still rescaled by ``1 / p_ij``,
so the estimator stays unbiased -- ``E[W_pruned] = W`` -- and the rescale
remains derived rather than stipulated.  Because the draw is stochastic it
keeps only ~42% of the largest-magnitude edges, unlike deterministic magnitude
pruning, and because ``p_ij`` is proportional to ``|w_ij|`` the rescaled
survivor ``w_ij / p_ij = sign(w_ij) / c`` has constant magnitude.

The contrast against ``noise_prune`` therefore isolates the covariance term and
nothing else, which is the decomposition the reviewer's closing paragraph asks
for ("how much of the improvement is gain and how much is the covariance
information").  The uniform-p and shuffled-p controls cannot answer it, since
they destroy all per-edge information at once.

Standard
--------
8 Mod-Cog tasks x 3 trained-network seeds x 4 sparsities, 3 pruning seeds for
every stochastic arm, giving n = 24 trained networks after averaging
pruning-seed replicates -- identical to the main task-preservation suite and to
score_control_full, so the arms merge with those tables directly.

Regression arms
---------------
``magnitude`` (deterministic) and ``lnp_rescale`` (pruning seed 0) are re-run
here at the same seeds as score_control_full and must reproduce it exactly.
They are the correctness gate: ``magnitude`` proves the evaluation protocol is
identical, and ``lnp_rescale`` proves the *stochastic* code path is -- which is
the path that carried the unseeded-RNG defect (errata E12), so a deterministic
check alone would not be sufficient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covariance_ablation_full_p50_80"
REFERENCE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80"
CONFIG_DIR = Path("configs/covariance_ablation")
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

# (label, strategy, pruning seeds, extra options)
METHODS = (
    ("lnp_magnitude", "noise_prune_magnitude_rescale", PRUNING_SEEDS, {}),
    ("snp_magnitude", "simulation_noise_prune_magnitude_rescale", PRUNING_SEEDS, {}),
    # regression gates -- must reproduce score_control_full exactly
    ("magnitude", "l1_unstructured", (0,), {}),
    ("lnp_rescale", "noise_prune", (0,), {}),
)

SNP_STRATEGIES = {"simulation_noise_prune_magnitude_rescale"}
CONTROL_STRATEGIES = {
    "noise_prune_magnitude_rescale",
    "simulation_noise_prune_magnitude_rescale",
}
PROB_CONTROL_SEED_BASE = 900_000

REQUIRED_DEFAULT_KEYS = {
    "hidden_size", "train_steps", "ft_steps", "last_only", "device",
    "movement_batches", "ng_T", "ng_B", "eval_sample_batches",
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


def shard_suffix(amount) -> str:
    return "" if amount is None else f"_p{int(amount * 100)}"


def config_path(task_label: str, amount=None) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task_label}{shard_suffix(amount)}.json"


def output_csv(task_label: str, amount=None) -> Path:
    return OUTPUT_DIR / f"{task_label}{shard_suffix(amount)}.csv"


def validate_inputs() -> None:
    missing = []
    for network_seed in NETWORK_SEEDS:
        for task_label, _t, _n in TASKS:
            req = [checkpoint_path(task_label, network_seed), eval_batches_path(task_label)]
            req += [score_batches_path(task_label, s) for s in PRUNING_SEEDS]
            missing += [p for p in req if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def validate_default_schema(defaults: dict) -> None:
    missing = sorted(REQUIRED_DEFAULT_KEYS - set(defaults))
    if missing:
        raise ValueError(f"Suite defaults missing required keys: {', '.join(missing)}")
    if defaults["eval_sample_batches"] <= 0:
        raise ValueError("eval_sample_batches must be > 0 for deterministic evaluation in suites.")


def defaults_block() -> dict:
    """Byte-identical to score_control_full, so the tables merge directly."""
    return {
        "reset_results": False, "resume": True, "model_type": "ctrnn",
        "hidden_size": 512, "activation": "tanh", "train_steps": 0, "ft_steps": 0,
        "skip_training": True, "last_only": False, "eval_last_only": False,
        "device": "cpu", "movement_batches": 20, "ng_T": 0, "ng_B": 256,
        "eval_sample_batches": 128,
        "eval_steps_pre0": 100, "eval_steps_pre": 100,
        "eval_steps_post0": 100, "eval_steps_post": 100,
        "noise_sigma": 1.0, "noise_eps": 0.3, "noise_leak_shift": 0.0,
        "noise_matched_diagonal": False,
        "sim_np_sigma": None, "sim_np_sigma_source": "natural_voltage",
        "sim_np_observable_space": "rate", "sim_np_inject_space": "rate",
        "sim_np_centering": "trajectory_mean", "sim_np_max_samples": 25000,
        "sim_np_burn_in_steps": 300,
    }


def assert_defaults_match_reference() -> None:
    """Fail if this suite's defaults drift from score_control_full's.

    The whole point of the suite is that its arms sit on the same footing as the
    reference tables; a silent difference in, say, ``eval_sample_batches`` would
    make every cross-suite comparison wrong while still producing clean-looking
    numbers.
    """
    ref_dir = Path("configs/score_control_full")
    ref_files = sorted(ref_dir.glob("*.json"))
    if not ref_files:
        print("  (reference configs absent; skipping defaults cross-check)")
        return
    ref = json.loads(ref_files[0].read_text())["defaults"]
    mine = defaults_block()
    diff = {k: (ref.get(k), mine.get(k)) for k in set(ref) | set(mine)
            if ref.get(k) != mine.get(k)}
    if diff:
        raise SystemExit(
            "Suite defaults differ from score_control_full, so the arms would not be "
            f"comparable: {diff}")
    print(f"  defaults cross-check vs {REFERENCE_STEM}: identical ({len(mine)} keys)")


def make_task_suite(task_label, task, ng_t, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for network_seed in NETWORK_SEEDS:
        common = {
            "task": task, "ng_T": ng_t,
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
                "run_id": f"covabl_{task_label}_netseed{network_seed}_baseline",
                "strategy": "none", "amount": 0.0, "no_prune": True,
                "seed": network_seed, **common,
            })
        for label, strategy, seeds, extra in METHODS:
            for amount in amounts:
                for pruning_seed in seeds:
                    run = {
                        "run_id": (f"covabl_{task_label}_netseed{network_seed}_{label}_"
                                   f"p{int(amount * 100)}_pruneseed{pruning_seed}"),
                        "strategy": strategy, "amount": amount, "seed": pruning_seed,
                        "pruning_seed": pruning_seed, "noise_rng_seed": pruning_seed,
                        **common, **extra,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--shard-by-sparsity", action="store_true",
                    help="one config per (task, sparsity) -- 32 units instead of 8, so more "
                         "than len(TASKS) workers can run concurrently")
    args = ap.parse_args()
    if args.validate_inputs:
        validate_inputs()
    assert_defaults_match_reference()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task_label, task, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            suite = make_task_suite(task_label, task, ng_t, amounts=amounts,
                                    include_baseline=(i == 0), shard=shard)
            validate_default_schema(suite["defaults"])
            config_path(task_label, shard).write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"]); n_cfg += 1
    print(f"\nwrote {n_cfg} configs to {CONFIG_DIR}")
    print(f"total runs: {total}")
    print(f"analysis units: {len(TASKS) * len(NETWORK_SEEDS)} trained networks x "
          f"{len(AMOUNTS)} sparsities  (n = 24, official suite standard)")


if __name__ == "__main__":
    main()
