#!/usr/bin/env python3
"""Generate the two revision suites requested by peer review.

Both reuse the paper's 24 trained networks, the frozen held-out evaluation
batches (seed 200000) and the frozen pruning-score batches (seeds 100000-2), so
every number is directly comparable to Figs. 1-2.

``low_sparsity``  (reviewer 2, comment 3a)
    The paper reports 50-80% only, justified at ll. 113-114 by "all methods
    performed well at sparsities below 50%".  That is currently documented only
    for magnitude and the two deterministic variants, at 30%, in a file that did
    not make the release.  This suite runs all seven methods at 10/20/30/40% so
    the full sparsity-vs-retention curve can be shown.

    It additionally re-runs S-NP rescale at 50-80%.  That arm is unchanged
    scientifically; the purpose is to capture the new ``kept_edges`` statistic
    (Bernoulli survivors before the exact-density top-k), which the original
    runs did not record and which reviewer 1's minor point 2 asks for.

``noise_necessity``  (reviewer 2, comment 4a)
    Sweeps the injected-noise scale as a multiple of each network's own natural
    variability, sigma = factor * sigma_nat.  The paper already shows that the
    *task-driven* component is not necessary (conditional centering, which
    isolates noise-induced deviation, performs the same).  The missing control
    is the converse: with sigma -> 0 the covariance is driven purely by
    across-trial task variability, so if scores remain good then noise is not
    necessary either.

    sigma = 0 exactly is undefined (the probability formula divides by sigma^2
    and the score normalisation is scale-free), so the noise-free limit is
    approached with factor 1e-3.  Note this control is only meaningful with
    trajectory-mean centering: under conditional centering the deviation is
    identically zero as sigma -> 0.

One config is written per task, each with its own output CSV, so the suites can
be run by the parallel launcher without workers contending for a file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BATCH_CACHE_DIR = Path("results/fixed_batches/tanh_h512_modcog_revised8_taskpres")
CONFIG_ROOT = Path("configs")
RESULT_ROOT = Path("results")

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

# --- low-sparsity suite -----------------------------------------------------
LOW_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40"
LOW_AMOUNTS = (0.1, 0.2, 0.3, 0.4)
LOW_METHODS = (
    ("random", "random_unstructured", PRUNING_SEEDS),
    ("magnitude", "l1_unstructured", (0,)),
    ("obs", "obs_compensated", (0,)),
    ("lnp_mask", "vanilla_mask_only", (0,)),
    ("lnp_rescale", "noise_prune", PRUNING_SEEDS),
    ("snp_mask", "simulation_noise_prune_mask_only", PRUNING_SEEDS),
    ("snp_rescale", "simulation_noise_prune_rescale", PRUNING_SEEDS),
)
# S-NP rescale re-run at the paper's sparsities purely to record `kept_edges`.
TOPK_AMOUNTS = (0.5, 0.6, 0.7, 0.8)

# --- noise-necessity suite --------------------------------------------------
NOISE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80"
NOISE_AMOUNTS = (0.5, 0.6, 0.7, 0.8)
NOISE_FACTORS = (0.001, 0.25, 0.5, 1.0, 2.0)   # 1.0 reproduces the paper's setting
NOISE_METHODS = (
    ("snp_mask", "simulation_noise_prune_mask_only"),
    ("snp_rescale", "simulation_noise_prune_rescale"),
)
NOISE_PRUNING_SEED = 0   # matches the existing S-NP ablation convention

SNP_STRATEGIES = {
    "simulation_noise_prune_mask_only",
    "simulation_noise_prune_rescale",
}
REQUIRED_DEFAULT_KEYS = {
    "hidden_size", "train_steps", "ft_steps", "last_only", "device",
    "movement_batches", "ng_T", "ng_B", "eval_sample_batches",
}


def checkpoint_path(task: str, seed: int) -> str:
    return (f"checkpoints/tanh_h512_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{seed}/"
            f"modcog_{task}_seed{seed}.pt")


def eval_batches_path(task: str) -> str:
    return str(BATCH_CACHE_DIR / "eval" / f"{task}_eval_seed200000.pt")


def score_batches_path(task: str, seed: int) -> str:
    return str(BATCH_CACHE_DIR / "score" / f"{task}_score_seed{100_000 + seed}.pt")


def validate_inputs() -> None:
    missing = []
    for seed in NETWORK_SEEDS:
        for task, _t, _n in TASKS:
            need = [checkpoint_path(task, seed), eval_batches_path(task)]
            need += [score_batches_path(task, s) for s in PRUNING_SEEDS]
            missing += [p for p in need if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def defaults_block() -> dict:
    return {
        "reset_results": False, "resume": True, "model_type": "ctrnn",
        "hidden_size": 512, "activation": "tanh",
        "train_steps": 0, "ft_steps": 0, "skip_training": True,
        "last_only": False, "eval_last_only": False, "device": "cpu",
        "movement_batches": 20, "ng_T": 0, "ng_B": 256,
        "eval_sample_batches": 128,
        "eval_steps_pre0": 100, "eval_steps_pre": 100,
        "eval_steps_post0": 100, "eval_steps_post": 100,
        "noise_sigma": 1.0, "noise_eps": 0.3, "noise_leak_shift": 0.0,
        "noise_matched_diagonal": False,
        "sim_np_sigma": None, "sim_np_sigma_source": "natural_voltage",
        "sim_np_sigma_factor": 1.0,
        "sim_np_observable_space": "rate", "sim_np_inject_space": "rate",
        "sim_np_centering": "trajectory_mean",
        "sim_np_max_samples": 25000, "sim_np_burn_in_steps": 300,
        # OBS options are deliberately NOT set here: the runner rejects unknown
        # kwargs for non-OBS strategies, and the pruner's own defaults already
        # match the paper (damping 1e-3, max_samples 25,000).
    }


def _common(task: str, ng_t: int, seed: int) -> dict:
    return {
        "task": f"modcog:{task}", "ng_T": ng_t,
        "load_model_path": checkpoint_path(task, seed),
        "source_model_label": f"{task}_tanh_h512_12k_lr0006_seqbest_no_l2_seed{seed}",
        "source_run_id": f"continue_modcog_{task}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{seed}",
        "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
        "eval_seed": 200_000, "eval_batches_path": eval_batches_path(task),
    }


def low_sparsity_suite(task: str, ng_t: int) -> dict:
    runs = []
    for seed in NETWORK_SEEDS:
        common = _common(task, ng_t, seed)
        runs.append({"run_id": f"lowsp_{task}_netseed{seed}_baseline", "strategy": "none",
                     "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for label, strategy, seeds in LOW_METHODS:
            amounts = LOW_AMOUNTS
            if strategy == "simulation_noise_prune_rescale":
                amounts = LOW_AMOUNTS + TOPK_AMOUNTS
            for amount in amounts:
                for ps in seeds:
                    run = {
                        "run_id": f"lowsp_{task}_netseed{seed}_{label}_p{int(amount*100)}_pruneseed{ps}",
                        "strategy": strategy, "amount": amount,
                        "seed": ps, "pruning_seed": ps, "noise_rng_seed": ps, **common,
                    }
                    if strategy in SNP_STRATEGIES:
                        run["score_batch_seed"] = 100_000 + ps
                        run["score_batches_path"] = score_batches_path(task, ps)
                    runs.append(run)
    return {"run_id": f"{LOW_STEM}_{task}",
            "output_csv": str(RESULT_ROOT / LOW_STEM / f"{task}.csv"),
            "defaults": defaults_block(), "runs": runs}


def noise_necessity_suite(task: str, ng_t: int) -> dict:
    runs = []
    for seed in NETWORK_SEEDS:
        common = _common(task, ng_t, seed)
        runs.append({"run_id": f"noisenec_{task}_netseed{seed}_baseline", "strategy": "none",
                     "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for label, strategy in NOISE_METHODS:
            for factor in NOISE_FACTORS:
                tag = f"sf{str(factor).replace('.','p')}"
                for amount in NOISE_AMOUNTS:
                    runs.append({
                        "run_id": f"noisenec_{task}_netseed{seed}_{label}_{tag}_p{int(amount*100)}",
                        "strategy": strategy, "amount": amount,
                        "seed": NOISE_PRUNING_SEED, "pruning_seed": NOISE_PRUNING_SEED,
                        "noise_rng_seed": NOISE_PRUNING_SEED,
                        "sim_np_sigma_factor": factor,
                        "score_batch_seed": 100_000 + NOISE_PRUNING_SEED,
                        "score_batches_path": score_batches_path(task, NOISE_PRUNING_SEED),
                        **common,
                    })
    return {"run_id": f"{NOISE_STEM}_{task}",
            "output_csv": str(RESULT_ROOT / NOISE_STEM / f"{task}.csv"),
            "defaults": defaults_block(), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--suite", choices=("low_sparsity", "noise_necessity", "both"), default="both")
    args = ap.parse_args()
    if args.validate_inputs:
        validate_inputs()

    builders = []
    if args.suite in ("low_sparsity", "both"):
        builders.append((LOW_STEM, low_sparsity_suite))
    if args.suite in ("noise_necessity", "both"):
        builders.append((NOISE_STEM, noise_necessity_suite))

    for stem, builder in builders:
        cfg_dir = CONFIG_ROOT / stem
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (RESULT_ROOT / stem).mkdir(parents=True, exist_ok=True)
        total = 0
        for task, _t, ng_t in TASKS:
            suite = builder(task, ng_t)
            missing = sorted(REQUIRED_DEFAULT_KEYS - set(suite["defaults"]))
            if missing:
                raise ValueError(f"defaults missing {missing}")
            (cfg_dir / f"{stem}_{task}.json").write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"])
        print(f"{stem}\n  configs -> {cfg_dir}\n  outputs -> {RESULT_ROOT / stem}/<task>.csv\n"
              f"  total runs: {total} ({total // len(TASKS)} per task)")


if __name__ == "__main__":
    main()
