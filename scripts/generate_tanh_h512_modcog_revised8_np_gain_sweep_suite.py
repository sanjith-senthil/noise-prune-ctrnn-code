#!/usr/bin/env python3
"""Generate a gain sweep for NOISE-PRUNE (companion to the magnitude gain sweep).

The magnitude sweep established that a deterministic mask has a sharply peaked
gain optimum near ``rho_pruned = rho_unpruned``.  Noise-prune already sits at
that point without being told to: its ``1 / p`` rescale makes the pruned matrix
an unbiased estimator of the original, and the measured ratio is 1.0005.  The
open question is whether that is where noise-prune actually performs *best*.

If noise-prune has its own optimum away from 1.0 -- plausible, since the ``1/p``
amplification injects variance that the cap-percentile sweep only partly
controls -- then the comparison against a gain-corrected magnitude baseline
should be made with each method at its own best scalar, not with one method
corrected and the other not.  If instead noise-prune peaks at 1.0, then it is
already operating at its optimum and no scalar will lift it.

The sweep brackets 1.0 on both sides, since noise-prune could be over- or
under-gained relative to its optimum.

``gain = 1.0`` is the identity control: it must reproduce plain ``noise_prune``
exactly, which has been verified directly (max |dW| = 0.000e+00).

Scope matches round 2: one network seed, one pruning seed, eight tasks.
Descriptive only -- not the n = 24 inferential unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_np_gain_sweep_p50_80"
CONFIG_DIR = Path("configs/np_gain_sweep")
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
NETWORK_SEED = 0
PRUNING_SEED = 0
AMOUNTS = (0.5, 0.8)
# Brackets 1.0 on both sides: noise-prune sits at rho ratio ~1.0005, so its
# optimum could lie below (over-gained by the 1/p tail) or above.
GAINS = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.36, 1.60)

REQUIRED_DEFAULT_KEYS = {
    "hidden_size", "train_steps", "ft_steps", "last_only", "device",
    "movement_batches", "ng_T", "ng_B", "eval_sample_batches",
}


def checkpoint_path(task_label: str) -> str:
    return (
        f"checkpoints/tanh_h512_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{NETWORK_SEED}/"
        f"modcog_{task_label}_seed{NETWORK_SEED}.pt"
    )


def eval_batches_path(task_label: str) -> str:
    return str(BATCH_CACHE_DIR / "eval" / f"{task_label}_eval_seed200000.pt")


def config_path(task_label: str) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task_label}.json"


def gain_tag(gain: float) -> str:
    return f"g{int(round(gain * 100)):03d}"


def validate_inputs() -> None:
    missing = [p for task, _t, _n in TASKS
               for p in (checkpoint_path(task), eval_batches_path(task))
               if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def validate_default_schema(defaults: dict) -> None:
    missing = sorted(REQUIRED_DEFAULT_KEYS - set(defaults))
    if missing:
        raise ValueError(f"Suite defaults missing required keys: {', '.join(missing)}")
    if defaults["eval_sample_batches"] <= 0:
        raise ValueError("eval_sample_batches must be > 0 for deterministic evaluation in suites.")


def defaults_block() -> dict:
    """Identical to rounds 1 and 2 so all three are directly comparable."""
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


def make_task_suite(task_label: str, task: str, ng_t: int) -> dict:
    common = {
        "task": task, "ng_T": ng_t,
        "load_model_path": checkpoint_path(task_label),
        "source_model_label": f"{task_label}_tanh_h512_12k_lr0006_seqbest_no_l2_seed{NETWORK_SEED}",
        "source_run_id": f"continue_modcog_{task_label}_tanh_h512_to12k_lr0006_seqbest_no_l2_seed{NETWORK_SEED}",
        "source_network_seed": NETWORK_SEED,
        "source_recurrent_l2_lambda": 0.0,
        "eval_seed": 200_000,
        "eval_batches_path": eval_batches_path(task_label),
    }
    runs = [{
        "run_id": f"npgainsweep_{task_label}_netseed{NETWORK_SEED}_baseline",
        "strategy": "none", "amount": 0.0, "no_prune": True, "seed": NETWORK_SEED, **common,
    }]
    for amount in AMOUNTS:
        # reference point: the spectral convention, whose factor the sweep brackets
        runs.append({
            "run_id": (f"npgainsweep_{task_label}_netseed{NETWORK_SEED}_spectral_"
                       f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
            "strategy": "noise_prune_gain", "gain_mode": "match_spectral_radius",
            "amount": amount, "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
            "noise_rng_seed": PRUNING_SEED, **common,
        })
        for gain in GAINS:
            runs.append({
                "run_id": (f"npgainsweep_{task_label}_netseed{NETWORK_SEED}_{gain_tag(gain)}_"
                           f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                "strategy": "noise_prune_gain", "gain_mode": "fixed", "gain_value": gain,
                "amount": amount, "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                "noise_rng_seed": PRUNING_SEED, **common,
            })
    return {
        "run_id": f"{SUITE_STEM}_{task_label}",
        "output_csv": str(OUTPUT_DIR / f"{task_label}.csv"),
        "defaults": defaults_block(),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    args = ap.parse_args()
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
    print(f"\ntotal runs: {total}   gains swept: {GAINS}   sparsities: {AMOUNTS}")
    print("gain = 1.00 is the identity control and must match plain noise_prune exactly.")


if __name__ == "__main__":
    main()
