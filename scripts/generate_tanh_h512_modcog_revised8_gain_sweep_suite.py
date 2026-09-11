#!/usr/bin/env python3
"""Generate a gain sweep for magnitude pruning.

Round 2 found that magnitude pruning followed by a single scalar restoring
``rho(W_rec)`` outperforms noise-prune at every sparsity.  That result is only
interpretable once one further question is answered: is *matching* the unpruned
spectral radius what helps, or does performance simply rise with gain across the
whole range, with the matched point nothing special?

The two readings lead to different responses to the reviewer:

* if retention peaks at or near ``rho / rho_0 = 1``, then magnitude pruning has
  a principled one-parameter correction and the matched-gain baseline the
  reviewer asked for is genuinely strong;
* if retention keeps climbing past 1, then "matched gain" is not a meaningful
  operating point, the arm is really just a tuned scalar, and it belongs in the
  same category as the fine-tuned baselines that are out of scope.

So this suite sweeps a *fixed* multiplicative gain over a range that brackets
the spectral-matching factor (which is x1.05 at 50% sparsity rising to x1.36 at
80%), applied to the same magnitude mask.

The sweep includes ``gain = 1.0``, which is the identity control: it must
reproduce the plain ``l1_unstructured`` arm exactly.  If it does, the only
difference between every arm in this family is the scalar, and no other code
path is implicated.

Scope matches round 2: one network seed, one pruning seed, eight tasks.
Descriptive only -- not the n = 24 inferential unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_gain_sweep_p50_80"
CONFIG_DIR = Path("configs/gain_sweep")
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
# Brackets the spectral factor at both sparsities (x1.05 at 50%, x1.36 at 80%)
# and extends well past it so a monotone response would be visible.
GAINS = (1.00, 1.05, 1.10, 1.20, 1.36, 1.60, 2.00)

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
        "run_id": f"gainsweep_{task_label}_netseed{NETWORK_SEED}_baseline",
        "strategy": "none", "amount": 0.0, "no_prune": True, "seed": NETWORK_SEED, **common,
    }]
    for amount in AMOUNTS:
        # reference point: the spectral convention, whose factor the sweep brackets
        runs.append({
            "run_id": (f"gainsweep_{task_label}_netseed{NETWORK_SEED}_spectral_"
                       f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
            "strategy": "l1_unstructured_gain", "gain_mode": "match_spectral_radius",
            "amount": amount, "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
            "noise_rng_seed": PRUNING_SEED, **common,
        })
        for gain in GAINS:
            runs.append({
                "run_id": (f"gainsweep_{task_label}_netseed{NETWORK_SEED}_{gain_tag(gain)}_"
                           f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                "strategy": "l1_unstructured_gain", "gain_mode": "fixed", "gain_value": gain,
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
    print("gain = 1.00 is the identity control and must match plain l1_unstructured exactly.")


if __name__ == "__main__":
    main()
