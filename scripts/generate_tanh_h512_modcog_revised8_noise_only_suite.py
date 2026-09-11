#!/usr/bin/env python3
"""Noise-only control: is task input needed at all to estimate the covariance?

Reviewer 2, comment 4 asks whether the S-NP pruning scores reflect task-driven
or noise-induced activity, since the covariance is currently estimated with both
present.  Two controls bracket that question:

  task input, no injected noise   -> the sigma sweep (``noise_necessity``,
                                     ``sigma_factor = 0.001``)
  injected noise, no task input   -> **this suite**

The second is the sharper half.  If the method still works with the task input
zeroed, the covariance it relies on is genuinely noise-induced and the name
"noise-prune" is earned; if it collapses, the scores were reading task activity
and the injected noise was incidental.

Implementation note
-------------------
``zero_task_input`` zeroes the input *after* sigma is resolved.  Sigma is
estimated from the natural variability of the task-driven network, so zeroing
first would drive that estimate to ~0 and the arm would differ from its
reference in two ways at once.  Zeroing afterwards holds the injected noise
scale identical, so task drive is the only thing that changes.  Verified on
dm1seqr: sigma_used is bit-identical between the two arms, and the empirical
covariance trace collapses to 0.29 when *both* task input and noise are removed
(a fixed point), against 129.1 for task-only and 693.0 for noise-only.

Design
------
Both arms are run here at three pruning seeds rather than merging the reference
from another suite, so the paired comparison is self-contained and needs no
cross-suite assumption.  The reference arm's pruning-seed-0 rows additionally
reproduce the original ``noise_necessity`` suite exactly, which is the check
that this suite sits on the same footing as the ablation it extends.

8 tasks x 3 network seeds x 4 sparsities x 3 pruning seeds x 2 arms, giving
n = 24 trained networks after averaging pruning-seed replicates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_only_p50_80"
CONFIG_DIR = Path("configs/noise_only")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM

TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
AMOUNTS = _rs.NOISE_AMOUNTS
PRUNING_SEEDS = (0, 1, 2)
SIGMA_FACTOR = 1.0          # the paper's setting; the sweep lives in noise_necessity

# (label, zero_task_input)
ARMS = (
    ("snp_rescale_noiseonly", True),
    ("snp_rescale_taskplusnoise", False),   # reference; seed 0 regresses on noise_necessity
)


def shard_suffix(amount) -> str:
    return "" if amount is None else f"_p{int(amount * 100)}"


def config_path(task: str, amount=None) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task}{shard_suffix(amount)}.json"


def output_csv(task: str, amount=None) -> Path:
    return OUTPUT_DIR / f"{task}{shard_suffix(amount)}.csv"


def validate_inputs() -> None:
    missing = []
    for seed in NETWORK_SEEDS:
        for task, _t, _n in TASKS:
            need = [_rs.checkpoint_path(task, seed), _rs.eval_batches_path(task)]
            need += [_rs.score_batches_path(task, s) for s in PRUNING_SEEDS]
            missing += [p for p in need if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def make_task_suite(task: str, ng_t: int, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = _rs._common(task, ng_t, seed)
        if include_baseline:
            runs.append({"run_id": f"noiseonly_{task}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **common})
        for label, zero_input in ARMS:
            for pseed in PRUNING_SEEDS:
                for amount in amounts:
                    runs.append({
                        "run_id": (f"noiseonly_{task}_netseed{seed}_{label}_"
                                   f"p{int(amount * 100)}_pruneseed{pseed}"),
                        "strategy": "simulation_noise_prune_rescale", "amount": amount,
                        "seed": pseed, "pruning_seed": pseed, "noise_rng_seed": pseed,
                        "sim_np_sigma_factor": SIGMA_FACTOR,
                        "sim_np_zero_task_input": zero_input,
                        "score_batch_seed": 100_000 + pseed,
                        "score_batches_path": _rs.score_batches_path(task, pseed),
                        **common,
                    })
    return {"run_id": f"{SUITE_STEM}_{task}{shard_suffix(shard)}",
            "output_csv": str(output_csv(task, shard)),
            "defaults": _rs.defaults_block(), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--shard-by-sparsity", action="store_true")
    args = ap.parse_args()
    if args.validate_inputs:
        validate_inputs()
    print(f"defaults inherited from the noise_necessity generator "
          f"({len(_rs.defaults_block())} keys)")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task, _t, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            suite = make_task_suite(task, ng_t, amounts=amounts,
                                    include_baseline=(i == 0), shard=shard)
            config_path(task, shard).write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {CONFIG_DIR}")
    print(f"total runs: {total}   arms: {[a for a, _ in ARMS]}")
    print(f"analysis units: {len(TASKS) * len(NETWORK_SEEDS)} trained networks x "
          f"{len(AMOUNTS)} sparsities, {len(PRUNING_SEEDS)} pruning seeds averaged (n = 24)")


if __name__ == "__main__":
    main()
