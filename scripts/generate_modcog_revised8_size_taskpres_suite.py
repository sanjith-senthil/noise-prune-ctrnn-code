#!/usr/bin/env python3
"""Repeat the paper's task-preservation suite at a different hidden size.

Reviewer 1, comment 4 asks for a second network size. This emits the *same seven
methods at the same four sparsities with the same pruning-seed counts* as the
paper's main suite, changing only ``hidden_size`` and the checkpoint paths, so
the size axis is added without introducing a second experimental design.

Method set and seed counts, copied from the paper's suite exactly
-----------------------------------------------------------------
    lnp_rescale       noise_prune                         3 pruning seeds
    snp_rescale       simulation_noise_prune_rescale       3
    snp_mask          simulation_noise_prune_mask_only     3
    random            random_unstructured                  3
    lnp_mask          vanilla_mask_only                    1  (deterministic)
    magnitude         l1_unstructured                      1  (deterministic)
    obs_compensated   obs_compensated                      1  (deterministic)

Plus the covariance ablation, carried over from the reviewer-1 controls:

    lnp_magnitude     noise_prune_magnitude_rescale        3
    snp_magnitude     simulation_noise_prune_magnitude_rescale  3

These drop the covariance factor from ``p_ij``, leaving it proportional to
``|w_ij|`` alone. At H = 512 the covariance contributes only +0.02 to +0.04 and
is not significant at 80%; running the same ablation here tests whether that
contribution *grows* with network size, which is what the theory predicts
(``K = 8 ln N / (eps^2 sigma^2)`` tightens with N).

24 networks x 4 sparsities gives 2,016 pruned runs plus 24 unpruned baselines.

Read this suite as a self-contained comparison *within* H = 1024
----------------------------------------------------------------
H = 1024 trains far better than H = 512 under the same budget (+0.27 mean
sequence accuracy, with five of eight tasks at or above 0.997). Retention is
normalised to the unpruned network, so a direct cross-size retention comparison
would confound method quality with how much headroom each size has. Every
comparison drawn from this suite is therefore between methods *at a fixed size*,
where the unpruned baseline is common to all arms and cancels.

Evaluation data is shared with H=512, deliberately
--------------------------------------------------
The cached batches are task inputs and targets with shape
``(time, batch, input_dim/output_dim)`` -- no hidden dimension -- so they are
hidden-size independent and are reused verbatim rather than regenerated. Both
sizes are therefore scored on *identical* evaluation and covariance-estimation
data, which removes one source of difference from the size comparison. The
paths keep their ``tanh_h512_...`` directory name because that is where they
live; nothing about their content is H=512-specific.

Safety
------
Refuses ``--hidden-size 512``: that suite already exists and is frozen. Also
refuses to emit any run whose checkpoint path does not exist, so a missing or
still-training network fails at generation rather than producing a silent gap
in the results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
PRUNING_SEEDS = (0, 1, 2)
BATCH_CACHE = Path("results/fixed_batches/tanh_h512_modcog_revised8_taskpres")

OBS_OPTIONS = {
    "obs_damping": 0.001, "obs_max_samples": 25000,
    "obs_compensation_mode": "diagonal", "obs_exact_block_threshold": 128,
}

# (label, strategy, pruning seeds, needs_score_batches, needs_noise_rng, extras)
METHODS = (
    ("lnp_rescale", "noise_prune", PRUNING_SEEDS, False, True, {}),
    ("snp_rescale", "simulation_noise_prune_rescale", PRUNING_SEEDS, True, True, {}),
    ("snp_mask", "simulation_noise_prune_mask_only", PRUNING_SEEDS, True, True, {}),
    ("random", "random_unstructured", PRUNING_SEEDS, False, False, {}),
    ("lnp_mask", "vanilla_mask_only", (0,), False, False, {}),
    ("magnitude", "l1_unstructured", (0,), False, False, {}),
    ("obs_compensated", "obs_compensated", (0,), True, False, OBS_OPTIONS),
    ("lnp_magnitude", "noise_prune_magnitude_rescale", PRUNING_SEEDS, False, True, {}),
    ("snp_magnitude", "simulation_noise_prune_magnitude_rescale", PRUNING_SEEDS, True, True, {}),
)
# Arms whose retention probabilities are replaced by a control; they take a
# dedicated seed so the control draw is reproducible independently of the
# Bernoulli stream. Base value matches the H=512 control suites.
CONTROL_STRATEGIES = {
    "noise_prune_magnitude_rescale",
    "simulation_noise_prune_magnitude_rescale",
}
PROB_CONTROL_SEED_BASE = 900_000


def suite_stem(h: int) -> str:
    return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_revised_task_only_p50_80"


def checkpoint_path(h: int, task_label: str, seed: int) -> str:
    return (f"checkpoints/tanh_h{h}_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{seed}/"
            f"modcog_{task_label}_seed{seed}.pt")


def eval_batches_path(task_label: str) -> str:
    return str(BATCH_CACHE / "eval" / f"{task_label}_eval_seed200000.pt")


def score_batches_path(task_label: str, pruning_seed: int) -> str:
    return str(BATCH_CACHE / "score" / f"{task_label}_score_seed{100_000 + pruning_seed}.pt")


def defaults_block(h: int) -> dict:
    """The paper's task-preservation defaults, with hidden_size as the only change."""
    return {
        "reset_results": False, "resume": True, "model_type": "ctrnn",
        "hidden_size": h, "activation": "tanh",
        "train_steps": 0, "ft_steps": 0, "skip_training": True,
        "last_only": False, "eval_last_only": False, "device": "cpu",
        "movement_batches": 20, "ng_T": 0, "ng_B": 256,
        "eval_sample_batches": 128,
        "eval_steps_pre0": 100, "eval_steps_pre": 100,
        "eval_steps_post0": 100, "eval_steps_post": 100,
        "noise_sigma": 1.0, "noise_eps": 0.3, "noise_leak_shift": 0.0,
        "noise_matched_diagonal": False,
        "sim_np_sigma": None, "sim_np_sigma_source": "natural_voltage",
        "sim_np_observable_space": "rate", "sim_np_inject_space": "rate",
        "sim_np_centering": "trajectory_mean", "sim_np_max_samples": 25000,
        "sim_np_burn_in_steps": 300,
        # NOTE: the obs_* options deliberately do NOT live here, even though the
        # paper's own config file puts them in `defaults`. `defaults` merges into
        # EVERY run, and the runner only consumes obs_damping / obs_max_samples /
        # obs_compensation_mode / obs_exact_block_threshold inside the
        # `strategy == "obs_compensated"` branch (runner.py ~line 358). For any
        # other strategy they survive as unconsumed kwargs and the run dies with
        # "Unsupported keyword arguments for run_prune_experiment: obs_damping".
        # They are attached per-run to the OBS arm instead, via METHODS extras.
    }


def shard_suffix(amount) -> str:
    return "" if amount is None else f"_p{int(amount * 100)}"


def validate(h: int, require_checkpoints: bool) -> None:
    if h == 512:
        raise SystemExit("Refusing H=512: that suite exists and is frozen.")
    missing = []
    for seed in NETWORK_SEEDS:
        for t, _task, _ng in TASKS:
            if require_checkpoints and not Path(checkpoint_path(h, t, seed)).exists():
                missing.append(checkpoint_path(h, t, seed))
            if not Path(eval_batches_path(t)).exists():
                missing.append(eval_batches_path(t))
            for s in PRUNING_SEEDS:
                if not Path(score_batches_path(t, s)).exists():
                    missing.append(score_batches_path(t, s))
    if missing:
        raise SystemExit(
            f"{len(set(missing))} required file(s) absent — refusing to emit a suite with "
            f"silent gaps:\n" + "\n".join(sorted(set(missing))[:12]))


def make_unit(h: int, task_label: str, task: str, ng_t: int,
              amounts=AMOUNTS, include_baseline=True, shard=None) -> dict:
    runs = []
    for seed in NETWORK_SEEDS:
        common = {
            "task": task, "ng_T": ng_t,
            "load_model_path": checkpoint_path(h, task_label, seed),
            "source_model_label": f"{task_label}_tanh_h{h}_12k_lr0006_seqbest_no_l2_seed{seed}",
            "source_run_id": (f"continue_modcog_{task_label}_tanh_h{h}_to12k_lr0006"
                              f"_seqbest_no_l2_seed{seed}"),
            "source_network_seed": seed,
            "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000,
            "eval_batches_path": eval_batches_path(task_label),
        }
        if include_baseline:
            runs.append({
                "run_id": f"taskpres_h{h}_{task_label}_netseed{seed}_baseline",
                "strategy": "none", "amount": 0.0, "no_prune": True,
                "seed": seed, **common,
            })
        for label, strategy, seeds, needs_score, needs_rng, extras in METHODS:
            for amount in amounts:
                for pseed in seeds:
                    run = {
                        "run_id": (f"taskpres_h{h}_{task_label}_netseed{seed}_{label}_"
                                   f"p{int(amount * 100)}_pruneseed{pseed}"),
                        "strategy": strategy, "amount": amount,
                        "seed": pseed, "pruning_seed": pseed, **common, **extras,
                    }
                    if needs_rng:
                        run["noise_rng_seed"] = pseed
                    if needs_score:
                        run["score_batch_seed"] = 100_000 + pseed
                        run["score_batches_path"] = score_batches_path(task_label, pseed)
                    if strategy in CONTROL_STRATEGIES:
                        run["prob_control_seed"] = PROB_CONTROL_SEED_BASE + pseed
                    runs.append(run)
    return {
        "run_id": f"{suite_stem(h)}_{task_label}{shard_suffix(shard)}",
        "output_csv": f"results/{suite_stem(h)}/{task_label}{shard_suffix(shard)}.csv",
        "defaults": defaults_block(h),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--shard-by-sparsity", action="store_true",
                    help="one config per (task, sparsity): 32 units instead of 8")
    ap.add_argument("--skip-checkpoint-check", action="store_true",
                    help="emit configs before the networks finish training (for dry planning)")
    args = ap.parse_args()
    h = args.hidden_size
    validate(h, require_checkpoints=not args.skip_checkpoint_check)

    cfg_dir = Path(f"configs/{suite_stem(h)}")
    out_dir = Path(f"results/{suite_stem(h)}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task_label, task, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            u = make_unit(h, task_label, task, ng_t, amounts=amounts,
                          include_baseline=(i == 0), shard=shard)
            (cfg_dir / f"{suite_stem(h)}_{task_label}{shard_suffix(shard)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {cfg_dir}")
    print(f"  H = {h}; total runs {total}")
    print(f"  {len(METHODS)} methods x {len(AMOUNTS)} sparsities x "
          f"{len(TASKS) * len(NETWORK_SEEDS)} networks, paper seed counts")
    print(f"  evaluation + score batches shared with H=512 (hidden-size independent)")


if __name__ == "__main__":
    main()
