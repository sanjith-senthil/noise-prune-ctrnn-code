#!/usr/bin/env python3
"""Can a better-estimated covariance raise accuracy -- and where does it show up?

The published covariance is estimated from the final timestep of each trial
(``burn_in_steps`` 300 against a 30-40 step trial), with trajectory-mean
centering, injecting and observing in the same (rate) space. Direct measurement
of the estimator shows the result is nearly isotropic, so the covariance factor
``C_ii + C_jj -/+ 2 C_ij`` is nearly constant across edges and ``p_ij`` collapses
towards ``|w_ij| x const``.

Estimating over the whole trial, with conditional centering, injecting into rate
and observing in *voltage* raises the spread of that factor from 1.4-2x to
2.4-12x (CV 0.12-0.34 -> 0.47-1.21) and drops the resulting mask's overlap with
pure magnitude from 0.90-0.95 to 0.71-0.89. This suite asks whether that buys
accuracy, in both of the settings where it could:

  deterministic   mask_paper   vs  mask_rv       (and vs mag_mask)
  rescaled        rescale_paper vs rescale_rv_wf

The split matters because the two settings have very different headroom. In the
frozen suite the covariance is worth +0.127/+0.165/+0.114/+0.060 at 50-80% when
the mask is taken deterministically (magnitude 0.748/0.416/0.188/0.087 against
S-NP mask 0.875/0.582/0.303/0.147), but only +0.026/+0.043/+0.038/+0.008 once
survivors are rescaled by 1/p (round-2 controls, covariance-dropped 0.923/0.784/
0.559/0.363 against full 0.949/0.827/0.597/0.371).

That gap has a mechanical cause. Under Bernoulli(p) retention with 1/p
rescaling the estimator is unbiased for any p, so E[W_pruned] = W regardless of
which edges survive; what p controls is variance. Minimising the total variance
Sum w^2 (1/p - 1) subject to Sum p = K gives p proportional to |w| exactly -- so
magnitude-proportional sampling is already the variance-optimal sampler, and the
covariance factor can only buy the difference between entrywise-optimal and
spectrally-optimal sampling. Deterministically there is no such compensation:
deleted mass is a systematic bias and edge choice is everything. If a better
covariance is worth anything, the deterministic arms are where it must appear.

``rescale_paper_wf`` additionally isolates the probability-normalisation fix
(water-filling instead of clipping) on the published estimator, since clipping
alone costs the published method 2.2% of its edge budget at 60%.

Both ``*_paper`` arms are configured identically to the frozen task-preservation
suite and must reproduce it bit-for-bit.

n = 24 trained networks (8 tasks x 3 network seeds), one pruning seed, 50-70%.
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

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covverdict_p50_70"
CONFIG_DIR = Path("configs/cov_verdict")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM

TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7)
PRUNING_SEED = 0
ROLLOUTS = 98          # the published effective budget: ceil(25_000 / ng_B=256)

SIM = ("simulation_noise_prune_mask_only", "simulation_noise_prune_rescale")

# label, strategy, burn_in, centering, observable, inject, sigma_source,
# sigma_factor, prob_normalize, num_rollouts
ARMS = (
    ("mag_mask",         "l1_unstructured",                  None, None, None, None, None, None, None, None),
    ("mask_paper",       "simulation_noise_prune_mask_only",  300, "trajectory_mean", "rate",    "rate", "natural_voltage", 1.0, "clip",      None),
    ("mask_rv",          "simulation_noise_prune_mask_only",    0, "conditional",     "voltage", "rate", "natural_voltage", 1.0, "clip",      ROLLOUTS),
    ("rescale_paper",    "simulation_noise_prune_rescale",    300, "trajectory_mean", "rate",    "rate", "natural_voltage", 1.0, "clip",      None),
    ("rescale_paper_wf", "simulation_noise_prune_rescale",    300, "trajectory_mean", "rate",    "rate", "natural_voltage", 1.0, "waterfill", None),
    ("rescale_rv_wf",    "simulation_noise_prune_rescale",      0, "conditional",     "voltage", "rate", "natural_voltage", 1.0, "waterfill", ROLLOUTS),
)


def shard_suffix(a) -> str:
    return "" if a is None else f"_p{int(a * 100)}"


def config_path(task, a=None) -> Path:
    return CONFIG_DIR / f"{SUITE_STEM}_{task}{shard_suffix(a)}.json"


def output_csv(task, a=None) -> Path:
    return OUTPUT_DIR / f"{task}{shard_suffix(a)}.csv"


def validate_inputs() -> None:
    missing = []
    for seed in NETWORK_SEEDS:
        for task, _t, _n in TASKS:
            need = [_rs.checkpoint_path(task, seed), _rs.eval_batches_path(task),
                    _rs.score_batches_path(task, PRUNING_SEED)]
            missing += [p for p in need if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(set(missing))))


def make_task_suite(task, ng_t, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = _rs._common(task, ng_t, seed)
        if include_baseline:
            runs.append({"run_id": f"covv_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for (label, strategy, burn, cen, obs, inj, src, sf, norm, nroll) in ARMS:
            for amount in amounts:
                run = {
                    "run_id": (f"covv_{task}_netseed{seed}_{label}_"
                               f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                    "strategy": strategy, "amount": amount,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED, **common,
                }
                if strategy in SIM:
                    run.update({
                        "sim_np_burn_in_steps": burn,
                        "sim_np_centering": cen,
                        "sim_np_observable_space": obs,
                        "sim_np_inject_space": inj,
                        "sim_np_sigma_source": src,
                        "sim_np_sigma_factor": sf,
                        "sim_np_prob_normalize": norm,
                        "score_batch_seed": 100_000 + PRUNING_SEED,
                        "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED),
                    })
                    if nroll is not None:
                        run["sim_np_num_rollouts"] = nroll
                runs.append(run)
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
    print(f"total runs: {total}   arms: {len(ARMS)}")


if __name__ == "__main__":
    main()
