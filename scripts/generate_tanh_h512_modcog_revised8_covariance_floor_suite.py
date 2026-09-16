#!/usr/bin/env python3
"""Can the covariance be made to measure the network rather than the injected noise?

Diagnosis.  With rate noise injected and rate observed, the noise drawn at step
``t`` enters the observable without passing through ``W``, so exactly

    C_observed = C_propagated + sigma^2 I

Only ``C_propagated`` carries connectivity information.  Measured on the frozen
score batches, the floor is **27-84% of the covariance diagonal**:

    task               floor share at the paper's sigma
    dm1seqr                        0.76
    dmsintseq                      0.78
    ctxdlydm1intseq                0.80
    multidlydmintseq               0.84

Because the score uses the covariance multiplicatively, through
``C_ii + C_jj -/+ 2 C_ij``, a large flat constant does not add noise to the
ranking -- it *dilutes* the informative variation.  That is why the score ends up
0.977 rank-correlated with ``|w|`` alone and the diff-cov heatmap shows
row/column striping (the additive ``C_ii + C_jj`` part) rather than pairwise
structure.

Crucially, **lowering sigma cannot fix this.**  In the linear regime the floor
and the propagated response both scale as sigma^2, so their ratio is
sigma-independent: on ctxdlydm1intseq the share is 0.797 at the paper's sigma and
0.784 at one-twentieth of it.  Reducing sigma helps only where tanh saturation is
compressing the propagated part (dm1seqr 0.76 -> 0.38, dmsintseq 0.78 -> 0.27).

Subtracting the floor does fix it.  Diagonal CV rises 3-6x and diff-cov CV 3-5x:

    task              diag CV raw -> floor-subtracted   diff-cov CV raw -> sub
    multidlydmintseq        0.032 -> 0.192                   0.033 -> 0.199
    ctxdlydm1intseq         0.063 -> 0.312                   0.063 -> 0.310
    dmsintseq               0.087 -> 0.401                   0.123 -> 0.572
    dm1seqr                 0.133 -> 0.552                   0.116 -> 0.482

and the score decorrelates from magnitude (Spearman 0.999 -> 0.975, 0.997 ->
0.946, 0.987 -> 0.801, 0.990 -> 0.880).

Design.  Every arm here is a **deterministic mask**.  Under 1/p rescaling the
pruned matrix has W as its mask-average for any score, so the score has only
second-order leverage and cannot discriminate between estimators; deterministic
selection is where a better covariance must show up if it is better at all.
Whatever wins here gets tested in the rescale arm afterwards.

n = 24 trained networks (8 tasks x 3 network seeds), one pruning seed, 50-70%.
``mag_mask`` and ``mask_paper`` must reproduce the frozen task-preservation suite.
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

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covfloor_p50_70"
CONFIG_DIR = Path("configs/cov_floor")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM

TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7)
PRUNING_SEED = 0
ROLLOUTS = 98          # the published effective budget: ceil(25_000 / ng_B=256)

MASK = "simulation_noise_prune_mask_only"

# label, strategy, burn_in, centering, observable, inject, sigma_factor, subtract_floor, rollouts
ARMS = (
    ("mag_mask",            "l1_unstructured", None, None, None, None, None, None, None),
    # published estimator, and the floor fix applied to it with nothing else changed
    ("mask_paper",          MASK, 300, "trajectory_mean", "rate", "rate", 1.0,  False, None),
    ("paper_floor",         MASK, 300, "trajectory_mean", "rate", "rate", 1.0,  True,  None),
    # whole trial + conditional centering, with and without the floor fix
    ("full_cond",           MASK,   0, "conditional",     "rate", "rate", 1.0,  False, ROLLOUTS),
    ("full_cond_floor",     MASK,   0, "conditional",     "rate", "rate", 1.0,  True,  ROLLOUTS),
    # sigma, which only helps by escaping saturation -- crossed with the fix
    ("full_cond_sf25_floor", MASK,  0, "conditional",     "rate", "rate", 0.25, True,  ROLLOUTS),
    ("full_cond_sf05_floor", MASK,  0, "conditional",     "rate", "rate", 0.05, True,  ROLLOUTS),
    # structural alternative: observe where the injection cannot land directly,
    # so the floor is zero by construction rather than by subtraction
    ("full_cond_rv",        MASK,   0, "conditional",  "voltage", "rate", 1.0,  False, ROLLOUTS),
)

SIM = {MASK}


def shard_suffix(a): return "" if a is None else f"_p{int(a * 100)}"
def config_path(t, a=None): return CONFIG_DIR / f"{SUITE_STEM}_{t}{shard_suffix(a)}.json"
def output_csv(t, a=None): return OUTPUT_DIR / f"{t}{shard_suffix(a)}.csv"


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
            runs.append({"run_id": f"covf_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for (label, strategy, burn, cen, obs, inj, sf, floor, nroll) in ARMS:
            for amount in amounts:
                run = {
                    "run_id": (f"covf_{task}_netseed{seed}_{label}_"
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
                        "sim_np_sigma_source": "natural_voltage",
                        "sim_np_sigma_factor": sf,
                        "sim_np_subtract_floor": floor,
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
    a = ap.parse_args()
    if a.validate_inputs:
        validate_inputs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in AMOUNTS] if a.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if a.shard_by_sparsity else None
            suite = make_task_suite(task, ng_t, amounts=amounts,
                                    include_baseline=(i == 0), shard=shard)
            config_path(task, shard).write_text(json.dumps(suite, indent=2) + "\n")
            total += len(suite["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {CONFIG_DIR}")
    print(f"total runs: {total}   arms: {len(ARMS)} (all deterministic masks)")


if __name__ == "__main__":
    main()
