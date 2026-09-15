#!/usr/bin/env python3
"""Can the S-NP covariance estimate be made more informative?

Motivation
----------
The covariance term currently contributes little.  Dropping it entirely
(magnitude-proportional probabilities at the same expected density) costs only
+0.014 / +0.023 / +0.017 retention at 50 / 60 / 70% sparsity.  Measuring the
estimator directly says why: at the paper's settings the empirical covariance is
very nearly isotropic, so ``C_ii + C_jj -/+ 2 C_ij`` is very nearly a constant
and ``p_ij ∝ |w_ij| x const`` -- i.e. the method degenerates towards
magnitude-proportional sampling.  Measured on the frozen score batches
(20 rollouts, netseed 0), coefficient of variation of that covariance factor
across edges, and mean |off-diagonal| / mean diagonal:

    task                paper est.  CV      off/diag
    dm1seqr                         0.115   0.053
    dmsintseq                       0.078   0.056
    ctxdlydm1intseq                 0.237   0.095
    multidlydmintseq                0.344   0.092

Three defects feed that, and they interact, so they are crossed here rather
than swept one at a time.

1. **The window is one timestep.**  ``burn_in_steps`` defaults to 300 while a
   Mod-Cog trial is 30-40 steps, so ``start = min(300, T-1) = T-1`` and the
   covariance is estimated from the *final timestep only*.  97% of every
   simulated trajectory is discarded, and the retained slice is the one where
   task drive is least informative.

2. **Centering mixes noise response with task-condition variance.**  At the
   terminal step, ``trajectory_mean`` subtracts the across-trial mean, so what
   is left is trial-to-trial variability -- injected noise *plus* condition
   differences.  ``conditional`` subtracts the matched noise-free trajectory and
   isolates the noise-induced deviation, which is the quantity the Lyapunov
   theory that S-NP replaces is actually about.

3. **The observable barely sees the connectivity.**  Noise is injected into
   *rate* at a scale taken from *voltage* variability -- a units mismatch that
   makes sigma 1.2-2.4x too large (mean 1.54x), and 0.85-2.9x the noise-free
   rate std.  Worse, injecting and observing in the same space means most of
   the measured variance is the freshly injected noise, which carries no
   connectivity information: with alpha = 0.1 the recurrently propagated part is
   O(alpha rho(W)) of the total.  Observing *voltage* while injecting into
   *rate* means every unit of measured covariance has passed through W once.
   Measured (full trial, conditional centering, CV / off-diag ratio):

    task                inject->observe   CV      off/diag
    dm1seqr             rate->rate        0.116   0.049
                        volt->volt        0.685   0.100
                        rate->volt        1.209   0.181
    ctxdlydm1intseq     rate->rate        0.063   0.032
                        rate->volt        0.654   0.155
    multidlydmintseq    rate->rate        0.033   0.017
                        rate->volt        0.467   0.101

   10-14x more edge-to-edge structure, and 2-6x more cross-covariance.

Budget
------
Every arm gets the same *rollout* budget (98, the paper's effective count:
ceil(25000 / 256)).  The sample cap cannot be used to compare windows, because
under a 25,000-sample cap a whole-trial window is reached in 3 rollouts against
98 for a one-step window -- a 32x difference in independent noise realisations,
and only 3 of the 20 frozen score batches touched.  ``sim_np_num_rollouts``
fixes rollouts and lets the sample count follow, so the arms differ in what is
measured and not in how much simulation paid for it.  ``arm_paper`` deliberately
keeps the legacy sample-capped path so it regresses bit-for-bit against the
frozen results; ``arm_budget_bridge`` is the same estimator on the new budget
path and isolates any effect of the budget change itself.

A fourth defect, found while smoke-testing the arms above
-----------------------------------------------------------
Retention probabilities are normalised by scaling to the target density and
then *clipping* at 1. The clipped mass is simply lost, so the expected retained
density lands below target -- and it lands further below the more structured the
score vector is, because structure means a heavier tail. On dm1seqr at 60%:

    arm                 edges clipped at p=1   realised / target edges
    paper                       9,438                 0.980
    cond_full                  11,464                 0.969
    cond_full_vv               14,952                 0.886
    cond_full_rv               21,021                 0.764

The exact-density top-k then pads the mask with zero-weight entries, so
``post_sparsity_recurrent`` still reads 0.60 while the network actually carries
76% of the edges it should -- effectively 69.5% sparsity, not 60%. That biases
the comparison directly against the estimators this suite is trying to test, so
every arm is crossed with ``sim_np_prob_normalize``: ``clip`` (as published) and
``waterfill``, which pins over-1 entries at 1 and redistributes their excess
over the rest, holding sum(p) to the target to 8e-16.

Scope
-----
50 / 60 / 70% only -- the range the revision cares about -- at one pruning seed.
This is a screen: the arms that win here get the full three-pruning-seed
treatment afterwards.  n = 24 trained networks (8 tasks x 3 network seeds).
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

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covquality_p50_70"
CONFIG_DIR = Path("configs/cov_quality")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM

TASKS = _rs.TASKS
NETWORK_SEEDS = _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7)
PRUNING_SEED = 0

# The paper's effective rollout count: ceil(25_000 / ng_B) with ng_B = 256.
ROLLOUTS = 98

# label, burn_in, centering, observable_space, inject_space, sigma_source,
# sigma_factor, zero_task_input, num_rollouts (None = legacy sample cap)
NORMALIZERS = ("clip", "waterfill")

# Suffix appended to the arm label for each normaliser; "clip" is the published
# behaviour and keeps the bare label so `paper` remains the regression anchor.
NORM_SUFFIX = {"clip": "", "waterfill": "wf"}

ARMS = (
    # -- references ----------------------------------------------------------
    ("paper",            300, "trajectory_mean", "rate",    "rate",    "natural_voltage", 1.0,  False, None),
    ("budget_bridge",    300, "trajectory_mean", "rate",    "rate",    "natural_voltage", 1.0,  False, ROLLOUTS),
    # -- defect 1 and 2: window x centering, at the paper's spaces -----------
    ("cond_term",        300, "conditional",     "rate",    "rate",    "natural_voltage", 1.0,  False, ROLLOUTS),
    ("traj_full",          0, "trajectory_mean", "rate",    "rate",    "natural_voltage", 1.0,  False, ROLLOUTS),
    ("cond_full",          0, "conditional",     "rate",    "rate",    "natural_voltage", 1.0,  False, ROLLOUTS),
    # -- defect 3: what is injected, and what is observed --------------------
    ("cond_full_vv",       0, "conditional",     "voltage", "voltage", "natural_voltage", 1.0,  False, ROLLOUTS),
    ("cond_full_rv",       0, "conditional",     "voltage", "rate",    "natural_voltage", 1.0,  False, ROLLOUTS),
    ("cond_full_vr",       0, "conditional",     "rate",    "voltage", "natural_voltage", 1.0,  False, ROLLOUTS),
    # -- noise scale, crossed with the two strongest spaces -------------------
    ("cond_full_rr_sf050", 0, "conditional",     "rate",    "rate",    "natural_voltage", 0.5,  False, ROLLOUTS),
    ("cond_full_rr_sf025", 0, "conditional",     "rate",    "rate",    "natural_voltage", 0.25, False, ROLLOUTS),
    ("cond_full_rv_sf050", 0, "conditional",     "voltage", "rate",    "natural_voltage", 0.5,  False, ROLLOUTS),
    ("cond_full_rv_sf025", 0, "conditional",     "voltage", "rate",    "natural_voltage", 0.25, False, ROLLOUTS),
    ("cond_full_vv_sf025", 0, "conditional",     "voltage", "voltage", "natural_voltage", 0.25, False, ROLLOUTS),
    # -- does the window still matter once the observable carries structure? --
    ("cond_term_rv",     300, "conditional",     "voltage", "rate",    "natural_voltage", 1.0,  False, ROLLOUTS),
    # -- units-consistent sigma: a principled default rather than a tuned one -
    ("cond_full_rr_sigrate", 0, "conditional",   "rate",    "rate",    "natural_rate",    1.0,  False, ROLLOUTS),
    ("cond_full_rv_sigrate", 0, "conditional",   "voltage", "rate",    "natural_rate",    1.0,  False, ROLLOUTS),
    # -- is task drive needed once the estimator is fixed? --------------------
    ("cond_full_rr_noinput", 0, "conditional",   "rate",    "rate",    "natural_voltage", 1.0,  True,  ROLLOUTS),
    ("cond_full_rv_noinput", 0, "conditional",   "voltage", "rate",    "natural_voltage", 1.0,  True,  ROLLOUTS),
)


def arm_label(label: str, normalize: str) -> str:
    suffix = NORM_SUFFIX[normalize]
    return label if not suffix else f"{label}__{suffix}"


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
            runs.append({"run_id": f"covq_{task}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **common})
        for (label, burn, centering, obs, inj, sigsrc, sf, zero_in, nroll) in ARMS:
          for normalize in NORMALIZERS:
            full_label = arm_label(label, normalize)
            for amount in amounts:
                run = {
                    "run_id": (f"covq_{task}_netseed{seed}_{full_label}_"
                               f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                    "strategy": "simulation_noise_prune_rescale", "amount": amount,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED,
                    "sim_np_burn_in_steps": burn,
                    "sim_np_centering": centering,
                    "sim_np_observable_space": obs,
                    "sim_np_inject_space": inj,
                    "sim_np_sigma_source": sigsrc,
                    "sim_np_sigma_factor": sf,
                    "sim_np_zero_task_input": zero_in,
                    "sim_np_prob_normalize": normalize,
                    "score_batch_seed": 100_000 + PRUNING_SEED,
                    "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED),
                    **common,
                }
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
    print(f"total runs: {total}   arms: {len(ARMS) * len(NORMALIZERS)} "
          f"({len(ARMS)} estimators x {len(NORMALIZERS)} normalisers)")
    print(f"analysis units: {len(TASKS) * len(NETWORK_SEEDS)} trained networks x "
          f"{len(AMOUNTS)} sparsities, {PRUNING_SEED} as the single pruning seed (n = 24)")


if __name__ == "__main__":
    main()
