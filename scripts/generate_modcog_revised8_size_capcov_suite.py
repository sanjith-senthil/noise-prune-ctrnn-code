#!/usr/bin/env python3
"""Does capping restore the covariance contribution at H = 1024?

THE GAP THIS FILLS.  At H = 512 the covariance term is worth almost nothing once
survivors are rescaled (+0.015/+0.020/+0.012 at 50/60/70% sparsity) because
``E[W_hat_ij] = w_ij`` for *any* score -- the retention probability cancels out
of the expectation.  Capping the amplification at ``c`` breaks that
cancellation, ``E[W_hat_ij] = w_ij * min(1, p_ij * c)``, and the covariance's
contribution rises to +0.060/+0.091/+0.076 at q50 (Suite F).

Reviewer concern #1 is that none of this survives at a larger size, and the
existing H = 1024 data is consistent with that fear: the *uncapped* covariance
contribution collapses from +0.020 to +0.001 at 60% sparsity.  But the H = 1024
capped suite was run without a covariance-dropped capped arm, so every H = 1024
covariance comparison on disk is an *uncapped* one -- the regime we now know
understates the covariance at both sizes.  The central Suite F claim is
therefore untested at H = 1024.  This suite tests it.

WHAT IS NEW HERE.  The ``*_mag`` arms are
``simulation_noise_prune_capped_magnitude_rescale``: identical machinery, with
``p_ij`` proportional to ``|w_ij|`` alone instead of
``|w_ij| (C_ii + C_jj -/+ 2 C_ij)``.  Paired against the capped full-score arms
they isolate the covariance under the cap.  q50 and q60 pair against arms that
already exist in ``..._capped_q50_q60_p50_80``; q30 and q70 need both halves
because neither quantile was run at this size.

WHY q30 AND q70 ARE IN HERE TOO.  At H = 512 retention peaks at q50 and the
covariance contribution rises monotonically as the cap tightens (q70 -> q50 ->
q30 gives +0.053 -> +0.091 -> +0.151 at 60%).  At H = 1024 the two quantiles on
disk run the other way -- q60 beats q50 -- which would mean the optimal cap
*loosens* with size, exactly the size-dependence the reviewer asked about.  Two
points cannot distinguish "the optimum moved to q60" from "the optimum moved
past q70".  q30 and q70 bracket it.

Pruning seed 0 only, matching the H = 512 Suite F protocol: n = 24 units =
8 tasks x 3 network seeds.  600 runs (576 pruned + 24 unpruned baselines).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "size_taskpres", Path(__file__).with_name("generate_modcog_revised8_size_taskpres_suite.py"))
_st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_st)

TASKS = _st.TASKS
NETWORK_SEEDS = _st.NETWORK_SEEDS
AMOUNTS = _st.AMOUNTS
PRUNING_SEED = 0

FULL_CAPPED = "simulation_noise_prune_capped_rescale"
MAG_CAPPED = "simulation_noise_prune_capped_magnitude_rescale"

# (label, strategy, cap quantile)
#   *_mag at q50/q60 pairs against the existing capped suite.
#   q30 / q70 are run as complete pairs because neither exists at this size.
# Ordered by decisiveness, and emitted arm-major (all three network seeds of one
# arm before the next arm starts) so a partial run answers the central question
# across every task rather than answering every question on a few tasks. The
# harness appends to the unit CSV per run and resumes per run_id, so this
# ordering costs nothing and an interrupted suite resumes exactly where it
# stopped.
#   q50_mag / q60_mag  pair against arms already on disk -> the covariance
#                      contribution under a cap, the reason this suite exists
#   q70_*              brackets the optimum on the loose side, where the two
#                      H=1024 points on disk (q60 > q50) say it has moved
#   q30_*              brackets it on the tight side, and matches the H=512
#                      cap-tightness trend
ARMS = (
    ("q50_mag",  MAG_CAPPED,  0.50),
    ("q60_mag",  MAG_CAPPED,  0.60),
    ("q70_full", FULL_CAPPED, 0.70),
    ("q70_mag",  MAG_CAPPED,  0.70),
    ("q30_full", FULL_CAPPED, 0.30),
    ("q30_mag",  MAG_CAPPED,  0.30),
)


def suite_stem(h: int) -> str:
    return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_capcov_p50_80"


def config_dir(h: int) -> Path:
    return Path(f"configs/{suite_stem(h)}")


def make_unit(h, task_label, task, ng_t, amounts=AMOUNTS, include_baseline=True, shard=None):
    def common_for(seed):
        return {
            "task": task, "ng_T": ng_t,
            "load_model_path": _st.checkpoint_path(h, task_label, seed),
            "source_model_label": f"{task_label}_tanh_h{h}_12k_lr0006_seqbest_no_l2_seed{seed}",
            "source_run_id": (f"continue_modcog_{task_label}_tanh_h{h}_to12k_lr0006"
                              f"_seqbest_no_l2_seed{seed}"),
            "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000,
            "eval_batches_path": _st.eval_batches_path(task_label),
        }

    runs = []
    if include_baseline:
        for seed in NETWORK_SEEDS:
            runs.append({"run_id": f"capcov{h}_{task_label}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **common_for(seed)})
    for label, strategy, cap_q in ARMS:
        for seed in NETWORK_SEEDS:
            common = common_for(seed)
            for amount in amounts:
                runs.append({
                    "run_id": (f"capcov{h}_{task_label}_netseed{seed}_{label}_"
                               f"p{int(amount * 100)}_pruneseed{PRUNING_SEED}"),
                    "strategy": strategy, "amount": amount,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED,
                    "prob_control_seed": PRUNING_SEED,
                    "score_batch_seed": 100_000 + PRUNING_SEED,
                    "score_batches_path": _st.score_batches_path(task_label, PRUNING_SEED),
                    "rescale_cap_mode": "quantile", "rescale_cap_quantile": cap_q,
                    **common,
                })
    return {"run_id": f"{suite_stem(h)}_{task_label}{_st.shard_suffix(shard)}",
            "output_csv": f"results/{suite_stem(h)}/{task_label}{_st.shard_suffix(shard)}.csv",
            "defaults": _st.defaults_block(h), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--shard-by-sparsity", action="store_true")
    args = ap.parse_args()
    h = args.hidden_size
    _st.validate(h, require_checkpoints=True)

    cfg_dir = config_dir(h); cfg_dir.mkdir(parents=True, exist_ok=True)
    Path(f"results/{suite_stem(h)}").mkdir(parents=True, exist_ok=True)
    total = n_cfg = 0
    for task_label, task, ng_t in TASKS:
        shards = [(a,) for a in AMOUNTS] if args.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            shard = amounts[0] if args.shard_by_sparsity else None
            u = make_unit(h, task_label, task, ng_t, amounts=amounts,
                          include_baseline=(i == 0), shard=shard)
            (cfg_dir / f"{suite_stem(h)}_{task_label}{_st.shard_suffix(shard)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n_cfg += 1
    print(f"wrote {n_cfg} configs to {cfg_dir}")
    print(f"  H = {h}; total runs {total}; arms {[a[0] for a in ARMS]}; pruning seed {PRUNING_SEED}")


if __name__ == "__main__":
    main()
