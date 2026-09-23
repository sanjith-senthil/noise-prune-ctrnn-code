#!/usr/bin/env python3
"""Cap-quantile sweep at a second hidden size: does the cap transfer with N?

Reviewer 1, comment 4 asks whether results hold at another network size. The
paper's cap-percentile curve peaks at q50-q60 at H = 512, and the revision plan
argues the cap value has a *size-independent anchor*: probabilities are
normalised so ``sum(p) = density * N(N-1)``, hence ``1/p = (1/density) *
(1/shape)`` and the cap at quantile q is ``c(q)/density`` with density a design
parameter rather than a function of N. ``cap_value * density`` is constant to
machine precision within every network (spread 6.7e-16). If that reasoning is
right, q50-q60 should remain the useful range at H = 1024.

This suite tests it directly.

CORRECTION (2026-09-22): an earlier version of this docstring claimed the
amplification tail is far heavier at H = 1024, so that capping would have more
to bite on. That is backwards. Mean ``prune_inv_p_mean`` is **23.6 at H = 1024
against 38.8 at H = 512** at 50% sparsity, and the per-network maxima are
comparable; amplification is *milder* at the larger size. Suite G confirms the
consequence: the optimal cap quantile moves q50 -> q60 and a q30 cap, which is
the best cap for the covariance at H = 512, is significantly harmful here.

Arms
----
    lnp_capped_q50 / q60   noise_prune_capped_rescale                3 seeds
    snp_capped_q50 / q60   simulation_noise_prune_capped_rescale     3 seeds
    snp_rescale            uncapped reference, pruning seed 0 only -- regression
                           against the completed H=1024 task-preservation suite,
                           which is what licenses comparing capped against
                           uncapped across the two suites.

Leak-shift budget
-----------------
The defaults are inherited from the task-preservation generator, which sets
``noise_max_attempts = 8``. That matters here: at H = 1024 the L-NP Lyapunov
solve needs shift 4 on six tasks and shift 8 on ``multidlydmintseq`` and
``dmsintseq`` (max Re(lambda) = 5.0-5.4, above the threshold of 5 that the
default budget of 5 attempts can reach). With 8 attempts all L-NP arms complete,
verified by smoke test on multidlydmintseq -- the hardest task -- at both
quantiles. Read ``prune_leak_shift`` when interpreting: a shift-8 run had its
covariance computed for A = W - 9I.
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

QUANTILES = (0.5, 0.6)
PRUNING_SEEDS = _st.PRUNING_SEEDS
AMOUNTS = _st.AMOUNTS
TASKS = _st.TASKS
NETWORK_SEEDS = _st.NETWORK_SEEDS


def suite_stem(h: int) -> str:
    return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_capped_q50_q60_p50_80"


def arms():
    out = []
    for q in QUANTILES:
        tag = f"q{int(q * 100)}"
        out.append((f"lnp_capped_{tag}", "noise_prune_capped_rescale",
                    PRUNING_SEEDS, False, True,
                    {"rescale_cap_mode": "quantile", "rescale_cap_quantile": q}))
        out.append((f"snp_capped_{tag}", "simulation_noise_prune_capped_rescale",
                    PRUNING_SEEDS, True, True,
                    {"rescale_cap_mode": "quantile", "rescale_cap_quantile": q}))
    # uncapped reference at pruning seed 0: must reproduce the completed suite
    out.append(("snp_rescale", "simulation_noise_prune_rescale", (0,), True, True, {}))
    return tuple(out)


def make_unit(h, task_label, task, ng_t, amounts=AMOUNTS, include_baseline=True, shard=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = {
            "task": task, "ng_T": ng_t,
            "load_model_path": _st.checkpoint_path(h, task_label, seed),
            "source_model_label": f"{task_label}_tanh_h{h}_12k_lr0006_seqbest_no_l2_seed{seed}",
            "source_run_id": (f"continue_modcog_{task_label}_tanh_h{h}_to12k_lr0006"
                              f"_seqbest_no_l2_seed{seed}"),
            "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000,
            "eval_batches_path": _st.eval_batches_path(task_label),
        }
        if include_baseline:
            runs.append({"run_id": f"cap{h}_{task_label}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **common})
        for label, strategy, seeds, needs_score, needs_rng, extras in arms():
            for amount in amounts:
                for pseed in seeds:
                    r = {"run_id": (f"cap{h}_{task_label}_netseed{seed}_{label}_"
                                    f"p{int(amount * 100)}_pruneseed{pseed}"),
                         "strategy": strategy, "amount": amount,
                         "seed": pseed, "pruning_seed": pseed, **common, **extras}
                    if needs_rng:
                        r["noise_rng_seed"] = pseed
                    if needs_score:
                        r["score_batch_seed"] = 100_000 + pseed
                        r["score_batches_path"] = _st.score_batches_path(task_label, pseed)
                    runs.append(r)
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

    cfg_dir = Path(f"configs/{suite_stem(h)}"); cfg_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"  H = {h}; total runs {total}; quantiles {QUANTILES}")
    print(f"  expect L-NP arms to fail on multidlydmintseq and dmsintseq (Lyapunov solve)")


if __name__ == "__main__":
    main()
