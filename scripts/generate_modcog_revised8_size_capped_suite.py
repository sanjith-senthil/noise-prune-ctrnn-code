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

This suite tests it directly, and the test matters more here than at H = 512:
the amplification tail is far heavier at the larger size (``prune_inv_p_max``
reaches 2.3e6), so capping has more to bite on.

Arms
----
    lnp_capped_q50 / q60   noise_prune_capped_rescale                3 seeds
    snp_capped_q50 / q60   simulation_noise_prune_capped_rescale     3 seeds
    snp_rescale            uncapped reference, pruning seed 0 only -- regression
                           against the completed H=1024 task-preservation suite,
                           which is what licenses comparing capped against
                           uncapped across the two suites.

Known limitation carried over
-----------------------------
At H = 1024 the L-NP Lyapunov solve fails outright on ``multidlydmintseq`` and
``dmsintseq`` (max Re(lambda) = 5.0-5.4 against a shift ladder that tops out at
A = W - 5I). The capped variant uses the same solve, so its L-NP arms will fail
on those two tasks too and report n = 18 rather than 24. That is expected, not a
new defect; the summarizer must be run with --allow-missing and reports which
cells are absent.
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
