#!/usr/bin/env python3
"""The size task-preservation suite at 10-40% sparsity.

WHY THIS EXISTS.  ``generate_modcog_revised8_size_taskpres_suite.py`` repeated
the paper's seven methods at H = 1024, but only at 50/60/70/80% -- the four
sparsities the paper's main figure used when it was written.  Every other figure
in the revision now spans 10-80%, so the size comparison is the one panel that
cannot show the same range, and the H = 1024 curves start mid-plot.

It also leaves the central puzzle untestable.  At H = 1024 absolute accuracy is
*lower* than at H = 512 at every shared sparsity despite a far higher unpruned
baseline (0.97 against 0.68), and the drop between 50% and 60% is a cliff rather
than the gradual decline H = 512 shows.  Two readings fit the 50-80% data
equally well:

    (a) H = 1024 is worse everywhere, and 50% already sits past its knee;
    (b) H = 1024 is *better* below its knee and the knee simply sits earlier,
        so the cliff is a shifted threshold rather than a uniform deficit.

These differ in what they say about the paper.  (a) says large networks are
genuinely less prunable; (b) says prunability has a size-dependent threshold and
the paper's 50-80% window happens to straddle it at H = 1024 while sitting below
it at H = 512.  Only sub-50% data separates them.

METHOD SET.  Identical to the 50-80% suite -- same seven methods, same pruning-
seed counts, same defaults -- so the two ranges concatenate into one dataset
with no design change across the join.  Arms are emitted in the order below and
the suite shards by (task, sparsity), so an interrupted run still yields
complete curves for the methods the figure plots:

    obs_compensated  magnitude  snp_rescale   the three the size figure plots
    random                                    the floor, for reading the others
    lnp_rescale  snp_mask  lnp_mask           completes the seven

BASELINES.  Re-run here rather than read across from the 50-80% directory.  They
are unpruned evaluations, so they cost little, and carrying them in the same CSV
keeps retention computable from this suite alone.  A previous low-sparsity suite
omitted them, every retention came out NaN, and the rows were silently dropped
from a figure; do not repeat that.

24 networks x 4 sparsities x 15 runs = 1,440 pruned runs, plus 24 baselines.
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
PRUNING_SEEDS = _st.PRUNING_SEEDS
AMOUNTS = (0.1, 0.2, 0.3, 0.4)

# Same seven methods as the 50-80% suite, reordered so the three the size figure
# plots come first. (label, strategy, pruning seeds, needs_score, needs_rng, extras)
_BY_LABEL = {m[0]: m for m in _st.METHODS}
ORDER = ("obs_compensated", "magnitude", "snp_rescale", "random",
         "lnp_rescale", "snp_mask", "lnp_mask")
METHODS = tuple(_BY_LABEL[k] for k in ORDER)
assert len(METHODS) == len(_st.METHODS), "method set drifted from the 50-80% suite"


def suite_stem(h: int) -> str:
    return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_revised_task_only_p10_40"


def make_unit(h: int, task_label: str, task: str, ng_t: int,
              amounts=AMOUNTS, include_baseline=True, shard=None) -> dict:
    runs = []
    for seed in NETWORK_SEEDS:
        common = {
            "task": task, "ng_T": ng_t,
            "load_model_path": _st.checkpoint_path(h, task_label, seed),
            "source_model_label": f"{task_label}_tanh_h{h}_12k_lr0006_seqbest_no_l2_seed{seed}",
            "source_run_id": (f"continue_modcog_{task_label}_tanh_h{h}_to12k_lr0006"
                              f"_seqbest_no_l2_seed{seed}"),
            "source_network_seed": seed,
            "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000,
            "eval_batches_path": _st.eval_batches_path(task_label),
        }
        if include_baseline:
            runs.append({
                "run_id": f"taskpreslow_h{h}_{task_label}_netseed{seed}_baseline",
                "strategy": "none", "amount": 0.0, "no_prune": True,
                "seed": seed, **common,
            })
        for label, strategy, seeds, needs_score, needs_rng, extras in METHODS:
            for amount in amounts:
                for pseed in seeds:
                    run = {
                        "run_id": (f"taskpreslow_h{h}_{task_label}_netseed{seed}_{label}_"
                                   f"p{int(amount * 100)}_pruneseed{pseed}"),
                        "strategy": strategy, "amount": amount,
                        "seed": pseed, "pruning_seed": pseed, **common, **extras,
                    }
                    if needs_rng:
                        run["noise_rng_seed"] = pseed
                    if needs_score:
                        run["score_batch_seed"] = 100_000 + pseed
                        run["score_batches_path"] = _st.score_batches_path(task_label, pseed)
                    if strategy in _st.CONTROL_STRATEGIES:
                        run["prob_control_seed"] = _st.PROB_CONTROL_SEED_BASE + pseed
                    runs.append(run)
    return {
        "run_id": f"{suite_stem(h)}_{task_label}{_st.shard_suffix(shard)}",
        "output_csv": f"results/{suite_stem(h)}/{task_label}{_st.shard_suffix(shard)}.csv",
        "defaults": _st.defaults_block(h),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--shard-by-sparsity", action="store_true",
                    help="one config per (task, sparsity): 32 units instead of 8")
    args = ap.parse_args()
    h = args.hidden_size
    _st.validate(h, require_checkpoints=True)

    cfg_dir = Path(f"configs/{suite_stem(h)}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"  H = {h}; total runs {total}; sparsities {[int(a * 100) for a in AMOUNTS]}")
    print(f"  arm order: {[m[0] for m in METHODS]}")


if __name__ == "__main__":
    main()
