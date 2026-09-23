#!/usr/bin/env python3
"""Complete the constant-factor cap: 80% sparsity, and a second network size.

The narrative's Fig 4 introduces the capped rescale **using a constant factor**,
so that arm has to cover the paper's whole sparsity range and has to be shown to
survive a change of network size. Neither held before this suite:

    on disk already   H = 512, 50-70%, constants 2.369 / 2.961 / 3.948
                      (per-sparsity) and 3.092 (one global constant)
    missing           80% at H = 512; every sparsity at H = 1024

**Choice of constants.** The per-sparsity q50 cap means at H = 512 are
2.369 / 2.961 / 3.948 / 5.922 at 50/60/70/80%; their mean over 50-70% is 3.092
(the constant the existing suite used) and over the full 50-80% range 3.800.
Since the paper reports 50-80%, 3.800 is the constant a reader would actually be
handed, so it is run across the whole range at both sizes.

**The size test is a transfer test, deliberately.** H = 1024 runs the constants
*fitted at H = 512*, unchanged. If they work, the constant transfers and Fig 4
can state it plainly. Cap value at a matched quantile differs by only ~3% across
the two sizes, so the prediction is that they do -- but that is an inference from
the quantile-to-value map, and this measures it.

Both score arms are run at every cap so the covariance contribution stays
measurable under a constant cap, not just absolute retention.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE / name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


_rs = _load("generate_tanh_h512_modcog_revised8_revision_suites.py")
_st = _load("generate_modcog_revised8_size_taskpres_suite.py")

TASKS = _st.TASKS
NETWORK_SEEDS = (0, 1, 2)
PRUNING_SEED = 0

# q50 cap means measured at H = 512, used unchanged at both sizes
PER_SPARSITY = {0.5: 2.369, 0.6: 2.961, 0.7: 3.948, 0.8: 5.922}
GLOBAL_50_70 = 3.092   # the constant the existing H=512 suite used
GLOBAL_50_80 = 3.800   # the constant the paper's 50-80% range implies

FULL = "simulation_noise_prune_capped_rescale"
MAG = "simulation_noise_prune_capped_magnitude_rescale"

# (label, strategy, cap resolver, sparsities) -- what is MISSING, per size
PLAN = {
    512: (
        ("fixedps_full",  FULL, lambda a: PER_SPARSITY[round(a, 2)], (0.8,)),
        ("fixedps_mag",   MAG,  lambda a: PER_SPARSITY[round(a, 2)], (0.8,)),
        ("fixedg38_full", FULL, lambda a: GLOBAL_50_80, (0.5, 0.6, 0.7, 0.8)),
        ("fixedg38_mag",  MAG,  lambda a: GLOBAL_50_80, (0.5, 0.6, 0.7, 0.8)),
    ),
    1024: (
        ("fixedg38_full", FULL, lambda a: GLOBAL_50_80, (0.5, 0.6, 0.7, 0.8)),
        ("fixedg38_mag",  MAG,  lambda a: GLOBAL_50_80, (0.5, 0.6, 0.7, 0.8)),
        ("fixedg31_full", FULL, lambda a: GLOBAL_50_70, (0.5, 0.6, 0.7)),
        ("fixedg31_mag",  MAG,  lambda a: GLOBAL_50_70, (0.5, 0.6, 0.7)),
    ),
}
ALL_AMOUNTS = (0.5, 0.6, 0.7, 0.8)


def suite_stem(h): return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_fixedcapfull_p50_80"
def shard(a): return "" if a is None else f"_p{int(a*100)}"


def common_for(h, task, ng_t, seed):
    if h == 512:
        return _rs._common(task, ng_t, seed)
    return {
        "task": f"modcog:{task}", "ng_T": ng_t,
        "load_model_path": _st.checkpoint_path(h, task, seed),
        "source_model_label": f"{task}_tanh_h{h}_12k_lr0006_seqbest_no_l2_seed{seed}",
        "source_run_id": (f"continue_modcog_{task}_tanh_h{h}_to12k_lr0006"
                          f"_seqbest_no_l2_seed{seed}"),
        "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
        "eval_seed": 200_000, "eval_batches_path": _st.eval_batches_path(task),
    }


def make_unit(h, task, ng_t, amounts, baseline, sh):
    runs = []
    if baseline:
        for seed in NETWORK_SEEDS:
            runs.append({"run_id": f"fcf{h}_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed,
                         **common_for(h, task, ng_t, seed)})
    for label, strat, capfn, arm_amounts in PLAN[h]:
        for seed in NETWORK_SEEDS:
            c = common_for(h, task, ng_t, seed)
            for amount in amounts:
                if amount not in arm_amounts:
                    continue
                runs.append({
                    "run_id": (f"fcf{h}_{task}_netseed{seed}_{label}_"
                               f"p{int(amount*100)}_pruneseed{PRUNING_SEED}"),
                    "strategy": strat, "amount": amount,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED, "prob_control_seed": PRUNING_SEED,
                    "rescale_cap_mode": "fixed", "rescale_cap_value": capfn(amount),
                    "score_batch_seed": 100_000 + PRUNING_SEED,
                    "score_batches_path": _st.score_batches_path(task, PRUNING_SEED),
                    **c})
    defaults = _rs.defaults_block() if h == 512 else _st.defaults_block(h)
    return {"run_id": f"{suite_stem(h)}_{task}{shard(sh)}",
            "output_csv": f"results/{suite_stem(h)}/{task}{shard(sh)}.csv",
            "defaults": defaults, "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, required=True, choices=(512, 1024))
    ap.add_argument("--shard-by-sparsity", action="store_true")
    a = ap.parse_args()
    h = a.hidden_size
    cfg = Path(f"configs/{suite_stem(h)}"); cfg.mkdir(parents=True, exist_ok=True)
    Path(f"results/{suite_stem(h)}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in ALL_AMOUNTS] if a.shard_by_sparsity else [ALL_AMOUNTS]
        for i, amounts in enumerate(shards):
            sh = amounts[0] if a.shard_by_sparsity else None
            u = make_unit(h, task, ng_t, amounts, baseline=(i == 0), sh=sh)
            if not u["runs"]:
                continue
            (cfg / f"{suite_stem(h)}_{task}{shard(sh)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {cfg}; H={h}; total runs {total}")


if __name__ == "__main__":
    main()
