#!/usr/bin/env python3
"""The cap-percentile sweep, re-run under the corrected normalisation.

Figure 5 panel a plots retention against the cap percentile of each network's
own 1/p distribution, with the unbounded rule as a reference point to the right.
That sweep was last run in June, before the probability normalisation was
corrected, while the unbounded reference in the same panel now comes from the
re-run -- so one panel was drawn from two different pruning rules, and the
interior optimum it reports was measured on the superseded one.

The 1/p distribution is exactly what the correction changed, so this sweep is
the single most affected analysis in the paper. Its conclusion (the optimum is
interior, and lands near the 50th percentile) has to be re-established on the
corrected rule rather than assumed to carry over.

9 quantiles x 4 sparsities x 8 tasks x 3 network seeds x 3 pruning seeds =
2,592 runs. Sharded by (task, sparsity): 32 units of 81 runs.

Sparsities stay at 50-80%, matching the panel. Below 50% the corrected rule
pins most edges at p = 1, so the median amplification is 1.0 and a percentile
cap has almost nothing to bite on; extending the sweep there would double the
cost to show a flat line.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"generate_{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_rs = _load("tanh_h512_modcog_revised8_revision_suites")
_rn = _load("modcog_revised8_renorm_suite")

AMOUNTS = (0.5, 0.6, 0.7, 0.8)
QUANTILES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
NETWORK_SEEDS = _rn.NETWORK_SEEDS
PRUNING_SEEDS = _rn.PRUNING_SEEDS
STRATEGY = "simulation_noise_prune_capped_rescale"
SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capsweep_p50_80"


def shard(a): return f"_p{int(round(a * 100))}"


def make_unit(task_label, ng_t, amount):
    runs = []
    for seed in NETWORK_SEEDS:
        c = _rs._common(task_label, ng_t, seed)
        for q in QUANTILES:
            for ps in PRUNING_SEEDS:
                runs.append({
                    "run_id": (f"capsweep_{task_label}_netseed{seed}_q{int(round(q*100))}"
                               f"_p{int(round(amount*100))}_pruneseed{ps}"),
                    "strategy": STRATEGY, "amount": amount,
                    "seed": ps, "pruning_seed": ps, "noise_rng_seed": ps,
                    "score_batch_seed": 100_000 + ps,
                    "score_batches_path": _rs.score_batches_path(task_label, ps),
                    "rescale_cap_quantile": q, **c})
    d = dict(_rs.defaults_block())
    d["sim_np_prob_normalize"] = _rn.NORMALIZE
    return {"run_id": f"{SUITE_STEM}_{task_label}{shard(amount)}",
            "output_csv": f"results/{SUITE_STEM}/{task_label}{shard(amount)}.csv",
            "defaults": d, "runs": runs}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    _rs.validate_inputs()
    cfg = Path(f"configs/{SUITE_STEM}")
    cfg.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task_label, _t, ng_t in _rs.TASKS:
        for amount in AMOUNTS:
            u = make_unit(task_label, ng_t, amount)
            (cfg / f"{SUITE_STEM}_{task_label}{shard(amount)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {cfg}; total runs {total}")
    print(f"  quantiles {[int(q*100) for q in QUANTILES]}; sparsities "
          f"{[int(a*100) for a in AMOUNTS]}; normalisation {_rn.NORMALIZE}")


if __name__ == "__main__":
    main()
