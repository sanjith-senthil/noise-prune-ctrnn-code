#!/usr/bin/env python3
"""Measure the q50 amplification cap under the corrected normalisation.

The constant cap the paper uses, 3.8, was obtained as the mean over networks of
the q50 of each network's own 1/p distribution, measured BEFORE the probability
normalisation was corrected. The correction raises K, which lowers every 1/p, so
that constant no longer describes the distribution it was derived from and has
to be re-measured the same way.

This suite runs the two capped arms with `rescale_cap_quantile = 0.5`, which
makes the pruner derive the cap per network and record it as
`prune_rescale_cap_value`. The mean of that column, per sparsity and overall, is
the recalibrated constant.

Both scores are measured. The published constant came from the covariance
(noise) arm, and that convention is kept; the magnitude arm is measured
alongside because its probabilities are magnitude-proportional and therefore a
different distribution, and it is worth knowing whether one constant still
serves both.

2 arms x 8 sparsities x 8 tasks x 3 network seeds x 3 pruning seeds = 1,152
runs. Sharded by (task, sparsity).
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

AMOUNTS = _rn.AMOUNTS
NETWORK_SEEDS = _rn.NETWORK_SEEDS
PRUNING_SEEDS = _rn.PRUNING_SEEDS
QUANTILE = 0.50
SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capq50_p10_80"

ARMS = (
    ("q50_full", "simulation_noise_prune_capped_rescale"),
    ("q50_mag",  "simulation_noise_prune_capped_magnitude_rescale"),
)


def shard(a): return f"_p{int(round(a * 100))}"


def make_unit(task_label, ng_t, amount, include_baseline):
    runs = []
    for seed in NETWORK_SEEDS:
        c = _rs._common(task_label, ng_t, seed)
        if include_baseline:
            runs.append({"run_id": f"capq50_{task_label}_netseed{seed}_baseline",
                         "strategy": "none", "amount": 0.0, "no_prune": True,
                         "seed": seed, **c})
        for label, strategy in ARMS:
            for ps in PRUNING_SEEDS:
                r = {"run_id": (f"capq50_{task_label}_netseed{seed}_{label}"
                                f"_p{int(round(amount * 100))}_pruneseed{ps}"),
                     "strategy": strategy, "amount": amount,
                     "seed": ps, "pruning_seed": ps, "noise_rng_seed": ps,
                     "score_batch_seed": 100_000 + ps,
                     "score_batches_path": _rs.score_batches_path(task_label, ps),
                     "rescale_cap_quantile": QUANTILE, **c}
                if strategy == "simulation_noise_prune_capped_magnitude_rescale":
                    r["prob_control_seed"] = _rn.PROB_CONTROL_SEED_BASE + ps
                runs.append(r)
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
        for i, amount in enumerate(AMOUNTS):
            u = make_unit(task_label, ng_t, amount, include_baseline=(i == 0))
            (cfg / f"{SUITE_STEM}_{task_label}{shard(amount)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {cfg}; total runs {total}")
    print(f"  arms {[a[0] for a in ARMS]}; quantile {QUANTILE}; normalisation {_rn.NORMALIZE}")


if __name__ == "__main__":
    main()
