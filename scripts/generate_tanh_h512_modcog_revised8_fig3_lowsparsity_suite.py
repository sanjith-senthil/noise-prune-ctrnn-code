#!/usr/bin/env python3
"""The Figure 3 arms at 10-40% sparsity.

Figure 3 contrasts, on the magnitude mask, doing nothing against two
deterministic constant-factor rescales and against sample-and-rescale. Three of
those arms had only ever been run at 50-80%, so the figure could not span the
same 10-80% range as Figures 1 and 2. This fills the gap.

    magnitude_gain_invdensity  one constant = 1 / retained density   (~3.2x)
    magnitude_gain_matchl1     one constant restoring total |W|      (~1.7x)
    snp_magnitude              sample-and-rescale with p proportional to |w|

Protocols match the suites that produced the 50-80% cells exactly: the two gain
arms are deterministic and take one pruning seed; snp_magnitude is stochastic
and takes three, with prob_control_seed 900000 as in the covariance-ablation
suite.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fig3_lowsparsity_p10_40"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
AMOUNTS = (0.1, 0.2, 0.3, 0.4)
PROB_CONTROL_SEED = 900_000

# (label, strategy, pruning seeds, extras)
ARMS = (
    ("magnitude_gain_invdensity", "l1_unstructured_gain", (0,), {"gain_mode": "inv_density"}),
    ("magnitude_gain_matchl1",    "l1_unstructured_gain", (0,), {"gain_mode": "match_l1"}),
    ("snp_magnitude", "simulation_noise_prune_magnitude_rescale", (0, 1, 2), {}),
)
NEEDS_SCORE = {"simulation_noise_prune_magnitude_rescale"}


def shard(a): return f"_p{int(a * 100)}"


def make_unit(task, ng_t, amount):
    runs = []
    for label, strat, pseeds, extras in ARMS:
        for seed in NETWORK_SEEDS:
            common = _rs._common(task, ng_t, seed)
            for ps in pseeds:
                r = {"run_id": (f"f3low_{task}_netseed{seed}_{label}_"
                                f"p{int(amount*100)}_pruneseed{ps}"),
                     "strategy": strat, "amount": amount,
                     "seed": ps, "pruning_seed": ps, "noise_rng_seed": ps,
                     **extras, **common}
                if strat in NEEDS_SCORE:
                    r["prob_control_seed"] = PROB_CONTROL_SEED
                    r["score_batch_seed"] = 100_000 + ps
                    r["score_batches_path"] = _rs.score_batches_path(task, ps)
                runs.append(r)
    return {"run_id": f"{SUITE_STEM}_{task}{shard(amount)}",
            "output_csv": f"results/{SUITE_STEM}/{task}{shard(amount)}.csv",
            "defaults": _rs.defaults_block(), "runs": runs}


def main():
    argparse.ArgumentParser().parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in TASKS:
        for amount in AMOUNTS:
            u = make_unit(task, ng_t, amount)
            (CONFIG_DIR / f"{SUITE_STEM}_{task}{shard(amount)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}; total runs {total}")


if __name__ == "__main__":
    main()
