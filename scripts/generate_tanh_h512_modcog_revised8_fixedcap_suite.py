#!/usr/bin/env python3
"""Does the cap need to be network-adaptive, or will a constant factor do?

Suite F capped at the q50 *quantile* of each network's own 1/p distribution, so
the absolute cap is re-derived per network. That is harder to justify as a
mechanism than a fixed bound on synaptic amplification, and harder to state in a
paper. The question is whether a constant reproduces it.

The quantile caps turn out to be very stable across networks -- spread only 1.15x
min-to-max at every sparsity, sd ~4% of the mean -- so a per-sparsity constant
should substitute closely:

    sparsity   mean q50 cap (covariance arm, n=24)
      50%          2.369
      60%          2.961
      70%          3.948
    global mean    3.092

Across sparsities the cap moves 1.67x, so a single global constant is tested
separately: it is too high at 50% and too low at 70%, and how much that costs is
the point of the comparison.

Both score arms are run at each cap so the covariance contribution stays
measurable under a fixed cap, not just the absolute retention.

4 arms x 8 tasks x 3 network seeds x 3 sparsities = 288 runs, n = 24.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fixedcap_p50_70"
CONFIG_DIR = Path("configs/fixed_cap")
OUTPUT_DIR = _rs.RESULT_ROOT / SUITE_STEM
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7)
PRUNING_SEED = 0

# mean q50 cap of the covariance arm, measured per sparsity over the 24 networks
PER_SPARSITY = {0.5: 2.369, 0.6: 2.961, 0.7: 3.948}
GLOBAL_CAP = 3.092

FULL = "simulation_noise_prune_capped_rescale"
MAG  = "simulation_noise_prune_capped_magnitude_rescale"

# label, strategy, cap-value resolver
ARMS = (
    ("fixedps_full",   FULL, lambda a: PER_SPARSITY[round(a, 2)]),
    ("fixedps_mag",    MAG,  lambda a: PER_SPARSITY[round(a, 2)]),
    ("fixedglob_full", FULL, lambda a: GLOBAL_CAP),
    ("fixedglob_mag",  MAG,  lambda a: GLOBAL_CAP),
)


def shard(a): return "" if a is None else f"_p{int(a*100)}"
def cfg(t, a=None): return CONFIG_DIR / f"{SUITE_STEM}_{t}{shard(a)}.json"
def out(t, a=None): return OUTPUT_DIR / f"{t}{shard(a)}.csv"


def make(task, ng_t, amounts=AMOUNTS, baseline=True, sh=None):
    runs = []
    for seed in NETWORK_SEEDS:
        common = _rs._common(task, ng_t, seed)
        if baseline:
            runs.append({"run_id": f"fc_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for (label, strat, capfn) in ARMS:
            for amount in amounts:
                runs.append({
                    "run_id": f"fc_{task}_netseed{seed}_{label}_p{int(amount*100)}_pruneseed{PRUNING_SEED}",
                    "strategy": strat, "amount": amount,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED, "prob_control_seed": PRUNING_SEED,
                    "rescale_cap_mode": "fixed", "rescale_cap_value": capfn(amount),
                    "score_batch_seed": 100_000 + PRUNING_SEED,
                    "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED),
                    **common})
    return {"run_id": f"{SUITE_STEM}_{task}{shard(sh)}", "output_csv": str(out(task, sh)),
            "defaults": _rs.defaults_block(), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-inputs", action="store_true")
    ap.add_argument("--shard-by-sparsity", action="store_true")
    a = ap.parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tot = n = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in AMOUNTS] if a.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            sh = amounts[0] if a.shard_by_sparsity else None
            s = make(task, ng_t, amounts, baseline=(i == 0), sh=sh)
            cfg(task, sh).write_text(json.dumps(s, indent=2) + "\n"); tot += len(s["runs"]); n += 1
    print(f"wrote {n} configs; total runs: {tot}  ({len(ARMS)} arms)")


if __name__ == "__main__":
    main()
