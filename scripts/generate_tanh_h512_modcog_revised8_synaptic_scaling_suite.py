#!/usr/bin/env python3
"""Synaptic scaling on the noise-prune mask (open question 3).

Reviewer 2 objects that sample-and-rescale has no obvious synaptic mechanism:
there is no way for a synapse to be boosted by the inverse of a sampling
probability it cannot observe. Synaptic scaling is the established homeostatic
alternative -- rescale all weights into a neuron by one constant so the neuron's
total input weight is unchanged -- and it is *local*: row `i`'s factor depends
only on the weights neuron `i` receives.

What was already known: on a **magnitude** mask synaptic scaling collapses
(`magnitude_gain_matchl1`, n = 24: 0.130 / 0.070 / 0.057 / 0.062 at 50/60/70/80%
against 0.779 for plain magnitude). Magnitude pruning retains L1 mass far more
poorly than it retains count or spectrum, so L1-matching demands a x2.4-2.5
factor, well past the x1.36 that helps, and it destabilises.

What was missing, and is the arm the objection actually points at: the same rule
on the **noise-prune** mask. The mask is then chosen by the noise score and only
the restoration rule changes, so this is the like-for-like replacement of
`1 / p` by a biologically defensible factor.

Arms, all on the S-NP deterministic top-k mask:

    snpgain_synaptic   match_l1_rowwise  -- per-neuron, local, homeostatic
    snpgain_invdensity inv_density       -- one global factor; the reviewer's
                                            own suggestion, on this mask

Both pair directly against `simulation_noise_prune_mask_only` (same mask, no
restoration) and `simulation_noise_prune_rescale` (same score, `1 / p`), which
are already on disk at n = 24, so no reference arms need re-running.

2 arms x 8 tasks x 3 network seeds x 4 sparsities = 192 runs, n = 24.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_synaptic_p50_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6, 0.7, 0.8)
PRUNING_SEED = 0
STRATEGY = "simulation_noise_prune_gain"

ARMS = (
    ("snpgain_synaptic", "match_l1_rowwise"),
    ("snpgain_invdensity", "inv_density"),
)


def shard(a): return "" if a is None else f"_p{int(a*100)}"


def make_unit(task, ng_t, amounts, baseline, sh):
    runs = []
    if baseline:
        for seed in NETWORK_SEEDS:
            runs.append({"run_id": f"syn_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed,
                         **_rs._common(task, ng_t, seed)})
    for label, gain_mode in ARMS:
        for seed in NETWORK_SEEDS:
            c = _rs._common(task, ng_t, seed)
            for amount in amounts:
                runs.append({
                    "run_id": (f"syn_{task}_netseed{seed}_{label}_"
                               f"p{int(amount*100)}_pruneseed{PRUNING_SEED}"),
                    "strategy": STRATEGY, "amount": amount,
                    "gain_mode": gain_mode,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED,
                    "score_batch_seed": 100_000 + PRUNING_SEED,
                    "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED),
                    **c})
    return {"run_id": f"{SUITE_STEM}_{task}{shard(sh)}",
            "output_csv": f"results/{SUITE_STEM}/{task}{shard(sh)}.csv",
            "defaults": _rs.defaults_block(), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-by-sparsity", action="store_true")
    a = ap.parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in AMOUNTS] if a.shard_by_sparsity else [AMOUNTS]
        for i, amounts in enumerate(shards):
            sh = amounts[0] if a.shard_by_sparsity else None
            u = make_unit(task, ng_t, amounts, baseline=(i == 0), sh=sh)
            (CONFIG_DIR / f"{SUITE_STEM}_{task}{shard(sh)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}; total runs {total}")


if __name__ == "__main__":
    main()
