#!/usr/bin/env python3
"""The noise score with a GLOBAL rescale, so the controls are score-matched.

The paper's sampling claim is that sample-and-rescale does something a single
global factor cannot. The control that carried it compared noise-prune with
sample-and-rescale against MAGNITUDE with a global factor -- which changes the
score and the rescale rule at the same time, so a reader cannot tell which one
the gap is about.

This suite supplies the missing cells, giving a clean 2x2:

                      sample-and-rescale        global factor
    noise score       (already have)            THIS SUITE
    magnitude score   (already have)            (already have)

Two global conventions are run, matching what exists for magnitude:

  ``match_l1``      one constant restoring the retained off-diagonal |W| sum to
                    the unpruned value -- the sum-based rule. It exists for the
                    magnitude score at 10-80% and did not exist for the noise
                    score at all, so it is run over the full grid here.
  ``inv_density``   multiply survivors by 1 / retained density. Already run at
                    50-80% by the synaptic-scaling suite, so only 10-40% is run
                    here; re-running the overlap would deposit the same cells
                    twice under two stems.

The mask is the deterministic top-k on the noise score, so these arms never
draw a Bernoulli mask and are untouched by the probability-normalisation
correction -- which is why the existing 50-80% inv_density rows stay valid and
only the gap needs filling. One pruning seed for the same reason: with a
deterministic mask and a deterministic factor there is nothing to average.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_matched_gain_p10_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
PRUNING_SEED = 0
STRATEGY = "simulation_noise_prune_gain"

# (label, gain_mode, amounts) -- amounts differ by arm so nothing already on
# disk is produced a second time.
ARMS = (
    ("snpgain_matchl1", "match_l1", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
    ("snpgain_invdensity_low", "inv_density", (0.1, 0.2, 0.3, 0.4)),
)


def shard(a):
    return "" if a is None else f"_p{int(round(a * 100))}"


def make_unit(task, ng_t, amount):
    runs = []
    for label, gain_mode, amounts in ARMS:
        if amount not in amounts:
            continue
        for seed in NETWORK_SEEDS:
            c = _rs._common(task, ng_t, seed)
            runs.append({
                "run_id": (f"scorematch_{task}_netseed{seed}_{label}_"
                           f"p{int(round(amount * 100))}_pruneseed{PRUNING_SEED}"),
                "strategy": STRATEGY, "amount": amount,
                "gain_mode": gain_mode,
                "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                "noise_rng_seed": PRUNING_SEED,
                "score_batch_seed": 100_000 + PRUNING_SEED,
                "score_batches_path": _rs.score_batches_path(task, PRUNING_SEED),
                **c})
    return {"run_id": f"{SUITE_STEM}_{task}{shard(amount)}",
            "output_csv": f"results/{SUITE_STEM}/{task}{shard(amount)}.csv",
            "defaults": _rs.defaults_block(), "runs": runs}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    _rs.validate_inputs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    amounts = sorted({a for _, _, amts in ARMS for a in amts})
    total = n = 0
    for task, _t, ng_t in TASKS:
        for amount in amounts:
            u = make_unit(task, ng_t, amount)
            if not u["runs"]:
                continue
            (CONFIG_DIR / f"{SUITE_STEM}_{task}{shard(amount)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"])
            n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}; total runs {total}")
    for label, gain_mode, amts in ARMS:
        print(f"  {label:<24} {gain_mode:<12} sparsities "
              f"{[int(a * 100) for a in amts]}")


if __name__ == "__main__":
    main()
