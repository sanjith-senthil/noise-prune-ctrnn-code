#!/usr/bin/env python3
"""The random mask with a global rescale, completing the control grid.

Figure 3 asks whether a single global factor can stand in for per-connection
sample-and-rescale. That question was answered for the noise score and for
magnitude; random was missing the same treatment, so the grid read:

                      1/density          |W|-matched constant
    noise score       have               have
    magnitude score   have               have
    random mask       50-80% only        MISSING

Random matters here precisely because it is the uninformative mask. If a global
factor rescued it the way sampling rescues magnitude, that would mean the gain
restoration -- not the choice of connections -- was doing the work. Leaving
random out of the constant-factor comparison is the one gap that could carry
that alternative explanation.

Two arms, each scoped to what is missing so nothing already on disk is produced
a second time:

  ``inv_density``   survivors multiplied by 1 / retained density. Exists at
                    50-80% from the score-control suite, so only 10-40% is run.
  ``match_l1``      one constant restoring the retained off-diagonal |W| sum to
                    the unpruned value. Never run for the random mask at any
                    sparsity, so the whole 10-80% grid is run.

THREE PRUNING SEEDS, unlike the magnitude gain arms, which take one. The
magnitude mask is deterministic; the random mask is not, so each network needs
its draws averaged before it enters the statistics, exactly as for the other
stochastic arms.
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

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_random_gain_p10_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
PRUNING_SEEDS = (0, 1, 2)
STRATEGY = "random_unstructured_gain"

# (label, gain_mode, amounts)
ARMS = (
    ("random_gain_matchl1", "match_l1",
     (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
    ("random_gain_invdensity_low", "inv_density",
     (0.1, 0.2, 0.3, 0.4)),
)


def shard(a):
    return f"_p{int(round(a * 100))}"


def make_unit(task, ng_t, amount):
    runs = []
    for label, gain_mode, amounts in ARMS:
        if amount not in amounts:
            continue
        for seed in NETWORK_SEEDS:
            c = _rs._common(task, ng_t, seed)
            for ps in PRUNING_SEEDS:
                runs.append({
                    "run_id": (f"randgain_{task}_netseed{seed}_{label}_"
                               f"p{int(round(amount * 100))}_pruneseed{ps}"),
                    "strategy": STRATEGY, "amount": amount,
                    "gain_mode": gain_mode,
                    "seed": ps, "pruning_seed": ps,
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
        print(f"  {label:<28} {gain_mode:<12} sparsities "
              f"{[int(a * 100) for a in amts]}  x {len(PRUNING_SEEDS)} draws")


if __name__ == "__main__":
    main()
