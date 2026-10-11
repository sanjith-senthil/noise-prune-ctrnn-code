#!/usr/bin/env python3
"""Pruning seeds 1 and 2 for the constant-factor capped arms at H = 512.

The constant-factor cap is a *stochastic* rule -- survivors are drawn
Bernoulli(p) -- so it must carry the same three pruning draws as the other
stochastic arms in the paper's figures. It did not: `simulation_noise_prune_rescale`
and `random_unstructured` average three draws per network while the
constant-factor capped arm averaged one, which inflates its between-network
spread relative to the curve plotted next to it.

(The *quantile*-capped arm was already run at three draws across q10-q90 at
50-80%. Only the fixed-factor variant was short, and only it is in the figures.)

Both score arms get the extra draws, so the covariance contrast in Figure 4
stays symmetric -- adding draws to one side only would make the paired
difference noisier on one arm than the other.

    fixedg38_full  10-80%   2 new seeds x 8 sparsities x 24 networks = 384
    fixedg38_mag   50-80%   2 new seeds x 4 sparsities x 24 networks = 192
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fcc", Path(__file__).with_name("generate_modcog_revised8_fixedcap_completion_suite.py"))
_fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fc)

H = 512
STEM = _fc.suite_stem(H)
NEW_SEEDS = (0, 1, 2)   # seed 0 already exists for 50-80%; resume skips it
# The covariance-dropped capped arm is extended to 10-40% too: Figure 5 pairs
# it against the full-score capped arm to show what the covariance term is worth
# under a cap, and that contrast has to span the same range as every other plot.
ARMS = (("fixedg38_full", _fc.FULL, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
        ("fixedg38_mag",  _fc.MAG,  (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
CAP = _fc.GLOBAL_50_80


def make_unit(task, ng_t, amount):
    runs = []
    for label, strat, amts in ARMS:
        if amount not in amts:
            continue
        for seed in _fc.NETWORK_SEEDS:
            common = _fc.common_for(H, task, ng_t, seed)
            for pseed in NEW_SEEDS:
                runs.append({
                    "run_id": (f"fcf{H}_{task}_netseed{seed}_{label}_"
                               f"p{int(amount*100)}_pruneseed{pseed}"),
                    "strategy": strat, "amount": amount,
                    "seed": pseed, "pruning_seed": pseed,
                    "noise_rng_seed": pseed, "prob_control_seed": pseed,
                    "rescale_cap_mode": "fixed", "rescale_cap_value": CAP,
                    "score_batch_seed": 100_000 + pseed,
                    "score_batches_path": _fc._st.score_batches_path(task, pseed),
                    **common})
    return {"run_id": f"{STEM}_{task}_p{int(amount*100)}_pruneseeds12",
            "output_csv": f"results/{STEM}/{task}_p{int(amount*100)}.csv",
            "defaults": _fc._rs.defaults_block(), "runs": runs}


def main():
    argparse.ArgumentParser().parse_args()
    cfg = Path(f"configs/{STEM}_pruneseeds12"); cfg.mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in _fc.TASKS:
        for amount in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            u = make_unit(task, ng_t, amount)
            if not u["runs"]:
                continue
            (cfg / f"{STEM}_{task}_p{int(amount*100)}_pruneseeds12.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {cfg}; total runs {total}")


if __name__ == "__main__":
    main()
