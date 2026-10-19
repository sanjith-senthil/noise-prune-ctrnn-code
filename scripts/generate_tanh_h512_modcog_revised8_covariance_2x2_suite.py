"""The task-input x noise 2x2, re-run under the corrected normalisation.

Reviewer 2's comment 4a asks which part of the covariance the method actually
uses. The answer is a 2x2 over what drives the fluctuations:

                       noise ON (sigma_nat)      noise OFF (sigma -> 0)
    task input ON      reference                 task-driven only
    task input OFF     noise-driven only         fixed point (degenerate)

The original version of this ran on 2026-09-11, before the probability
normalisation was corrected. Both cells use the RESCALE variant, so both draw a
Bernoulli mask and both are hit by the defect: the old rows realise 52.9%
density at a requested 50% (56.5% in the seed-extension suite) against 49.96%
under the corrected rule. Restricting the sparsity range does not fix this --
the bias is present at every level and is not common across cells, because the
noise-off arm has a different covariance and therefore clips differently. So
the whole design is re-run rather than patched.

THE REFERENCE CELL IS RE-RUN HERE TOO, even though a corrected
``simulation_noise_prune_rescale`` already exists at 10-80%. The sigma controls
require trajectory-mean centering (under conditional centering the deviation is
identically zero as sigma -> 0), and a comparison whose reference differs from
its treatment cells in centering would confound the thing being measured. All
three cells therefore carry identical settings and differ only in the 2x2
factors.

sigma = 0 exactly is undefined -- the probability formula divides by sigma^2 --
so the noise-free limit is approached at factor 1e-3, as before. The fourth
cell is a fixed point with no fluctuations to score and is not run.

``zero_task_input`` zeroes the input AFTER sigma is resolved. Sigma is estimated
from the natural variability of the task-driven network, so zeroing first would
drive it to ~0 and the arm would differ in two ways at once.
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

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_cov2x2_p10_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
AMOUNTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
PRUNING_SEEDS = (0, 1, 2)
STRATEGY = "simulation_noise_prune_rescale"
NOISE_FREE_FACTOR = 0.001

# (label, sigma_factor, zero_task_input)
CELLS = (
    ("ref_task_noise", 1.0, False),              # both drives present
    ("task_only", NOISE_FREE_FACTOR, False),     # noise off
    ("noise_only", 1.0, True),                   # task input off
)


def make_unit(task, ng_t, amount):
    runs = []
    for seed in NETWORK_SEEDS:
        c = _rs._common(task, ng_t, seed)
        for label, sf, zero_in in CELLS:
            for ps in PRUNING_SEEDS:
                runs.append({
                    "run_id": (f"cov2x2_{task}_netseed{seed}_{label}_"
                               f"p{int(round(amount * 100))}_pruneseed{ps}"),
                    "strategy": STRATEGY, "amount": amount,
                    "sim_np_sigma_factor": sf,
                    "sim_np_zero_task_input": zero_in,
                    "sim_np_centering": "trajectory_mean",
                    "seed": ps, "pruning_seed": ps, "noise_rng_seed": ps,
                    "score_batch_seed": 100_000 + ps,
                    "score_batches_path": _rs.score_batches_path(task, ps),
                    **c})
    d = dict(_rs.defaults_block())
    d["sim_np_prob_normalize"] = "bisect"        # the corrected rule
    return {"run_id": f"{SUITE_STEM}_{task}_p{int(round(amount * 100))}",
            "output_csv": f"results/{SUITE_STEM}/{task}_p{int(round(amount * 100))}.csv",
            "defaults": d, "runs": runs}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    _rs.validate_inputs()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in TASKS:
        for amount in AMOUNTS:
            u = make_unit(task, ng_t, amount)
            (CONFIG_DIR / f"{SUITE_STEM}_{task}_p{int(round(amount*100))}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}; total runs {total}")
    print(f"  cells {[c[0] for c in CELLS]}; sparsities {[int(a*100) for a in AMOUNTS]}; "
          f"{len(PRUNING_SEEDS)} pruning seeds")


if __name__ == "__main__":
    main()
