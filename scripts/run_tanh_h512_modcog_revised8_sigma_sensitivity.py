#!/usr/bin/env python3
"""Run final-suite S-NP sigma sensitivity.

This is a paper-facing rerun of the sigma-dependence check for the final
H512 revised-8 seq-best checkpoints. It measures top-k mask overlap against
the sigma = sigma_nat baseline, using the selected S-NP implementation:
rate-space injection, rate-space scoring, trajectory-mean centering,
25k samples, and 300-step burn-in.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUITE_CSV = (
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_3net_3seed_p50_80_fixedbatches.csv"
)
ROWS_CSV = "results/sim_np_sigma_sensitivity_tanh_h512_modcog_revised8_12k_seqbest_rows.csv"
SUMMARY_CSV = "results/sim_np_sigma_sensitivity_tanh_h512_modcog_revised8_12k_seqbest_summary.csv"
PLOT_PATH = "results/sim_np_sigma_sensitivity_tanh_h512_modcog_revised8_12k_seqbest.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-csv", default=SUITE_CSV)
    parser.add_argument("--rows-csv", default=ROWS_CSV)
    parser.add_argument("--summary-csv", default=SUMMARY_CSV)
    parser.add_argument("--plot", default=PLOT_PATH)
    parser.add_argument("--amounts", nargs="+", default=["0.8"])
    parser.add_argument("--sigma-factors", nargs="+", default=["0.75", "1.0", "1.5"])
    parser.add_argument("--rng-seeds", nargs="+", default=["0"])
    parser.add_argument("--checkpoint-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/analyze_sim_np_sigma_sensitivity.py",
        "--suite_csv",
        args.suite_csv,
        "--sample_batches",
        "8",
        "--batch_seed",
        "100000",
        "--amounts",
        *args.amounts,
        "--sigma_factors",
        *args.sigma_factors,
        "--rng_seeds",
        *args.rng_seeds,
        "--eps",
        "0.3",
        "--observable_space",
        "rate",
        "--inject_space",
        "rate",
        "--centering",
        "trajectory_mean",
        "--max_samples",
        "25000",
        "--burn_in_steps",
        "300",
        "--output_rows_csv",
        args.rows_csv,
        "--output_summary_csv",
        args.summary_csv,
        "--output_plot",
        args.plot,
    ]
    if args.checkpoint_limit is not None:
        cmd.extend(["--checkpoint_limit", str(args.checkpoint_limit)])
    if args.force:
        cmd.append("--force")

    print(" ".join(cmd))
    Path(args.rows_csv).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
