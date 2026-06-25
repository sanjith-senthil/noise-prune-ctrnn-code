#!/usr/bin/env python3
"""Aggregate frozen rescale-value histograms into wider log10 bins.

The frozen histogram is already binned in log10(1 / p_ij) with 0.05-wide bins.
For any wider bin width that is an integer multiple of 0.05, summing adjacent
frozen bins is equivalent to recomputing the histogram with the wider bins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INDIR = Path("paper_artifacts/official_h512_24net/data/rescale_value_distributions")
DEFAULT_BY_RUN = DEFAULT_INDIR / "rescale_value_histogram_snp_lnp_p80_by_run.csv"
DEFAULT_OVERALL = DEFAULT_INDIR / "rescale_value_histogram_snp_lnp_p80_overall.csv"
RUN_GROUP_COLS = [
    "method_family",
    "method",
    "amount",
    "task",
    "source_model_label",
    "pruning_seed",
    "score_batch_seed",
]
OVERALL_GROUP_COLS = ["method_family", "method", "amount"]


def width_tag(width: float) -> str:
    return f"logbin{width:.2f}".replace(".", "p")


def assign_wide_bins(df: pd.DataFrame, *, width: float) -> pd.DataFrame:
    out = df.copy()
    left = pd.to_numeric(out["log10_rescale_left"], errors="raise").astype(float)
    # The small epsilon avoids assigning values like 0.5999999999999999 to the
    # previous bin because of CSV floating-point roundtrip.
    coarse_index = np.floor((left.to_numpy() + 1e-12) / float(width)).astype(int)
    out["log10_rescale_left"] = coarse_index * float(width)
    out["log10_rescale_right"] = out["log10_rescale_left"] + float(width)
    out["rescale_left"] = 10.0 ** out["log10_rescale_left"]
    out["rescale_right"] = 10.0 ** out["log10_rescale_right"]
    out["bin_width_log10"] = float(width)
    return out


def aggregate(df: pd.DataFrame, *, group_cols: list[str], width: float) -> pd.DataFrame:
    work = assign_wide_bins(df, width=width)
    grouped = (
        work.groupby(
            [*group_cols, "log10_rescale_left", "log10_rescale_right", "rescale_left", "rescale_right", "bin_width_log10"],
            as_index=False,
        )
        .agg(count=("count", "sum"))
        .sort_values([*group_cols, "log10_rescale_left"])
        .reset_index(drop=True)
    )
    grouped["count"] = grouped["count"].astype(int)
    return grouped


def validate_totals(original: pd.DataFrame, coarse: pd.DataFrame, *, label: str) -> None:
    keys = ["method_family", "method", "amount"]
    original_totals = original.groupby(keys)["count"].sum().sort_index()
    coarse_totals = coarse.groupby(keys)["count"].sum().sort_index()
    if not original_totals.equals(coarse_totals):
        diff = (coarse_totals - original_totals).dropna()
        raise ValueError(f"{label}: coarse totals changed: {diff[diff != 0].to_dict()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-run-input", type=Path, default=DEFAULT_BY_RUN)
    parser.add_argument("--overall-input", type=Path, default=DEFAULT_OVERALL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_INDIR)
    parser.add_argument("--bin-width-log10", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width = float(args.bin_width_log10)
    if width <= 0:
        raise ValueError("--bin-width-log10 must be positive")
    tag = width_tag(width)
    by_run = pd.read_csv(args.by_run_input)
    overall = pd.read_csv(args.overall_input)

    by_run_coarse = aggregate(by_run, group_cols=RUN_GROUP_COLS, width=width)
    overall_coarse = aggregate(overall, group_cols=OVERALL_GROUP_COLS, width=width)
    validate_totals(by_run, by_run_coarse, label="by-run")
    validate_totals(overall, overall_coarse, label="overall")

    # The two aggregation paths should agree after dropping run-level metadata.
    by_run_overall = aggregate(by_run_coarse, group_cols=OVERALL_GROUP_COLS, width=width)
    check_cols = [*OVERALL_GROUP_COLS, "log10_rescale_left", "log10_rescale_right", "count"]
    expected = overall_coarse[check_cols].sort_values(check_cols[:-1]).reset_index(drop=True)
    actual = by_run_overall[check_cols].sort_values(check_cols[:-1]).reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError("overall coarse histogram does not match aggregation of by-run coarse histogram")

    args.outdir.mkdir(parents=True, exist_ok=True)
    by_run_output = args.outdir / f"rescale_value_histogram_snp_lnp_p80_by_run_{tag}.csv"
    overall_output = args.outdir / f"rescale_value_histogram_snp_lnp_p80_overall_{tag}.csv"
    by_run_coarse.to_csv(by_run_output, index=False)
    overall_coarse.to_csv(overall_output, index=False)
    print(f"wrote {by_run_output} rows={len(by_run_coarse)}")
    print(f"wrote {overall_output} rows={len(overall_coarse)}")
    print("candidate-edge totals by method family:")
    print(overall_coarse.groupby("method_family")["count"].sum().to_string())


if __name__ == "__main__":
    main()
