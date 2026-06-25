"""Utilities for summarising experiment CSV outputs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def summarize_csv(
    csv_path: str,
    group_fields: Sequence[str] = ("strategy", "amount"),
    metrics: Sequence[str] = ("post_acc_sequence", "post_loss"),
    output_path: str | None = None,
    filters: Dict[str, str] | None = None,
) -> List[Dict[str, float]]:
    """Summarise experiment results grouped by selected columns."""
    groups: Dict[Tuple, List[Dict[str, float]]] = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filters:
                matched = True
                for key, value in filters.items():
                    if str(row.get(key, "")) != str(value):
                        matched = False
                        break
                if not matched:
                    continue
            key = tuple(row.get(field, "") for field in group_fields)
            metric_row = {}
            skip = False
            for metric in metrics:
                value = row.get(metric)
                if value is None or value == "":
                    skip = True
                    break
                try:
                    fv = float(value)
                except ValueError:
                    skip = True
                    break
                if not math.isfinite(fv):
                    skip = True
                    break
                metric_row[metric] = fv
            if skip:
                continue
            groups[key].append(metric_row)

    summaries: List[Dict[str, float]] = []
    for key, rows in groups.items():
        summary: Dict[str, float] = {field: value for field, value in zip(group_fields, key)}
        for metric in metrics:
            values = [r[metric] for r in rows if metric in r]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = math.sqrt(var)
            summary[f"{metric}_count"] = len(values)
        summaries.append(summary)

    if output_path:
        if output_path.endswith(".json"):
            import json

            with open(output_path, "w") as f:
                json.dump(summaries, f, indent=2)
        elif output_path.endswith(".csv"):
            fieldnames = sorted({k for row in summaries for k in row.keys()})
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in summaries:
                    writer.writerow(row)
        else:
            raise ValueError("output_path must end with .csv or .json")
    return summaries


__all__ = ["summarize_csv"]
