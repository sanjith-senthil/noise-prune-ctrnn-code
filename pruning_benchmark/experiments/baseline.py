"""Utilities for training baseline checkpoints prior to pruning sweeps."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .runner import run_prune_experiment
from ..utils import make_run_id


def _slugify(task: str) -> str:
    return task.replace(":", "_").replace("-", "_")


def _merge_config(defaults: Dict, specific: Dict) -> Dict:
    merged = dict(defaults)
    for key, value in specific.items():
        if key in {"task", "seeds", "checkpoint_name"}:
            continue
        merged[key] = value
    return merged


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _record_manifest_entry(
    manifest: Dict[str, Dict],
    checkpoint_path: Path,
    *,
    task: str,
    seed: int,
    run_id: str,
    train_steps: int,
    ft_steps: int,
    status: str,
) -> None:
    digest = _hash_file(checkpoint_path)
    manifest[str(checkpoint_path)] = {
        "task": task,
        "seed": int(seed),
        "run_id": run_id,
        "train_steps": int(train_steps),
        "ft_steps": int(ft_steps),
        "hash": digest,
        "status": status,
        "updated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "path": str(checkpoint_path),
        "abs_path": str(checkpoint_path.resolve()),
    }


def train_baselines(config_path: str, *, overwrite: bool = False) -> List[str]:
    """
    Train (or reuse) baseline models described in a JSON configuration file.

    Returns the list of checkpoint paths produced.
    """
    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    tasks: List[Dict] = cfg.get("tasks", [])
    if not tasks:
        raise ValueError("Baseline config contains no tasks.")

    defaults: Dict = cfg.get("defaults", {})
    out_dir = Path(cfg.get("output_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[str] = []
    manifest_path = out_dir / "baseline_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    total_runs = sum(len(spec.get("seeds", [0])) for spec in tasks)
    completed = 0

    for spec in tasks:
        task_name = spec["task"]
        slug = _slugify(task_name)
        seeds = spec.get("seeds", [0])
        merged = _merge_config(defaults, spec)
        for seed in seeds:
            run_id = spec.get("run_id_prefix", f"baseline_{slug}") + f"_seed{seed}"
            checkpoint_template = spec.get("checkpoint_name", f"{slug}_seed{{seed}}.pt")
            checkpoint_path = out_dir / checkpoint_template.format(seed=seed, task=slug)
            train_steps = int(merged.get("train_steps", 600))
            ft_steps = int(merged.get("ft_steps", 0))
            if checkpoint_path.exists() and not overwrite:
                completed += 1
                print(
                    f"[baseline {completed}/{total_runs}] pre-existing checkpoint for {task_name} seed {seed} -> {checkpoint_path}"
                )
                _record_manifest_entry(
                    manifest,
                    checkpoint_path,
                    task=task_name,
                    seed=seed,
                    run_id=run_id,
                    train_steps=train_steps,
                    ft_steps=ft_steps,
                    status="reused",
                )
                produced.append(str(checkpoint_path))
                continue

            completed += 1
            print(
                f"[baseline {completed}/{total_runs}] training {task_name} seed {seed} (steps={train_steps})"
            )

            run_prune_experiment(
                strategy="none",
                amount=0.0,
                train_steps=train_steps,
                ft_steps=ft_steps,
                last_only=bool(merged.get("last_only", True)),
                seed=int(seed),
                device=merged.get("device", "cpu"),
                movement_batches=int(merged.get("movement_batches", 20)),
                task=task_name,
                no_prune=True,
                run_id=run_id,
                model_type=merged.get("model_type", "ctrnn"),
                hidden_size=merged.get("hidden_size"),
                ng_kwargs=merged.get("ng_kwargs"),
                ng_dataset_kwargs=merged.get("ng_dataset_kwargs"),
                ng_T=merged.get("ng_T"),
                ng_B=merged.get("ng_B"),
                save_model_path=str(checkpoint_path),
            )
            _record_manifest_entry(
                manifest,
                checkpoint_path,
                task=task_name,
                seed=seed,
                run_id=run_id,
                train_steps=train_steps,
                ft_steps=ft_steps,
                status="trained",
            )
            produced.append(str(checkpoint_path))

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return produced


__all__ = ["train_baselines"]
